import copy
import multiprocessing
import os
import pickle
from math import ceil, isnan, modf

import numpy as np
import torch
import tqdm
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .ecg_utils import load_ann, load_record, load_wave_ann


class MITBIHDataset(BaseDataset):
    """
    MIT-BIH Arrhythmia dataset.
    """

    def __init__(
        self,
        data_prefix=None,
        token_size=4,
        data_size=32500,
        signal_names=["MLII", "V1", "V2", "V4", "V5"],
        symbol_names=["N", "L", "R", "V", "A"],
        ecg_process_method="dwt",
        ecg_wave_kinds="PRT",
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = ceil(data_size / token_size)
        self.signal_names = {
            signal_name: i for i, signal_name in enumerate(signal_names)
        }
        self.symbol_names = {
            symbol_name: i for i, symbol_name in enumerate(symbol_names)
        }
        self.ecg_process_method = ecg_process_method
        self.ecg_wave_kinds = ecg_wave_kinds

        if data_prefix is None:
            data_prefix = dict(data_path="", ann_path="", cache_path="cache")
        data_prefix["cache_path"] = f"{ecg_process_method}_{data_prefix['cache_path']}"
        super().__init__(data_prefix=data_prefix, **kwargs)

    def _join_prefix(self):
        super()._join_prefix()
        for p in self.data_prefix.values():
            os.makedirs(p, exist_ok=True)

    def load_data_list(self):
        name_list = list_from_file(self.ann_file)
        uncached_list = []

        for name in name_list:
            if not os.path.exists(
                os.path.join(self.data_prefix["cache_path"], name + ".pkl")
            ):
                uncached_list.append(name)

        if uncached_list:
            self.cache_wave_ann(
                uncached_list,
                self.data_prefix["data_path"],
                self.data_prefix["cache_path"],
            )

        data_list = []
        for name in name_list:
            data_list.extend(self.calculate_data(name))

        return data_list

    @staticmethod
    def collate_fn(batch):
        collate_fn_map = copy.copy(default_collate_fn_map)
        collate_fn_map[list] = lambda batch, collate_fn_map: batch

        return collate(batch, collate_fn_map=collate_fn_map)

    @staticmethod
    def cache_wave_ann(data_list, data_path, cache_path, num_processes=2, **kwargs):
        pool = multiprocessing.Pool(num_processes)
        bar = tqdm.tqdm(total=len(data_list))

        data_list = [
            [
                name,
                pool.apply_async(
                    load_wave_ann,
                    args=(data_path, name),
                    kwds=kwargs,
                    callback=lambda *args, **kwargs: bar.update(1),
                ),
            ]
            for name in data_list
        ]

        for name, task in data_list:
            _, wave_ann = task.get()
            pickle.dump(
                wave_ann,
                open(
                    os.path.join(cache_path, name) + ".pkl",
                    "wb",
                ),
            )

        pool.close()
        pool.join()

    def calculate_data(self, name):
        record = load_record(self.data_prefix["data_path"], name)
        ann = load_ann(self.data_prefix["ann_path"], name)
        wave_ann = pickle.load(
            open(
                os.path.join(self.data_prefix["cache_path"], name + ".pkl"),
                "rb",
            )
        )

        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        signal = signal.reshape(*signal.shape[:-1], -1, self.token_size)

        signal_embedding = signal.new_zeros((signal.shape[0],), dtype=torch.long)
        wave_embedding = signal.new_zeros((*signal.shape[:2], len(self.ecg_wave_kinds)))
        for sig_index, sig_name in enumerate(record.sig_name):
            signal_embedding[sig_index] = self.signal_names[sig_name]
            for wave_index, wave_name in enumerate(self.ecg_wave_kinds):
                on_sets = (
                    np.array(wave_ann[sig_name][f"ECG_{wave_name}_Onsets"])
                    / self.token_size
                )
                off_sets = (
                    np.array(wave_ann[sig_name][f"ECG_{wave_name}_Offsets"])
                    / self.token_size
                )
                for i in range(len(on_sets)):
                    if not isnan(on_sets[i]) and not isnan(off_sets[i]):
                        start_percent, start_token = modf(on_sets[i])
                        end_percent, end_token = modf(off_sets[i])
                        start_token = int(start_token)
                        end_token = int(end_token)

                        if start_token == end_token:
                            wave_embedding[sig_index, start_token, wave_index] = (
                                end_percent - start_percent
                            )
                        else:
                            wave_embedding[sig_index, start_token, wave_index] = (
                                1 - start_percent
                            )
                            wave_embedding[
                                sig_index,
                                start_token + 1 : end_token,
                                wave_index,
                            ] = 1
                            wave_embedding[
                                sig_index, end_token, wave_index
                            ] = end_percent

        symbol_target = signal.new_full((1, signal.shape[1]), -1, dtype=torch.long)

        for i, symbol in zip(ann.sample, ann.symbol):
            if symbol in self.symbol_names:
                symbol_target[0, int(i / self.token_size)] = self.symbol_names[symbol]

        aux_note = {
            int(i / self.token_size): aux_note.rstrip("\x00")
            for i, aux_note in zip(ann.sample, ann.aux_note)
            if aux_note
        }

        data_list = []
        end_data_index = min(self.data_size, signal.shape[1])
        while end_data_index <= signal.shape[1]:
            start_data_index = max(0, end_data_index - self.data_size)

            cur_aux_note = [
                (key - start_data_index, value)
                for key, value in aux_note.items()
                if start_data_index <= key < end_data_index
            ]

            cur_symbol_target = symbol_target[:, start_data_index:end_data_index]

            if (cur_symbol_target >= 0).sum() > 0:
                data_list.append(
                    {
                        "name": name,
                        "signal": signal[:, start_data_index:end_data_index, ...],
                        "signal_name": record.sig_name,
                        "signal_embedding": signal_embedding,
                        "wave_embedding": wave_embedding[
                            :, start_data_index:end_data_index, ...
                        ],
                        "symbol_target": symbol_target[
                            :, start_data_index:end_data_index
                        ],
                        "aux_note": cur_aux_note,
                    }
                )

            end_data_index += ceil(self.data_size / 2)

        return data_list
