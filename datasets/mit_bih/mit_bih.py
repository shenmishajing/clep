import copy
import multiprocessing
import os
import pickle
from collections import OrderedDict, defaultdict
from math import ceil, modf

import psutil
import torch
import tqdm
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .ecg_utils import load_ann, load_record, load_wave_ann
from .wave_utils import find_index


class MITBIHDataset(BaseDataset):
    """
    MIT-BIH Arrhythmia dataset.
    """

    def __init__(
        self,
        data_prefix=None,
        token_size=4,
        data_size=512,
        around_period_num=1,
        signal_names=["MLII", "V1", "V2", "V4", "V5"],
        symbol_names=["N", "L", "R", "V", "A"],
        ecg_process_method="dwt",
        ecg_wave_kinds="PRT",
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = data_size
        self.around_period_num = around_period_num
        self.period_num = around_period_num * 2 + 1
        self.signal_names = {
            signal_name: i for i, signal_name in enumerate(signal_names)
        }
        self.symbol_names = {
            symbol_name: i for i, symbol_name in enumerate(symbol_names)
        }
        self.ecg_process_method = ecg_process_method
        self.ecg_wave_kinds = ecg_wave_kinds
        self.cls_token_size = len(self.ecg_wave_kinds)
        self.total_size = self.cls_token_size + self.data_size

        if data_prefix is None:
            data_prefix = dict(data_path="", ann_path="", cache_path="cache")

        super().__init__(data_prefix=data_prefix, lazy_init=True, **kwargs)

        self.cache_info = OrderedDict()
        self.cache_info["wave_ann"] = {
            "path": self.ecg_process_method,
            "suffix": ".pkl",
            "kwargs": {
                "method": self.ecg_process_method,
                "data_path": self.data_prefix["data_path"],
            },
        }
        self.parse_cache_info()

        self.cache_info["wave_ann_filted"] = {
            "path": f"around_period_num_{self.around_period_num}",
            "suffix": ".pkl",
            "kwargs": {
                "ann_path": self.data_prefix["ann_path"],
                "wave_ann_path": self.cache_info["wave_ann"]["path"],
                "symbol_names": self.symbol_names,
                "around_period_num": self.around_period_num,
            },
        }
        self.parse_cache_info()

        self.full_init()

    def parse_cache_info(self):
        for cache_name, cache_info in self.cache_info.items():
            cache_path = os.path.join(
                self.data_prefix["cache_path"],
                cache_name + "_cache",
            )
            if "path" in cache_info:
                cache_info["path"] = os.path.join(cache_path, cache_info["path"])
            else:
                cache_info["path"] = cache_path

            if "func" not in cache_info:
                cache_info["func"] = self.cache_func

                if "kwargs" not in cache_info:
                    cache_info["kwargs"] = {}

                if "func" not in cache_info["kwargs"] and hasattr(
                    self, "cache_" + cache_name
                ):
                    cache_info["kwargs"]["func"] = getattr(self, "cache_" + cache_name)

            if "uncached" not in cache_info:
                cache_info["uncached"] = []

    @staticmethod
    def collate_fn(batch):
        collate_fn_map = copy.copy(default_collate_fn_map)
        collate_fn_map[list] = lambda batch, collate_fn_map: batch

        return collate(batch, collate_fn_map=collate_fn_map)

    def load_data_list(self):
        name_list = list_from_file(self.ann_file)

        self.prepare_cache(name_list)
        data_list = []
        for name in name_list:
            data_list.extend(self.calculate_data(name))

        return data_list

    def prepare_cache(self, name_list):
        for name in name_list:
            for info in self.cache_info.values():
                if not os.path.exists(
                    os.path.join(info["path"], name) + info["suffix"]
                ):
                    info["uncached"].append(name)

        for info in self.cache_info.values():
            if info["uncached"]:
                info["func"](
                    info["uncached"],
                    cache_path=info["path"],
                    **info["kwargs"],
                )

                info["uncached"] = []

    def cache_func(self, data_list, cache_path, func, num_processes=None, **kwargs):
        os.makedirs(cache_path, exist_ok=True)

        if num_processes is None:
            num_processes = psutil.cpu_count(False)

        num_processes = min(num_processes, len(data_list))

        pool = multiprocessing.Pool(num_processes)
        bar = tqdm.tqdm(total=len(data_list))

        for name in data_list:
            pool.apply_async(
                func,
                args=(name, cache_path),
                kwds=kwargs,
                callback=lambda *args, **kwargs: bar.update(1),
            )

        pool.close()
        pool.join()

    @staticmethod
    def cache_wave_ann(name, cache_path, data_path, **kwargs):
        _, wave_ann = load_wave_ann(data_path, name, **kwargs)
        pickle.dump(
            wave_ann,
            open(
                os.path.join(cache_path, name) + ".pkl",
                "wb",
            ),
        )

    @staticmethod
    def cache_wave_ann_filted(
        name, cache_path, ann_path, wave_ann_path, symbol_names, around_period_num
    ):
        ann = load_ann(ann_path, name)
        wave_ann = pickle.load(
            open(
                os.path.join(wave_ann_path, name + ".pkl"),
                "rb",
            )
        )

        symbol = {
            i: symbol
            for i, symbol in zip(ann.sample, ann.symbol)
            if symbol in symbol_names
        }

        wave_ann_filted = defaultdict(list)
        for cur_name, cur_ann in wave_ann.items():
            for wave_T_index in range(len(cur_ann["ECG_T_Offsets"])):
                result = find_index(cur_ann, wave_T_index, 2 * around_period_num + 1)
                if not result:
                    continue

                cur_symbol = [
                    i
                    for i in symbol
                    if result[around_period_num]["period_start"]
                    <= i
                    <= result[around_period_num]["period_end"]
                ]
                if len(cur_symbol) != 1:
                    continue
                cur_symbol = symbol[cur_symbol[0]]

                wave_ann_filted[cur_name].append(
                    {
                        "symbol": cur_symbol,
                        "res": result,
                    }
                )

        wave_ann_filted = sorted(
            [(res, name, len(res)) for name, res in wave_ann_filted.items()],
            key=lambda x: x[-1],
            reverse=True,
        )

        pickle.dump(
            wave_ann_filted, open(os.path.join(cache_path, name) + ".pkl", "wb")
        )

    def calculate_data(self, name):
        record = load_record(self.data_prefix["data_path"], name)
        results = pickle.load(
            open(
                os.path.join(self.cache_info["wave_ann_filted"]["path"], name + ".pkl"),
                "rb",
            )
        )[0][0]

        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        signal = signal.reshape(*signal.shape[:-1], -1, self.token_size)

        data_list = []

        for result in results:
            start_token = result["res"][0]["period_start"] // self.token_size
            end_token = ceil(result["res"][-1]["period_end"] / self.token_size) + 1

            if end_token - start_token > self.data_size:
                continue

            cur_signal = signal[:, start_token:end_token, ...]
            data_length = cur_signal.shape[1]
            cur_signal = torch.cat(
                [
                    cur_signal,
                    signal.new_zeros(
                        (
                            cur_signal.shape[0],
                            self.data_size - cur_signal.shape[1],
                            *cur_signal.shape[2:],
                        )
                    ),
                ],
                dim=1,
            )

            signal_embedding = signal.new_zeros((signal.shape[0],), dtype=torch.long)
            for sig_index, sig_name in enumerate(record.sig_name):
                signal_embedding[sig_index] = self.signal_names[sig_name]

            wave_embedding = signal.new_zeros((self.total_size, self.cls_token_size))

            attention_mask = signal.new_zeros(
                (self.total_size, self.total_size), dtype=torch.bool
            )
            attention_mask[:, self.cls_token_size + data_length :] = True
            attention_mask[: self.cls_token_size] = True

            for res in result["res"]:
                for wave_index, wave_name in enumerate(self.ecg_wave_kinds):
                    start_percent, start = modf(
                        res[f"ECG_{wave_name}_Onsets"] / self.token_size
                    )
                    end_percent, end = modf(
                        res[f"ECG_{wave_name}_Offsets"] / self.token_size
                    )
                    start = int(start) - start_token + 1
                    end = int(end) - start_token + 1

                    if start == end:
                        wave_embedding[start, wave_index] = end_percent - start_percent
                    else:
                        wave_embedding[start, wave_index] = 1 - start_percent
                        wave_embedding[start + 1 : end, wave_index] = 1
                        wave_embedding[end, wave_index] = end_percent

                    attention_mask[wave_index, start : end + 1] = False

            data_list.append(
                {
                    "name": name,
                    "signal": cur_signal,
                    "attention_mask": attention_mask,
                    "signal_name": record.sig_name,
                    "signal_embedding": signal_embedding,
                    "wave_embedding": wave_embedding,
                    "symbol_target": self.symbol_names[result["symbol"]],
                    # "aux_note": cur_aux_note,
                }
            )

        return data_list
