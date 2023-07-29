import copy
import multiprocessing
import os
import pickle
from collections import OrderedDict
from math import ceil, modf
from queue import Queue

import numpy as np
import psutil
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

    # SymbolClasses = "NLRAV"
    # SymbolClasses = "NLRejAaJSVEFf"
    SymbolClasses = "AaJSVEF"
    SymbolClassNum = len(SymbolClasses)
    SymbolClassToIndex = {name: i for i, name in enumerate(SymbolClasses)}

    # SymbolSuperClasses = OrderedDict(
    #     [("N", "NLRej"), ("SVEB", "AaJS"), ("VEB", "VE"), ("F", "F"), ("Q", "Qf")]
    # )
    SymbolSuperClasses = OrderedDict([("SVEB", "AaJS"), ("VEB", "VE"), ("F", "F")])
    SymbolSuperClassesNum = len(SymbolSuperClasses)
    SymbolSuperClassToIndex = {name: i for i, name in enumerate(SymbolSuperClasses)}

    # SymbolClassToSuperClassIndex = defaultdict(
    #     lambda: MITBIHDataset.SymbolSuperClassToIndex["Q"]
    # )
    SymbolClassToSuperClassIndex = {}

    for i, (s, name) in enumerate(SymbolSuperClasses.items()):
        for n in name:
            SymbolClassToSuperClassIndex[n] = i

    ECGWaves = "PRT"
    ECGWaveToIndex = {name: i for i, name in enumerate(ECGWaves)}
    ECGWaveNum = len(ECGWaves)

    def __init__(
        self,
        data_prefix=None,
        token_size=4,
        data_size=512,
        around_period_num=1,
        multi_label=False,
        wave_num_cls_token=True,
        signal_names=["MLII", "V1", "V2", "V4", "V5"],
        ecg_process_method="dwt",
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = data_size
        self.around_period_num = around_period_num
        self.period_num = around_period_num * 2 + 1
        self.multi_label = multi_label
        self.signal_names = {
            signal_name: i for i, signal_name in enumerate(signal_names)
        }
        self.ecg_process_method = ecg_process_method
        self.wave_num_cls_token = wave_num_cls_token
        self.cls_token_num = self.ECGWaveNum if wave_num_cls_token else 1
        self.total_size = self.cls_token_num + self.data_size

        if data_prefix is None:
            data_prefix = dict(data_path="", ann_path="", cache_path="cache")

        super().__init__(data_prefix=data_prefix, lazy_init=True, **kwargs)

        self.cache_info = OrderedDict()
        self.cache_info["wave_ann"] = {
            "path": self.ecg_process_method,
            "kwargs": {
                "method": self.ecg_process_method,
                "data_path": self.data_prefix["data_path"],
            },
        }
        self.cache_info["wave_ann_filted"] = {
            "path": f"around_period_num_{self.around_period_num}",
            "kwargs": {
                "ann_path": self.data_prefix["ann_path"],
                "wave_ann_path": self.cache_info["wave_ann"]["path"],
                "around_period_num": self.around_period_num,
            },
        }
        self.parse_cache_info()
        self.cache_info["wave_ann_filted"]["kwargs"]["wave_ann_path"] = self.cache_info[
            "wave_ann"
        ]["path"]

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

            if "suffix" not in cache_info:
                cache_info["suffix"] = "pkl"

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
                    os.path.join(info["path"], name) + "." + info["suffix"]
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

        if num_processes:
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
        else:
            for name in tqdm.tqdm(data_list):
                func(name, cache_path, **kwargs)

    @staticmethod
    def cache_wave_ann(name, cache_path, data_path, **kwargs):
        _, wave_ann = load_wave_ann(data_path, name, **kwargs)
        pickle.dump(wave_ann, open(os.path.join(cache_path, name) + ".pkl", "wb"))

    @staticmethod
    def cache_wave_ann_filted(
        name,
        cache_path,
        ann_path,
        wave_ann_path,
        around_period_num,
    ):
        period_num = around_period_num * 2 + 1
        ann = load_ann(ann_path, name)
        wave_ann = pickle.load(open(os.path.join(wave_ann_path, name + ".pkl"), "rb"))
        lead_name = sorted(
            [(lead_name, len(a)) for lead_name, a in wave_ann.items()],
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        wave_ann = wave_ann[lead_name]

        symbol = sorted(
            [[i, symbol] for i, symbol in zip(ann.sample, ann.symbol)],
            key=lambda x: x[0],
        )

        wave_ann_filted = []
        result = Queue(period_num)
        inds = {wave: 0 for wave in MITBIHDataset.ECGWaves}
        for i in range(1, len(symbol) - 1):
            cur_res = {
                "period": [symbol[i - 1][0], symbol[i + 1][0]],
                "symbol": symbol[i][1],
                "peak": symbol[i][0],
                "waves": [],
            }
            for wave_name in MITBIHDataset.ECGWaves:
                while inds[wave_name] < len(wave_ann[f"ECG_{wave_name}_Offsets"]) and (
                    np.isnan(wave_ann[f"ECG_{wave_name}_Offsets"][inds[wave_name]])
                    or wave_ann[f"ECG_{wave_name}_Offsets"][inds[wave_name]]
                    < cur_res["period"][0]
                ):
                    inds[wave_name] += 1

                while inds[wave_name] < len(wave_ann[f"ECG_{wave_name}_Onsets"]):
                    if np.isnan(
                        wave_ann[f"ECG_{wave_name}_Onsets"][inds[wave_name]]
                    ) or np.isnan(
                        wave_ann[f"ECG_{wave_name}_Offsets"][inds[wave_name]]
                    ):
                        inds[wave_name] += 1
                        continue

                    if (
                        wave_ann[f"ECG_{wave_name}_Onsets"][inds[wave_name]]
                        > cur_res["period"][1]
                    ):
                        break
                    elif (
                        wave_ann[f"ECG_{wave_name}_Onsets"][inds[wave_name]]
                        < cur_res["period"][0]
                    ):
                        start = cur_res["period"][0]
                    else:
                        start = wave_ann[f"ECG_{wave_name}_Onsets"][inds[wave_name]]

                    if (
                        wave_ann[f"ECG_{wave_name}_Offsets"][inds[wave_name]]
                        > cur_res["period"][1]
                    ):
                        cur_res["waves"].append(
                            [wave_name, start, cur_res["period"][1]]
                        )
                        break
                    else:
                        end = wave_ann[f"ECG_{wave_name}_Offsets"][inds[wave_name]]
                        inds[wave_name] += 1
                        cur_res["waves"].append([wave_name, start, end])

            result.put(cur_res)

            if result.full():
                wave_ann_filted.append(list(result.queue))
                result.get()

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
        )

        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        signal = signal.reshape(*signal.shape[:-1], -1, self.token_size)

        data_list = []

        for result in results:
            start_token = result[0]["period"][0] // self.token_size
            end_token = ceil(result[-1]["period"][1] / self.token_size) + 1

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

            position_embedding = torch.arange(0, self.total_size, device=signal.device)
            position_embedding = (
                position_embedding - result[self.around_period_num]["peak"]
            )

            wave_embedding = signal.new_zeros((self.total_size, self.ECGWaveNum))

            attention_mask = signal.new_zeros(
                (self.total_size, self.total_size), dtype=torch.bool
            )
            attention_mask[:, self.cls_token_num + data_length :] = True
            # attention_mask[: self.cls_token_num] = True

            for i, res in enumerate(result):
                for wave_name, start, end in res["waves"]:
                    wave_index = self.ECGWaveToIndex[wave_name]
                    start_percent, start = modf(start / self.token_size)
                    end_percent, end = modf(end / self.token_size)
                    start = int(start) - start_token + self.cls_token_num
                    end = int(end) - start_token + self.cls_token_num

                    if start == end:
                        wave_embedding[start, wave_index] = end_percent - start_percent
                    else:
                        wave_embedding[start, wave_index] = 1 - start_percent
                        wave_embedding[start + 1 : end, wave_index] = 1
                        wave_embedding[end, wave_index] = end_percent

                    if i == self.around_period_num:
                        attention_mask[wave_index, start : end + 1] = False

            if self.multi_label:
                target = signal.new_zeros((self.SymbolSuperClassesNum), dtype=torch.int)
                if (
                    result[self.around_period_num]["symbol"]
                    in self.SymbolClassToSuperClassIndex
                ):
                    target[
                        self.SymbolClassToSuperClassIndex[
                            result[self.around_period_num]["symbol"]
                        ]
                    ] = 1
                target_single_class = signal.new_zeros(
                    (self.SymbolClassNum), dtype=torch.int
                )
                if result[self.around_period_num]["symbol"] in self.SymbolClassToIndex:
                    target_single_class[
                        self.SymbolClassToIndex[
                            result[self.around_period_num]["symbol"]
                        ]
                    ] = 1
            else:
                if (
                    result[self.around_period_num]["symbol"]
                    in self.SymbolClassToSuperClassIndex
                ):
                    target = signal.new_full(
                        (),
                        self.SymbolClassToSuperClassIndex[
                            result[self.around_period_num]["symbol"]
                        ],
                        dtype=torch.int,
                    )
                else:
                    continue

                if result[self.around_period_num]["symbol"] in self.SymbolClassToIndex:
                    target_single_class = signal.new_full(
                        (),
                        self.SymbolClassToIndex[
                            result[self.around_period_num]["symbol"]
                        ],
                        dtype=torch.int,
                    )
                else:
                    target_single_class = signal.new_full((), 0, dtype=torch.int)

            if not ((~attention_mask).sum(-1) == 0).any():
                data_list.append(
                    {
                        "name": name,
                        "signal": cur_signal,
                        "attention_mask": attention_mask,
                        "signal_name": record.sig_name,
                        "signal_embedding": signal_embedding,
                        "position_embedding": position_embedding,
                        "wave_embedding": wave_embedding,
                        "target": target,
                        "target_single_class": target_single_class,
                        # "aux_note": cur_aux_note,
                    }
                )

        return data_list
