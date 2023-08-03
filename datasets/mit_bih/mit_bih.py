import os
import pickle
from collections import OrderedDict
from math import ceil, modf
from queue import Queue

import numpy as np
import torch
from mmengine.fileio import list_from_file

from ..cache_dataset import CacheDataset
from ..utils.ecg_utils import load_ann, load_record, load_wave_ann


class MITBIHDataset(CacheDataset):
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

    MaxLength = 650000
    ECGWaves = "PRT"
    ECGWaveToIndex = {name: i for i, name in enumerate(ECGWaves)}
    ECGWaveNum = len(ECGWaves)

    def __init__(
        self,
        data_prefix=None,
        token_size=1,
        data_size=1024,
        around_period_num=0,
        wave_fliter=True,
        multi_label=False,
        signal_names=["MLII", "V1", "V2", "V4", "V5"],
        ecg_process_method="dwt",
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = data_size
        self.around_period_num = around_period_num
        self.period_num = around_period_num * 2 + 1
        self.wave_fliter = wave_fliter
        self.multi_label = multi_label
        self.signal_names = {
            signal_name: i for i, signal_name in enumerate(signal_names)
        }
        self.ecg_process_method = ecg_process_method

        if data_prefix is None:
            data_prefix = dict(data_path="", ann_path="", cache_path="cache")

        super().__init__(data_prefix=data_prefix, **kwargs)

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

    def load_data_list(self):
        name_list = list_from_file(self.ann_file)

        self.prepare_cache(name_list)
        data_list = []
        for name in name_list:
            data_list.extend(self.calculate_data(name))

        return data_list

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
            [
                (
                    lead_name,
                    0
                    if a is None
                    else min(
                        [len([c for c in b if not np.isnan(c)]) for b in a.values()]
                    ),
                )
                for lead_name, a in wave_ann.items()
            ],
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
        for i in range(len(symbol)):
            cur_res = {
                "period": [
                    0 if i == 0 else symbol[i - 1][0],
                    MITBIHDataset.MaxLength
                    if i == len(symbol) - 1
                    else symbol[i + 1][0],
                ],
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

            wave_embedding = signal.new_zeros((self.data_size, self.ECGWaveNum))
            for res in result:
                for wave_name, start, end in res["waves"]:
                    wave_index = self.ECGWaveToIndex[wave_name]
                    start_percent, start = modf(start / self.token_size)
                    end_percent, end = modf(end / self.token_size)
                    start = int(start) - start_token
                    end = int(end) - start_token

                    if start == end:
                        wave_embedding[start, wave_index] = end_percent - start_percent
                    else:
                        wave_embedding[start, wave_index] = 1 - start_percent
                        wave_embedding[start + 1 : end, wave_index] = 1
                        wave_embedding[end, wave_index] = end_percent

            if self.wave_fliter and (wave_embedding.sum(0) == 0).any():
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

            peak_position = signal.new_full(
                (), result[self.around_period_num]["peak"] - result[0]["period"][0]
            )

            attention_mask = signal.new_zeros((self.data_size,), dtype=torch.bool)
            attention_mask[data_length:] = True

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

            data_list.append(
                {
                    "name": name,
                    "signal": cur_signal,
                    "attention_mask": attention_mask,
                    "signal_name": record.sig_name,
                    "signal_embedding": signal_embedding,
                    "peak_position": peak_position,
                    "wave_embedding": wave_embedding,
                    "target": target,
                    "target_single_class": target_single_class,
                    # "aux_note": cur_aux_note,
                }
            )

        return data_list
