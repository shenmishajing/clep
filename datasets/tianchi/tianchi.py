import json
import os
import pickle
from collections import OrderedDict, deque
from math import ceil, modf

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.utils.ecg_utils import calculate_wave_ann

from ..cache_dataset import CacheDataset


class TianChiDataset(CacheDataset):
    """
    TianChi dataset.
    """

    SignalNames = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    SignalNameToIndex = {name: i for i, name in enumerate(SignalNames)}

    FramePerSencond = 500
    MaxLength = 5000
    Factor = 4.88
    ECGWaves = "PRT"
    ECGWaveToIndex = {name: i for i, name in enumerate(ECGWaves)}
    ECGWaveNum = len(ECGWaves)

    def __init__(
        self,
        data_prefix=None,
        class_names=None,
        token_size=1,
        data_size=None,
        around_period_num=0,
        total_record=True,
        wave_fliter=True,
        ecg_process_method="dwt",
        tokenizer="openlm-research/open_llama_3b",
        description_max_length=768,
        debug_len=None,
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = data_size
        self.total_record = total_record
        self.around_period_num = around_period_num
        self.period_num = (
            None if around_period_num is None else around_period_num * 2 + 1
        )
        self.wave_fliter = wave_fliter
        self.ecg_process_method = ecg_process_method
        self.description_max_length = description_max_length
        self.debug_len = debug_len

        if data_prefix is None:
            data_prefix = dict(
                data_path="ecg",
                cache_path="cache",
                symbol_path="symbols_chatgpt/symbol_description.json",
            )

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
                "wave_ann_path": self.cache_info["wave_ann"]["path"],
                "around_period_num": self.around_period_num,
            },
        }
        self.parse_cache_info()
        self.cache_info["wave_ann_filted"]["kwargs"]["wave_ann_path"] = self.cache_info[
            "wave_ann"
        ]["path"]

        if class_names is None:
            class_names = os.path.join(
                os.path.dirname(self.ann_file), "class_names.txt"
            )

        if isinstance(class_names, str):
            if not os.path.exists(class_names):
                class_names = os.path.join(self.data_root, class_names)
            with open(class_names, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line]
        self.class_names = {name: i for i, name in enumerate(class_names)}
        self.ann = pd.read_csv(self.ann_file)
        self.ann["id"] = self.ann["id"].astype(str)

        self.name_list = self.ann["id"].to_list()
        if self.debug_len is not None:
            self.name_list = self.name_list[: self.debug_len]

        self.ann_id_to_disease_name = {
            i: disease for i, disease in zip(self.ann["id"], self.ann["disease"])
        }

        self.ann_id_to_labels = self.ann[["id"] + class_names]
        self.ann_id_to_labels = self.ann_id_to_labels.set_index("id").to_dict("split")
        self.ann_id_to_labels = {
            ind: torch.tensor(data, dtype=torch.int32)
            for ind, data in zip(
                self.ann_id_to_labels["index"], self.ann_id_to_labels["data"]
            )
        }

        self.symbols = json.load(open(self.data_prefix["symbol_path"], "r"))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.symbols = {
            k: tokenizer(v, return_tensors="pt")["input_ids"][0]
            for k, v in self.symbols.items()
        }
        del tokenizer

        self.full_init()

    def load_data_list(self):
        self.prepare_cache(self.name_list)
        data_list = []
        for name in tqdm(self.name_list, desc="calculate data"):
            data_list.extend(self.calculate_data(name))

        return data_list

    @staticmethod
    def load_ecg_record(data_path, name):
        df = pd.read_csv(os.path.join(data_path, name + ".txt"), sep=" ")
        df["III"] = df["II"] - df["I"]
        df["aVR"] = -(df["II"] + df["I"]) / 2
        df["aVL"] = (df["I"] - df["II"]) / 2
        df["aVF"] = (df["II"] - df["I"]) / 2
        return df[TianChiDataset.SignalNames] * TianChiDataset.Factor

    @staticmethod
    def cache_wave_ann(name, cache_path, data_path, **kwargs):
        data = TianChiDataset.load_ecg_record(data_path, name)
        wave_ann = calculate_wave_ann(
            data.values,
            TianChiDataset.FramePerSencond,
            TianChiDataset.SignalNames,
            **kwargs,
        )
        pickle.dump(wave_ann, open(os.path.join(cache_path, name) + ".pkl", "wb"))

    @staticmethod
    def cache_wave_ann_filted(
        name,
        cache_path,
        wave_ann_path,
        around_period_num,
    ):
        if around_period_num is None:
            return
        period_num = around_period_num * 2 + 1
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

        wave_ann_filted = []
        result = deque(maxlen=period_num)
        for i in range(
            min([len([c for c in b if not np.isnan(c)]) for b in wave_ann.values()])
        ):
            cur_res = {
                "period": [
                    0 if i == 0 else wave_ann["ECG_T_Offsets"][i - 1],
                    TianChiDataset.MaxLength
                    if i == len(wave_ann) - 1
                    else wave_ann["ECG_P_Onsets"][i + 1],
                ],
                "peak": wave_ann["ECG_R_Peaks"][i],
                "waves": [],
            }
            for wave_name in TianChiDataset.ECGWaves:
                cur_res["waves"].append(
                    [
                        wave_name,
                        wave_ann[f"ECG_{wave_name}_Onsets"][i],
                        wave_ann[f"ECG_{wave_name}_Offsets"][i],
                    ]
                )

            if any(
                [
                    np.isnan(x)
                    for x in [
                        *cur_res["period"],
                        cur_res["peak"],
                        *[x[1] for x in cur_res["waves"]],
                        *[x[2] for x in cur_res["waves"]],
                    ]
                ]
            ):
                result.clear()
            else:
                result.append(cur_res)

            if len(result) == result.maxlen:
                wave_ann_filted.append(list(result))

        pickle.dump(
            wave_ann_filted, open(os.path.join(cache_path, name) + ".pkl", "wb")
        )

    def calculate_data(self, name):
        signal = TianChiDataset.load_ecg_record(self.data_prefix["data_path"], name)
        signal = torch.tensor(signal.values, dtype=torch.float32).T
        signal = signal.reshape(*signal.shape[:-1], -1, self.token_size)

        signal_embedding = signal.new_zeros((signal.shape[0],), dtype=torch.long)
        for sig_name, sig_index in TianChiDataset.SignalNameToIndex.items():
            signal_embedding[sig_index] = sig_index

        results = pickle.load(
            open(
                os.path.join(self.cache_info["wave_ann_filted"]["path"], name + ".pkl"),
                "rb",
            )
        )

        description = self.symbols[self.ann_id_to_labels[name]]
        description_attention_mask = description.new_ones(
            (self.description_max_length,), dtype=torch.int32
        )
        description_attention_mask[description.shape[0] :] = 0
        description = torch.cat(
            [
                description,
                description.new_zeros(
                    self.description_max_length - description.shape[0]
                ),
            ]
        )
        data_list = []

        if self.total_record:
            if self.data_size is None:
                attention_mask = signal.new_zeros((signal.shape[1],), dtype=torch.bool)
            elif signal.shape[1] >= self.data_size:
                return []
            else:
                attention_mask = signal.new_zeros((self.data_size,), dtype=torch.bool)
                attention_mask[signal.shape[1] :] = True

                signal = torch.cat(
                    [
                        signal,
                        signal.new_zeros(
                            (
                                signal.shape[0],
                                self.data_size - signal.shape[1],
                                *signal.shape[2:],
                            )
                        ),
                    ],
                    dim=1,
                )

            peak_position = signal.new_full((), 0)

            wave_embedding = signal.new_zeros((signal.shape[1], self.ECGWaveNum))
            for result in results:
                for res in result:
                    for wave_name, start, end in res["waves"]:
                        wave_index = self.ECGWaveToIndex[wave_name]
                        start_percent, start = modf(start / self.token_size)
                        end_percent, end = modf(end / self.token_size)
                        start = int(start)
                        end = int(end)

                        if start == end:
                            wave_embedding[start, wave_index] = (
                                end_percent - start_percent
                            )
                        else:
                            wave_embedding[start, wave_index] = 1 - start_percent
                            wave_embedding[start + 1 : end, wave_index] = 1
                            wave_embedding[end, wave_index] = end_percent

            if self.wave_fliter and (wave_embedding.sum(0) == 0).any():
                return []

            data_list.append(
                {
                    "name": name,
                    "signal": signal,
                    "attention_mask": attention_mask,
                    "signal_name": TianChiDataset.SignalNames,
                    "signal_embedding": signal_embedding,
                    "peak_position": peak_position,
                    "wave_embedding": wave_embedding,
                    "target": self.ann_id_to_labels[name],
                    "description": description,
                    "description_attention_mask": description_attention_mask,
                }
            )
        else:
            for result in results:
                start_token = result[0]["period"][0] // self.token_size
                end_token = ceil(result[-1]["period"][1] / self.token_size) + 1

                if end_token - start_token > self.data_size:
                    continue

                cur_signal = signal[:, start_token:end_token, ...]

                attention_mask = signal.new_zeros((self.data_size,), dtype=torch.bool)
                attention_mask[cur_signal.shape[1] :] = True

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

                peak_position = signal.new_full(
                    (), result[self.around_period_num]["peak"] - result[0]["period"][0]
                )

                wave_embedding = signal.new_zeros((self.data_size, self.ECGWaveNum))
                for res in result:
                    for wave_name, start, end in res["waves"]:
                        wave_index = self.ECGWaveToIndex[wave_name]
                        start_percent, start = modf(start / self.token_size)
                        end_percent, end = modf(end / self.token_size)
                        start = int(start) - start_token
                        end = int(end) - start_token

                        if start == end:
                            wave_embedding[start, wave_index] = (
                                end_percent - start_percent
                            )
                        else:
                            wave_embedding[start, wave_index] = 1 - start_percent
                            wave_embedding[start + 1 : end, wave_index] = 1
                            wave_embedding[end, wave_index] = end_percent

                if self.wave_fliter and (wave_embedding.sum(0) == 0).any():
                    continue

                data_list.append(
                    {
                        "name": name,
                        "signal": cur_signal,
                        "attention_mask": attention_mask,
                        "signal_name": TianChiDataset.SignalNames,
                        "signal_embedding": signal_embedding,
                        "peak_position": peak_position,
                        "wave_embedding": wave_embedding,
                        "target": self.ann_id_to_labels[name],
                        "description_input": self.symbols[name],
                        "description_attention_mask": description_attention_mask,
                    }
                )

        return data_list
