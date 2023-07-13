import multiprocessing
import os
import pickle
from math import ceil, isnan, modf

import numpy as np
import torch
import tqdm
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file

from datasets.ecg_utils import load_ann, load_record, load_wave_ann


class MITBIHDataset(BaseDataset):
    """
    MIT-BIH Arrhythmia dataset.
    """

    def __init__(
        self,
        data_prefix=None,
        token_size=4,
        data_size=32500,
        ecg_process_method="dwt",
        ecg_wave_kinds="PRT",
        **kwargs,
    ):
        self.token_size = token_size
        self.data_size = ceil(data_size / token_size)
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
            record = load_record(self.data_prefix["data_path"], name)
            ann = load_ann(self.data_prefix["ann_path"], name)
            wave_ann = pickle.load(
                open(
                    os.path.join(self.data_prefix["cache_path"], name + ".pkl"),
                    "rb",
                )
            )

            signal = torch.tensor(record.p_signal).T
            signal = signal.reshape(*signal.shape[:-1], -1, self.token_size)

            wave_embedding = signal.new_zeros(
                (*signal.shape[:2], len(self.ecg_wave_kinds))
            )
            for sig_index, sig_name in enumerate(record.sig_name):
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

            symbols = {
                int(i / self.token_size): symbol
                for i, symbol in enumerate(ann.symbol)
                if symbol not in "N"
            }
            aux_note = {
                int(i / self.token_size): aux_note
                for i, aux_note in enumerate(ann.aux_note)
                if aux_note
            }

            start_data_index = 0
            while start_data_index < signal.shape[1]:
                end_data_index = min(start_data_index + self.data_size, signal.shape[1])

                data_list.append(
                    {
                        "name": name,
                        "signal": signal[:, start_data_index:end_data_index, ...],
                        "signal_name": record.sig_name,
                        "wave_embedding": wave_embedding[
                            :, start_data_index:end_data_index, ...
                        ],
                        "symbols": {
                            key - start_data_index: value
                            for key, value in symbols.items()
                            if start_data_index <= key < end_data_index
                        },
                        "aux_note": {
                            key - start_data_index: value
                            for key, value in aux_note.items()
                            if start_data_index <= key < end_data_index
                        },
                    }
                )

                start_data_index += ceil(self.data_size / 2)

        return data_list

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


def main():
    MITBIHDataset(
        ann_file="RECORDS",
        data_root="data/mit-bih-arrhythmia-database-1.0.0",
        ecg_process_method="dwt",
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MIT-BIH Arrhythmia dataset.")
    parser.add_argument(
        "--data-root",
        default="data/mit-bih-arrhythmia-database-1.0.0",
        help="dataset root directory",
    )
    parser.add_argument(
        "--ann-file",
        default="RECORDS",
        help="annotation file",
    )
    parser.add_argument(
        "--data-path",
        default="",
        help="data path",
    )
    parser.add_argument(
        "--cache-path",
        default="cache",
        help="cache path",
    )
    parser.add_argument(
        "--method", default="dwt", help="method to delineate the ECG sign"
    )
    parser.add_argument(
        "--num-processes",
        default=2,
        type=int,
        help="number of processes",
    )
    parser.add_argument("--size", default=2, type=int, help="size of data to cache")
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="index of data to cache",
    )

    args = parser.parse_args()

    args.ann_file = os.path.join(args.data_root, args.ann_file)
    args.data_path = os.path.join(args.data_root, args.data_path)
    args.cache_path = os.path.join(args.data_root, f"{args.method}_{args.cache_path}")

    os.makedirs(args.cache_path, exist_ok=True)

    return args


def cache_wave_ann_per_split():
    args = parse_args()

    data_list = list_from_file(args.ann_file)
    print(f"split: {args.index}/{ceil(len(data_list)/args.size)-1}")
    data_list = data_list[
        args.index * args.size : min((args.index + 1) * args.size, len(data_list))
    ]

    MITBIHDataset.cache_wave_ann(
        data_list,
        data_path=args.data_path,
        cache_path=args.cache_path,
        num_processes=args.num_processes,
        method=args.method,
    )


if __name__ == "__main__":
    main()
