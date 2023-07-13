import os
from math import ceil

from mmengine.fileio import list_from_file

from datasets import MITBIHDataset


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


def main():
    MITBIHDataset(
        ann_file="RECORDS",
        data_root="data/mit-bih-arrhythmia-database-1.0.0",
        ecg_process_method="dwt",
    )


if __name__ == "__main__":
    main()
