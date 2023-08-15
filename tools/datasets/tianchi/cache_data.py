from datasets import TianChiDataset


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MIT-BIH Arrhythmia dataset.")
    parser.add_argument(
        "--data-root",
        default="data/tianchi",
        help="dataset root directory",
    )
    parser.add_argument(
        "--ann-file",
        default="ann/round_all/all.csv",
        help="annotation file",
    )
    parser.add_argument(
        "--method", default="dwt", help="method to delineate the ECG sign"
    )
    parser.add_argument("--debug-len", default=None, help="debug max length")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    TianChiDataset(
        ann_file=args.ann_file,
        data_root=args.data_root,
        ecg_process_method=args.method,
        debug_len=args.debug_len,
    ).full_init()


if __name__ == "__main__":
    main()
