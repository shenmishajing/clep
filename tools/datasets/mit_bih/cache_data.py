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
        "--method", default="dwt", help="method to delineate the ECG sign"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    MITBIHDataset(
        ann_file=args.ann_file,
        data_root=args.data_root,
        ecg_process_method=args.method,
    )


if __name__ == "__main__":
    main()
