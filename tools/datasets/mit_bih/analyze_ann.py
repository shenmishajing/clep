import os
from collections import defaultdict

from datasets.mit_bih.ecg_utils import load_ann


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    data_list = open(os.path.join(data_path, "RECORDS"), "r").readlines()
    data_list = [x.strip() for x in data_list]

    symbols = defaultdict(int)
    aux_note = defaultdict(int)
    for file in data_list:
        ann = load_ann(data_path, file)

        for symbol in ann.symbol:
            symbols[symbol] += 1

        for note in ann.aux_note:
            aux_note[note] += 1

    print(f"symbols: {symbols}, len: {len(symbols)}")
    print(f"aux_note: {aux_note}, len: {len(aux_note)}")


if __name__ == "__main__":
    main()
