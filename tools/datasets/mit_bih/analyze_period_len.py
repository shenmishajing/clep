import os
import pickle
from math import ceil

import numpy as np

from datasets.mit_bih.ecg_utils import load_ann
from datasets.mit_bih.wave_utils import find_index


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    around_period_num = 1
    period_num = around_period_num * 2 + 1
    token_size = 4
    period_len = []
    for name in open(os.path.join(data_path, "RECORDS"), "r").readlines():
        name = name.strip()
        ann = load_ann(data_path, name)
        wave_ann = pickle.load(
            open(
                os.path.join(data_path, "cache/wave_ann_cache/dwt", name) + ".pkl", "rb"
            )
        )

        symbol = {i: symbol for i, symbol in zip(ann.sample, ann.symbol)}

        for cur_name, cur_ann in wave_ann.items():
            for wave_T_index in range(len(cur_ann["ECG_T_Offsets"])):
                cur_res = find_index(cur_ann, wave_T_index, period_num)

                if cur_res:
                    assert len(cur_res) == period_num
                    start_token = cur_res[0]["period_start"] // token_size
                    end_token = ceil(cur_res[-1]["period_end"] / token_size)

                    cur_symbol = [
                        i
                        for i in symbol
                        if cur_res[around_period_num]["period_start"]
                        <= i
                        <= cur_res[around_period_num]["period_end"]
                    ]
                    if len(cur_symbol) != 1:
                        continue
                    cur_symbol = symbol[cur_symbol[0]]
                    if cur_symbol not in ["N", "L", "R", "V", "A"]:
                        continue

                    period_len.append(
                        (end_token - start_token + 1, cur_res, name, cur_name)
                    )

    period_len = sorted(period_len, key=lambda x: x[0], reverse=True)

    period_len = [x[0] for x in period_len]

    print(
        f"period_len: {np.mean(period_len)}, {np.std(period_len)}, {np.max(period_len)}, {np.min(period_len)}"
    )


if __name__ == "__main__":
    main()
