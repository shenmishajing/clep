import os
import pickle

import numpy as np


def main():
    dataset = "tianchi"
    around_period_num = 0
    token_size = 1

    datasets = {"mit-bih": "mit-bih-arrhythmia-database-1.0.0", "tianchi": "tianchi"}
    data_path = f"data/{datasets[dataset]}/cache/wave_ann_filted_cache/around_period_num_{around_period_num}"

    period_len = []
    for name in os.listdir(data_path):
        data = pickle.load(open(os.path.join(data_path, name), "rb"))
        for d in data:
            period_len.append(
                ((d[-1]["period"][1] - d[0]["period"][0]) / token_size, d, name)
            )

    period_len = sorted(period_len, key=lambda x: x[0], reverse=True)

    period_len = [x[0] for x in period_len]

    print(
        f"period_len: {np.mean(period_len)}, {np.std(period_len)}, {np.max(period_len)}, {np.min(period_len)}"
    )


if __name__ == "__main__":
    main()
