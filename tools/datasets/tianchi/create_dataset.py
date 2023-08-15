import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


def main():
    data_path = "data/tianchi"

    general_columns = ["id", "age", "gender"]
    disease_columns = ["f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]

    for round in [1, 2, "all"]:
        output_path = os.path.join(data_path, f"ann/round_{round}")
        analyze_path = os.path.join(data_path, f"analyze/round_{round}")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(analyze_path, exist_ok=True)

        round_list = [1, 2] if round == "all" else [round]
        data = [
            pd.read_csv(
                os.path.join(data_path, f"raw/hf_round{r}_label.txt"),
                sep="\t",
                names=general_columns + disease_columns,
            )
            for r in round_list
        ]
        data = pd.concat(data).reset_index(drop=True)
        data["disease"] = 0

        class_names = set()
        for r in round_list:
            class_names |= set(
                [
                    l.strip()
                    for l in open(
                        os.path.join(data_path, f"raw/hf_round{r}_arrythmia.txt")
                    ).readlines()
                    if l
                ]
            )
        class_names = sorted(class_names)  # 读入心电异常事件列表

        with open(os.path.join(output_path, "class_names.txt"), "w") as f:
            f.write("\n".join(class_names))

        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    np.zeros([data.shape[0], len(class_names)], dtype=np.int32),
                    columns=class_names,
                    index=data.index,
                ),
            ],
            axis=1,
        )

        for i in tqdm(range(data.shape[0])):
            dises = set(data.iloc[i, 3:12].unique())
            dises = sorted([d for d in dises if isinstance(d, str)])
            for d in dises:
                data.loc[i, d] = 1
            data.loc[i, "disease"] = "_".join(dises)
        data["id"] = data["id"].apply(lambda x: x.removesuffix(".txt"))
        data = data[general_columns + ["disease"] + class_names]
        data.to_csv(os.path.join(output_path, "all.csv"), index=False)
        data_train, data_val = train_test_split(data, test_size=0.2)
        data_train.to_csv(os.path.join(output_path, "train.csv"), index=False)
        data_val.to_csv(os.path.join(output_path, "val.csv"), index=False)

        for name in ["age", "gender"] + class_names:
            data[name].hist(density=True, histtype="step", label="all")
            data_train[name].hist(density=True, histtype="step", label="train")
            data_val[name].hist(density=True, histtype="step", label="val")
            plt.legend()
            plt.savefig(os.path.join(analyze_path, f"{name}.png"))
            plt.cla()


if __name__ == "__main__":
    main()
