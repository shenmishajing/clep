import math
import os
import random


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    data = open(os.path.join(data_path, "RECORDS"), "r").readlines()
    data = [x.strip() for x in data]

    random.shuffle(data)

    train_num = math.ceil(len(data) * 0.8)

    with open(os.path.join(data_path, "train.txt"), "w") as f:
        f.write("\n".join(sorted(data[:train_num])))

    with open(os.path.join(data_path, "val.txt"), "w") as f:
        f.write("\n".join(sorted(data[train_num:])))


if __name__ == "__main__":
    main()
