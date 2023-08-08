import pandas as pd


def main():
    data_path = "data/tianchi/ann/round_all/all.csv"
    data = pd.read_csv(data_path)
    num = {}
    for name in data:
        if name in ["id", "age", "gender"]:
            continue
        num[name] = data[name].sum()
    for name in num:
        num[name] = num[name] / len(data)
    num = sorted(num.items(), key=lambda x: x[1], reverse=True)
    for k, v in num:
        print(k, v, sep="\t\t")

    num = sorted([n[0] for n in num if n[1] > 0.01])
    for k in num:
        print(k)


if __name__ == "__main__":
    main()
