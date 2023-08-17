import json

from transformers import AutoTokenizer


def main():
    tokenizer = "Deci/DeciCoder-1b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    data = json.load(open("data/tianchi/symbols_chatgpt/symbol_description.json"))

    data = [
        (k, len(tokenizer(v, return_tensors="pt")["input_ids"][0]))
        for k, v in data.items()
    ]

    data = sorted(data, key=lambda x: x[1], reverse=True)
    print(data)


if __name__ == "__main__":
    main()
