import json
import os
import pickle
from time import sleep

import torch
from openai import InvalidRequestError

from utils import openai


def get_llm_results(messages, model="gpt-3.5-turbo", temperature=0):
    try_num = 0
    while True:
        try:
            return openai.ChatCompletion.create(
                messages=messages, model=model, temperature=temperature
            )
        except InvalidRequestError as e:
            if e.code == "context_length_exceeded":
                if model == "gpt-3.5-turbo":
                    model = "gpt-3.5-turbo-16k"
                    continue
                else:
                    print(f"context length exceeded, skip, error: {e}")
            else:
                raise e
        except Exception as e:
            try_num += 1
            print(f"error: {e}, try num: {try_num}, retry after {try_num+10} min")
            sleep((try_num + 10) * 60)


def get_symbol_raw_description(data_root, leads, symbols):
    if not os.path.exists(os.path.join(data_root, "symbol_description_raw.pkl")):
        data = {}
        for lead in leads:
            data[lead] = {}
            for symbol, disease in symbols.items():
                print(f"description lead: {lead}, symbol: {symbol}, disease: {disease}")
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"In ecg, what does a {disease} look like in lead {lead}? Describe it in P wave, QRS wave and T wave respectively, starting with '<wave_name> wave: '. For example, describe P wave starting with P wave: . If there is no special characteristic for some wave in such a lead, just say None.",
                    },
                ]
                data[lead][symbol] = get_llm_results(messages)

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_description_raw.pkl"), "wb")
        )


def get_symbol_description(data_root):
    data = pickle.load(
        open(os.path.join(data_root, "symbol_description_raw.pkl"), "rb")
    )
    for lead in data:
        for symbol in data[lead]:
            data[lead][symbol] = data[lead][symbol]["choices"][0]["message"]["content"]

    json.dump(
        data,
        open(os.path.join(data_root, "symbol_description.json"), "w"),
        indent=4,
    )


def get_symbol_sentence(data_root):
    data = json.load(open(os.path.join(data_root, "symbol_description.json")))
    for lead in data:
        for symbol in data[lead]:
            res = {}
            sentences = data[lead][symbol].split("\n")
            for wave in ["P", "QRS", "T"]:
                for sentence in sentences:
                    ind = sentence.find(f"{wave} wave: ")
                    if ind != -1:
                        res[wave] = sentence[ind + len(f"{wave} wave: ") :]
                        break
                if wave in res and "None" in res[wave]:
                    res.pop(wave)
            if "QRS" in res:
                res["R"] = res.pop("QRS")
            data[lead][symbol] = res

    json.dump(
        data,
        open(os.path.join(data_root, "symbol_sentence.json"), "w"),
        indent=4,
    )


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try_num = 0
    while True:
        try:
            return openai.Embedding.create(input=[text], model=model)
        except Exception as e:
            try_num += 1
            print(f"error: {e}, try num: {try_num}, retry after {try_num+10} min")
            sleep((try_num + 10) * 60)


def get_symbol_raw_embedding(data_root):
    if not os.path.exists(os.path.join(data_root, "symbol_embedding_raw.pkl")):
        data = json.load(open(os.path.join(data_root, "symbol_sentence.json")))
        for lead in data:
            for symbol in data[lead]:
                for wave in data[lead][symbol]:
                    print(f"embedding lead: {lead}, symbol: {symbol}, wave: {wave}")
                    data[lead][symbol][wave] = get_embedding(data[lead][symbol][wave])

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "wb")
        )


def get_symbol_embedding(data_root):
    data = pickle.load(open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "rb"))

    for lead in data:
        for symbol in data[lead]:
            for wave in data[lead][symbol]:
                data[lead][symbol][wave] = torch.tensor(
                    data[lead][symbol][wave]["data"][0]["embedding"]
                )

    pickle.dump(data, open(os.path.join(data_root, "symbol_embedding.pkl"), "wb"))


def main():
    data_root = "data/mit-bih-arrhythmia-database-1.0.0/symbols_chatgpt"
    leads = ["MLII", "V1", "V2", "V4", "V5"]
    symbols = {
        "N": "Normal beat",
        "SVEB": "Supraventricular ectopic beat",
        "VEB": "Ventricular ectopic beat",
        "F": "Fusion beat",
    }
    # symbols = {
    #     # N
    #     "N": "Normal beat",
    #     "L": "Left bundle branch block beat",
    #     "R": "Right bundle branch block beat",
    #     "e": "Atrial escape beat",
    #     "j": "Nodal (junctional) escape beat",
    #     # SVEB
    #     "A": "Atrial premature beat",
    #     "a": "Aberrated atrial premature beat",
    #     "J": "Nodal (junctional) premature beat",
    #     "S": "Supraventricular premature",
    #     # VE
    #     "V": "Premature ventricular contraction",
    #     "E": "Ventricular escape beat",
    #     # F
    #     "F": "Fusion of ventricular and normal beat",
    #     # Q
    #     "f": "Fusion of paced and normal beat",
    #     # "Q": "Unclassifiable beat",
    #     # # other
    #     # "[": "Start of ventricular flutter/fibrillation",
    #     # "!": "Ventricular flutter wave",
    #     # "]": "End of ventricular flutter/fibrillation",
    #     # "/": "Paced beat",
    #     # "x": "Non-conducted P-wave (blocked APC)",
    #     # "|": "Isolated QRS-like artifact",
    # }

    os.makedirs(data_root, exist_ok=True)
    get_symbol_raw_description(data_root, leads, symbols)
    get_symbol_description(data_root)
    get_symbol_sentence(data_root)
    get_symbol_raw_embedding(data_root)
    get_symbol_embedding(data_root)


if __name__ == "__main__":
    main()
