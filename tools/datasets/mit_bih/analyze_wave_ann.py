import os
import pickle

import numpy as np


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    for name in open(os.path.join(data_path, "RECORDS"), "r").readlines():
        name = name.strip()
        wave_ann = pickle.load(
            open(os.path.join(data_path, "dwt_cache", name) + ".pkl", "rb")
        )

        for cur_name, cur_ann in wave_ann.items():
            duration = {x: [] for x in "PRT"}
            for wave_name in duration:
                for i in range(len(cur_ann[f"ECG_{wave_name}_Peaks"])):
                    if not np.isnan(
                        cur_ann[f"ECG_{wave_name}_Onsets"][i]
                    ) and not np.isnan(cur_ann[f"ECG_{wave_name}_Offsets"][i]):
                        duration[wave_name].append(
                            cur_ann[f"ECG_{wave_name}_Offsets"][i]
                            - cur_ann[f"ECG_{wave_name}_Onsets"][i]
                        )

                    if wave_name == "R":
                        if (
                            not np.isnan(cur_ann["ECG_Q_Peaks"][i])
                            and not np.isnan(cur_ann["ECG_R_Onsets"][i])
                            and not np.isnan(cur_ann["ECG_R_Offsets"][i])
                            and not np.isnan(cur_ann["ECG_S_Peaks"][i])
                            and (
                                cur_ann["ECG_Q_Peaks"][i] < cur_ann["ECG_R_Onsets"][i]
                                or cur_ann["ECG_S_Peaks"][i]
                                > cur_ann["ECG_R_Offsets"][i]
                            )
                        ):
                            print(
                                "QRS range error",
                                name,
                                i,
                                cur_ann["ECG_R_Onsets"][i],
                                cur_ann["ECG_Q_Peaks"][i],
                                cur_ann["ECG_S_Peaks"][i],
                                cur_ann["ECG_R_Offsets"][i],
                            )

            print(
                name,
                cur_name,
                {
                    k: f"mean: {np.mean(v)}, min: {np.min(v)}, max: {np.max(v)}"
                    for k, v in duration.items()
                },
            )


if __name__ == "__main__":
    main()
