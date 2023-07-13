import os
import pickle

import numpy as np


def main():
    data_path = "data/mit-bih-arrhythmia-database-1.0.0"

    for name in open(os.path.join(data_path, "RECORDS"), "r").readlines():
        name = name.strip()
        wave_peaks = pickle.load(
            open(os.path.join(data_path, "cache", name) + ".pkl", "rb")
        )

        for signal_name, signal_peaks in wave_peaks.items():
            duration = {x: [] for x in "PRT"}
            for wave_name in duration:
                for i in range(len(signal_peaks[f"ECG_{wave_name}_Peaks"])):
                    if not np.isnan(
                        signal_peaks[f"ECG_{wave_name}_Onsets"][i]
                    ) and not np.isnan(signal_peaks[f"ECG_{wave_name}_Offsets"][i]):
                        duration[wave_name].append(
                            signal_peaks[f"ECG_{wave_name}_Offsets"][i]
                            - signal_peaks[f"ECG_{wave_name}_Onsets"][i]
                        )

                    if wave_name == "R":
                        if (
                            not np.isnan(signal_peaks["ECG_Q_Peaks"][i])
                            and not np.isnan(signal_peaks["ECG_R_Onsets"][i])
                            and not np.isnan(signal_peaks["ECG_R_Offsets"][i])
                            and not np.isnan(signal_peaks["ECG_S_Peaks"][i])
                            and (
                                signal_peaks["ECG_Q_Peaks"][i]
                                < signal_peaks["ECG_R_Onsets"][i]
                                or signal_peaks["ECG_S_Peaks"][i]
                                > signal_peaks["ECG_R_Offsets"][i]
                            )
                        ):
                            print(
                                "QRS range error",
                                name,
                                i,
                                signal_peaks["ECG_R_Onsets"][i],
                                signal_peaks["ECG_Q_Peaks"][i],
                                signal_peaks["ECG_S_Peaks"][i],
                                signal_peaks["ECG_R_Offsets"][i],
                            )

            print(
                name,
                signal_name,
                {
                    k: f"mean: {np.mean(v)}, min: {np.min(v)}, max: {np.max(v)}"
                    for k, v in duration.items()
                },
            )


if __name__ == "__main__":
    main()
