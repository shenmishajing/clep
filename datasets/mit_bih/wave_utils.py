import numpy as np


def find_next_non_nan_index(wave_ann, wave_index, min=None, max=None):
    while (
        np.isnan(wave_ann[wave_index]) or min is not None and wave_ann[wave_index] < min
    ):
        wave_index += 1
        if wave_index >= len(wave_ann):
            return None

    if max is not None and wave_ann[wave_index] > max:
        return None
    return wave_index


def find_prev_non_nan_index(wave_ann, wave_index, max=None):
    while (
        np.isnan(wave_ann[wave_index]) or max is not None and wave_ann[wave_index] > max
    ):
        wave_index -= 1
        if wave_index < 0:
            return 0

    return wave_index


def find_index(wave_ann, wave_T_index=0, period_num=3):
    result = []
    wave_index = {
        "T": wave_T_index,
        "P": find_prev_non_nan_index(
            wave_ann["ECG_P_Onsets"],
            wave_T_index,
            wave_ann["ECG_T_Offsets"][wave_T_index],
        ),
        "R": find_prev_non_nan_index(
            wave_ann["ECG_R_Onsets"],
            wave_T_index,
            wave_ann["ECG_T_Offsets"][wave_T_index],
        ),
    }

    while len(result) < period_num:
        cur_res = {}

        # find the end of next T wave
        wave_index["T"] = find_next_non_nan_index(
            wave_ann["ECG_T_Offsets"], wave_index["T"]
        )
        if wave_index["T"] is None or wave_index["T"] + 1 >= len(
            wave_ann["ECG_T_Offsets"]
        ):
            return []
        cur_res["period_start"] = wave_ann["ECG_T_Offsets"][wave_index["T"]]
        cur_res["period_end"] = wave_ann["ECG_T_Offsets"][wave_index["T"] + 1]
        if np.isnan(cur_res["period_end"]):
            return []

        # find the next P R T wave
        for wave_name in "PRT":
            wave_index[f"{wave_name}"] = find_next_non_nan_index(
                wave_ann[f"ECG_{wave_name}_Onsets"],
                wave_index[f"{wave_name}"],
                wave_ann["ECG_T_Offsets"][wave_index["T"]],
                wave_ann["ECG_T_Offsets"][wave_index["T"] + 1],
            )
            if wave_index[f"{wave_name}"] is None or np.isnan(
                wave_ann[f"ECG_{wave_name}_Offsets"][wave_index[f"{wave_name}"]]
            ):
                return []
            else:
                for event in ["Peaks", "Onsets", "Offsets"]:
                    cur_res[f"ECG_{wave_name}_" + event] = wave_ann[
                        f"ECG_{wave_name}_" + event
                    ][wave_index[f"{wave_name}"]]
                cur_res[f"ECG_{wave_name}_Index"] = wave_index[f"{wave_name}"]
        else:
            # if do not break
            result.append(cur_res)

    return result
