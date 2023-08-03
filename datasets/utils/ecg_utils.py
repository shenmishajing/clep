import os

import neurokit2 as nk
import wfdb


def load_record(data_path, name):
    return wfdb.rdrecord(os.path.join(data_path, name))


def load_ann(ann_path, name):
    return wfdb.rdann(os.path.join(ann_path, name), "atr")


def calculate_wave_ann(signal, fs, signal_name, method="dwt"):
    wave_peaks = {}

    for i, name in enumerate(signal_name):
        try:
            _, rpeaks = nk.ecg_peaks(signal[:, i], sampling_rate=fs)
            # Delineate the ECG signal
            _, cur_wave_peaks = nk.ecg_delineate(
                signal[:, i],
                rpeaks,
                sampling_rate=fs,
                method=method,
            )
            wave_peaks[name] = cur_wave_peaks
            wave_peaks[name]["ECG_R_Peaks"] = rpeaks["ECG_R_Peaks"].tolist()
        except Exception as e:
            wave_peaks[name] = None

    return wave_peaks


def load_wave_ann(data_path, name, **kwargs):
    record = load_record(data_path, name)
    wave_ann = calculate_wave_ann(record.p_signal, record.fs, record.sig_name, **kwargs)
    return record, wave_ann


def load_record_and_ann(data_path, ann_path, name, **kwargs):
    record, wave_ann = load_wave_ann(data_path, name, **kwargs)
    ann = load_ann(ann_path, name)
    return record, ann, wave_ann
