import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def estimate_rowwise_bpm(df: pd.DataFrame, window_seconds: float = 10.0) -> pd.Series:
    phase = np.unwrap(df["phase"].to_numpy())
    timestamps = df["timestamp"].to_numpy()

    duration = float(timestamps[-1] - timestamps[0]) if len(df) > 1 else 0.0
    fs = len(df) / duration if duration > 0 else 0.0

    if fs <= 0 or fs / 2 <= 3:
        return pd.Series(np.zeros(len(df)), index=df.index, name="heart_rate_bpm")

    low = 0.8 / (fs / 2)
    high = 3.0 / (fs / 2)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, phase)

    peaks, _ = find_peaks(filtered, distance=max(1, int(fs * 0.4)))
    peak_times = timestamps[peaks]
    global_bpm = (len(peaks) * 60.0 / duration) if duration > 0 and len(peaks) else 0.0

    bpm_values = np.full(len(df), np.nan)
    half_window = window_seconds / 2.0

    for i, current_time in enumerate(timestamps):
        window_start = max(timestamps[0], current_time - half_window)
        window_end = min(timestamps[-1], current_time + half_window)
        beats_in_window = np.count_nonzero(
            (peak_times >= window_start) & (peak_times <= window_end)
        )
        actual_window = window_end - window_start

        if actual_window >= 3.0 and beats_in_window > 0:
            bpm_values[i] = beats_in_window * 60.0 / actual_window

    bpm_series = pd.Series(bpm_values, index=df.index, name="heart_rate_bpm")
    bpm_series = bpm_series.bfill().ffill()

    if bpm_series.isna().any():
        bpm_series = bpm_series.fillna(global_bpm)

    if global_bpm > 0:
        bpm_series = bpm_series.clip(lower=max(40.0, global_bpm - 20.0),
                                     upper=min(180.0, global_bpm + 20.0))

    return bpm_series.round(1)


def estimate_session_bpm(df: pd.DataFrame, window_seconds: float = 10.0) -> float:
    bpm_series = estimate_rowwise_bpm(df, window_seconds=window_seconds)
    nonzero = bpm_series[bpm_series > 0]

    if nonzero.empty:
        return 0.0

    return float(nonzero.tail(min(len(nonzero), 20)).median())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate rolling heart rate from mmWave phase data and write it to CSV."
    )
    parser.add_argument("csv_path", help="Path to the CSV file to update.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Rolling window size used for the BPM estimate.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    required_columns = {"timestamp", "phase"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["heart_rate_bpm"] = estimate_rowwise_bpm(df, window_seconds=args.window_seconds)
    df.to_csv(csv_path, index=False)

    session_bpm = estimate_session_bpm(df, window_seconds=args.window_seconds)
    if session_bpm > 0:
        print(
            f"Updated {csv_path} with rolling BPM values. "
            f"Predicted heart rate: {session_bpm:.1f} BPM"
        )
    else:
        print(f"Updated {csv_path}, but no BPM values were stable enough to report yet.")


if __name__ == "__main__":
    main()
