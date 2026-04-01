import argparse
import csv
import datetime
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial

from pulse_extract import estimate_rowwise_bpm, estimate_session_bpm


CFG_FILE = Path(__file__).with_name("xwr68xx_profile_VitalSigns_20fps_Front.cfg")

CLI_PORT = "COM10"
DATA_PORT = "COM9"

CLI_BAUD = 115200
DATA_BAUD = 921600

DEFAULT_DURATION = 30
DEFAULT_WINDOW_SECONDS = 10.0

SAVE_DIR = Path(__file__).resolve().parent
SAVE_DIR.mkdir(exist_ok=True)

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def send_config(cli: serial.Serial) -> None:
    print("Sending config...")
    with CFG_FILE.open() as cfg_file:
        for line in cfg_file:
            line = line.strip()
            if line and not line.startswith("%"):
                cli.write((line + "\n").encode())
                time.sleep(0.05)
    print("Config sent.")


def capture_sensor_data(duration: int, csv_path: Path) -> pd.DataFrame:
    print("Connecting...")
    cli = serial.Serial(CLI_PORT, CLI_BAUD)
    data = serial.Serial(DATA_PORT, DATA_BAUD)

    rows = []

    try:
        time.sleep(1)
        cli.write(b"sensorStop\n")
        time.sleep(1)
        send_config(cli)

        start_time = time.time()

        while time.time() - start_time < duration:
            if data.read(8) != MAGIC_WORD:
                continue

            header = data.read(32)
            total_len = struct.unpack("<I", header[4:8])[0]
            frame_num = struct.unpack("<I", header[12:16])[0]

            payload_len = total_len - 40
            payload = data.read(payload_len)

            offset = 0

            while offset + 8 <= len(payload):
                tlv_type, tlv_length = struct.unpack_from("<II", payload, offset)
                offset += 8

                if tlv_type == 2:
                    values = struct.unpack(
                        "<" + "h" * (tlv_length // 2),
                        payload[offset:offset + tlv_length],
                    )

                    num_bins = tlv_length // 4
                    mags = []
                    complex_bins = []

                    for i in range(num_bins):
                        i_val = values[i * 2]
                        q_val = values[i * 2 + 1]
                        complex_bins.append((i_val, q_val))
                        mags.append(np.sqrt(i_val * i_val + q_val * q_val))

                    target_bin = int(np.argmax(mags))
                    i_val, q_val = complex_bins[target_bin]
                    phase = np.arctan2(q_val, i_val)
                    timestamp = time.time() - start_time

                    rows.append(
                        [timestamp, frame_num, target_bin, i_val, q_val, phase]
                    )

                offset += tlv_length
    finally:
        print("Stopping...")
        cli.write(b"sensorStop\n")
        cli.close()
        data.close()

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "frame", "range_bin", "I", "Q", "phase"],
    )
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record mmWave pulse data for a few seconds and predict heart rate."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Number of seconds to record sensor data.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Rolling window size used for BPM estimation.",
    )
    args = parser.parse_args()

    csv_path = SAVE_DIR / f"pulse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = capture_sensor_data(args.duration, csv_path)

    if df.empty:
        print("No sensor data was captured.")
        return

    df["heart_rate_bpm"] = estimate_rowwise_bpm(df, window_seconds=args.window_seconds)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    predicted_bpm = estimate_session_bpm(df, window_seconds=args.window_seconds)

    print(f"Saved: {csv_path}")
    if predicted_bpm > 0:
        print(f"Predicted heart rate: {predicted_bpm:.1f} BPM")
    else:
        print("Predicted heart rate could not be estimated from this recording.")


if __name__ == "__main__":
    main()
