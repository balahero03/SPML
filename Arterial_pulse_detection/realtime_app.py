import serial
import time
import struct
import csv
import datetime
import os
import numpy as np
import asyncio
import threading
from scipy.signal import butter, lfilter, find_peaks
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ===============================
# APP INITIALIZATION
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SENSOR SETTINGS — Edit these if your COM ports differ
# ===============================
CFG_FILE = r"C:\Users\bala3\Downloads\mm\xwr68xx_profile_VitalSigns_20fps_Front.cfg"
CLI_PORT = 'COM4'
DATA_PORT = 'COM3'
CLI_BAUD = 115200
DATA_BAUD = 921600
SAVE_DIR = r"C:\Users\bala3\Downloads\mm"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_FILE = os.path.join(SAVE_DIR, f"pulse_live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# 600 samples = ~30 seconds at 20 fps — used for accurate BPM calculation
WINDOW_SIZE = 600

# ===============================
# SHARED STATE
# ===============================
is_running = True
phase_buffer = []      # Sliding window: list of (timestamp, phase)
bpm_value = 0          # Latest computed BPM
radar_status = "connecting"  # "connecting" | "ok" | "error"
error_message = ""

# CSV file opened once, written to continuously
csv_file = open(CSV_FILE, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp", "frame", "range_bin", "I", "Q", "phase", "heart_rate_bpm"])
csv_file.flush()

# Serial handles (initialized inside thread)
cli_serial = None
data_serial = None

# ===============================
# HELPER: Butterworth Bandpass
# ===============================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 1 - 1e-6)
    b, a = butter(order, [low, high], btype='band')
    return b, a

# ===============================
# THREAD 1: Radar Data Capture
# ===============================
def radar_thread():
    global is_running, phase_buffer, radar_status, error_message
    global cli_serial, data_serial

    print(f"[Radar] Connecting to {CLI_PORT} and {DATA_PORT}...")
    try:
        cli_serial = serial.Serial(CLI_PORT, CLI_BAUD, timeout=2)
        data_serial = serial.Serial(DATA_PORT, DATA_BAUD, timeout=2)
    except serial.SerialException as e:
        radar_status = "error"
        error_message = str(e)
        print(f"[Radar] ERROR: {e}")
        return

    time.sleep(1)
    cli_serial.write(b'sensorStop\n')
    time.sleep(1)

    print("[Radar] Sending configuration...")
    try:
        with open(CFG_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('%'):
                    cli_serial.write((line + '\n').encode())
                    time.sleep(0.05)
    except Exception as e:
        radar_status = "error"
        error_message = f"Config error: {e}"
        print(f"[Radar] Config ERROR: {e}")
        return

    radar_status = "ok"
    print("[Radar] Listening for data...")
    start_time = time.time()

    while is_running:
        try:
            magic = data_serial.read(8)
            if magic != MAGIC_WORD:
                continue

            header = data_serial.read(32)
            if len(header) < 32:
                continue

            total_len = struct.unpack('<I', header[4:8])[0]
            frame_num = struct.unpack('<I', header[12:16])[0]

            payload_len = total_len - 40
            if payload_len <= 0:
                continue

            payload = data_serial.read(payload_len)
            offset = 0

            while offset + 8 <= len(payload):
                tlv_type, tlv_length = struct.unpack_from('<II', payload, offset)
                offset += 8

                if tlv_type == 2 and tlv_length > 0 and (offset + tlv_length) <= len(payload):
                    try:
                        values = struct.unpack('<' + 'h' * (tlv_length // 2), payload[offset:offset + tlv_length])
                    except struct.error:
                        break

                    num_bins = tlv_length // 4
                    mags = []
                    complex_bins = []

                    for i in range(num_bins):
                        I = values[i * 2]
                        Q = values[i * 2 + 1]
                        complex_bins.append((I, Q))
                        mags.append(np.sqrt(I * I + Q * Q))

                    if not mags:
                        break

                    target_bin = int(np.argmax(mags))
                    I, Q = complex_bins[target_bin]
                    phase = np.arctan2(Q, I)
                    timestamp = time.time() - start_time

                    # Save to CSV — include current BPM so each row has context
                    writer.writerow([timestamp, frame_num, target_bin, I, Q, phase, bpm_value])
                    csv_file.flush()

                    # Update sliding window — 600 samples = 30 seconds at 20fps
                    phase_buffer.append((timestamp, phase))
                    if len(phase_buffer) > WINDOW_SIZE:
                        phase_buffer.pop(0)

                offset += tlv_length

        except Exception as e:
            if is_running:
                print(f"[Radar] Read error: {e}")
            break

    print("[Radar] Thread exiting.")


# ===============================
# THREAD 2: BPM Calculation
# ===============================
def calculate_bpm_thread():
    global bpm_value, phase_buffer, is_running

    while is_running:
        time.sleep(1)  # Recalculate every second

        # Need at least half the window (15 sec) before we start calculating
        if len(phase_buffer) < WINDOW_SIZE // 2:
            continue

        current_data = list(phase_buffer)
        timestamps = np.array([pt[0] for pt in current_data])
        raw_phases = np.array([pt[1] for pt in current_data])

        phases = np.unwrap(raw_phases)
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            continue

        fs = len(phases) / duration

        try:
            b, a = butter_bandpass(0.8, 3.0, fs, order=4)
            filtered = lfilter(b, a, phases)
            peaks, _ = find_peaks(filtered, distance=fs * 0.4)

            if len(peaks) > 1:
                bpm = len(peaks) / (duration / 60)
                bpm_value = round(bpm, 1)
        except Exception as e:
            print(f"[BPM] Calculation error: {e}")

    print("[BPM] Thread exiting.")


# ===============================
# LIFECYCLE
# ===============================
@app.on_event("startup")
def startup_event():
    threading.Thread(target=radar_thread, daemon=True).start()
    threading.Thread(target=calculate_bpm_thread, daemon=True).start()
    print("[Server] Started. Open http://localhost:8000 in your browser.")


@app.on_event("shutdown")
def shutdown_event():
    global is_running
    is_running = False
    csv_file.close()
    try:
        if cli_serial and cli_serial.is_open:
            cli_serial.write(b'sensorStop\n')
            cli_serial.close()
        if data_serial and data_serial.is_open:
            data_serial.close()
    except Exception:
        pass
    print(f"[Server] Shutdown complete. Data saved to: {CSV_FILE}")


# ===============================
# ROUTES
# ===============================
@app.get("/")
def get_html():
    html_path = r"C:\Users\bala3\Downloads\mm\index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/status")
def get_status():
    return JSONResponse({
        "radar": radar_status,
        "error": error_message,
        "bpm": bpm_value,
        "samples": len(phase_buffer)
    })


@app.post("/stop")
def stop_capture():
    global is_running
    is_running = False
    csv_file.flush()
    csv_file.close()
    try:
        if cli_serial and cli_serial.is_open:
            cli_serial.write(b'sensorStop\n')
            cli_serial.close()
        if data_serial and data_serial.is_open:
            data_serial.close()
    except Exception:
        pass
    return JSONResponse({"status": "success", "file": CSV_FILE})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.1)  # 10 updates per second
            if len(phase_buffer) > 0:
                last_pt = phase_buffer[-1]
                await websocket.send_json({
                    "timestamp": round(last_pt[0], 3),
                    "phase": round(last_pt[1], 6),
                    "bpm": bpm_value,
                    "radar": radar_status
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Client disconnected: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  mmWave Real-Time Pulse Monitor")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
