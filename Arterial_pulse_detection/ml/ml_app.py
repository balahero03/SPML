import serial
import time
import struct
import csv
import datetime
import os
import numpy as np
import asyncio
import threading
import pandas as pd
import pickle
from collections import deque
from scipy.signal import butter, lfilter, find_peaks
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ML Loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pulse_rf_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        ml_model = pickle.load(f)
    print("✅ ML Model loaded successfully into core!")
except FileNotFoundError:
    print("Warning: ML Model not compiled yet. Run train_ml_model.py")
    ml_model = None

# ===============================
# APP INITIALIZATION
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    threading.Thread(target=radar_thread, daemon=True).start()
    threading.Thread(target=calculate_bpm_thread, daemon=True).start()
    print("[Server] Started. Open http://localhost:8000 in your browser.")
    yield
    # Shutdown
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

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SENSOR SETTINGS — Edit these if your COM ports differ
# ===============================
CFG_FILE = os.path.join(os.path.dirname(__file__), "xwr68xx_profile_VitalSigns_20fps_Front.cfg")
CLI_PORT = 'COM10'
DATA_PORT = 'COM9'
CLI_BAUD = 115200
DATA_BAUD = 921600
SAVE_DIR = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_FILE = os.path.join(SAVE_DIR, f"pulse_live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# 600 samples = ~30 seconds at 20 fps — used for accurate BPM calculation
WINDOW_SIZE = 600

# ===============================
# SHARED STATE
# ===============================
is_running = True
phase_buffer = deque(maxlen=600)
bpm_value = 0          # Latest computed BPM
radar_status = "connecting"  # "connecting" | "ok" | "error"
error_message = ""
current_gender = 0
current_age = 22

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

                offset += tlv_length

        except Exception as e:
            if is_running:
                print(f"[Radar] Read error: {e}")
            break

    print("[Radar] Thread exiting.")


# ===============================
# THREAD 2: BPM Calculation
# ===============================
def extract_raw_features(phase_array):
    features = {}
    features['std'] = np.std(phase_array)
    features['var'] = np.var(phase_array)
    features['mad'] = np.mean(np.abs(phase_array - np.mean(phase_array)))
    features['max'] = np.max(phase_array)
    features['min'] = np.min(phase_array)
    features['energy'] = np.sum(np.square(phase_array))
    
    fft_val = np.abs(np.fft.rfft(phase_array))
    freqs = np.fft.rfftfreq(len(phase_array), d=1/20.0)
    valid_idx = np.where((freqs >= 0.8) & (freqs <= 2.5))[0]
    
    features['dom_freq'] = freqs[valid_idx[np.argmax(fft_val[valid_idx])]] if len(valid_idx) > 0 else 0
    return features


def calculate_bpm_thread():
    """ Runs continuously, calculating ML BPM from radar phase """
    global bpm_value, is_running
    
    while is_running:
        time.sleep(1)  # Recalculate every second

        # Need at least 10 sec (200 samples) before we start predicting
        if len(phase_buffer) < 200:
            continue

        raw_phase = [pt[1] for pt in list(phase_buffer)[-200:]]
        
        try:
            if ml_model:
                feat = extract_raw_features(raw_phase)
                feat['gender'] = current_gender
                feat['age'] = current_age
                
                # Form array matching Training
                X = pd.DataFrame([[feat['std'], feat['var'], feat['mad'], feat['max'], 
                                   feat['min'], feat['energy'], feat['dom_freq'], feat['gender'], feat['age']]],
                                 columns=['std', 'var', 'mad', 'max', 'min', 'energy', 'dom_freq', 'gender', 'age'])
                
                bpm_pred = ml_model.predict(X)[0]
                bpm_value = round(bpm_pred, 1)
            else:
                bpm_value = 0
        except Exception as e:
            print(f"[BPM ML] Calculation error: {e}")

    print("[BPM] Thread exiting.")


# ===============================
# LIFECYCLE (Moved to lifespan context manager)
# ===============================


# ===============================
# ROUTES
# ===============================
@app.get("/")
def get_html():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
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


@app.post("/start_calibration")
async def start_calibration(request: Request):
    global phase_buffer, bpm_value, current_gender, current_age
    phase_buffer.clear()
    bpm_value = 0
    
    try:
        param = await request.json()
        if param:
            current_gender = 0 if param.get('gender') == 'male' else 1
            # Convert age bracket string back to pure int mean for Model
            age_str = param.get('age', '25')
            if age_str == "18-35": current_age = 26
            elif age_str == "36-55": current_age = 45
            elif age_str == "56+": current_age = 65
            else: current_age = 30
    except Exception:
        pass
        
    return JSONResponse({"status": "cleared"})


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
