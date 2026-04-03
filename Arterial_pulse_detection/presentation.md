# Arterial Pulse Detection using mmWave Radar
*A Non-Contact, Real-Time Vital Signs Monitoring System*

---

## Slide 1: Title Slide
**Arterial Pulse Detection using mmWave Radar**  
*Integrating Digital Signal Processing & Machine Learning for Real-Time Monitoring*

- **Project Focus:** Non-invasive physiological monitoring.
- **Core Technology:** Frequency Modulated Continuous Wave (FMCW) radar.
- **Presenter:** [Your Name/Team]

---

## Slide 2: The Problem Statement
**Why do we need a new way to measure pulse?**

- **Contact Sensors are Restrictive:** Traditional methods (ECG grids, pulse oximeters) require physical attachment, which stresses patients long-term (e.g., burn units, sleep studies).
- **Camera Methods are Fragile:** Photoplethysmography (cameras) suffers heavily in poor lighting and raises major privacy concerns in bedrooms or clinical spaces.
- **The Gap:** We need a continuous, safe, privacy-preserving, and completely contactless method to monitor cardiovascular health.

---

## Slide 3: Proposed Solution & Objectives
**The mmWave Radar Approach**

- **The Concept:** A millimeter-wave radar can detect microscopic chest-wall displacements (fractions of a millimeter) caused by the heartbeat.
- **Key Objectives:**
  1. Build a real-time data streaming pipeline via hardware serial ports.
  2. Isolate the faint cardiac pulse from massive breathing artifacts using advanced Digital Signal Processing (DSP).
  3. Integrate Machine Learning to make the heart-rate prediction robust and personalized.
  4. Display the live data on a zero-latency web dashboard.

---

## Slide 4: High-Level System Architecture
**Bridging Hardware to Software**

- **Data Capture:** Texas Instruments mmWave Sensor (xWR68xx) running a 20 FPS physiological configuration profile.
- **Hardware Interface:** USB-Serial duplex communication connecting a Command Line Interface (CLI) and a High-Speed Binary Data port.
- **Backend Hub:** An asynchronous Python server (FastAPI/Uvicorn) that reads the data stream, unpacks binary TLV packets, and threads the processing logic.
- **Frontend Dashboard:** A web interface powered by WebSockets to render the live waveform and current Beats Per Minute (BPM) instantly.

---

## Slide 5: The Physics – Phase Extraction
**How do we 'see' a heartbeat?**

- When radar chirps bounce off the human chest, the system evaluates the complex baseband signals.
- **Targeting:** We find the distance ("range bin") with the strongest reflection magnitude `sqrt(I² + Q²)`.
- **Phase Shift:** The arterial pulse physically shifts the chest wall, mathematically shifting the radar phase. 
- **The Equation:** By extracting the phase angle `np.arctan2(Q, I)` and unwrapping it, we get a 1D timeline of physical chest displacement over time.

---

## Slide 6: Traditional DSP Pipeline
**Filtering the Noise (The Mathematical Approach)**

- **The Challenge:** Breathing creates massive waveforms that easily hide the pulse.
- **Butterworth Bandpass Filter:** We pass the signal through a strict 4th-order filter configured for **0.8 Hz to 3.0 Hz**.
  - *0.8 Hz (48 BPM):* Erases slow respiration waves.
  - *3.0 Hz (180 BPM):* Erases high-frequency sensor static.
- **Peak Counting:** Using SciPy's `find_peaks`, we literally map local extremas with a distance constraint to count every individual heartbeat in a 10-second window.

---

## Slide 7: Why Introduce Machine Learning?
**Moving away from rigid mathematics**

- **The DSP Limitation:** Counting peaks mathematically is rigid. If a patient suddenly shifts or shifts their breathing, it creates "fake peaks" and causes the BPM output to spike erratically.
- **The ML Paradigm Shift:** Instead of aggressively counting individual humps on a graph, what if we looked at the *overall energy and shape* of a 10-second window?
- Machine Learning provides a robust, statistical safety net that is much harder to fool with random noise artifacts.

---

## Slide 8: The ML Model & Feature Extraction
**Random Forest Regression**

- **The Architecture:** We implemented a Random Forest Regressor (`scikit-learn`), utilizing 100 decision trees to prevent overfitting.
- **The Inputs (9 Features):** 
  - *7 Signal Statistics:* Variance, Standard Deviation, Mean Absolute Deviation (MAD), Max, Min, Total Energy, and Dominant Frequency.
  - *2 Demographics:* Age and Gender.
- **The Output:** By feeding these 9 data points into our trained model (`pulse_rf_model.pkl`), it instantaneously predicts the target Beats Per Minute (BPM).

---

## Slide 9: Software & Real-Time Performance
**Zero-Latency Web Integration**

- A real-time system is useless if it lags.
- **Threading:** We utilize background Python daemon threads for reading radar and calculating BPM so the web server loop is never blocked.
- **WebSockets:** Unlike standard HTTP polling, our active WebSocket pushes JSON payloads `(timestamp, phase, bpm)` ~10 times per second directly to the user's browser.
- **End-to-End Latency:** Kept securely under 100 milliseconds for a flawlessly smooth UI visual.

---

## Slide 10: Performance Validation & Results
**Accuracy and Stability**

- **Throughput:** Maintained perfectly at 20 frames per second using hardware-optimized native NumPy arrays for math operations instead of standard Python lists.
- **ML Efficacy:** Validated via rigorous R-Squared Analysis against our `ml_dataset.csv` benchmark.
- **Graceful Error Handling:** Serial ports are aggressively managed to prevent memory leaks and ensure the radar safely transitions to a standby state upon shutdown.

---

## Slide 11: Real-World Applications
**Where does this technology belong?**

- **Tele-Health & Elderly Care:** Non-invasive, completely passive monitoring inside a bedroom—recording baseline vitals without attaching electrodes to the patient.
- **Automotive Safety:** Built into a car dashboard to monitor driver cardiovascular markers, actively tracking drowsiness or sudden cardiac events without cameras.
- **Burn Units & NICUs:** Perfect for patients whose skin cannot tolerate sticky ECG pads securely.

---

## Slide 12: Conclusion & Future Enhancements
**What comes next for the platform?**

- **Conclusion:** We successfully bridged mmWave FMCW hardware, rigorous DSP mathematics, and Random Forest Machine Learning into a cohesive, non-contact pulse tracker.
- **Future Scope 1:** Adding Respiration Rate tracking by analyzing the frequencies below 0.5 Hz.
- **Future Scope 2:** Shifting the Random Forest inference entirely onto a local microcontroller (Edge Computing) to remove the need for a PC workstation entirely.
- **Future Scope 3:** Implementing Deep Learning (LSTMs) to track cardiovascular variability continuously over multi-hour sessions.

---
*Questions & Discussion*
