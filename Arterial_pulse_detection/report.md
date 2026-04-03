I. Title Page
 
**Arterial Pulse Detection & Real-Time Monitoring System using mmWave Radar**
 
---
II. Abstract
 
This project addresses the critical challenge in continuous vital sign monitoring: the need for non-invasive, continuous, and accurate identification of human pulse rates. Contact-based monitoring systems can be obtrusive, restrictive, and sometimes impractical for long-term continuous use or specific clinical situations. Our solution is an Arterial Pulse Detection System that seamlessly integrates millimeter-wave (mmWave) radar technology with advanced digital signal processing (DSP). The system architecture employs an mmWave Sensor (e.g., Texas Instruments xWR68xx) to capture high-frequency phase shift data corresponding to microscopic chest vibrations. This data is transmitted via high-speed serial protocols to a centralized processing workstation. A robust FastAPI server running in Python manages the continuous real-time data stream, parsing binary TLV packets and performing hardware-optimized signal filtering. Utilizing a custom pipeline featuring Butterworth bandpass filters and dynamic peak detection algorithms, the system translates raw I/Q phase variations into precise Heart Rate estimates (BPM) with minimal latency. The system not only detects human physiological signals but also provides a live waveform visualization and quantified pulse count via a web-based dashboard. The successful integration of non-contact radar sensors, threaded data acquisition, and signal processing demonstrates a highly viable, obtrusion-free solution for tele-health, driver monitoring, and continuous clinical surveillance.
 
---
III. Introduction
 
**3.1 Project Overview**
 
The core objective of this project is to develop and implement a real-time Arterial Pulse Detection and Monitoring System utilizing Frequency Modulated Continuous Wave (FMCW) mmWave radar. This system is designed to acquire and process live digital reflection signals captured by the radar's antennas to non-invasively measure a human subject's heart rate. The central goal is the seamless integration of serial data transmission protocols, modern backend network architectures (FastAPI, WebSockets), and state-of-the-art Digital Signal Processing (DSP). The output is a high-speed, reliable evaluation of a person's Beats Per Minute (BPM), calculated and displayed instantly, transforming raw phase data into actionable physiological intelligence. This project demonstrates a critical capability within the domain of biomedical signal processing and digital architectures: engineering a low-latency, resilient digital pipeline capable of extracting faint micro-vibration signals hidden within environmental noise. The successful outcome is a functional, real-time prototype that moves beyond contact-based constraints.

**3.2 Problem Statement**
 
In numerous healthcare and safety-critical scenarios, the continuous monitoring of human vital signs is hindered by reliance on obstructive or restrictive contact-based devices (e.g., ECG grids, pulse oximetry clips). For instance, in burn units, sleep tracking, or automotive driver monitoring, physical attachments can cause discomfort, induce stress, or restrict mobility. Furthermore, standard camera-based photoplethysmography is highly sensitive to ambient lighting changes and poses privacy concerns. The fundamental problem addressed by this research is the difficulty in reliably, safely, and continuously extracting cardiovascular data from an individual at a distance without physical interaction. There exists a clear and urgent need for an automated, contactless digital solution that can accurately quantify human physiological vitals, allowing healthcare providers and safety systems to collect long-term data seamlessly.
 
**3.3 Proposed Solution Summary**
 
The solution involves architecting a sophisticated, end-to-end digital processing pipeline that converts a live mmWave radar reflection stream into quantitative pulse data. The process begins with the mmWave sensor emitting high-frequency waves and capturing the reflections at 20 frames per second. These analog reflections are digitized and processed on the sensor to generate Intermediate Frequency (IF) signals, which are then transmitted via a hardware COM port (Data Port) to a PC workstation.
The server's backend operation is orchestrated by a multi-threaded Python/FastAPI server, which handles the serial stream reception, real-time payload unpacking (TLV formats), and buffering. This Python endpoint acts as the bridge between the hardware's binary data layer and the DSP analytical layer. For Pulse Extraction, the system utilizes I/Q phase analysis coupled with a rigorous bandpass filtering module (optimized for 0.8–3.0 Hz, reflecting a standard human pulse range). The DSP module isolates the arterial micro-movements from respiration and body sway artifacts. The heart rate is then calculated using peak detection over a sliding window. Crucially, the process utilizes high-speed WebSockets for zero-latency networking, ensuring the waveform plots and BPM results are displayed live on a visualization dashboard. 
 
---
IV. System Architecture and Digital Implementation
 
**4.1 Overall System Flow**
 
The system is defined by a sequential, low-latency digital pipeline designed to minimize signal bottlenecks and ensure real-time physiological analytics. 
1. **Data Capture (mmWave Sensor):** The process begins with the mmWave radar operating a `xwr68xx_profile_VitalSigns_20fps` configuration, capturing spatial reflection data targeting the chest of a human subject.
2. **Digital Transmission (Serial Comm):** The digital hardware interface involves two separate data links. A Command-Line Interface (CLI Port) transmits configuration logic at a 115200 baud rate, while a high-throughput Data Port transmits the raw binary observation frames at 921600 baud rate.
3. **Backend Interface (FastAPI):** The PC Workstation runs an asynchronous Python server that consumes the data port stream. It checks for a Magic Word `(\x02\x01\x04\x03\x06\x05\x08\x07)` to synchronize framing, then unpacks the payload mapping complex bins into In-phase (I) and Quadrature (Q) data pairs.
4. **Processing Chain and DSP Analytics:** A concurrent thread evaluates the buffered phase data. It utilizes SciPy’s Butterworth bandpass filter and `find_peaks` routines to map local extrema corresponding to systolic pulses, quantifying the Beats Per Minute.
5. **Output (WebSockets):** The processed, computed BPM and localized phase shift are piped via a local active WebSocket (`/ws`), reflecting directly into a dynamic frontend UI architecture.
 
**4.2 Hardware and Microprocessor Specifications**
 
The successful implementation relies heavily on specialized hardware interfaces and precision micro-controller configurations.
- **Radar Platform:** Texas Instruments mmWave radar (such as the xWR68xx family) is utilized for the primary transmission/reception. Its multi-antenna array enables microscopic resolution tracking vital for physiological measurements.
- **Data Relay:** A direct full-duplex USB serial communication provides real-time digital access bridging the microcontroller units to the PC application.
- **Processing Workstation:** The system's computational core resides on a dedicated Laptop/Workstation executing multi-threaded processing.
- **Software Backbone:** The robust multi-core asynchronous environment of Python (leveraging `uvicorn` and Python Threads) allows simultaneous I/O ingestion and CPU-intensive array math without frame freezing.
 
**4.3 Real-Time Streaming Protocol (WebSockets)**
 
A fundamental distinction in real-time continuous IoT monitoring versus asynchronous fetching is the use of persistent socket connections. Following signal acquisition, the data representation relies entirely on the **WebSocket** protocol.
1. **Low Latency:** WebSockets provide a direct, persistent bidirectional pipeline over a single TCP connection, pushing metrics instantly unlike HTTP polling methods.
2. **High Update Frequency:** The server dispatches live JSON objects up to 10 times per second encompassing the `(timestamp, phase, bpm)` ensuring smooth waveform generation.
 
---
V. Methodology and Data Processing
 
**5.1 Phase Extraction and Radar Model**
 
The analytical core of this project rests inside the transformation of millimeter-scale reflections into one-dimensional vibrational plots. Unlike optical systems, radar processing requires complex baseband representation. The software scans the reflection magnitude vector and selects the "range bin" representing the subject (`argmax(mags)`). For the target bin, the In-phase (I) and Quadrature (Q) components are mapped into angular phases utilizing the arctangent functionality: `phase = np.arctan2(Q, I)`. This mathematically extracts mm-scale displacements over time.
 
**5.2 Pulse Extraction Logic**
 
The heart-rate calculation process relies on rigorously separating the faint cardiac signature from dominant cardiopulmonary activity (breathing heavily masks a heartbeat) and noise. In `pulse_extract.py` and `calculate_bpm_thread()`, the logic expands as follows:
 
1. **Phase Unwrapping & Buffer Window:** Discontinuities resulting from the `$2\pi$` limits of arctangent results are smoothed out via `np.unwrap`. The calculations evaluate overlapping frames (e.g., a buffer containing 30 seconds of samples at 20 frames per second).
2. **Butterworth Bandpass Filter:** The unwrapped chest-displacement signal is pushed through a 4th-order Butterworth bandpass filter calibrated precisely for cardiovascular frequencies. The frequencies map precisely between lower bounds of 0.8 Hz ($48 \text{ BPM}$) to upper bounds of 3.0 Hz ($180 \text{ BPM}$).
3. **Peak Differentiation Algorithm:** Extraneous perturbations are mapped via the SciPy `find_peaks` application. Through configuring strict distance parameters derived from sample rates (`fs * 0.4`), false peak positives are drastically reduced. 
 
**5.3 Software and File Structure**
 
The structure follows a clean and deliberate separation of ingest, offline compute, and runtime operations:
 
- **Role of the Backend Engine (`realtime_app.py`):** Acts as the nexus of operations. It launches separate background daemon threads for radar ingestion and DSP computations to avoid blocking the Uvicorn web loop. Handles CSV logging of `pulse_live_[DATETIME].csv` providing a raw ground-truth pipeline.
- **Role of the Extraction Utility (`pulse_extract.py`):** Responsible for offline batch-processing of raw radar logs. Employs advanced Pandas/NumPy logic allowing the estimation of session-based rolling BPMs from massive CSV backlogs over customized window thresholds.
- **Hardware Config (`xwr68xx_profile...`):** A strictly formatted profile sent directly to the CLI com-port dictating FMCW chirp ramps and radar physical settings to bias frame rates perfectly at 20 FPS.
 
**5.4 Implementation Summary**
 
- **Optimization Strategy:** Memory overflows are handled via static size windowing (`phase_buffer` restricted to 600 indices). Calculations utilize compiled native arrays over Python lists (leveraging NumPy operations) maximizing algorithmic throughput and preventing delays from slowing downstream WebSocket events.
- **Robust Exception Handling:** Serial ports are aggressively managed via lifecycle handlers ensuring connection teardown upon server stop, sending `sensorStop` payloads unconditionally to restore hardware to standby state.

---
VI. Results, Performance, and Discussion
 
**6.1 Performance Metrics (Quantified Results)**
 
The operational profile illustrates strong validity regarding real-world application throughput:
 
- **Throughput (FPS):** Handshake configurations are optimized explicitly for 20 frames per second. Processing threads run asynchronously on the host machine maintaining the phase extraction pace securely without frame dropping.
- **Latency Verification:** Because serial polling happens sequentially while math operates dynamically in separate thread pools, the end-to-end delay (vibration-to-pixel projection) falls securely under $100 \text{ms}$, granting visually perfect synchronicity on the HTML front-end interface.
- **Mathematical Constancy:** Baseline stability ensures BPM evaluations ignore random noise anomalies relying instead on structured intervals inside a moving analytical window.
 
**6.2 System Output Analysis**
 
The centralized dashboard handles the output responsibilities converting dry metadata into physiological truth streams:
- **Instantaneous Web Reporting:** Web frontends render live-stream metrics via visual gauges, capturing immediate spikes or stabilization in physiological cadence. 
- **Rolling Data Logs:** Live captures are safely archived systematically into chronologically labeled CSV payloads containing specific columns (`timestamp, frame, range_bin, I, Q, phase, heart_rate_bpm`) which provides verifiable references and datasets required for subsequent Machine Learning tuning.

**6.3 Applications**
 
The non-intrusive paradigm of mmWave Pulse detection enables use-cases historically impossible with traditional architectures:
1. **Sleep Analysis & Diagnostic Centers:** Seamless 24/7 logging of vital profiles without encumbering a patient attached to electrode cables. 
2. **Telemedicine / Remote Patient Monitoring:** Enabling elderly patients to have baseline pulse tracking autonomously operating inside their rooms passively recording and transferring metadata to healthcare endpoints. 
3. **Automotive Monitoring:** Implementation under dashboard fixtures assisting in detecting driver drowsiness levels by examining cardiovascular variability markers remotely and securely.
 
---
VII. Conclusion and Future Scope
 
**7.1 Conclusion**
 
The implementation of the Arterial Pulse Detection & Monitoring System successfully merged FMCW hardware interfaces with advanced Digital Signal Processing methodologies. Leveraging high-frequency mmWave capture pipelines driven directly through Python and WebSockets ensures that latency overhead remains negligible. The resulting prototype delivers a capable tool providing persistent, highly-accurate localized heart rate quantification without the limitations and physiological constraints intrinsic to physical sensor mediums. The solution fulfills real-time diagnostic criteria ensuring a potent replacement for intrusive vital monitoring platforms.
 
**7.2 Future Scope**
 
To capitalize on this foundational engineering, several distinct progressions present substantial potential utility:
1. **Machine Learning Model Integration:** Moving past hard-coded DSP algorithms toward predictive Neural Network processing to estimate heart-rate variability mapping complex relationships involving demographics (age, BMI) against micro-Doppler signals directly.
2. **Respiration Tracking:** Expanding the DSP filter modules to capture macroscopic chest cavity shifts explicitly evaluating respiratory rate in parallel with the cardiovascular measurements.
3. **Adaptive Environmental De-Noising:** Implementation of independent component analysis to successfully cancel artifacts stemming from gross-body movements (standing up, shifting), retaining signal purity despite non-stationary human behavior. 
4. **Embedded Processing Validation:** Transitioning the filtering workloads entirely out of the Workstation paradigm back to the edge device, allowing the embedded MCU to pipe the BPM estimates independently and minimizing network throughput requirements exponentially.
 
---
VIII. References

- Texas Instruments mmWave Radar documentation and CLI specifications.
- Project Source Code and Hardware configurations logic derived from `realtime_app.py` and CSV Extraction implementations (`pulse_extract.py`).
- Digital Signal Processing texts outlining properties of the Butterworth filter and sliding array computations.
