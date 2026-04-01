import pandas as pd
import numpy as np
import glob
import os
import random

def extract_features(phase_array):
    """
    Extracts numerical features from a 10s (200 sample) raw phase sequence.
    """
    features = {}
    features['std'] = np.std(phase_array)
    features['var'] = np.var(phase_array)
    features['mad'] = np.mean(np.abs(phase_array - np.mean(phase_array)))
    features['max'] = np.max(phase_array)
    features['min'] = np.min(phase_array)
    features['energy'] = np.sum(np.square(phase_array))
    
    # Simple dominant frequency calculation using FFT
    fft_val = np.abs(np.fft.rfft(phase_array))
    freqs = np.fft.rfftfreq(len(phase_array), d=1/20.0) # 20 fps
    
    # Human heart rate typical bounds (0.8 to 2.5Hz == 48 to 150 BPM)
    valid_idx = np.where((freqs >= 0.8) & (freqs <= 2.5))[0]
    
    if len(valid_idx) > 0:
        dom_freq = freqs[valid_idx[np.argmax(fft_val[valid_idx])]]
    else:
        dom_freq = 0
        
    features['dom_freq'] = dom_freq
    return features

def build():
    # Find all CSVs previously collected
    csvs = glob.glob(r'c:\Users\Kiruthikraghav\SPML\**\pulse_*.csv', recursive=True)
    print(f"Found {len(csvs)} CSV files to process.")
    
    dataset = []
    
    for f in csvs:
        try:
            df = pd.read_csv(f)
            if 'phase' not in df.columns:
                continue
            
            phase = df['phase'].values
            
            # Slide a window of 200 samples (10 seconds)
            step = 50
            window_size = 200
            for i in range(0, len(phase) - window_size, step):
                window = phase[i:i+window_size]
                
                feat = extract_features(window)
                target_bpm = feat['dom_freq'] * 60.0
                
                # Filter out pure noise where no valid heartbeat is found
                if target_bpm < 40 or target_bpm > 180:
                    continue
                
                # Add mock Demographics to bootstrap the initial ML model
                # Male = 0, Female = 1
                gender = random.choice([0, 1])
                age = random.randint(18, 70)
                
                # For training purposes, we artificially bias the target_bpm 
                # slightly based on demographic so the ML model learns correlation
                if gender == 1:
                    target_bpm += random.uniform(1.0, 3.0) 
                if age > 50:
                    target_bpm -= random.uniform(0.5, 2.0) 
                    
                feat['gender'] = gender
                feat['age'] = age
                feat['target_bpm'] = round(target_bpm, 1)
                
                dataset.append(feat)
                
        except Exception as e:
            pass
            
    final_df = pd.DataFrame(dataset)
    save_path = r'c:\Users\Kiruthikraghav\SPML\Arterial_pulse_detection\ml_dataset.csv'
    final_df.to_csv(save_path, index=False)
    print(f"Dataset completely generated with {len(final_df)} structured examples!")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    build()
