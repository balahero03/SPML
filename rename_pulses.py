import os
import sys
from pathlib import Path
import pandas as pd

# Add Arterial_pulse_detection to path so we can import pulse_extract
sys.path.append(os.path.join(os.path.dirname(__file__), 'Arterial_pulse_detection'))
try:
    from pulse_extract import estimate_rowwise_bpm, estimate_session_bpm
except ImportError as e:
    print(f"Error importing pulse_extract: {e}")
    sys.exit(1)

spml_dir = Path(__file__).parent

for csv_file in spml_dir.glob("pulse_*.csv"):
    if "bpm" in csv_file.name:
        continue # skip already renamed
        
    print(f"Processing {csv_file.name}...")
    try:
        df = pd.read_csv(csv_file)
        
        required_columns = {"timestamp", "phase"}
        missing = required_columns - set(df.columns)
        if missing:
            print(f"  Missing columns {missing}, skipping.")
            continue
            
        if "heart_rate_bpm" not in df.columns:
            df["heart_rate_bpm"] = estimate_rowwise_bpm(df)
            df.to_csv(csv_file, index=False)
            
        bpm = estimate_session_bpm(df)
        
        if bpm > 0:
            new_name = f"{csv_file.stem}_{int(round(bpm))}bpm.csv"
            new_path = csv_file.with_name(new_name)
            csv_file.rename(new_path)
            print(f"  Renamed to {new_name}")
        else:
            print(f"  No stable BPM found for {csv_file.name}, keeping original name.")
            
    except Exception as e:
        print(f"  Error processing {csv_file.name}: {e}")
