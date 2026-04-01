import pickle
import numpy as np
from build_dataset import extract_features

def predict_pulse_rate(raw_phase_data_10s, gender, age):
    """
    Demo function that uses your newly trained ML model to predict BPM in real-time.
    
    Parameters:
    - raw_phase_data_10s: A numpy array or list of 200 phase values from your sensor
    - gender: 0 (Male) or 1 (Female)
    - age: Integer age (e.g. 24)
    """
    model_path = r'c:\Users\Kiruthikraghav\SPML\Arterial_pulse_detection\pulse_rf_model.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found! Please run train_ml_model.py first.")
        return None
        
    # 1. Extract geometric and frequency features from the raw mmWave signal
    feat = extract_features(raw_phase_data_10s)
    
    # 2. Add demographics
    feat['gender'] = gender
    feat['age'] = age
    
    # Needs to match the model training feature selection exactly
    # ['std', 'var', 'mad', 'max', 'min', 'energy', 'dom_freq', 'gender', 'age']
    feature_array = [
        feat['std'], feat['var'], feat['mad'], feat['max'], 
        feat['min'], feat['energy'], feat['dom_freq'], feat['gender'], feat['age']
    ]
    
    import pandas as pd
    X_input = pd.DataFrame([feature_array], columns=[
        'std', 'var', 'mad', 'max', 'min', 'energy', 'dom_freq', 'gender', 'age'
    ])
    
    # Predict BPM Using Machine Learning (Random Forest Tree Ensemble)
    predicted_bpm = model.predict(X_input)[0]
    
    return round(predicted_bpm, 1)

if __name__ == "__main__":
    print("Initializing Pulse Rate Predictive ML Model Verification Mode")
    print("Generating simulated 10s radar phase signal (200 empty items)...")
    
    # Create fake sensor data that oscillates like a heartbeat to test!
    fake_time = np.linspace(0, 10, 200)
    fake_phase = np.sin(2 * np.pi * 1.5 * fake_time) + np.random.normal(0, 0.1, 200) # Roughly 90 BPM signal
    
    user_gender = 0 # Male
    user_age = 22
    
    print("\n--- Running Inference ---")
    prediction = predict_pulse_rate(fake_phase, user_gender, user_age)
    if prediction:
        print(f"Demographics: Male, 22 yrs old")
        print(f"[PREDICTION] Final ML Predicted Heart Rate: {prediction} BPM")
