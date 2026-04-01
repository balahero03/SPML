import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    dataset_path = r'c:\Users\Kiruthikraghav\SPML\Arterial_pulse_detection\ml_dataset.csv'
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Please run build_dataset.py first!")
        return
        
    if len(df) < 10:
        print("Not enough data to train a robust ML model. Please collect more CSV sessions!")
        return
        
    print(f"Loaded dataset with {len(df)} examples.")
    
    # Feature Selection
    # X includes raw phase extraction features + demographics
    features = ['std', 'var', 'mad', 'max', 'min', 'energy', 'dom_freq', 'gender', 'age']
    X = df[features]
    y = df['target_bpm']
    
    # Split into 80% Training and 20% Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training ML Model on {len(X_train)} instances...")
    
    # Random Forest is highly robust against overfitting on mixed numeric/categorical features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("--- Model Evaluation ---")
    print(f"Mean Absolute Error: {mae:.2f} BPM")
    print(f"R-Squared Accuracy: {r2:.3f}")
    
    # Save the model 
    save_path = r'c:\Users\Kiruthikraghav\SPML\Arterial_pulse_detection\pulse_rf_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\n[SUCCESS] Model perfectly trained and saved to: {save_path}")
    print("You can now load this .pkl file in your realtime application to predict BPM using Demographics & Phase!")

if __name__ == "__main__":
    train_model()
