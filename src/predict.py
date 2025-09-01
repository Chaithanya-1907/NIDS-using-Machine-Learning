# src/predict.py

import pandas as pd
import joblib
import os

# Import configuration from our config file
from config import MODEL_PATH, COLUMNS

def load_model(path=MODEL_PATH):
    """Loads the trained model pipeline from the specified path."""
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        return None
    
    print(f"Loading model from {path}...")
    model = joblib.load(path)
    print("Model loaded successfully.")
    return model

def predict_intrusion(data, model):
    """
    Makes a prediction on new data using the loaded model.
    'data' should be a pandas DataFrame with columns matching the training data.
    """
    # The model's pipeline will handle preprocessing
    predictions = model.predict(data)
    prediction_labels = ['Normal' if p == 0 else 'Attack' for p in predictions]
    return prediction_labels

if __name__ == '__main__':
    # Load the trained model
    nids_model = load_model()

    if nids_model:
        # --- Create Sample Data for Demonstration ---
        # The columns must be in the exact same order as the training data, excluding 'attack' and 'difficulty'
        feature_columns = [col for col in COLUMNS if col not in ['attack', 'difficulty']]

        # Sample 1: A record that looks like normal traffic
        sample_normal = pd.DataFrame([[
            0, 'tcp', 'http', 'SF', 215, 45076, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 8, 8, 1.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0
        ]], columns=feature_columns)

        # Sample 2: A record that looks like a SYN flood attack (high count, serror_rate)
        sample_attack = pd.DataFrame([[
            0, 'tcp', 'ecr_i', 'S0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            250, 15, 1.0, 1.0, 0.0, 0.0, 0.06, 0.07, 0.0, 255, 15, 0.06, 0.07, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
        ]], columns=feature_columns)

        # --- Make Predictions ---
        print("\n--- Making Predictions on Sample Data ---")
        
        # Predict on the normal sample
        prediction_1 = predict_intrusion(sample_normal, nids_model)
        print(f"\nPrediction for Sample 1 (Normal-like): {prediction_1[0]}")
        print("Data:", sample_normal.iloc[0].to_dict())

        # Predict on the attack sample
        prediction_2 = predict_intrusion(sample_attack, nids_model)
        print(f"\nPrediction for Sample 2 (Attack-like): {prediction_2[0]}")
        print("Data:", sample_attack.iloc[0].to_dict())