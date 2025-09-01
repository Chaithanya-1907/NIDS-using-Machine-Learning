# src/api.py
"""
This script creates a simple Flask web API to serve predictions from the trained NIDS model.
"""
import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify

from config import MODEL_PATH, COLUMNS

# 1. Initialize the Flask application
app = Flask(__name__)

# 2. Load the model ONCE when the application starts
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded for API.")
else:
    model = None
    print("Model file not found. API will not be able to make predictions.")

# Get the list of feature names the model expects
feature_columns = [col for col in COLUMNS if col not in ['attack', 'difficulty']]

# 3. Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Receives connection data in JSON format and returns a prediction."""
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # 4. Get JSON data from the request
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON input."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {e}"}), 400

    # 5. Convert the JSON data into a pandas DataFrame
    # The keys in the JSON must match the feature names.
    try:
        input_df = pd.DataFrame([data], columns=feature_columns)
        # Fill any missing columns with 0, a safe default
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    except Exception as e:
        return jsonify({"error": f"Failed to create DataFrame: {e}"}), 400
        
    # 6. Make a prediction
    try:
        prediction_numeric = model.predict(input_df)
        result = 'Attack' if prediction_numeric[0] == 1 else 'Normal'
        
        # 7. Return the result as JSON
        return jsonify({
            "prediction": result,
            "is_attack": int(prediction_numeric[0])
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# 8. Define a simple root endpoint to confirm the API is running
@app.route('/', methods=['GET'])
def index():
    return "NIDS Prediction API is running. Use the /predict endpoint for predictions."

# 9. Run the Flask app
if __name__ == '__main__':
    # host='0.0.0.0' makes the API accessible from other devices on the same network
    app.run(host='0.0.0.0', port=5000, debug=True)