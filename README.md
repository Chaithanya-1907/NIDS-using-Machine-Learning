# Network Intrusion Detection System using Machine Learning

This project implements a high-performance Network Intrusion Detection System (NIDS) using the **XGBoost** machine learning model. The system is trained on the benchmark NSL-KDD dataset to accurately classify network traffic as either "normal" or "attack."

The project includes a complete end-to-end workflow: from data preprocessing and optimized model training to deployment as both a real-time command-line monitor and a scalable web API.

## Project Structure

```
Cyber_Project/
│
├── data/
│   └── raw/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
│
├── models/
│   └── xgboost_nids_model.pkl
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── api.py                 # Flask API for on-demand predictions
│   ├── config.py              # Central configuration file
│   ├── download_data.py       # (Optional) Script to download dataset
│   ├── predict.py             # Script for making a single prediction
│   ├── realtime_detector.py   # Real-time network traffic monitor
│   └── train.py               # Script for training the XGBoost model
│
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine.

### 2. Create a Virtual Environment (Recommended)
It is highly recommended to create a virtual environment to keep project dependencies isolated.

```bash
# Navigate to the project root directory
cd path/to/Cyber_Project

# Create a virtual environment
python -m venv sklearn-env

# Activate the virtual environment
# On Windows:
sklearn-env\Scripts\activate
# On macOS/Linux:
source sklearn-env/bin/activate
```

### 3. Install Dependencies
Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the NSL-KDD dataset files (`KDDTrain+.txt` and `KDDTest+.txt`) and place them inside the `data/raw/` directory. You can typically find them by searching for "NSL-KDD dataset download" from the University of New Brunswick (UNB) website.

## How to Run the Project

Make sure you are in the project's root directory (`Cyber_Project/`) and your virtual environment is activated for all commands.

### Step 1: Train the Model
This script will load the dataset, preprocess it, train the XGBoost model, and save the final pipeline to the `models/` directory.

```bash
python src/train.py
```

### Step 2: Run the Real-Time Detector
This script uses `scapy` to monitor your live network traffic and classify packets in real-time. **This requires administrator/root privileges.**

*   **On Windows:** Open Command Prompt or PowerShell **as an Administrator** and run:
    ```bash
    python src/realtime_detector.py
    ```
*   **On macOS/Linux:**
    ```bash
    sudo python src/realtime_detector.py
    ```
Press `Ctrl+C` to stop the detector.

### Step 3: Run the API Server
This script will start a local web server, making your model available for on-demand predictions.

```bash
python src/api.py
```
Leave this terminal running. The API will be available at `http://127.0.0.1:5000`.

### Step 4: Test the API
Open a **new, separate terminal** and use a tool like `curl` to send requests to your running API.

*   **Test with an attack-like sample:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{ "protocol_type": "tcp", "service": "private", "flag": "S0", "count": 200, "serror_rate": 1.0 }' http://127.0.0.1:5000/predict
    ```
    *Expected Response:* `{"is_attack":1,"prediction":"Attack"}`

*   **Test with a normal-like sample:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{ "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 300, "dst_bytes": 5000 }' http://127.0.0.1:5000/predict
    ```
    *Expected Response:* `{"is_attack":0,"prediction":"Normal"}`