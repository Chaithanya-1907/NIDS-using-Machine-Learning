# src/realtime_detector.py
"""
This script captures live network traffic, processes it in time-based batches,
and makes accurate, stateful intrusion predictions.

--- TIME-BASED BATCHING IMPLEMENTATION ---
This is the most robust version. It decouples packet capture from prediction.
A dedicated sniffer thread gathers connection data continuously. A separate
worker thread wakes up at a fixed interval (e.g., 60 seconds), analyzes all
traffic from that period, makes predictions, and then clears the state.

This approach provides the highest accuracy by building a rich context for
each connection, closely mimicking how the training dataset was generated and
dramatically reducing false positives.
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
import threading
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

from config import MODEL_PATH, COLUMNS, CATEGORICAL_FEATURES

# --- CONFIGURATION ---
PREDICTION_INTERVAL = 60  # seconds. Analyze traffic every 60 seconds.

print("--- Time-Based Real-Time Network Intrusion Detector ---")

# 1. Load the trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
    exit()

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# --- STATEFUL TRACKING SETUP (THREAD-SAFE) ---
# This dictionary is shared between the sniffer and predictor threads.
connections = defaultdict(lambda: {
    'start_time': time.time(),
    'last_seen_time': time.time(),
    'packet_count': 0,
    'src_bytes': 0,
    'flags': set()
})
# A lock to prevent race conditions when accessing the shared 'connections' dict.
connections_lock = threading.Lock()

# Prepare feature lists once
feature_columns = [col for col in COLUMNS if col not in ['attack', 'difficulty']]
numerical_features = [col for col in feature_columns if col not in CATEGORICAL_FEATURES]


def generate_features_and_predict(conn_key, conn_data):
    """Calculates features from aggregated data and makes a prediction."""
    # Calculate duration
    duration = conn_data['last_seen_time'] - conn_data['start_time']
    # Avoid division by zero if duration is 0
    duration = max(duration, 1e-6)

    # --- Create feature dictionary with placeholders ---
    features = {key: 0 for key in feature_columns}

    # Fill in the features we can calculate
    features['duration'] = duration
    features['protocol_type'] = conn_key[4]
    features['src_bytes'] = conn_data['src_bytes']
    features['count'] = conn_data['packet_count']
    features['srv_count'] = conn_data['packet_count']  # Simplified assumption

    # Placeholders for complex features
    features['service'] = 'private'
    features['flag'] = list(conn_data['flags'])[0] if conn_data['flags'] else 'SF'

    # Create DataFrame
    df = pd.DataFrame([features], columns=feature_columns)

    # Set data types
    try:
        df[numerical_features] = df[numerical_features].astype(np.float64)
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    except Exception as e:
        print(f"Error during data type conversion: {e}")
        return

    # --- Make a prediction ---
    try:
        prediction_numeric = model.predict(df)
        result = 'Attack' if prediction_numeric[0] == 1 else 'Normal'

        protocol, src, sport, dst, dport = conn_key[4], conn_key[0], conn_key[1], conn_key[2], conn_key[3]
        print(
            f"[PREDICTION] Flow: {protocol.upper()} {src}:{sport} -> {dst}:{dport} | "
            f"Pkts: {features['count']:<4} | Duration: {duration:<5.2f}s | "
            f"Result: {result}"
        )
    except Exception as e:
        print(f"Error during prediction: {e}")


def prediction_worker():
    """
    A worker that runs in a separate thread to make predictions periodically.
    """
    while True:
        time.sleep(PREDICTION_INTERVAL)

        # Make a copy of the current connections to analyze
        with connections_lock:
            if not connections:
                continue  # Skip if no traffic was captured

            # This is a snapshot of the connections from the last interval
            current_connections = connections.copy()
            # Clear the shared dictionary to start the next collection window
            connections.clear()

        print(
            f"\n--- Analyzing {len(current_connections)} traffic flows from the last {PREDICTION_INTERVAL} seconds ---")

        for key, data in current_connections.items():
            generate_features_and_predict(key, data)


def packet_callback(packet):
    """
    The function called by Scapy for each packet.
    Its only job is to update the shared 'connections' dictionary.
    """
    if not packet.haslayer(IP):
        return

    ip_layer = packet.getlayer(IP)
    proto, src_port, dst_port, flags = 'other', 0, 0, set()

    if packet.haslayer(TCP):
        proto = 'tcp'
        tcp_layer = packet.getlayer(TCP)
        src_port, dst_port = tcp_layer.sport, tcp_layer.dport
        for flag in tcp_layer.flags.flagrepr(): flags.add(flag)
    elif packet.haslayer(UDP):
        proto = 'udp'
        udp_layer = packet.getlayer(UDP)
        src_port, dst_port = udp_layer.sport, udp_layer.dport

    conn_key = (ip_layer.src, src_port, ip_layer.dst, dst_port, proto)

    # Lock the dictionary to prevent conflicts between threads
    with connections_lock:
        conn = connections[conn_key]
        conn['packet_count'] += 1
        conn['src_bytes'] += len(ip_layer.payload)
        conn['last_seen_time'] = time.time()
        conn['flags'].update(flags)


def start_sniffing():
    """Starts the packet sniffer and the prediction worker thread."""
    # Create and start the background prediction thread
    predictor_thread = threading.Thread(target=prediction_worker, daemon=True)
    predictor_thread.start()

    print(
        f"\nStarting packet capture. Predictions will be made every {PREDICTION_INTERVAL} seconds. Press Ctrl+C to stop.")
    try:
        sniff(prn=packet_callback, store=0)
    except PermissionError:
        print("\nError: Permission denied. You need to run this script with root/administrator privileges.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    start_sniffing()