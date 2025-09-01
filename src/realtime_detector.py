"""
This script captures live network traffic, processes it to match the model's
expected feature format, and makes real-time intrusion predictions.
"""
import pandas as pd
import joblib
import os
from scapy.all import sniff, IP, TCP, UDP

from config import MODEL_PATH, COLUMNS

# --- IMPORTANT ---
# This is a simplified feature extraction for real-time demonstration.
# The NSL-KDD dataset has complex time-based and connection-based features
# (e.g., 'count', 'srv_serror_rate') that cannot be generated from a single packet.
# We will use placeholder values (0) for these features and extract what we can
# from the packet itself. A real-world NIDS would require a stateful engine.
# -----------------

print("--- Real-Time Network Intrusion Detector ---")

# 1. Load the trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
    exit()

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Get the list of feature names the model was trained on
feature_columns = [col for col in COLUMNS if col not in ['attack', 'difficulty']]

def packet_to_features(packet):
    """Extracts features from a Scapy packet and formats them for the model."""
    features = {key: 0 for key in feature_columns} # Initialize all features to 0

    if packet.haslayer(IP):
        ip_layer = packet.getlayer(IP)
        features['src_bytes'] = len(ip_layer.payload)
        features['dst_bytes'] = 0 # Can't determine from a single packet

    proto = 'other'
    if packet.haslayer(TCP):
        proto = 'tcp'
    elif packet.haslayer(UDP):
        proto = 'udp'

    protocol_map={'tcp':0,'udp':1,'other':2}
    service_map={'private':0}
    flag_map={'SF':0}
    features['protocol_type'] = protocol_map.get(proto,2)
    # Use placeholder values for service and flag, as they are hard to infer
    features['service'] =service_map.get( 'private',0)
    features['flag'] = flag_map.get('SF',0)

    # Create a pandas DataFrame from the single record
    return pd.DataFrame([features], columns=feature_columns)

def packet_callback(packet):
    """
    This function is called for every packet captured by Scapy.
    """
    # 2. Extract features from the packet
    feature_vector = packet_to_features(packet)
    
    # 3. Make a prediction
    prediction_numeric = model.predict(feature_vector)
    result = 'Attack' if prediction_numeric[0] == 1 else 'Normal'

    # 4. Display the result
    print(f"Packet: {packet.summary()}  ==>  Prediction: {result}")

def start_sniffing():
    """Starts the Scapy packet sniffer."""
    print("\nStarting packet capture... Press Ctrl+C to stop.")
    try:
        # 'prn' specifies the callback function for each packet. 'store=0' prevents storing packets in memory.
        sniff(prn=packet_callback, store=0)
    except PermissionError:
        print("\nError: Permission denied. You need to run this script with root/administrator privileges.")
        print("Try running with 'sudo python src/realtime_detector.py'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    start_sniffing()