# src/config.py
"""
This file contains the configuration settings and constants for the project.
"""

# --- Path Configurations ---
# Relative paths are defined from the root of the project directory.
DATA_DIR = "data/"
RAW_DATA_PATH = DATA_DIR + "raw/KDDTrain+.txt"
TEST_DATA_PATH = DATA_DIR + "raw/KDDTest+.txt"
MODEL_DIR = "models/"
MODEL_PATH = MODEL_DIR + "svm_nids_model.pkl"

# --- Data Columns ---
# Defines the schema for the NSL-KDD dataset.
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'difficulty'
]

# --- Model Features ---
# Define which columns are treated as categorical for preprocessing.
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# Define the target variable we want to predict.
TARGET_COLUMN = 'attack'