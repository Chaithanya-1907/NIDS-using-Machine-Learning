import os
import urllib.request

# Folder where we want to save the dataset
data_dir = "data/raw"
os.makedirs(data_dir, exist_ok=True)

# Dataset URLs
urls = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
}

# Download files
for filename, url in urls.items():
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Saved to {file_path}")
    else:
        print(f"{filename} already exists. Skipping...")

print("✅ All dataset files are ready!")