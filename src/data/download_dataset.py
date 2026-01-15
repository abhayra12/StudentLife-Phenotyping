import os
import requests
import tarfile
import sys
from tqdm import tqdm

DATASET_URL = "https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2"
RAW_DIR = os.path.join("data", "raw")
TAR_PATH = os.path.join(RAW_DIR, "dataset.tar.bz2")
EXTRACT_PATH = os.path.join(RAW_DIR, "dataset")

def download_file(url, filename):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB

    print(f"Downloading {url}...")
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_sensing_data(tar_path, extract_to):
    """Extract only the 'sensing' folder from the tar archive."""
    print(f"Extracting 'sensing' folder from {tar_path}...")
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            # Filter for sensing folder members
            members = [m for m in tar.getmembers() if m.name.startswith("dataset/sensing")]
            
            if not members:
                print("Error: 'dataset/sensing' folder not found in archive.")
                return

            tar.extractall(path=RAW_DIR, members=members) # Extract to data/raw/dataset/sensing
            print(f"Extraction complete. Data located at: {os.path.join(RAW_DIR, 'dataset', 'sensing')}")
            
    except Exception as e:
        print(f"Extraction failed: {e}")

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    # 1. Download if not exists
    if not os.path.exists(TAR_PATH):
        try:
            download_file(DATASET_URL, TAR_PATH)
        except KeyboardInterrupt:
            print("\nDownload cancelled. Deleting partial file.")
            if os.path.exists(TAR_PATH):
                os.remove(TAR_PATH)
            sys.exit(1)
    else:
        print(f"Archive already exists at {TAR_PATH}, skipping download.")

    # 2. Extract
    # Check if sensing dir already exists to avoid re-extraction
    sensing_path = os.path.join(RAW_DIR, "dataset", "sensing")
    if os.path.exists(sensing_path):
         print(f"Sensing data already exists at {sensing_path}. Skipping extraction.")
    else:
        extract_sensing_data(TAR_PATH, RAW_DIR)

    # 3. Cleanup (Optional - user might want to keep the tar)
    # os.remove(TAR_PATH)
    print("\n✅ Data Setup Complete.")

if __name__ == "__main__":
    main()
