"""
StudentLife Dataset Downloader
Downloads the StudentLife dataset from Dartmouth College.

Dataset: 53GB compressed, ~100GB extracted
URL: https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar."""
    
    # Make sure directory exists
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if destination.exists():
        print(f"✅ File already exists: {destination}")
        print(f"📊 Size: {destination.stat().st_size / (1024**3):.2f} GB")
        response = input("Download again? (y/N): ")
        if response.lower() != 'y':
            return
    
    print(f"📥 Downloading StudentLife dataset...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    print(f"   Size: ~53 GB (this will take time!)")
    print()
    
    # Stream download with progress bar
    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)
    
    print(f"\n✅ Download complete!")
    print(f"📦 Saved to: {destination}")
    print(f"📊 Final size: {destination.stat().st_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    # Dataset URL
    url = "https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2"
    
    # Destination path
    destination = Path("data/raw/dataset.tar.bz2")
    
    try:
        download_file(url, destination)
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative: Download manually from:")
        print("https://studentlife.cs.dartmouth.edu/dataset.html")
