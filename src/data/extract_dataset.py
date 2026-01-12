"""
Extract StudentLife Dataset
Extracts the downloaded tar.bz2 archive.
"""

import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_dataset(archive_path, extract_to):
    """Extract tar.bz2 archive with progress."""
    
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        print("Run download_dataset.py first!")
        return
    
    print(f"📦 Extracting StudentLife dataset...")
    print(f"   Archive: {archive_path}")
    print(f"   Destination: {extract_to}")
    print(f"   This may take 10-30 minutes...")
    print()
    
    # Create extraction directory
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Extract with progress
    try:
        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getmembers()
            print(f"📄 Total files: {len(members)}")
            
            for member in tqdm(members, desc="Extracting"):
                # Sanitize filename for Windows
                original_name = member.name
                member.name = member.name.replace("?", "_").replace(":", "_").replace('"', "_").replace("*", "_").replace("<", "_").replace(">", "_").replace("|", "_")
                
                if original_name != member.name:
                    # print(f"Renamed: {original_name} -> {member.name}")
                    pass
                    
                tar.extract(member, path=extract_to)
        
        print(f"\n✅ Extraction complete!")
        print(f"📁 Dataset extracted to: {extract_to}")
        
        # Show directory structure
        print("\n📂 Directory structure:")
        for item in sorted(extract_to.rglob("*"))[:20]:
            if item.is_dir():
                print(f"   📁 {item.relative_to(extract_to)}")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")

if __name__ == "__main__":
    archive_path = Path("data/raw/dataset.tar.bz2")
    extract_to = Path("data/raw")
    
    extract_dataset(archive_path, extract_to)
