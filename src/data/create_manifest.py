"""
Create sensing data manifest

This script analyzes the StudentLife sensing folder and generates
a comprehensive JSON manifest documenting all sensor types, participants,
file statistics, and coverage information.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd


def create_sensing_manifest(sensing_path: str) -> Dict:
    """
    Analyze sensing folder and create JSON manifest.
    
    Args:
        sensing_path: Path to sensing/ directory
        
    Returns:
        Dictionary containing:
        - sensor_types: List of 10 sensor types
        - participants: List of participant IDs
        - per_sensor_stats: Detailed statistics per sensor
        - global_stats: Overall dataset statistics
    """
    sensing_path = Path(sensing_path)
    
    if not sensing_path.exists():
        raise ValueError(f"Sensing path does not exist: {sensing_path}")
    
    # Expected sensor types
    expected_sensors = [
        'activity', 'audio', 'bluetooth', 'conversation', 'dark',
        'gps', 'phonecharge', 'phonelock', 'wifi', 'wifi_location'
    ]
    
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "sensing_path": str(sensing_path.absolute()),
        "sensor_types": [],
        "participants": set(),
        "per_sensor_stats": {},
        "global_stats": {}
    }
    
    total_files = 0
    total_size_bytes = 0
    
    # Analyze each sensor type
    for sensor in expected_sensors:
        sensor_dir = sensing_path / sensor
        
        if not sensor_dir.exists():
            print(f"Warning: Sensor directory not found: {sensor}")
            continue
            
        manifest["sensor_types"].append(sensor)
        
        # Get all CSV files
        csv_files = list(sensor_dir.glob("*.csv"))
        
        # Extract participant IDs from filenames
        participants_in_sensor = set()
        file_sizes = []
        dates = []
        
        for csv_file in csv_files:
            # Filename format: sensing_<sensor>_<participant_id>.csv
            # or <participant_id>.<sensor>.csv
            filename = csv_file.stem
            
            # Try to extract participant ID
            if filename.startswith(sensor):
                # Format: activity_u00.csv
                parts = filename.split('_')
                if len(parts) >= 2:
                    participant_id = parts[-1]
                    participants_in_sensor.add(participant_id)
            else:
                # Format: u00.activity.csv
                parts = filename.split('.')
                if len(parts) >= 2:
                    participant_id = parts[0]
                    participants_in_sensor.add(participant_id)
            
            file_sizes.append(csv_file.stat().st_size)
            total_size_bytes += csv_file.stat().st_size
            total_files += 1
            
            # Try to get date range from file
            try:
                df = pd.read_csv(csv_file, nrows=1)
                if 'timestamp' in df.columns:
                    dates.append(df['timestamp'].iloc[0])
            except:
                pass
        
        # Update participants
        manifest["participants"].update(participants_in_sensor)
        
        # Calculate sensor statistics
        manifest["per_sensor_stats"][sensor] = {
            "total_files": len(csv_files),
            "participants_covered": sorted(list(participants_in_sensor)),
            "num_participants": len(participants_in_sensor),
            "total_size_mb": round(sum(file_sizes) / (1024 * 1024), 2),
            "avg_file_size_kb": round(sum(file_sizes) / len(file_sizes) / 1024, 2) if file_sizes else 0,
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            }
        }
    
    # Convert participants set to sorted list
    manifest["participants"] = sorted(list(manifest["participants"]))
    
    # Global statistics
    manifest["global_stats"] = {
        "total_sensor_types": len(manifest["sensor_types"]),
        "total_participants": len(manifest["participants"]),
        "total_files": total_files,
        "total_size_gb": round(total_size_bytes / (1024 ** 3), 2),
        "avg_files_per_sensor": round(total_files / len(manifest["sensor_types"]), 1) if manifest["sensor_types"] else 0
    }
    
    return manifest


def save_manifest(manifest: Dict, output_path: str):
    """Save manifest to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Manifest saved to: {output_path}")


def print_summary(manifest: Dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("SENSING DATA MANIFEST SUMMARY")
    print("="*60)
    
    gs = manifest["global_stats"]
    print(f"\nDataset Overview:")
    print(f"  Total Sensor Types: {gs['total_sensor_types']}")
    print(f"  Total Participants: {gs['total_participants']}")
    print(f"  Total Files: {gs['total_files']}")
    print(f"  Total Size: {gs['total_size_gb']} GB")
    
    print(f"\nSensors: {', '.join(manifest['sensor_types'])}")
    print(f"\nParticipants: {', '.join(manifest['participants'][:10])}..." if len(manifest['participants']) > 10 else f"\nParticipants: {', '.join(manifest['participants'])}")
    
    print("\nPer-Sensor Statistics:")
    for sensor, stats in manifest["per_sensor_stats"].items():
        print(f"  {sensor:15s}: {stats['total_files']:3d} files, "
              f"{stats['num_participants']:2d} participants, "
              f"{stats['total_size_mb']:6.1f} MB")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Default paths
    sensing_path = "data/raw/dataset/sensing"
    output_path = "data/raw/sensing_manifest.json"
    
    print(f"Analyzing sensing data at: {sensing_path}")
    
    # Create manifest
    manifest = create_sensing_manifest(sensing_path)
    
    # Print summary
    print_summary(manifest)
    
    # Save to JSON
    save_manifest(manifest, output_path)
    
    print(f"\nManifest created successfully!")
    print(f"Output: {output_path}")
