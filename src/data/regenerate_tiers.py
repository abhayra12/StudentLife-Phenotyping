"""
Regenerate Participant Tiers

This script regenerates the participant_tiers.csv file by analyzing
sensor coverage in data/raw/dataset/sensing.
"""

import pandas as pd
from pathlib import Path

def regenerate_tiers():
    sensing_path = Path('data/raw/dataset/sensing')
    output_path = Path('data/processed/participant_tiers.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sensors = [
        'activity', 'audio', 'bluetooth', 'conversation', 'dark',
        'gps', 'phonecharge', 'phonelock', 'wifi', 'wifi_location'
    ]
    
    # Get all participants
    participants = set()
    for sensor in sensors:
        sensor_dir = sensing_path / sensor
        if sensor_dir.exists():
            files = list(sensor_dir.glob('*.csv'))
            for f in files:
                # Format: sensor_u00.csv or u00.sensor.csv
                name = f.stem
                if name.startswith(sensor):
                    pid = name.split('_')[-1]
                else:
                    pid = name.split('.')[0]
                participants.add(pid)
                
    participants = sorted(list(participants))
    
    # Calculate coverage
    data = []
    for pid in participants:
        sensor_count = 0
        missing_sensors = []
        
        for sensor in sensors:
            # Check both naming conventions
            f1 = sensing_path / sensor / f"{sensor}_{pid}.csv"
            f2 = sensing_path / sensor / f"{pid}.{sensor}.csv"
            
            if f1.exists() or f2.exists():
                sensor_count += 1
            else:
                missing_sensors.append(sensor)
                
        # Tier logic
        if sensor_count >= 8:
            tier = "Tier 1: Excellent"
        elif sensor_count >= 5:
            tier = "Tier 2: Good"
        else:
            tier = "Tier 3: Low Quality"
            
        data.append({
            'participant': pid,
            'quality_tier': tier,
            'sensor_count': sensor_count,
            'missing_sensors': str(missing_sensors)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Regenerated tiers for {len(df)} participants at {output_path}")
    print(df['quality_tier'].value_counts())

if __name__ == "__main__":
    regenerate_tiers()
