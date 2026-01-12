"""
Time Alignment Module

This module aligns disparate sensor streams onto a common hourly time grid.
It handles different sampling rates (continuous vs event-based) and
different aggregation strategies (sum, count, last, mode).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

def create_hourly_grid(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    """Create hourly time grid for the study period."""
    return pd.date_range(start=start_date, end=end_date, freq='h')

def get_timestamp_col(df: pd.DataFrame) -> str:
    """Identify timestamp column name."""
    candidates = ['timestamp', 'time', 'start_timestamp', 'start']
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found. Columns: {df.columns}")

def resample_sensor(df: pd.DataFrame, sensor_name: str, time_grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Resample sensor data to hourly grid.
    
    Args:
        df: Input sensor DataFrame
        sensor_name: Type of sensor
        time_grid: Target hourly index
        
    Returns:
        DataFrame with hourly index and aggregated features
    """
    if df.empty:
        return pd.DataFrame(index=time_grid)
        
    df = df.copy()
    
    # Standardize timestamp
    time_col = get_timestamp_col(df)
    
    # Convert to datetime if needed
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        # Check if it's unix timestamp (likely float or int)
        # StudentLife timestamps are usually unix seconds
        try:
            df[time_col] = pd.to_datetime(df[time_col], unit='s')
        except:
            df[time_col] = pd.to_datetime(df[time_col])
            
    # Set index
    df = df.set_index(time_col).sort_index()
    
    # Aggregation logic based on sensor type
    if sensor_name == 'activity':
        # Inference: 0=Stationary, 1=Walking, 2=Running, 3=Unknown
        # Goal: Active minutes (1+2)
        # Data is 1 min ON / 3 min OFF. 
        # But rows might be frequent within the ON minute.
        # We'll assume each row represents a duration until the next row, capped at e.g. 10s
        
        # Calculate duration of each sample
        df['duration'] = df.index.to_series().diff().shift(-1).dt.total_seconds()
        df['duration'] = df['duration'].fillna(0).clip(upper=60) # Cap at 60s
        
        # Filter for active states
        active_mask = df['activity inference'].isin([1, 2])
        df_active = df[active_mask].copy()
        
        # Resample sum of duration
        hourly = df_active['duration'].resample('h').sum() / 60.0 # Convert to minutes
        hourly.name = 'activity_active_minutes'
        
        # Also maybe 'unknown' minutes?
        unknown_mask = df['activity inference'] == 3
        df_unknown = df[unknown_mask].copy()
        hourly_unknown = df_unknown['duration'].resample('h').sum() / 60.0
        hourly_unknown.name = 'activity_unknown_minutes'
        
        return pd.concat([hourly, hourly_unknown], axis=1).reindex(time_grid).fillna(0)

    elif sensor_name == 'audio':
        # Inference: 0=Silence, 1=Voice, 2=Noise, 3=Unknown
        # Goal: Voice minutes, Noise minutes
        df['duration'] = df.index.to_series().diff().shift(-1).dt.total_seconds()
        df['duration'] = df['duration'].fillna(0).clip(upper=60)
        
        # Voice
        voice_mask = df['audio inference'] == 1
        hourly_voice = df[voice_mask]['duration'].resample('h').sum() / 60.0
        hourly_voice.name = 'audio_voice_minutes'
        
        # Noise
        noise_mask = df['audio inference'] == 2
        hourly_noise = df[noise_mask]['duration'].resample('h').sum() / 60.0
        hourly_noise.name = 'audio_noise_minutes'
        
        return pd.concat([hourly_voice, hourly_noise], axis=1).reindex(time_grid).fillna(0)

    elif sensor_name == 'conversation':
        # Columns: start_timestamp, end_timestamp
        # This is event-based with durations spanning potentially multiple hours
        # We need to split durations across hour boundaries
        
        # Create a series of minutes for every minute in the range? Too slow.
        # Better: Iterate and add to bins? Or simplified: assign to start hour.
        # For precision, let's just assign to start hour for now, or use simple resampling if duration is short.
        # Most conversations are short.
        # Let's use 'start_timestamp' as index.
        
        if 'end_timestamp' in df.columns:
            # Calculate duration if not present
            # Timestamps might need conversion first if not done above (we only converted index)
            # But we set index to time_col. If time_col was start_timestamp, we are good.
            pass
            
        # If we just resample on start time, a 2-hour conversation only counts for the first hour.
        # Correct approach: Expand events to minute-level and resample?
        # Or just sum duration based on start time (approximation).
        # Given 10 weeks of data, approximation is likely acceptable for "behavioral trends".
        
        # Let's calculate duration in minutes
        if ' duration' in df.columns: # Note space in some CSVs? Check schema.
             dur_col = ' duration'
        elif 'duration' in df.columns:
             dur_col = 'duration'
        else:
             # Calc from end - start
             end_col = 'end_timestamp' if 'end_timestamp' in df.columns else ' end_timestamp'
             # We need to ensure end_col is datetime too
             end_series = pd.to_datetime(df[end_col], unit='s')
             start_series = df.index
             df['duration'] = (end_series - start_series).dt.total_seconds()
             dur_col = 'duration'
             
        hourly = df[dur_col].resample('h').sum() / 60.0
        hourly.name = 'conversation_minutes'
        return hourly.reindex(time_grid).fillna(0).to_frame()

    elif sensor_name == 'gps':
        # Lat, Lon. 
        # Goal: Last location, or maybe "distance traveled"?
        # Let's do: Count of samples (coverage), and Last Lat/Lon
        
        hourly_count = df['latitude'].resample('h').count()
        hourly_count.name = 'gps_samples'
        
        hourly_lat = df['latitude'].resample('h').last()
        hourly_lat.name = 'gps_lat'
        
        hourly_lon = df['longitude'].resample('h').last()
        hourly_lon.name = 'gps_lon'
        
        return pd.concat([hourly_count, hourly_lat, hourly_lon], axis=1).reindex(time_grid) # NaNs allowed for lat/lon

    elif sensor_name == 'bluetooth':
        # Count unique MACs per hour
        # Index is time.
        hourly = df['MAC'].resample('h').nunique()
        hourly.name = 'bluetooth_unique_devices'
        return hourly.reindex(time_grid).fillna(0).to_frame()

    elif sensor_name == 'wifi':
        # Count unique BSSIDs
        hourly = df['BSSID'].resample('h').nunique()
        hourly.name = 'wifi_unique_aps'
        return hourly.reindex(time_grid).fillna(0).to_frame()
        
    elif sensor_name == 'wifi_location':
        # Mode of location
        # Resample apply lambda mode
        def get_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else None
            
        hourly = df['location'].resample('h').apply(get_mode)
        hourly.name = 'wifi_location_mode'
        return hourly.reindex(time_grid).to_frame() # Keep NaNs

    elif sensor_name == 'dark':
        # Duration based. Start, End.
        # Similar to conversation, approx by start time
        if 'end' in df.columns:
            end_series = pd.to_datetime(df['end'], unit='s')
            df['duration'] = (end_series - df.index).dt.total_seconds()
        
        hourly = df['duration'].resample('h').sum() / 60.0
        hourly.name = 'dark_minutes'
        return hourly.reindex(time_grid).fillna(0).to_frame()

    elif sensor_name == 'phonecharge':
        # Duration based
        if 'end' in df.columns:
            end_series = pd.to_datetime(df['end'], unit='s')
            df['duration'] = (end_series - df.index).dt.total_seconds()
            
        hourly = df['duration'].resample('h').sum() / 60.0
        hourly.name = 'phonecharge_minutes'
        return hourly.reindex(time_grid).fillna(0).to_frame()

    elif sensor_name == 'phonelock':
        # Duration (locked) + Count (unlocks?)
        # "start" is when it was locked? "end" is when unlocked?
        # Usually phonelock records "lock duration".
        if 'end' in df.columns:
            end_series = pd.to_datetime(df['end'], unit='s')
            df['duration'] = (end_series - df.index).dt.total_seconds()
            
        hourly_dur = df['duration'].resample('h').sum() / 60.0
        hourly_dur.name = 'phonelock_minutes'
        
        hourly_count = df['duration'].resample('h').count()
        hourly_count.name = 'phonelock_count'
        
        return pd.concat([hourly_dur, hourly_count], axis=1).reindex(time_grid).fillna(0)

    else:
        raise ValueError(f"Unknown sensor type: {sensor_name}")

def align_participant(participant_id: str, 
                     sensor_dfs: Dict[str, pd.DataFrame], 
                     start_date: datetime, 
                     end_date: datetime) -> pd.DataFrame:
    """
    Align all sensors for a participant.
    
    Args:
        participant_id: ID
        sensor_dfs: Dict of {sensor_name: dataframe}
        start_date: Study start
        end_date: Study end
        
    Returns:
        Aligned DataFrame with all sensor features
    """
    time_grid = create_hourly_grid(start_date, end_date)
    aligned_dfs = []
    
    for sensor, df in sensor_dfs.items():
        try:
            aligned = resample_sensor(df, sensor, time_grid)
            aligned_dfs.append(aligned)
        except Exception as e:
            print(f"Error aligning {sensor} for {participant_id}: {e}")
            # Continue with other sensors
            
    if not aligned_dfs:
        return pd.DataFrame(index=time_grid)
        
    # Merge all
    full_df = pd.concat(aligned_dfs, axis=1)
    full_df['participant_id'] = participant_id
    
    return full_df
