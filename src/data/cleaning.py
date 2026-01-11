"""
Data Cleaning Pipeline Module

This module provides functions for validating and cleaning sensor data,
including timestamp validation, value validation, outlier detection,
and missing data handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime

# Study constraints
STUDY_START_DATE = datetime(2013, 3, 1)
STUDY_END_DATE = datetime(2013, 6, 1)  # Approximate end of term

def validate_timestamps(df: pd.DataFrame, time_col: str = 'timestamp') -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean timestamps.
    
    Checks for:
    - Future timestamps
    - Timestamps before study start
    - Duplicate timestamps
    - Out-of-order timestamps
    
    Args:
        df: Input DataFrame
        time_col: Name of timestamp column
        
    Returns:
        Tuple containing:
        - Cleaned DataFrame
        - Dictionary of issues found
    """
    issues = {
        'future_timestamps': 0,
        'pre_study_timestamps': 0,
        'duplicates': 0,
        'out_of_order': 0
    }
    
    if df.empty:
        return df, issues
        
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        try:
            # Try converting from unix timestamp (seconds) if numeric
            if np.issubdtype(df[time_col].dtype, np.number):
                df[time_col] = pd.to_datetime(df[time_col], unit='s')
            else:
                df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            raise ValueError(f"Could not convert {time_col} to datetime: {e}")

    # Check for duplicates
    duplicates = df.duplicated(subset=[time_col])
    issues['duplicates'] = duplicates.sum()
    if issues['duplicates'] > 0:
        df = df[~duplicates]
        
    # Check for out of order
    if not df[time_col].is_monotonic_increasing:
        issues['out_of_order'] = 1 # Flag as present
        df = df.sort_values(time_col)
        
    # Check range
    now = pd.Timestamp.now()
    future_mask = df[time_col] > now
    issues['future_timestamps'] = future_mask.sum()
    
    pre_study_mask = df[time_col] < STUDY_START_DATE
    issues['pre_study_timestamps'] = pre_study_mask.sum()
    
    # Filter invalid ranges
    df = df[~future_mask & ~pre_study_mask]
    
    return df, issues

def validate_values(df: pd.DataFrame, sensor_type: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate sensor-specific values.
    
    Args:
        df: Input DataFrame
        sensor_type: Type of sensor ('activity', 'gps', 'conversation', 'phonelock', etc.)
        
    Returns:
        Tuple containing:
        - Cleaned DataFrame
        - Dictionary of invalid counts per column
    """
    invalid_counts = {}
    df = df.copy()
    
    if sensor_type == 'activity':
        # Activity inference: 0=Stationary, 1=Walking, 2=Running, 3=Unknown
        if 'activity_inference' in df.columns:
            valid_mask = df['activity_inference'].isin([0, 1, 2, 3])
            invalid_counts['activity_inference'] = (~valid_mask).sum()
            df = df[valid_mask]
            
    elif sensor_type == 'gps':
        # Lat: [-90, 90], Lon: [-180, 180]
        if 'latitude' in df.columns:
            lat_mask = df['latitude'].between(-90, 90)
            invalid_counts['latitude'] = (~lat_mask).sum()
            df = df[lat_mask]
            
        if 'longitude' in df.columns:
            lon_mask = df['longitude'].between(-180, 180)
            invalid_counts['longitude'] = (~lon_mask).sum()
            df = df[lon_mask]
            
    elif sensor_type == 'conversation':
        # Duration >= 0
        if 'duration' in df.columns: # Assuming column name, check actual data
            dur_mask = df['duration'] >= 0
            invalid_counts['duration'] = (~dur_mask).sum()
            df = df[dur_mask]
        elif 'duration_seconds' in df.columns: # Alternative name
             dur_mask = df['duration_seconds'] >= 0
             invalid_counts['duration_seconds'] = (~dur_mask).sum()
             df = df[dur_mask]

    elif sensor_type == 'phonelock':
        # Lock state usually binary or categorical. Assuming binary 0/1 for now based on typical data
        # Need to verify actual column names and values from EDA
        pass 
        
    return df, invalid_counts

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Index:
    """
    Detect outliers in a specific column.
    
    Args:
        df: Input DataFrame
        column: Column to check
        method: 'iqr' or 'zscore'
        
    Returns:
        Index of outlier rows
    """
    if column not in df.columns:
        return pd.Index([])
        
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        if std == 0:
            return pd.Index([])
        z_scores = np.abs((data - mean) / std)
        outliers = data[z_scores > 3]
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
        
    return outliers.index

def handle_missing_data(df: pd.DataFrame, 
                       time_col: str = 'timestamp',
                       strategy: str = 'forward_fill', 
                       max_gap_hours: float = 2.0) -> pd.DataFrame:
    """
    Handle missing data in time series.
    
    Args:
        df: Input DataFrame (must have timestamp column)
        time_col: Timestamp column name
        strategy: 'forward_fill', 'backward_fill', 'interpolate', 'drop'
        max_gap_hours: Maximum gap size to fill (in hours)
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Ensure sorted by time
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col)
        
    # Set timestamp as index for resampling/filling if needed, or just use as is
    # For filling, we often need a regular grid or just fill based on existing rows
    # Here we assume we are filling NaN values in OTHER columns based on time order
    
    if strategy == 'drop':
        return df.dropna()
        
    # For filling strategies, we need to respect the max_gap
    # Calculate time diffs to identify large gaps
    df['time_diff'] = df[time_col].diff().dt.total_seconds() / 3600.0 # in hours
    
    # Create groups of continuous segments
    # A new group starts whenever the gap is larger than max_gap_hours
    df['segment_id'] = (df['time_diff'] > max_gap_hours).cumsum()
    
    # Apply filling within segments
    columns_to_fill = [c for c in df.columns if c not in [time_col, 'time_diff', 'segment_id']]
    
    if strategy == 'forward_fill':
        df[columns_to_fill] = df.groupby('segment_id')[columns_to_fill].ffill()
        
    elif strategy == 'backward_fill':
        df[columns_to_fill] = df.groupby('segment_id')[columns_to_fill].bfill()
        
    elif strategy == 'interpolate':
        # Interpolation requires numeric index or datetime index
        # We'll set index temporarily
        df_indexed = df.set_index(time_col)
        # We can't easily group by segment and interpolate with datetime index in one go 
        # without iterating. 
        # Simpler approach: mask values near large gaps? 
        # Or just interpolate and then re-introduce NaNs where gaps were too large?
        
        # Let's use the limit argument in interpolate, but it's based on number of rows, not time
        # Time-based interpolation:
        df[columns_to_fill] = df.groupby('segment_id')[columns_to_fill].apply(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
    # Cleanup
    df = df.drop(columns=['time_diff', 'segment_id'])
    
    return df
