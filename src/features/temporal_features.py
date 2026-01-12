"""
Temporal Features

This module extracts time-based features from timestamps, including
cyclical encodings for periodic variables (hour, day, week) and
contextual features like day parts and academic term phases.
"""

import pandas as pd
import numpy as np

def encode_cyclical_time(df, col_map=None):
    """
    Encode cyclical features using Sine and Cosine transformations.
    
    Args:
        df: DataFrame containing time features.
        col_map: Dictionary mapping column names to their max values (period).
                 Default: {'hour_of_day': 24, 'day_of_week': 7, 'week_of_term': 10}
                 
    Returns:
        DataFrame with added sin/cos columns.
    """
    df = df.copy()
    
    if col_map is None:
        col_map = {
            'hour_of_day': 24,
            'day_of_week': 7,
            'week_of_term': 10
        }
        
    for col, period in col_map.items():
        if col in df.columns:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
            
    return df

def create_day_parts(df, hour_col='hour_of_day'):
    """
    Categorize hours into day parts.
    
    Parts:
    - Night: 00:00 - 06:00
    - Morning: 06:00 - 12:00
    - Afternoon: 12:00 - 18:00
    - Evening: 18:00 - 24:00
    
    Args:
        df: DataFrame with hour column.
        hour_col: Name of the hour column.
        
    Returns:
        DataFrame with 'day_part' column.
    """
    df = df.copy()
    
    if hour_col not in df.columns:
        raise ValueError(f"Column {hour_col} not found in DataFrame")
        
    # Vectorized categorization
    conditions = [
        (df[hour_col] >= 0) & (df[hour_col] < 6),
        (df[hour_col] >= 6) & (df[hour_col] < 12),
        (df[hour_col] >= 12) & (df[hour_col] < 18),
        (df[hour_col] >= 18) & (df[hour_col] < 24)
    ]
    choices = ['Night', 'Morning', 'Afternoon', 'Evening']
    
    df['day_part'] = np.select(conditions, choices, default='Unknown')
    
    return df

def add_academic_features(df):
    """
    Add academic context features.
    
    Features:
    - is_weekend: Boolean (True if Sat/Sun)
    - term_phase: 'Early', 'Midterm', 'Finals'
    
    Args:
        df: DataFrame with 'day_of_week' and 'week_of_term'.
        
    Returns:
        DataFrame with added features.
    """
    df = df.copy()
    
    # Weekend (5=Sat, 6=Sun)
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
    # Term Phase
    if 'week_of_term' in df.columns:
        conditions = [
            (df['week_of_term'] <= 3),
            (df['week_of_term'] > 3) & (df['week_of_term'] <= 7),
            (df['week_of_term'] > 7)
        ]
        choices = ['Early', 'Midterm', 'Finals']
        df['term_phase'] = np.select(conditions, choices, default='Unknown')
        
    return df

def process_temporal_features(df):
    """
    Apply all temporal feature engineering steps.
    """
    df = encode_cyclical_time(df)
    df = create_day_parts(df)
    df = add_academic_features(df)
    return df
