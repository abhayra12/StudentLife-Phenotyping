"""
Activity & Sleep Features

This module extracts behavioral features related to physical activity and sleep.
It includes heuristics for inferring sleep states from passive sensors
(Dark, Phone Lock, Audio) since ground-truth labels are not available.
"""

import pandas as pd
import numpy as np

def calculate_activity_features(df, window='24h'):
    """
    Calculate activity-related features.
    
    Features:
    - active_ratio: Proportion of hour spent active (walking/running).
    - sedentary_ratio: Proportion of hour spent sedentary.
    - rolling_activity: Rolling sum of active minutes over window.
    
    Args:
        df: DataFrame with 'activity_active_minutes'.
        window: Rolling window size (default '24h').
        
    Returns:
        DataFrame with added activity features.
    """
    df = df.copy()
    
    if 'activity_active_minutes' not in df.columns:
        # If missing, return df as is (or raise warning)
        return df
        
    # 1. Hourly Ratios
    # active_minutes is out of 60 (max)
    df['active_ratio'] = df['activity_active_minutes'] / 60.0
    df['active_ratio'] = df['active_ratio'].clip(0, 1)
    
    df['sedentary_ratio'] = 1.0 - df['active_ratio']
    
    # 2. Rolling Activity
    # Ensure sorted by time/participant for rolling
    # Assuming df is single participant or we group by participant
    if 'participant_id' in df.columns:
        df['rolling_activity_24h'] = df.groupby('participant_id')['activity_active_minutes'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    else:
        df['rolling_activity_24h'] = df['activity_active_minutes'].rolling(window, min_periods=1).sum()
        
    return df

def infer_sleep(df):
    """
    Infer sleep state using heuristic:
    Sleep = (Dark > 30m) AND (Locked > 30m) AND (Quiet) AND (Night Hours)
    
    Args:
        df: DataFrame with sensor columns.
        
    Returns:
        DataFrame with 'is_asleep' and sleep metrics.
    """
    df = df.copy()
    
    required_cols = ['dark_minutes', 'phonelock_minutes', 'audio_voice_minutes', 'hour_of_day']
    for col in required_cols:
        if col not in df.columns:
            # Cannot infer sleep without these
            return df
            
    # Heuristic Thresholds
    DARK_THRESH = 30.0      # Minutes
    LOCK_THRESH = 30.0      # Minutes
    VOICE_THRESH = 5.0      # Minutes (Max allowed voice)
    NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 10 PM to 9 AM
    
    # 1. Boolean Flags
    is_dark = df['dark_minutes'] >= DARK_THRESH
    is_locked = df['phonelock_minutes'] >= LOCK_THRESH
    is_quiet = df['audio_voice_minutes'] <= VOICE_THRESH
    is_night = df['hour_of_day'].isin(NIGHT_HOURS)
    
    # 2. Infer Sleep State
    df['is_asleep'] = (is_dark & is_locked & is_quiet & is_night).astype(int)
    
    # 3. Sleep Metrics (Rolling)
    # Sleep duration in last 24h
    if 'participant_id' in df.columns:
        df['sleep_duration_24h'] = df.groupby('participant_id')['is_asleep'].transform(
            lambda x: x.rolling('24h', min_periods=1).sum()
        )
    else:
        df['sleep_duration_24h'] = df['is_asleep'].rolling('24h', min_periods=1).sum()
        
    return df

def process_activity_sleep(df):
    """Apply all activity and sleep feature engineering."""
    df = calculate_activity_features(df)
    df = infer_sleep(df)
    return df
