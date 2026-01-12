"""
Tests for Time Alignment Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.alignment import (
    resample_sensor,
    create_hourly_grid,
    align_participant
)

@pytest.fixture
def time_grid():
    return pd.date_range(start='2013-01-01', periods=5, freq='h')

def test_resample_activity(time_grid):
    """Test activity resampling."""
    # Create dummy activity data
    # 2 samples in first hour, 1 in second
    timestamps = [
        time_grid[0] + timedelta(minutes=10),
        time_grid[0] + timedelta(minutes=11), # 1 min later
        time_grid[1] + timedelta(minutes=30)
    ]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'activity inference': [1, 1, 0] # Walk, Walk, Stationary
    })
    # Timestamps are unix seconds in real data, but function handles datetime too
    # Let's use datetime for simplicity here as function supports it
    
    aligned = resample_sensor(df, 'activity', time_grid)
    
    # First hour: 2 samples. 
    # Sample 1 duration: T1 - T0 = 1 min = 60s.
    # Sample 2 duration: T2 - T1 (next is T2 in next hour). 
    # Wait, logic uses shift(-1).
    # T0: 10m. T1: 11m. Diff = 1m = 60s. Active.
    # T1: 11m. Next is T2 (1h 30m). Diff is huge. Cap at 60s. Active.
    # Total active: 60s + 60s = 120s = 2 mins.
    
    assert aligned.loc[time_grid[0], 'activity_active_minutes'] == 2.0
    assert aligned.loc[time_grid[1], 'activity_active_minutes'] == 0.0 # Stationary

def test_resample_conversation(time_grid):
    """Test conversation resampling."""
    # Conversation starting in first hour
    t0 = time_grid[0] + timedelta(minutes=10)
    t1 = t0 + timedelta(minutes=5) # 5 min duration
    
    df = pd.DataFrame({
        'start_timestamp': [t0],
        'end_timestamp': [t1]
    })
    
    aligned = resample_sensor(df, 'conversation', time_grid)
    
    assert aligned.loc[time_grid[0], 'conversation_minutes'] == 5.0
    assert aligned.loc[time_grid[1], 'conversation_minutes'] == 0.0

def test_resample_gps(time_grid):
    """Test GPS resampling."""
    t0 = time_grid[0] + timedelta(minutes=10)
    
    df = pd.DataFrame({
        'time': [t0],
        'latitude': [45.0],
        'longitude': [90.0]
    })
    
    aligned = resample_sensor(df, 'gps', time_grid)
    
    assert aligned.loc[time_grid[0], 'gps_lat'] == 45.0
    assert aligned.loc[time_grid[0], 'gps_samples'] == 1
    assert np.isnan(aligned.loc[time_grid[1], 'gps_lat']) # Should be NaN

def test_align_participant(time_grid):
    """Test full participant alignment."""
    # Activity DF
    df_act = pd.DataFrame({
        'timestamp': [time_grid[0]],
        'activity inference': [1]
    })
    
    # GPS DF
    df_gps = pd.DataFrame({
        'time': [time_grid[0]],
        'latitude': [10.0],
        'longitude': [20.0]
    })
    
    sensors = {'activity': df_act, 'gps': df_gps}
    
    aligned = align_participant('u00', sensors, time_grid[0], time_grid[-1])
    
    assert 'activity_active_minutes' in aligned.columns
    assert 'gps_lat' in aligned.columns
    assert aligned.index.equals(create_hourly_grid(time_grid[0], time_grid[-1]))
