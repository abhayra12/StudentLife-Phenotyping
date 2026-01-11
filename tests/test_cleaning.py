"""
Tests for Data Cleaning Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.cleaning import (
    validate_timestamps, 
    validate_values, 
    detect_outliers, 
    handle_missing_data,
    STUDY_START_DATE
)

@pytest.fixture
def sample_activity_df():
    """Create sample activity dataframe."""
    dates = pd.date_range(start=STUDY_START_DATE, periods=10, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'activity_inference': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        'other_col': np.random.randn(10)
    })
    return df

def test_validate_timestamps(sample_activity_df):
    """Test timestamp validation."""
    df = sample_activity_df.copy()
    
    # Add invalid rows
    future_date = pd.Timestamp.now() + pd.Timedelta(days=365)
    past_date = datetime(2000, 1, 1)
    
    new_rows = pd.DataFrame({
        'timestamp': [future_date, past_date, df['timestamp'].iloc[0]], # Future, Past, Duplicate
        'activity_inference': [0, 0, 0],
        'other_col': [0, 0, 0]
    })
    
    df_dirty = pd.concat([df, new_rows], ignore_index=True)
    
    # Shuffle to test ordering
    df_dirty = df_dirty.sample(frac=1).reset_index(drop=True)
    
    cleaned_df, issues = validate_timestamps(df_dirty)
    
    assert len(cleaned_df) == 10 # Should match original length (unique valid dates)
    assert issues['future_timestamps'] == 1
    assert issues['pre_study_timestamps'] == 1
    assert issues['duplicates'] >= 1
    assert cleaned_df['timestamp'].is_monotonic_increasing

def test_validate_values_activity(sample_activity_df):
    """Test activity value validation."""
    df = sample_activity_df.copy()
    
    # Add invalid activity
    df.loc[0, 'activity_inference'] = 99
    
    cleaned_df, invalid_counts = validate_values(df, 'activity')
    
    assert len(cleaned_df) == 9
    assert invalid_counts['activity_inference'] == 1

def test_validate_values_gps():
    """Test GPS value validation."""
    df = pd.DataFrame({
        'latitude': [45.0, 91.0, -91.0, 0.0], # 2 invalid
        'longitude': [100.0, 181.0, -181.0, 0.0] # 2 invalid
    })
    
    cleaned_df, invalid_counts = validate_values(df, 'gps')
    
    # Should remove rows with ANY invalid coordinate? 
    # The current implementation filters sequentially.
    # Lat filter removes 2 rows (indices 1, 2). Remaining: 0, 3.
    # Lon filter on remaining: Index 0 (100.0) ok, Index 3 (0.0) ok.
    # Wait, indices 1 and 2 had invalid lats. 
    # Indices 1 and 2 also had invalid lons.
    # Let's trace:
    # 1. Lat check: removes 91.0 and -91.0. Keeps indices 0, 3.
    # 2. Lon check: checks indices 0, 3. 100.0 and 0.0 are valid.
    # Result: 2 rows.
    
    assert len(cleaned_df) == 2
    assert invalid_counts['latitude'] == 2
    # Note: longitude invalid count might be 0 because those rows were already removed by lat check
    # or it might be calculated on the filtered df. 
    # Current implementation: df = df[lat_mask], then checks lon. 
    # So invalid lons in rows with invalid lats won't be counted.
    
def test_detect_outliers():
    """Test outlier detection."""
    # IQR test
    df = pd.DataFrame({
        'val': [1, 2, 3, 2, 1, 100, 2, 3, 1, 2] # 100 is outlier
    })
    outliers = detect_outliers(df, 'val', method='iqr')
    assert 5 in outliers # Index of 100
    
    # Z-score test (needs more points for Z > 3)
    # 20 normal points (mean~0, std~1) and one outlier (100)
    normal_data = [0] * 20
    data = normal_data + [100]
    df_z = pd.DataFrame({'val': data})
    
    outliers_z = detect_outliers(df_z, 'val', method='zscore')
    assert 20 in outliers_z # Index of 100

def test_handle_missing_data():
    """Test missing data handling."""
    dates = pd.date_range(start=STUDY_START_DATE, periods=10, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'val': [1, np.nan, 3, 4, np.nan, np.nan, 7, 8, 9, 10]
    })
    
    # Gap at index 1 is 1 hour (fillable)
    # Gap at index 4, 5 is 2 hours total gap? 
    # Wait, timestamps are hourly. 
    # Index 3: T+3h, val=4
    # Index 4: T+4h, val=NaN
    # Index 5: T+5h, val=NaN
    # Index 6: T+6h, val=7
    # Diff between 3 and 4 is 1h. Diff between 4 and 5 is 1h.
    # The function calculates diff from previous row.
    # So all diffs are 1h. Max gap is 2h. So all should be filled.
    
    filled_df = handle_missing_data(df, strategy='forward_fill', max_gap_hours=2)
    assert filled_df['val'].isna().sum() == 0
    assert filled_df.iloc[1]['val'] == 1
    
    # Create a large gap
    dates_gap = list(dates[:3]) + [dates[3] + timedelta(hours=5)] # Gap of 5 hours
    df_gap = pd.DataFrame({
        'timestamp': dates_gap,
        'val': [1, 2, np.nan, 4] # NaN at index 2. 
        # Index 1: T+1h. Index 2: T+2h (NaN). Index 3: T+1h+5h = T+6h.
        # Wait, I need to construct it carefully.
    })
    
    # Let's just use explicit timestamps
    t0 = datetime(2013, 1, 1, 10, 0)
    df_gap = pd.DataFrame({
        'timestamp': [
            t0, 
            t0 + timedelta(hours=1), 
            t0 + timedelta(hours=10) # 9 hour gap
        ],
        'val': [1, np.nan, 3] # NaN is at index 1? No, value is associated with timestamp.
        # If we want to fill MISSING ROWS, that's resampling.
        # The function handle_missing_data currently fills NaN VALUES in existing rows.
        # It uses time_diff to determine if we should fill across the gap between rows.
        # If row N has NaN, we look at gap between N-1 and N? Or N and N+1?
        # Usually ffill propagates N-1 to N. So we check gap (N) - (N-1).
    })
    
    # Case: Row 1 has NaN. Gap from Row 0 is 1 hour. Should fill.
    # Case: Row 2 has value. Gap from Row 1 is 9 hours. 
    # If Row 2 had NaN, we wouldn't ffill from Row 1 because gap is large.
    
    df_gap.loc[1, 'val'] = np.nan # Explicitly set NaN
    
    # Gap between row 0 and 1 is 1 hour. Should fill.
    filled = handle_missing_data(df_gap, strategy='forward_fill', max_gap_hours=2)
    assert filled.loc[1, 'val'] == 1.0
    
    # Now case with large gap
    df_gap2 = pd.DataFrame({
        'timestamp': [t0, t0 + timedelta(hours=5)],
        'val': [1, np.nan]
    })
    # Gap is 5 hours. Should NOT fill.
    filled2 = handle_missing_data(df_gap2, strategy='forward_fill', max_gap_hours=2)
    assert np.isnan(filled2.loc[1, 'val'])
