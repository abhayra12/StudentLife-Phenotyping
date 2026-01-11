"""
Temporal Feature Engineering
---------------------------
Implements time-based features derived from the StudentLife paper.

Key Features:
1. Epoch-based aggregation (Day, Evening, Night)
2. Circular time encoding (Hour of day, Day of week)
3. Term lifecycle features (Week of term, Midterm/Finals phase)
"""

import pandas as pd
import numpy as np
from enum import Enum

class Epoch(Enum):
    NIGHT = (0, 9)    # 12 AM - 9 AM
    DAY = (9, 18)     # 9 AM - 6 PM
    EVENING = (18, 24) # 6 PM - 12 AM

def get_epoch(hour: int) -> str:
    """Returns the epoch name for a given hour."""
    if Epoch.NIGHT.value[0] <= hour < Epoch.NIGHT.value[1]:
        return "Night"
    elif Epoch.DAY.value[0] <= hour < Epoch.DAY.value[1]:
        return "Day"
    else:
        return "Evening"

def add_cyclical_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Adds sin/cos encoding for hour and day of week.
    
    Args:
        df: Input dataframe
        datetime_col: Name of the datetime column
        
    Returns:
        DataFrame with added columns: hour_sin, hour_cos, day_sin, day_cos
    """
    df = df.copy()
    dt = df[datetime_col].dt
    
    # Hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    
    # Day of week (0-6)
    df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
    
    return df

def add_term_lifecycle_features(df: pd.DataFrame, datetime_col: str, term_start_date: str) -> pd.DataFrame:
    """
    Adds features related to the academic term lifecycle.
    
    Args:
        df: Input dataframe
        datetime_col: Name of the datetime column
        term_start_date: String 'YYYY-MM-DD' of term start
        
    Returns:
        DataFrame with: week_of_term, is_midterm, is_finals
    """
    df = df.copy()
    start = pd.Timestamp(term_start_date)
    
    # Calculate week of term (1-indexed)
    df['week_of_term'] = ((df[datetime_col] - start).dt.days // 7) + 1
    
    # Define phases
    # Midterms: Weeks 4-7 (approx)
    # Finals: Weeks 8-10 (approx)
    df['is_midterm'] = df['week_of_term'].between(4, 7).astype(int)
    df['is_finals'] = df['week_of_term'].between(8, 10).astype(int)
    
    return df
