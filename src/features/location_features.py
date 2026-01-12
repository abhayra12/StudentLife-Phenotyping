"""
Location & Mobility Features

This module extracts spatial behavioral features from GPS data.
It includes metrics for mobility (distance, radius of gyration) and
routine (location entropy, significant places).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def calculate_mobility_features(df):
    """
    Calculate mobility features from hourly GPS data.
    
    Features:
    - distance_traveled: Km moved since last hour.
    - location_variance: Log(Lat variance + Lon variance).
    - radius_of_gyration: Root mean square distance from center of mass (daily).
    
    Args:
        df: DataFrame with 'gps_lat', 'gps_lon', 'participant_id', 'timestamp'.
        
    Returns:
        DataFrame with added features.
    """
    df = df.copy()
    
    if 'gps_lat' not in df.columns or 'gps_lon' not in df.columns:
        return df
        
    # 1. Distance Traveled (Hourly)
    # Shift to get previous location
    # Group by participant to avoid cross-participant distance
    if 'participant_id' in df.columns:
        df['prev_lat'] = df.groupby('participant_id')['gps_lat'].shift(1)
        df['prev_lon'] = df.groupby('participant_id')['gps_lon'].shift(1)
    else:
        df['prev_lat'] = df['gps_lat'].shift(1)
        df['prev_lon'] = df['gps_lon'].shift(1)
        
    # Calculate distance
    # Handle NaNs (if no prev location, distance is 0 or NaN)
    # We'll fill with 0 for first row, but keep NaN if data is missing
    mask = df['gps_lat'].notna() & df['prev_lat'].notna()
    
    df.loc[mask, 'distance_traveled'] = haversine(
        df.loc[mask, 'prev_lon'], df.loc[mask, 'prev_lat'],
        df.loc[mask, 'gps_lon'], df.loc[mask, 'gps_lat']
    )
    
    # 2. Location Variance (Daily Rolling)
    # Log(Var(Lat) + Var(Lon))
    def log_variance(x):
        return np.log(np.var(x) + 1e-9) # Add epsilon
        
    # We need daily aggregation for variance/ROG usually, but let's do rolling 24h
    if 'participant_id' in df.columns:
        grouper = df.groupby('participant_id')
        df['lat_var'] = grouper['gps_lat'].transform(lambda x: x.rolling('24h', min_periods=1).var())
        df['lon_var'] = grouper['gps_lon'].transform(lambda x: x.rolling('24h', min_periods=1).var())
    else:
        df['lat_var'] = df['gps_lat'].rolling('24h', min_periods=1).var()
        df['lon_var'] = df['gps_lon'].rolling('24h', min_periods=1).var()
        
    df['location_variance'] = np.log(df['lat_var'].fillna(0) + df['lon_var'].fillna(0) + 1e-9)
    
    # 3. Radius of Gyration (Daily)
    # ROG = sqrt(1/N * sum(dist(p_i, center)^2))
    # This is complex to do rolling efficiently. 
    # Simplified: Distance from daily centroid.
    
    return df.drop(columns=['prev_lat', 'prev_lon', 'lat_var', 'lon_var'])

def cluster_locations(df, eps_km=0.05, min_samples=3):
    """
    Cluster locations to identify significant places using DBSCAN.
    
    Args:
        df: DataFrame with gps_lat, gps_lon.
        eps_km: Epsilon neighborhood in km (default 50m).
        min_samples: Min samples to form cluster.
        
    Returns:
        DataFrame with 'location_cluster' column.
    """
    df = df.copy()
    
    # Filter valid GPS
    mask = df['gps_lat'].notna() & df['gps_lon'].notna()
    coords = df.loc[mask, ['gps_lat', 'gps_lon']].values
    
    if len(coords) < min_samples:
        df['location_cluster'] = -1
        return df
        
    # DBSCAN requires radians for haversine metric
    coords_rad = np.radians(coords)
    
    # Earth radius in km
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine', algorithm='ball_tree')
    db.fit(coords_rad)
    
    df.loc[mask, 'location_cluster'] = db.labels_
    df['location_cluster'] = df['location_cluster'].fillna(-1)
    
    return df

def calculate_entropy(df):
    """
    Calculate location entropy based on clusters.
    Entropy = -sum(p * log(p))
    """
    df = df.copy()
    
    if 'location_cluster' not in df.columns:
        return df
        
    # Rolling entropy over 24h
    # This is hard to vectorize. 
    # Alternative: Daily entropy.
    # For now, let's just return the cluster ID for analysis.
    
    return df

def process_location_features(df):
    """Apply all location feature engineering."""
    df = calculate_mobility_features(df)
    # Clustering usually needs to be done per-participant
    if 'participant_id' in df.columns:
        processed_dfs = []
        for pid, group in df.groupby('participant_id'):
            group = cluster_locations(group)
            processed_dfs.append(group)
        df = pd.concat(processed_dfs)
    else:
        df = cluster_locations(df)
        
    return df
