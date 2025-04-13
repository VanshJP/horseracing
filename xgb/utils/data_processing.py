import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Dictionary to store encoders
ENCODERS = {}

def preprocess_data(df, is_training=True):
    """
    Basic preprocessing of the horse racing data
    
    Args:
        df: DataFrame containing the raw data
        is_training: Whether this is training data or prediction data
        
    Returns:
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date columns to datetime
    if 'race_date' in processed_df.columns:
        processed_df['race_date'] = pd.to_datetime(processed_df['race_date'], errors='coerce')
    
    if 'last_race_date' in processed_df.columns:
        processed_df['last_race_date'] = pd.to_datetime(processed_df['last_race_date'], errors='coerce')
    
    # Handle missing values for critical columns
    if 'horse_name' in processed_df.columns:
        processed_df = processed_df.dropna(subset=['horse_name'])
    
    # Calculate days since last race
    if 'race_date' in processed_df.columns and 'last_race_date' in processed_df.columns:
        processed_df['days_since_last_race'] = (processed_df['race_date'] - processed_df['last_race_date']).dt.days
        processed_df['days_since_last_race'] = processed_df['days_since_last_race'].fillna(365)  # No previous race
    
    # Convert dollar_odds to numeric
    if 'dollar_odds' in processed_df.columns:
        processed_df['dollar_odds'] = pd.to_numeric(processed_df['dollar_odds'], errors='coerce')
    
    # Fill missing values in numeric columns with appropriate defaults
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(-1)
    
    # Fill missing values in categorical columns with 'Unknown'
    cat_cols = processed_df.select_dtypes(include=['object']).columns
    processed_df[cat_cols] = processed_df[cat_cols].fillna('Unknown')
    
    return processed_df


def engineer_features(df, historical_data=None, is_training=True):
    """
    Engineer horse-specific performance features under different conditions
    
    Args:
        df: Preprocessed DataFrame
        historical_data: Historical data for computing statistics (for prediction data)
        is_training: Whether this is training data
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    feature_df = df.copy()
    
    # If we're processing prediction data, we need historical data for statistics
    data_for_stats = historical_data if not is_training and historical_data is not None else feature_df
    
    # === Horse Performance Features ===
    
    # Group by horse_name to calculate overall performance metrics
    horse_stats = data_for_stats.groupby('horse_name', observed=True).agg(
        races_count=('horse_name', 'count'),
        win_count=('finish', lambda x: (x == 1).sum()),
        place_count=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
        show_count=('finish', lambda x: ((x == 1) | (x == 2) | (x == 3)).sum()),
        avg_finish=('finish', 'mean'),
        avg_odds=('dollar_odds', 'mean')
    ).reset_index()
    
    # Calculate win rates
    horse_stats['win_rate'] = horse_stats['win_count'] / horse_stats['races_count']
    horse_stats['place_rate'] = horse_stats['place_count'] / horse_stats['races_count']
    horse_stats['show_rate'] = horse_stats['show_count'] / horse_stats['races_count']
    
    # === Surface-specific Performance ===
    # Calculate how each horse performs on different surfaces
    if 'surface' in data_for_stats.columns:
        surface_perf = data_for_stats.groupby(['horse_name', 'surface'], observed=True).agg(
            surface_races=('finish', 'count'),
            surface_wins=('finish', lambda x: (x == 1).sum()),
            surface_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            surface_avg_finish=('finish', 'mean')
        ).reset_index()
        
        # Calculate win rate by surface
        surface_perf['surface_win_rate'] = surface_perf['surface_wins'] / surface_perf['surface_races']
        surface_perf['surface_place_rate'] = surface_perf['surface_places'] / surface_perf['surface_races']
        
        # Create pivot table for surface win rates
        surface_pivot = surface_perf.pivot_table(
            index='horse_name', 
            columns='surface', 
            values=['surface_win_rate', 'surface_place_rate', 'surface_avg_finish'],
            fill_value=0,
            observed=True
        )
        
        # Flatten the column names
        surface_pivot.columns = [f"{col[0]}_{col[1]}".lower() for col in surface_pivot.columns]
        surface_pivot = surface_pivot.reset_index()
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, surface_pivot, on='horse_name', how='left')
    
    # === Track Condition Performance ===
    # How does each horse perform in different track conditions?
    if 'track_condition' in data_for_stats.columns:
        condition_perf = data_for_stats.groupby(['horse_name', 'track_condition'], observed=True).agg(
            condition_races=('finish', 'count'),
            condition_wins=('finish', lambda x: (x == 1).sum()),
            condition_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            condition_avg_finish=('finish', 'mean')
        ).reset_index()
        
        # Calculate win rate by track condition
        condition_perf['condition_win_rate'] = condition_perf['condition_wins'] / condition_perf['condition_races']
        condition_perf['condition_place_rate'] = condition_perf['condition_places'] / condition_perf['condition_races']
        
        # Create pivot table for condition win rates
        condition_pivot = condition_perf.pivot_table(
            index='horse_name', 
            columns='track_condition', 
            values=['condition_win_rate', 'condition_place_rate', 'condition_avg_finish'],
            fill_value=0,
            observed=True
        )
        
        # Flatten the column names
        condition_pivot.columns = [f"{col[0]}_{col[1]}".lower() for col in condition_pivot.columns]
        condition_pivot = condition_pivot.reset_index()
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, condition_pivot, on='horse_name', how='left')
    
    # === Distance Performance ===
    # How does each horse perform at different distances?
    if 'distance' in data_for_stats.columns:
        # Create distance bins for analysis
        data_for_stats['distance_bin'] = pd.cut(
            data_for_stats['distance'], 
            bins=[0, 5, 6, 7, 8, 9, 10, 100],
            labels=['<5', '5-6', '6-7', '7-8', '8-9', '9-10', '>10']
        )
        
        # Add this bin to the feature dataframe too
        if 'distance' in feature_df.columns:
            feature_df['distance_bin'] = pd.cut(
                feature_df['distance'], 
                bins=[0, 5, 6, 7, 8, 9, 10, 100],
                labels=['<5', '5-6', '6-7', '7-8', '8-9', '9-10', '>10']
            )
        
        distance_perf = data_for_stats.groupby(['horse_name', 'distance_bin'], observed=True).agg(
            distance_races=('finish', 'count'),
            distance_wins=('finish', lambda x: (x == 1).sum()),
            distance_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            distance_avg_finish=('finish', 'mean')
        ).reset_index()
        
        # Calculate win rate by distance
        distance_perf['distance_win_rate'] = distance_perf['distance_wins'] / distance_perf['distance_races']
        distance_perf['distance_place_rate'] = distance_perf['distance_places'] / distance_perf['distance_races']
        
        # Create pivot table for distance win rates
        distance_pivot = distance_perf.pivot_table(
            index='horse_name', 
            columns='distance_bin', 
            values=['distance_win_rate', 'distance_place_rate', 'distance_avg_finish'],
            fill_value=0,
            observed=True  # Add this to fix warning
        )
        
        # Flatten the column names
        distance_pivot.columns = [f"{col[0]}_{col[1]}".lower() for col in distance_pivot.columns]
        distance_pivot = distance_pivot.reset_index()
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, distance_pivot, on='horse_name', how='left')
        
        # Also add horse-specific performance at current distance bin
        # First, create a lookup table of current distance bin for each horse
        horse_dist_lookup = feature_df[['horse_name', 'distance_bin']].drop_duplicates()
        
        # Now merge with distance performance data
        exact_distance_perf = pd.merge(
            horse_dist_lookup,
            distance_perf,
            on=['horse_name', 'distance_bin'],
            how='left'
        )
        
        # Drop the distance_bin to avoid duplication
        exact_distance_perf = exact_distance_perf.drop('distance_bin', axis=1)
        
        # Now merge these exact distance stats with the main feature dataframe
        feature_df = pd.merge(
            feature_df,
            exact_distance_perf,
            on='horse_name',
            how='left',
            suffixes=('', '_exact')
        )
    
    # === Jockey Performance ===
    # How well does this jockey perform?
    if 'jockey' in data_for_stats.columns:
        jockey_stats = data_for_stats.groupby('jockey', observed=True).agg(
            jockey_races=('jockey', 'count'),
            jockey_wins=('finish', lambda x: (x == 1).sum()),
            jockey_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            jockey_shows=('finish', lambda x: ((x == 1) | (x == 2) | (x == 3)).sum()),
            jockey_avg_finish=('finish', 'mean')
        ).reset_index()
        
        jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_races']
        jockey_stats['jockey_place_rate'] = jockey_stats['jockey_places'] / jockey_stats['jockey_races']
        jockey_stats['jockey_show_rate'] = jockey_stats['jockey_shows'] / jockey_stats['jockey_races']
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, jockey_stats, on='jockey', how='left')
    
    # === Jockey-Horse Combination ===
    # How well does this jockey perform with this specific horse?
    if 'jockey' in data_for_stats.columns and 'horse_name' in data_for_stats.columns:
        jockey_horse = data_for_stats.groupby(['jockey', 'horse_name'], observed=True).agg(
            jh_races=('finish', 'count'),
            jh_wins=('finish', lambda x: (x == 1).sum()),
            jh_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            jh_avg_finish=('finish', 'mean')
        ).reset_index()
        
        # Only calculate rates when there are enough races together
        jockey_horse['jh_win_rate'] = jockey_horse.apply(
            lambda x: x['jh_wins'] / x['jh_races'] if x['jh_races'] > 0 else 0, axis=1
        )
        
        jockey_horse['jh_place_rate'] = jockey_horse.apply(
            lambda x: x['jh_places'] / x['jh_races'] if x['jh_races'] > 0 else 0, axis=1
        )
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, jockey_horse, on=['jockey', 'horse_name'], how='left')
    
    # === Trainer Performance ===
    # How well does this trainer perform?
    if 'trainer' in data_for_stats.columns:
        trainer_stats = data_for_stats.groupby('trainer', observed=True).agg(
            trainer_races=('trainer', 'count'),
            trainer_wins=('finish', lambda x: (x == 1).sum()),
            trainer_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            trainer_avg_finish=('finish', 'mean')
        ).reset_index()
        
        trainer_stats['trainer_win_rate'] = trainer_stats['trainer_wins'] / trainer_stats['trainer_races']
        trainer_stats['trainer_place_rate'] = trainer_stats['trainer_places'] / trainer_stats['trainer_races']
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, trainer_stats, on='trainer', how='left')
    
    # === Track Performance ===
    # How well does this horse perform at this specific track?
    if 'track_code' in data_for_stats.columns and 'horse_name' in data_for_stats.columns:
        track_perf = data_for_stats.groupby(['horse_name', 'track_code'], observed=True).agg(
            track_races=('finish', 'count'),
            track_wins=('finish', lambda x: (x == 1).sum()),
            track_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            track_avg_finish=('finish', 'mean')
        ).reset_index()
        
        track_perf['track_win_rate'] = track_perf.apply(
            lambda x: x['track_wins'] / x['track_races'] if x['track_races'] > 0 else 0, axis=1
        )
        
        track_perf['track_place_rate'] = track_perf.apply(
            lambda x: x['track_places'] / x['track_races'] if x['track_races'] > 0 else 0, axis=1
        )
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, track_perf, on=['horse_name', 'track_code'], how='left')
    
    # === Post Position Advantage ===
    # Is there an advantage to certain post positions at this track?
    if 'post_position' in data_for_stats.columns and 'track_code' in data_for_stats.columns:
        post_stats = data_for_stats.groupby(['track_code', 'post_position'], observed=True).agg(
            post_races=('post_position', 'count'),
            post_wins=('finish', lambda x: (x == 1).sum()),
            post_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            post_avg_finish=('finish', 'mean')
        ).reset_index()
        
        post_stats['post_win_rate'] = post_stats.apply(
            lambda x: x['post_wins'] / x['post_races'] if x['post_races'] > 0 else 0, axis=1
        )
        
        post_stats['post_place_rate'] = post_stats.apply(
            lambda x: x['post_places'] / x['post_races'] if x['post_races'] > 0 else 0, axis=1
        )
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, post_stats, on=['track_code', 'post_position'], how='left')
    
    # === Recent Form ===
    # How has this horse been performing recently?
    if 'horse_name' in data_for_stats.columns and 'race_date' in data_for_stats.columns:
        # Get the most recent race before the current one for each horse
        data_for_stats_sorted = data_for_stats.sort_values(['horse_name', 'race_date'])
        
        # Group by horse and get last 3 races (excluding current)
        horse_recent = data_for_stats_sorted.groupby('horse_name', observed=True).apply(
            lambda x: x.iloc[-4:-1] if len(x) > 3 else x.iloc[:-1] if len(x) > 1 else pd.DataFrame()
        )
        
        if not horse_recent.empty:
            # Handle the MultiIndex properly
            if isinstance(horse_recent.index, pd.MultiIndex):
                # Check if horse_name already exists as a column
                if 'horse_name' in horse_recent.columns:
                    # Just drop the index completely and use the existing horse_name column
                    horse_recent = horse_recent.reset_index(drop=True)
                else:
                    # Extract the horse_name from the index
                    horse_recent = horse_recent.reset_index(level=1, drop=True).reset_index()
            
            # Calculate recent performance metrics
            recent_stats = horse_recent.groupby('horse_name', observed=True).agg(
                recent_races=('finish', 'count'),
                recent_wins=('finish', lambda x: (x == 1).sum()),
                recent_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
                recent_avg_finish=('finish', 'mean'),
                recent_best_finish=('finish', 'min'),
                recent_worst_finish=('finish', 'max')
            ).reset_index()
            
            recent_stats['recent_win_rate'] = recent_stats.apply(
                lambda x: x['recent_wins'] / x['recent_races'] if x['recent_races'] > 0 else 0, axis=1
            )
            
            recent_stats['recent_place_rate'] = recent_stats.apply(
                lambda x: x['recent_places'] / x['recent_races'] if x['recent_races'] > 0 else 0, axis=1
            )
            
            # Merge with feature dataframe
            feature_df = pd.merge(feature_df, recent_stats, on='horse_name', how='left')
    
    # === Last Race Performance ===
    if 'last_race_finish' in feature_df.columns:
        # Convert to numeric
        feature_df['last_race_finish'] = pd.to_numeric(feature_df['last_race_finish'], errors='coerce')
        # Create indicators
        feature_df['last_race_won'] = (feature_df['last_race_finish'] == 1).astype(int)
        feature_df['last_race_placed'] = ((feature_df['last_race_finish'] == 1) | 
                                          (feature_df['last_race_finish'] == 2)).astype(int)
        feature_df['last_race_showed'] = ((feature_df['last_race_finish'] == 1) | 
                                          (feature_df['last_race_finish'] == 2) | 
                                          (feature_df['last_race_finish'] == 3)).astype(int)
    
    # === Merge general horse stats ===
    feature_df = pd.merge(feature_df, horse_stats, on='horse_name', how='left')
    
    # === Add additional meaningful features ===
    
    # Performance relative to odds (value indicator)
    if 'avg_odds' in feature_df.columns and 'win_rate' in feature_df.columns:
        feature_df['value_indicator'] = feature_df['win_rate'] * feature_df['avg_odds']
    
    # Days since last race impact (rest factor)
    if 'days_since_last_race' in feature_df.columns:
        # Create rest factor - optimal rest is typically 30-45 days
        feature_df['optimal_rest'] = ((feature_df['days_since_last_race'] >= 30) & 
                                      (feature_df['days_since_last_race'] <= 45)).astype(int)
        
        # Too much rest (> 90 days) can be negative
        feature_df['excess_rest'] = (feature_df['days_since_last_race'] > 90).astype(int)
        
        # First time racing (no previous race data)
        feature_df['first_time_racing'] = (feature_df['days_since_last_race'] >= 365).astype(int)
    
    # === Cleanup ===
    # Fill NaN values for all the new features
    for col in feature_df.columns:
        if col not in df.columns and col != 'distance_bin':
            feature_df[col] = feature_df[col].fillna(0)
    
    return feature_df 