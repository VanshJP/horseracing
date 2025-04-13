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
        avg_odds=('dollar_odds', 'mean'),
        # New metrics
        std_finish=('finish', lambda x: x.std() if len(x) > 1 else 0),  # Prevent NaN for single race
        best_finish=('finish', 'min'),  # Best ever finish
        worst_finish=('finish', 'max'),  # Worst ever finish
        median_finish=('finish', 'median'),  # Typical performance
        finish_skew=('finish', lambda x: x.skew() if len(x) > 2 else 0),  # Performance distribution, prevent NaN
        avg_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),  # How dominant in wins
    ).reset_index()
    
    # Calculate win rates
    horse_stats['win_rate'] = horse_stats['win_count'] / horse_stats['races_count'].clip(1)
    horse_stats['place_rate'] = horse_stats['place_count'] / horse_stats['races_count'].clip(1)
    horse_stats['show_rate'] = horse_stats['show_count'] / horse_stats['races_count'].clip(1)
    
    # New performance metrics
    horse_stats['consistency_score'] = 1 / (1 + horse_stats['std_finish'].clip(0.001))  # Higher is more consistent, prevent division by zero
    horse_stats['improvement_rate'] = (horse_stats['best_finish'] - horse_stats['worst_finish']) / horse_stats['races_count'].clip(1)
    horse_stats['dominance_score'] = horse_stats['win_rate'] * (1 / (1 + horse_stats['avg_win_margin'].abs().clip(0.001)))  # Use absolute value and prevent division by zero
    
    # === Merge general horse stats first ===
    feature_df = pd.merge(feature_df, horse_stats, on='horse_name', how='left')
    
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
        
        # New last race metrics
        feature_df['last_race_improvement'] = feature_df['last_race_finish'] - feature_df['avg_finish']
        feature_df['last_race_consistency'] = (feature_df['last_race_finish'] - feature_df['avg_finish']).abs()
    
    # === Surface-specific Performance ===
    # Calculate how each horse performs on different surfaces
    if 'surface' in data_for_stats.columns:
        surface_perf = data_for_stats.groupby(['horse_name', 'surface'], observed=True).agg(
            surface_races=('finish', 'count'),
            surface_wins=('finish', lambda x: (x == 1).sum()),
            surface_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            surface_avg_finish=('finish', 'mean'),
            # New surface metrics
            surface_std_finish=('finish', 'std'),
            surface_best_finish=('finish', 'min'),
            surface_worst_finish=('finish', 'max'),
            surface_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        # Calculate win rate by surface
        surface_perf['surface_win_rate'] = surface_perf['surface_wins'] / surface_perf['surface_races']
        surface_perf['surface_place_rate'] = surface_perf['surface_places'] / surface_perf['surface_races']
        
        # New surface metrics
        surface_perf['surface_consistency'] = 1 / (1 + surface_perf['surface_std_finish'].clip(0.001))
        surface_perf['surface_improvement'] = (surface_perf['surface_best_finish'] - surface_perf['surface_worst_finish']) / surface_perf['surface_races'].clip(1)
        
        # Create pivot table for surface win rates
        surface_pivot = surface_perf.pivot_table(
            index='horse_name', 
            columns='surface', 
            values=['surface_win_rate', 'surface_place_rate', 'surface_avg_finish',
                   'surface_consistency', 'surface_improvement'],  # Removed surface_win_margin as it may have extreme values
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
            condition_avg_finish=('finish', 'mean'),
            # New condition metrics
            condition_std_finish=('finish', 'std'),
            condition_best_finish=('finish', 'min'),
            condition_worst_finish=('finish', 'max'),
            condition_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        # Calculate win rate by track condition
        condition_perf['condition_win_rate'] = condition_perf['condition_wins'] / condition_perf['condition_races'].clip(1)
        condition_perf['condition_place_rate'] = condition_perf['condition_places'] / condition_perf['condition_races'].clip(1)
        
        # New condition metrics
        condition_perf['condition_consistency'] = 1 / (1 + condition_perf['condition_std_finish'].clip(0.001))
        condition_perf['condition_improvement'] = (condition_perf['condition_best_finish'] - condition_perf['condition_worst_finish']) / condition_perf['condition_races'].clip(1)
        
        # Create pivot table for condition win rates
        condition_pivot = condition_perf.pivot_table(
            index='horse_name', 
            columns='track_condition', 
            values=['condition_win_rate', 'condition_place_rate', 'condition_avg_finish',
                   'condition_consistency', 'condition_improvement'],  # Removed condition_win_margin
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
            distance_avg_finish=('finish', 'mean'),
            # New distance metrics
            distance_std_finish=('finish', 'std'),
            distance_best_finish=('finish', 'min'),
            distance_worst_finish=('finish', 'max'),
            distance_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        # Calculate win rate by distance
        distance_perf['distance_win_rate'] = distance_perf['distance_wins'] / distance_perf['distance_races'].clip(1)
        distance_perf['distance_place_rate'] = distance_perf['distance_places'] / distance_perf['distance_races'].clip(1)
        
        # New distance metrics
        distance_perf['distance_consistency'] = 1 / (1 + distance_perf['distance_std_finish'].clip(0.001))
        distance_perf['distance_improvement'] = (distance_perf['distance_best_finish'] - distance_perf['distance_worst_finish']) / distance_perf['distance_races'].clip(1)
        
        # Create pivot table for distance win rates
        distance_pivot = distance_perf.pivot_table(
            index='horse_name', 
            columns='distance_bin', 
            values=['distance_win_rate', 'distance_place_rate', 'distance_avg_finish',
                   'distance_consistency', 'distance_improvement'],  # Removed distance_win_margin
            fill_value=0,
            observed=True
        )
        
        # Flatten the column names
        distance_pivot.columns = [f"{col[0]}_{col[1]}".lower() for col in distance_pivot.columns]
        distance_pivot = distance_pivot.reset_index()
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, distance_pivot, on='horse_name', how='left')
        
        # Also add horse-specific performance at current distance bin
        horse_dist_lookup = feature_df[['horse_name', 'distance_bin']].drop_duplicates()
        exact_distance_perf = pd.merge(
            horse_dist_lookup,
            distance_perf,
            on=['horse_name', 'distance_bin'],
            how='left'
        )
        exact_distance_perf = exact_distance_perf.drop('distance_bin', axis=1)
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
            jockey_avg_finish=('finish', 'mean'),
            # New jockey metrics
            jockey_std_finish=('finish', 'std'),
            jockey_best_finish=('finish', 'min'),
            jockey_worst_finish=('finish', 'max'),
            jockey_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_races'].clip(1)
        jockey_stats['jockey_place_rate'] = jockey_stats['jockey_places'] / jockey_stats['jockey_races'].clip(1)
        jockey_stats['jockey_show_rate'] = jockey_stats['jockey_shows'] / jockey_stats['jockey_races'].clip(1)
        
        # New jockey metrics
        jockey_stats['jockey_consistency'] = 1 / (1 + jockey_stats['jockey_std_finish'].clip(0.001))
        jockey_stats['jockey_improvement'] = (jockey_stats['jockey_best_finish'] - jockey_stats['jockey_worst_finish']) / jockey_stats['jockey_races'].clip(1)
        
        # Add better predictive metrics
        jockey_stats['jockey_win_efficiency'] = jockey_stats['jockey_win_rate'] * (1 / np.log1p(jockey_stats['jockey_races']))
        jockey_stats['jockey_form_indicator'] = jockey_stats['jockey_win_rate'] * (1 - jockey_stats['jockey_std_finish'].clip(0, 1))
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, jockey_stats, on='jockey', how='left')
    
    # === Jockey-Horse Combination ===
    # How well does this jockey perform with this specific horse?
    if 'jockey' in data_for_stats.columns and 'horse_name' in data_for_stats.columns:
        jockey_horse = data_for_stats.groupby(['jockey', 'horse_name'], observed=True).agg(
            jh_races=('finish', 'count'),
            jh_wins=('finish', lambda x: (x == 1).sum()),
            jh_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            jh_avg_finish=('finish', 'mean'),
            # New jockey-horse metrics
            jh_std_finish=('finish', 'std'),
            jh_best_finish=('finish', 'min'),
            jh_worst_finish=('finish', 'max'),
            jh_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        # Only calculate rates when there are enough races together
        jockey_horse['jh_win_rate'] = jockey_horse['jh_wins'] / jockey_horse['jh_races'].clip(1)
        jockey_horse['jh_place_rate'] = jockey_horse['jh_places'] / jockey_horse['jh_races'].clip(1)
        
        # New jockey-horse metrics
        jockey_horse['jh_consistency'] = 1 / (1 + jockey_horse['jh_std_finish'].clip(0.001))
        jockey_horse['jh_improvement'] = (jockey_horse['jh_best_finish'] - jockey_horse['jh_worst_finish']) / jockey_horse['jh_races'].clip(1)
        
        # Add interaction metrics - how good is this combo compared to jockey average?
        jockey_horse = pd.merge(jockey_horse, jockey_stats[['jockey', 'jockey_win_rate']], on='jockey', how='left')
        jockey_horse['jh_synergy'] = jockey_horse['jh_win_rate'] - jockey_horse['jockey_win_rate']  # Positive means better than jockey average
        jockey_horse['jh_synergy'] = jockey_horse['jh_synergy'].fillna(0)
        jockey_horse = jockey_horse.drop('jockey_win_rate', axis=1)
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, jockey_horse, on=['jockey', 'horse_name'], how='left')
    
    # === Trainer Performance ===
    # How well does this trainer perform?
    if 'trainer' in data_for_stats.columns:
        trainer_stats = data_for_stats.groupby('trainer', observed=True).agg(
            trainer_races=('trainer', 'count'),
            trainer_wins=('finish', lambda x: (x == 1).sum()),
            trainer_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            trainer_avg_finish=('finish', 'mean'),
            # New trainer metrics
            trainer_std_finish=('finish', 'std'),
            trainer_best_finish=('finish', 'min'),
            trainer_worst_finish=('finish', 'max'),
            trainer_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        trainer_stats['trainer_win_rate'] = trainer_stats['trainer_wins'] / trainer_stats['trainer_races'].clip(1)
        trainer_stats['trainer_place_rate'] = trainer_stats['trainer_places'] / trainer_stats['trainer_races'].clip(1)
        
        # New trainer metrics
        trainer_stats['trainer_consistency'] = 1 / (1 + trainer_stats['trainer_std_finish'].clip(0.001))
        trainer_stats['trainer_improvement'] = (trainer_stats['trainer_best_finish'] - trainer_stats['trainer_worst_finish']) / trainer_stats['trainer_races'].clip(1)
        
        # Add better predictive metrics
        trainer_stats['trainer_effectiveness'] = trainer_stats['trainer_win_rate'] * np.sqrt(trainer_stats['trainer_races'].clip(1))  # Rewards large sample sizes
        trainer_stats['trainer_reliability'] = (trainer_stats['trainer_win_rate'] * 
                                               (1 - trainer_stats['trainer_std_finish'] / trainer_stats['trainer_races'].clip(1).clip(0, 1)))
        
        # Calculate trainer-surface performance if surface data available
        if 'surface' in data_for_stats.columns:
            trainer_surface = data_for_stats.groupby(['trainer', 'surface'], observed=True).agg(
                ts_races=('finish', 'count'),
                ts_wins=('finish', lambda x: (x == 1).sum()),
                ts_avg_finish=('finish', 'mean')
            ).reset_index()
            
            trainer_surface['ts_win_rate'] = trainer_surface['ts_wins'] / trainer_surface['ts_races'].clip(1)
            
            # Pivot to create features for each surface
            ts_pivot = trainer_surface.pivot_table(
                index='trainer',
                columns='surface',
                values='ts_win_rate',
                fill_value=0,
                observed=True
            )
            ts_pivot.columns = [f"trainer_{col}_win_rate" for col in ts_pivot.columns]
            ts_pivot = ts_pivot.reset_index()
            
            # Merge with trainer stats
            trainer_stats = pd.merge(trainer_stats, ts_pivot, on='trainer', how='left')
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, trainer_stats, on='trainer', how='left')
    
    # === Track Performance ===
    # How well does this horse perform at this specific track?
    if 'track_code' in data_for_stats.columns and 'horse_name' in data_for_stats.columns:
        track_perf = data_for_stats.groupby(['horse_name', 'track_code'], observed=True).agg(
            track_races=('finish', 'count'),
            track_wins=('finish', lambda x: (x == 1).sum()),
            track_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            track_avg_finish=('finish', 'mean'),
            # New track metrics
            track_std_finish=('finish', 'std'),
            track_best_finish=('finish', 'min'),
            track_worst_finish=('finish', 'max'),
            track_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        track_perf['track_win_rate'] = track_perf['track_wins'] / track_perf['track_races'].clip(1)
        track_perf['track_place_rate'] = track_perf['track_places'] / track_perf['track_races'].clip(1)
        
        # New track metrics
        track_perf['track_consistency'] = 1 / (1 + track_perf['track_std_finish'].clip(0.001))
        track_perf['track_improvement'] = (track_perf['track_best_finish'] - track_perf['track_worst_finish']) / track_perf['track_races'].clip(1)
        
        # Track familiarity and specialization metrics
        # Get total races for each horse to calculate track specialization
        horse_total_races = data_for_stats.groupby('horse_name', observed=True)['horse_name'].count().reset_index(name='total_races')
        track_perf = pd.merge(track_perf, horse_total_races, on='horse_name', how='left')
        track_perf['track_specialization'] = track_perf['track_races'] / track_perf['total_races'].clip(1)
        track_perf['track_advantage'] = (track_perf['track_win_rate'] - 
                                        data_for_stats.groupby('horse_name', observed=True)['finish'].apply(
                                            lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0
                                        ).reindex(track_perf['horse_name']).values)
        track_perf['track_advantage'] = track_perf['track_advantage'].fillna(0)
        track_perf = track_perf.drop('total_races', axis=1)
        
        # Merge with feature dataframe
        feature_df = pd.merge(feature_df, track_perf, on=['horse_name', 'track_code'], how='left')
    
    # === Post Position Advantage ===
    # Is there an advantage to certain post positions at this track?
    if 'post_position' in data_for_stats.columns and 'track_code' in data_for_stats.columns:
        post_stats = data_for_stats.groupby(['track_code', 'post_position'], observed=True).agg(
            post_races=('post_position', 'count'),
            post_wins=('finish', lambda x: (x == 1).sum()),
            post_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
            post_avg_finish=('finish', 'mean'),
            # New post position metrics
            post_std_finish=('finish', 'std'),
            post_best_finish=('finish', 'min'),
            post_worst_finish=('finish', 'max'),
            post_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
        ).reset_index()
        
        post_stats['post_win_rate'] = post_stats['post_wins'] / post_stats['post_races'].clip(1)
        post_stats['post_place_rate'] = post_stats['post_places'] / post_stats['post_races'].clip(1)
        
        # New post position metrics
        post_stats['post_consistency'] = 1 / (1 + post_stats['post_std_finish'].clip(0.001))
        post_stats['post_importance'] = post_stats['post_win_rate'] * np.log1p(post_stats['post_races'])  # Higher weight to posts with more data
        
        # Calculate positional bias - inside vs outside posts
        post_stats['is_inside'] = (post_stats['post_position'] <= 4).astype(int)
        post_stats['is_outside'] = (post_stats['post_position'] >= 7).astype(int)
        post_stats['is_middle'] = ((post_stats['post_position'] > 4) & (post_stats['post_position'] < 7)).astype(int)
        
        # Compare position to average for the track
        track_avg_win = post_stats.groupby('track_code', observed=True)['post_win_rate'].transform('mean')
        post_stats['post_advantage'] = post_stats['post_win_rate'] - track_avg_win
        
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
                if 'horse_name' in horse_recent.columns:
                    horse_recent = horse_recent.reset_index(drop=True)
                else:
                    horse_recent = horse_recent.reset_index(level=1, drop=True).reset_index()
            
            # Calculate recent performance metrics
            recent_stats = horse_recent.groupby('horse_name', observed=True).agg(
                recent_races=('finish', 'count'),
                recent_wins=('finish', lambda x: (x == 1).sum()),
                recent_places=('finish', lambda x: ((x == 1) | (x == 2)).sum()),
                recent_avg_finish=('finish', 'mean'),
                recent_best_finish=('finish', 'min'),
                recent_worst_finish=('finish', 'max'),
                # New recent form metrics
                recent_std_finish=('finish', 'std'),
                recent_win_margin=('win_time', lambda x: x.diff().mean() if len(x) > 1 else 0),
                recent_improvement=('finish', lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0),
            ).reset_index()
            
            recent_stats['recent_win_rate'] = recent_stats['recent_wins'] / recent_stats['recent_races'].clip(1)
            recent_stats['recent_place_rate'] = recent_stats['recent_places'] / recent_stats['recent_races'].clip(1)
            
            # New recent form metrics with better safeguards
            recent_stats['recent_consistency'] = 1 / (1 + recent_stats['recent_std_finish'].clip(0.001))
            recent_stats['recent_momentum'] = recent_stats['recent_improvement'] / recent_stats['recent_races'].clip(1)
            
            # Add weighted recent performance metrics (more recent races matter more)
            if 'race_date' in horse_recent.columns:
                # Create time-weighted metrics for each horse
                recent_weighted = []
                
                for horse, group in horse_recent.groupby('horse_name'):
                    if len(group) > 1:
                        # Sort by date, most recent last
                        group = group.sort_values('race_date')
                        
                        # Calculate days from first race in group
                        group['days_from_first'] = (group['race_date'] - group['race_date'].min()).dt.days
                        max_days = group['days_from_first'].max()
                        
                        # Create time weights (more recent = higher weight)
                        if max_days > 0:
                            group['time_weight'] = group['days_from_first'] / max_days
                        else:
                            group['time_weight'] = 1.0
                            
                        # Calculate weighted finish and weighted win
                        weighted_finish = (group['finish'] * group['time_weight']).sum() / group['time_weight'].sum()
                        weighted_win = ((group['finish'] == 1) * group['time_weight']).sum() / group['time_weight'].sum()
                        
                        # Calculate trajectory with error handling
                        try:
                            if len(group) > 2:
                                trajectory = np.polyfit(range(len(group)), group['finish'], 1)[0]
                            else:
                                trajectory = 0
                        except (ValueError, np.linalg.LinAlgError):
                            trajectory = 0
                        
                        # Calculate form stability with error handling
                        try:
                            if group['finish'].mean() > 0:
                                form_stability = group['finish'].std() / group['finish'].mean()
                            else:
                                form_stability = 0
                        except (ZeroDivisionError, ValueError):
                            form_stability = 0
                            
                        # Add to results
                        recent_weighted.append({
                            'horse_name': horse,
                            'recent_weighted_finish': weighted_finish,
                            'recent_weighted_win_rate': weighted_win,
                            'recent_trajectory': trajectory,
                            'recent_form_stability': form_stability
                        })
                
                if recent_weighted:
                    # Convert to DataFrame and merge
                    recent_weighted_df = pd.DataFrame(recent_weighted)
                    # Replace any potential infinity or NaN values
                    recent_weighted_df = recent_weighted_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                    recent_stats = pd.merge(recent_stats, recent_weighted_df, on='horse_name', how='left')
            
            # Add form cycle indicators (is horse improving or declining?)
            recent_stats['recent_improving'] = (recent_stats['recent_improvement'] < 0).astype(int)  # Negative improvement means better finish (lower number)
            recent_stats['recent_declining'] = (recent_stats['recent_improvement'] > 0).astype(int)  # Positive improvement means worse finish (higher number)
            
            # Merge with feature dataframe
            feature_df = pd.merge(feature_df, recent_stats, on='horse_name', how='left')
    
    # === Add additional meaningful features ===
    
    # Performance relative to odds (value indicator)
    if 'avg_odds' in feature_df.columns and 'win_rate' in feature_df.columns:
        # Better value indicators
        feature_df['value_indicator'] = feature_df['win_rate'] * feature_df['avg_odds']
        feature_df['value_consistency'] = feature_df['consistency_score'] / (feature_df['avg_odds'].clip(1))
        
        # Odds-adjusted win rate (risk vs reward)
        feature_df['odds_adjusted_win'] = feature_df['win_rate'] - (1 / feature_df['avg_odds'].clip(1))
        
        # Kelly criterion for optimal betting (simplified)
        feature_df['kelly_criterion'] = (feature_df['win_rate'] * feature_df['avg_odds'] - 1) / (feature_df['avg_odds'] - 1).clip(0.001)
        feature_df['kelly_criterion'] = feature_df['kelly_criterion'].clip(-1, 1)  # Cap between -1 and 1
        
        # Value bet indicator (positive means value exists)
        feature_df['value_bet'] = (feature_df['win_rate'] - (1 / feature_df['avg_odds'].clip(1))).clip(-1, 1)
        
        # Odds-based multipliers for existing stats
        feature_df['odds_weight_recent'] = feature_df['recent_win_rate'] * (1 / feature_df['avg_odds'].clip(1))
        feature_df['odds_weight_jockey'] = feature_df['jockey_win_rate'] * feature_df['avg_odds']
        
        # Recent odds vs current odds comparison (if available)
        if 'dollar_odds' in feature_df.columns:
            feature_df['odds_drift'] = feature_df['dollar_odds'] - feature_df['avg_odds']
            feature_df['odds_drift_ratio'] = feature_df['dollar_odds'] / feature_df['avg_odds'].clip(0.1)
            feature_df['odds_movement_indicator'] = feature_df['odds_drift'] * feature_df['win_rate']
        
        # Odds rank adjustment features
        if 'odds_rank' in feature_df.columns and 'field_size' in feature_df.columns:
            # Normalize odds rank from 0-1
            feature_df['odds_rank_normalized'] = feature_df['odds_rank'] / feature_df['field_size']
            
            # Expected value based on odds rank
            feature_df['odds_rank_expected_value'] = (1 - feature_df['odds_rank_normalized']) * feature_df['avg_odds']
            
            # Odds ratio to field average
            if 'race_avg_odds' in feature_df.columns:
                feature_df['odds_ratio_to_avg'] = feature_df['dollar_odds'] / feature_df['race_avg_odds'].clip(0.1)
                feature_df['market_confidence'] = 1 / feature_df['odds_ratio_to_avg'].clip(0.001)
    
    # Days since last race impact (rest factor)
    if 'days_since_last_race' in feature_df.columns:
        # Create rest factor - optimal rest is typically 30-45 days
        feature_df['optimal_rest'] = ((feature_df['days_since_last_race'] >= 30) & 
                                      (feature_df['days_since_last_race'] <= 45)).astype(int)
        
        # Too much rest (> 90 days) can be negative
        feature_df['excess_rest'] = (feature_df['days_since_last_race'] > 90).astype(int)
        
        # First time racing (no previous race data)
        feature_df['first_time_racing'] = (feature_df['days_since_last_race'] >= 365).astype(int)
        
        # New rest metrics
        feature_df['rest_quality'] = np.exp(-(feature_df['days_since_last_race'] - 35)**2 / 200)  # Gaussian centered at 35 days
        feature_df['rest_consistency'] = (feature_df['days_since_last_race'] - feature_df['days_since_last_race'].mean()).abs()
    
    # === Race Dynamics Features ===
    # Add features about the current race dynamics
    if 'race_number' in feature_df.columns and 'track_code' in feature_df.columns:
        # Calculate field size
        field_size = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['horse_name'].count()
        field_size.name = 'field_size'
        feature_df = feature_df.merge(field_size, on=['race_date', 'track_code', 'race_number'], how='left')
        
        # Calculate average odds in the race
        race_odds = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['dollar_odds'].mean()
        race_odds.name = 'race_avg_odds'
        feature_df = feature_df.merge(race_odds, on=['race_date', 'track_code', 'race_number'], how='left')
        
        # Calculate odds rank (lower is better)
        feature_df['odds_rank'] = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['dollar_odds'].rank()
        
        # Calculate normalized odds rank (percentile-based)
        feature_df['odds_rank_pct'] = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['dollar_odds'].rank(pct=True)
        
        # Calculate field quality using transform to maintain index alignment
        if 'win_rate' in feature_df.columns:
            feature_df['field_quality'] = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['win_rate'].transform('mean')
            
            # Calculate variance in field quality
            feature_df['field_quality_std'] = feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['win_rate'].transform('std')
            
            # Calculate competitive index relative to field
            feature_df['competitive_index'] = feature_df['win_rate'] / feature_df['field_quality'].clip(0.001)
    
    # === Cleanup ===
    # Fill NaN values for all the new features
    for col in feature_df.columns:
        if col not in df.columns and col != 'distance_bin':
            feature_df[col] = feature_df[col].fillna(0)
    
    # Replace infinite values with appropriate finite values
    # For win rates and similar ratios, replace inf with 1 (perfect record)
    # For other metrics, replace with large but finite values
    for col in feature_df.columns:
        if feature_df[col].dtype in ['float64', 'float32']:
            # Replace positive infinity with a large but finite value
            feature_df[col] = feature_df[col].replace([np.inf], 1e6)
            # Replace negative infinity with a small but finite value
            feature_df[col] = feature_df[col].replace([-np.inf], -1e6)
            
            # For win rates and similar ratios, cap at 1
            if 'win_rate' in col or 'place_rate' in col or 'show_rate' in col:
                feature_df[col] = feature_df[col].clip(0, 1)
            
            # For odds and margins, cap at reasonable values
            if 'odds' in col.lower() or 'margin' in col.lower():
                feature_df[col] = feature_df[col].clip(-1000, 1000)
    
    # Create relative strength metrics
    if 'field_quality' in feature_df.columns and 'win_rate' in feature_df.columns:
        feature_df['win_rate_vs_field'] = feature_df['win_rate'] - feature_df['field_quality']
        
    if 'field_quality' in feature_df.columns and 'consistency_score' in feature_df.columns:
        feature_df['consistency_vs_field'] = feature_df['consistency_score'] - feature_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)['consistency_score'].transform('mean')
    
    # Favorite/longshot bias adjustment
    if 'odds_rank' in feature_df.columns and 'field_size' in feature_df.columns:
        feature_df['favorite_indicator'] = (feature_df['odds_rank'] == 1).astype(int)
        feature_df['longshot_indicator'] = (feature_df['odds_rank'] >= feature_df['field_size'] * 0.75).astype(int)
    
    return feature_df 