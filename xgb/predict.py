#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from utils.data_processing import preprocess_data, engineer_features
from utils.evaluation import calculate_score, track_performance
from models.xgb_model import train_model, predict_top_finishers
import argparse

# Define constants for file paths
DATA_DIR = 'data'
MODEL_DIR = 'xgb/models'
OUTPUT_DIR = 'xgb/output'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Define file paths
    historical_data_path = os.path.join(DATA_DIR, 'all_tracks_hackathon.csv')
    test_data_path = os.path.join(DATA_DIR, 'test_data.csv')  # Changed from keeneland_entries_path
    model_path = os.path.join(MODEL_DIR, 'race_predictor.json')
    predictions_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    features_path = os.path.join(MODEL_DIR, 'feature_list.txt')
    
    # Load historical data
    print("Loading historical data...")
    if not os.path.exists(historical_data_path):
        print(f"Error: Historical data not found at {historical_data_path}")
        return
    
    historical_data = pd.read_csv(historical_data_path, low_memory=False)
    
    # Check if test data exists
    if os.path.exists(test_data_path):
        print("Loading test race data...")
        test_races = pd.read_csv(test_data_path)
    else:
        print(f"Warning: No test race data found at {test_data_path}")
        test_races = None
    
    # Preprocess historical data
    print("Preprocessing historical data...")
    processed_data = preprocess_data(historical_data)
    
    # Engineer features - this creates horse-specific performance metrics 
    # for different track conditions, surfaces, distances, etc.
    print("Engineering features (analyzing horse performance in different conditions)...")
    feature_data = engineer_features(processed_data)
    
    # Print summary of how many features were created
    features_by_category = {
        'Surface': len([col for col in feature_data.columns if 'surface_' in col]),
        'Track Condition': len([col for col in feature_data.columns if 'condition_' in col]),
        'Distance': len([col for col in feature_data.columns if 'distance_' in col]),
        'Jockey': len([col for col in feature_data.columns if 'jockey_' in col]),
        'Trainer': len([col for col in feature_data.columns if 'trainer_' in col]),
        'Track': len([col for col in feature_data.columns if 'track_' in col]),
        'Horse Performance': len([col for col in feature_data.columns if any(x in col for x in ['win_rate', 'place_rate', 'show_rate'])]),
        'Recent Form': len([col for col in feature_data.columns if 'recent_' in col]),
    }
    
    print("\nFeature categories created:")
    for category, count in features_by_category.items():
        print(f"{category}: {count} features")
    
    # Prepare for training - identify columns that shouldn't be used in the model
    # but we need to keep for identification purposes
    id_cols = ['horse_name', 'race_date', 'track_code', 'race_number']
    target_col = 'finish'
    
    # Exclude columns that aren't numeric (except for the ones we specifically engineered)
    # This preserves all the performance metrics we created
    object_cols = feature_data.select_dtypes(include=['object']).columns
    exclude_cols = [col for col in object_cols if col not in id_cols + [target_col]]
    
    # Create the feature matrix (X) and target vector (y)
    X = feature_data.drop(exclude_cols + [target_col], axis=1, errors='ignore')
    y = feature_data[target_col]
    
    # Print data types to understand what we're working with
    print("\nData types in feature set:")
    type_counts = X.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Remove any remaining non-numeric columns for older versions of the code
    # Modern version handles these types properly in the model
    for col in X.select_dtypes(include=['object']).columns:
        if col not in id_cols:
            X = X.drop(col, axis=1)
    
    # Verify no unknown/unsupported types remain
    for col in X.columns:
        # Skip ID columns which might have various types
        if col in id_cols:
            continue
            
        col_type = X[col].dtype
        # Check if column type is supported
        is_numeric = pd.api.types.is_numeric_dtype(col_type)
        is_bool = pd.api.types.is_bool_dtype(col_type)
        is_categorical = isinstance(col_type, pd.CategoricalDtype)
        is_datetime = pd.api.types.is_datetime64_dtype(col_type)
        
        if not (is_numeric or is_bool or is_categorical or is_datetime):
            print(f"Warning: Column {col} has unsupported type {col_type}. Dropping.")
            X = X.drop(col, axis=1)
    
    # Filter for top 3 finishes to create a classification problem
    y = y.apply(lambda x: x if x <= 3 else 4)  # 4 = not in top 3
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Display feature information
    print(f"\nTraining XGBoost model with {X.shape[1] - len(id_cols)} performance-based features")
    print(f"These features analyze how horses perform in specific conditions")
    
    # Count categorical features
    cat_cols = X.select_dtypes(include=['category']).columns
    print(f"Including {len(cat_cols)} categorical features: {', '.join(cat_cols)}")
    
    # List some numeric features too
    num_cols = X.select_dtypes(include=['number']).columns
    print(f"Top numeric features: {', '.join(num_cols[:15])}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # If we have test races, make predictions
    if test_races is not None:
        print("\nPreparing test race data for prediction...")
        # Preprocess test data
        processed_test = preprocess_data(test_races, is_training=False)
        
        # Engineer features for test data
        # This will calculate performance metrics for each horse based on historical data
        print("Analyzing horses' past performance in conditions similar to test races...")
        feature_test = engineer_features(processed_test, historical_data=processed_data, is_training=False)
        
        # Filter for specific test race if requested
        if args.race:
            print(f"Making predictions for race {args.race} only...")
            test_df = feature_test[feature_test['race_number'] == args.race]
        else:
            print("Making predictions for race 5 only...")
            test_df = feature_test[feature_test['race_number'] == 5]
        
        if test_df.empty:
            print("\nNo data found for the specified race!")
        else:
            # Group by race
            race_groups = test_df.groupby(['race_date', 'track_code', 'race_number'], observed=True)
            
            # Make predictions for each race
            print("\nMaking predictions for the specified race...")
            all_predictions = []
            
            for (race_date, track_code, race_num), race_group in race_groups:
                race_id = f"{track_code} - {race_date} - Race {race_num}"
                print(f"\nPredicting for {race_id}")
                
                # Get race conditions
                conditions = {}
                if 'surface' in race_group.columns:
                    conditions['Surface'] = race_group['surface'].iloc[0]
                if 'track_condition' in race_group.columns:
                    conditions['Track Condition'] = race_group['track_condition'].iloc[0]
                if 'distance' in race_group.columns:
                    conditions['Distance'] = race_group['distance'].iloc[0]
                
                print("Race conditions:", ", ".join([f"{k}: {v}" for k, v in conditions.items()]))
                
                # Get horse names for this race
                horse_names = race_group['horse_name'].tolist()
                print(f"Horses in this race: {', '.join(horse_names)}")
                
                # Prepare race data for prediction - use the same columns as training
                race_X = race_group[X.columns.intersection(race_group.columns)]
                
                # Check for missing columns and add them with default values
                for col in X.columns:
                    if col not in race_X.columns:
                        race_X[col] = 0
                
                # Ensure same column order as training data
                race_X = race_X[X.columns]
                
                # Get top 5 predictions
                predictions = predict_top_finishers(model, race_X, horse_names, min_confidence=0.1)
                
                # Print top 5 predictions
                print(f"\nTop 5 predictions for {race_id}:")
                print()
                
                for i, (horse, horse_info) in enumerate(predictions, 1):
                    if isinstance(horse_info, dict):
                        confidence = horse_info.get('confidence', 0) * 100
                        print(f"{i}. {horse} (confidence: {confidence:.2f}%)")
                        
                        # Display odds-related information
                        key_factors = []
                        if 'jockey_win_rate' in race_group.columns:
                            horse_row = race_group[race_group['horse_name'] == horse]
                            if not horse_row.empty:
                                if 'jockey_win_rate' in horse_row.columns:
                                    jockey_win = horse_row['jockey_win_rate'].values[0] * 100
                                    key_factors.append(f"Jockey win rate: {jockey_win:.2f}%")
                                if 'days_since_last_race' in horse_row.columns:
                                    days = horse_row['days_since_last_race'].values[0]
                                    key_factors.append(f"Days since last race: {days}")
                                if 'post_position' in horse_row.columns:
                                    post = horse_row['post_position'].values[0]
                                    key_factors.append(f"Post position: {post}")
                                    
                        # Add odds-related factors
                        if 'dollar_odds' in horse_info:
                            key_factors.append(f"Odds: {horse_info['dollar_odds']:.2f}")
                        if 'odds_rank' in horse_info:
                            key_factors.append(f"Odds rank: {horse_info['odds_rank']:.0f}")
                        if 'safety_score' in horse_info:
                            safety = horse_info['safety_score'] * 100
                            key_factors.append(f"Safety score: {safety:.1f}%")
                        if 'win_value_score' in horse_info:
                            key_factors.append(f"Value score: {horse_info['win_value_score']:.2f}")
                        if 'kelly_value' in horse_info:
                            kelly = horse_info['kelly_value'] * 100
                            key_factors.append(f"Kelly value: {kelly:.2f}%")
                            
                        # Add bet recommendation based on odds rank and safety
                        if 'odds_rank' in horse_info and 'safety_score' in horse_info:
                            if horse_info['odds_rank'] <= 3 and horse_info['safety_score'] > 0.7:
                                key_factors.append("RECOMMENDATION: Safe bet")
                            elif horse_info['odds_rank'] <= 4 and horse_info['safety_score'] > 0.5:
                                key_factors.append("RECOMMENDATION: Moderate risk")
                            elif horse_info['odds_rank'] > 5:
                                key_factors.append("RECOMMENDATION: Higher risk")
                        
                        if key_factors:
                            print(f"   Key factors: {', '.join(key_factors)}")
                    else:
                        # Handle old format where horse_info is just confidence value
                        confidence = horse_info * 100 
                        print(f"{i}. {horse} (confidence: {confidence:.2f}%)")
                        
                        # Still try to show key factors from race data
                        if 'jockey_win_rate' in race_group.columns:
                            horse_row = race_group[race_group['horse_name'] == horse]
                            if not horse_row.empty:
                                key_factors = []
                                if 'jockey_win_rate' in horse_row.columns:
                                    jockey_win = horse_row['jockey_win_rate'].values[0] * 100
                                    key_factors.append(f"Jockey win rate: {jockey_win:.2f}%")
                                if 'days_since_last_race' in horse_row.columns:
                                    days = horse_row['days_since_last_race'].values[0]
                                    key_factors.append(f"Days since last race: {days}")
                                if 'post_position' in horse_row.columns:
                                    post = horse_row['post_position'].values[0]
                                    key_factors.append(f"Post position: {post}")
                                if key_factors:
                                    print(f"   Key factors: {', '.join(key_factors)}")
                    
                    print()
                
                race_result = {
                    'race_date': race_date,
                    'track_code': track_code, 
                    'race_number': race_num,
                    'predictions': predictions,
                    'conditions': conditions
                }
                all_predictions.append(race_result)
            
            # Save predictions
            print("\nSaving predictions...")
            # Convert predictions to DataFrame for easier saving
            pred_rows = []
            for race in all_predictions:
                for i, (horse, prob) in enumerate(race['predictions']):
                    pred_rows.append({
                        'race_date': race['race_date'],
                        'track_code': race['track_code'],
                        'race_number': race['race_number'],
                        'predicted_position': i+1,
                        'horse_name': horse,
                        'confidence': prob
                    })
            
            pred_df = pd.DataFrame(pred_rows)
            pred_df.to_csv(predictions_path, index=False)
            print(f"Predictions saved to {predictions_path}")
    
    # Save model for future use
    print("Saving model...")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature list for future reference
    with open(features_path, 'w') as f:
        f.write('\n'.join([col for col in X.columns if col not in id_cols]))
    print(f"Feature list saved to {features_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict horse positions using XGBoost")
    parser.add_argument("--race", type=int, help="Specify the race number to predict")
    args = parser.parse_args()
    main() 