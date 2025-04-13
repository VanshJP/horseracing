#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from utils.data_processing import preprocess_data, engineer_features
from utils.evaluation import calculate_score, track_performance
from models.xgb_model import train_model, predict_top_finishers

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
        
        # Filter for only race 1
        race1_data = feature_test[feature_test['race_number'] == 1]
        
        if race1_data.empty:
            print("\nNo data found for race 1 in the test data!")
        else:
            # Group by race
            race_groups = race1_data.groupby(['race_date', 'track_code', 'race_number'], observed=True)
            
            # Make predictions for each race
            print("\nMaking predictions for race 1 only...")
            all_predictions = []
            
            for (race_date, track_code, race_num), race_df in race_groups:
                race_id = f"{track_code} - {race_date} - Race {race_num}"
                print(f"\nPredicting for {race_id}")
                
                # Get race conditions
                conditions = {}
                if 'surface' in race_df.columns:
                    conditions['Surface'] = race_df['surface'].iloc[0]
                if 'track_condition' in race_df.columns:
                    conditions['Track Condition'] = race_df['track_condition'].iloc[0]
                if 'distance' in race_df.columns:
                    conditions['Distance'] = race_df['distance'].iloc[0]
                
                print("Race conditions:", ", ".join([f"{k}: {v}" for k, v in conditions.items()]))
                
                # Get horse names for this race
                horse_names = race_df['horse_name'].tolist()
                print(f"Horses in this race: {', '.join(horse_names)}")
                
                # Prepare race data for prediction - use the same columns as training
                race_X = race_df[X.columns.intersection(race_df.columns)]
                
                # Check for missing columns and add them with default values
                for col in X.columns:
                    if col not in race_X.columns:
                        race_X[col] = 0
                
                # Ensure same column order as training data
                race_X = race_X[X.columns]
                
                # Get top 3 predictions
                predictions = predict_top_finishers(model, race_X, horse_names)
                
                print(f"Top 3 predictions for {race_id}:")
                for i, (horse, prob) in enumerate(predictions):
                    print(f"{i+1}. {horse} (confidence: {prob:.2f})")
                    
                    # Find key performance stats for this horse
                    horse_row = race_df[race_df['horse_name'] == horse]
                    if not horse_row.empty:
                        # Show why this horse was selected (key performance indicators)
                        key_metrics = []
                        
                        # Overall performance
                        if 'win_rate' in horse_row.columns:
                            win_rate = horse_row['win_rate'].iloc[0]
                            if win_rate > 0:
                                key_metrics.append(f"Win rate: {win_rate:.2f}")
                        
                        # Surface-specific performance
                        if conditions.get('Surface') and f"surface_win_rate_{conditions['Surface']}".lower() in horse_row.columns:
                            surface_win = horse_row[f"surface_win_rate_{conditions['Surface']}".lower()].iloc[0]
                            if surface_win > 0:
                                key_metrics.append(f"Win rate on {conditions['Surface']}: {surface_win:.2f}")
                        
                        # Track condition performance
                        if conditions.get('Track Condition') and f"condition_win_rate_{conditions['Track Condition']}".lower() in horse_row.columns:
                            condition_win = horse_row[f"condition_win_rate_{conditions['Track Condition']}".lower()].iloc[0]
                            if condition_win > 0:
                                key_metrics.append(f"Win rate in {conditions['Track Condition']} conditions: {condition_win:.2f}")
                        
                        # Jockey performance
                        if 'jockey_win_rate' in horse_row.columns:
                            jockey_win = horse_row['jockey_win_rate'].iloc[0]
                            if jockey_win > 0:
                                key_metrics.append(f"Jockey win rate: {jockey_win:.2f}")
                        
                        # Recent form
                        if 'recent_win_rate' in horse_row.columns:
                            recent_win = horse_row['recent_win_rate'].iloc[0]
                            if recent_win > 0:
                                key_metrics.append(f"Recent win rate: {recent_win:.2f}")
                        
                        if key_metrics:
                            print(f"   Key factors: {', '.join(key_metrics)}")
                
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
    main() 