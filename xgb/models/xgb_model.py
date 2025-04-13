import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train an XGBoost model for horse race prediction
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Trained XGBoost model
    """
    # Identify ID columns that shouldn't be used for training
    id_cols = [col for col in X_train.columns if col in ['horse_name', 'race_date', 'track_code', 'race_number']]
    
    # Remove ID columns from training data
    X_train_model = X_train.drop(id_cols, axis=1, errors='ignore')
    
    if X_val is not None:
        X_val_model = X_val.drop(id_cols, axis=1, errors='ignore')
    
    # Create copied frames to avoid fragmentation warnings
    X_train_model = X_train_model.copy()
    if X_val is not None:
        X_val_model = X_val_model.copy()
    
    # Handle datetime columns - XGBoost can't process these directly
    datetime_cols = X_train_model.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        # Convert datetime to integer timestamp (days since epoch)
        X_train_model[f"{col}_timestamp"] = X_train_model[col].astype(np.int64) // 10**9 // 86400
        # Drop the original datetime column
        X_train_model = X_train_model.drop(col, axis=1)
        
        if X_val is not None and col in X_val_model.columns:
            X_val_model[f"{col}_timestamp"] = X_val_model[col].astype(np.int64) // 10**9 // 86400
            X_val_model = X_val_model.drop(col, axis=1)
    
    # Handle Categorical columns - we need to convert these to numeric
    # First identify categorical columns to handle them differently
    cat_cols = X_train_model.select_dtypes(include=['category']).columns
    
    # Convert categorical columns to numeric codes and drop the original
    for col in cat_cols:
        # Create a numeric version with the code
        X_train_model[f"{col}_code"] = X_train_model[col].cat.codes
        # Drop the original categorical column
        X_train_model = X_train_model.drop(col, axis=1)
        
        if X_val is not None and col in X_val_model.columns:
            X_val_model[f"{col}_code"] = X_val_model[col].cat.codes
            X_val_model = X_val_model.drop(col, axis=1)
    
    # Fill missing values in non-categorical columns
    num_cols = X_train_model.select_dtypes(include=['number']).columns
    X_train_model[num_cols] = X_train_model[num_cols].fillna(0)
    
    if X_val is not None:
        num_cols_val = X_val_model.select_dtypes(include=['number']).columns
        X_val_model[num_cols_val] = X_val_model[num_cols_val].fillna(0)
    
    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train_model, label=y_train)
    
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val_model, label=y_val)
        eval_set = [(dtrain, 'train'), (dval, 'validation')]
    else:
        eval_set = [(dtrain, 'train')]
    
    # Set XGBoost parameters optimized for horse racing prediction with emphasis on odds
    params = {
        'objective': 'multi:softprob',  # Multi-class probability
        'num_class': 5,  # 1=win, 2=place, 3=show, 4=other, plus 0 for indexing
        'max_depth': 7,  # Less depth to avoid overfitting to longshots
        'eta': 0.04,  # Slightly higher learning rate for more stable convergence
        'subsample': 0.85,  # Higher subsample to reduce variance
        'colsample_bytree': 0.8,  # Higher to include more features per tree
        'min_child_weight': 4,  # Higher to be more conservative
        'gamma': 0.1,  # Higher to require more information gain for splits
        'alpha': 0.2,  # L1 regularization 
        'lambda': 1.5,  # Higher L2 regularization to prefer simpler models
        'eval_metric': ['mlogloss', 'merror'],
        'tree_method': 'hist',  # Fast algorithm for large datasets
        'scale_pos_weight': 1.0,  # Balance positive/negative weights
    }
    
    # Train the model with more emphasis on odd features
    print("\nTraining XGBoost model with more emphasis on odds-related features...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,  # More boosting rounds for more detailed learning
        evals=eval_set,
        early_stopping_rounds=40,
        verbose_eval=20  # Print evaluation every 20 iterations
    )
    
    # Get feature importance
    importance = model.get_score(importance_type='gain')  # Using gain for more meaningful importance
    if importance:
        # Convert to DataFrame for easier handling
        importance_df = pd.DataFrame(
            {'feature': list(importance.keys()), 
             'importance': list(importance.values())}
        ).sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv('xgb/output/feature_importance.csv', index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 10))
        top_n = min(20, len(importance_df))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title('Most Important Performance Metrics for Predicting Horse Races', fontsize=16)
        plt.tight_layout()
        plt.savefig('xgb/output/feature_importance.png')
        
        # Print top features
        print("\nTop 15 most important horse performance metrics:")
        for i, (feature, score) in enumerate(zip(importance_df['feature'].head(15), 
                                              importance_df['importance'].head(15))):
            print(f"{i+1}. {feature}: {score:.2f}")
            
        # Group importance by category
        categories = {
            'Surface': [col for col in importance_df['feature'] if 'surface_' in col],
            'Track Condition': [col for col in importance_df['feature'] if 'condition_' in col],
            'Distance': [col for col in importance_df['feature'] if 'distance_' in col],
            'Jockey': [col for col in importance_df['feature'] if 'jockey_' in col],
            'Trainer': [col for col in importance_df['feature'] if 'trainer_' in col],
            'Track': [col for col in importance_df['feature'] if 'track_' in col],
            'Overall': [col for col in importance_df['feature'] if any(x in col for x in ['win_rate', 'place_rate', 'show_rate'])]
        }
        
        print("\nImportance by category:")
        for category, cols in categories.items():
            if cols:
                category_importance = importance_df[importance_df['feature'].isin(cols)]['importance'].sum()
                print(f"{category}: {category_importance:.2f}")
    
    # Evaluate the model if validation data is provided
    if X_val is not None and y_val is not None:
        y_pred = model.predict(dval)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_class):.4f}")
        print(f"Precision (macro): {precision_score(y_val, y_pred_class, average='macro'):.4f}")
        print(f"Recall (macro): {recall_score(y_val, y_pred_class, average='macro'):.4f}")
        print(f"F1 Score (macro): {f1_score(y_val, y_pred_class, average='macro'):.4f}")
        
        # Position-specific metrics
        for position in [1, 2, 3]:  # Win, Place, Show
            position_name = {1: "Win", 2: "Place", 3: "Show"}[position]
            y_val_position = (y_val == position).astype(int)
            y_pred_position = y_pred[:, position]
            
            # Calculate precision at different confidence thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            print(f"\n{position_name} position performance at different confidence thresholds:")
            for threshold in thresholds:
                y_pred_threshold = (y_pred_position >= threshold).astype(int)
                if y_pred_threshold.sum() > 0:
                    precision = (y_val_position & y_pred_threshold).sum() / y_pred_threshold.sum()
                    print(f"  Confidence >= {threshold:.1f}: Precision = {precision:.4f}, Count = {y_pred_threshold.sum()}")
    
    return model

def predict_top_finishers(model, X, horse_names, min_confidence=0.1):
    """
    Predict the top 5 finishers for a race
    
    Args:
        model: Trained XGBoost model
        X: Features for the race
        horse_names: List of horse names corresponding to the features
        min_confidence: Minimum confidence threshold for predictions
        
    Returns:
        List of (horse_name, probability) tuples for the top 5 predicted finishers
    """
    # Identify ID columns that shouldn't be used for prediction
    id_cols = [col for col in X.columns if col in ['horse_name', 'race_date', 'track_code', 'race_number']]
    
    # Extract key odds-related columns before dropping them
    odds_data = {}
    if 'dollar_odds' in X.columns:
        odds_data['dollar_odds'] = X['dollar_odds'].tolist()
    if 'odds_rank' in X.columns:
        odds_data['odds_rank'] = X['odds_rank'].tolist()
    if 'value_bet' in X.columns:
        odds_data['value_bet'] = X['value_bet'].tolist()
    if 'odds_ratio_to_avg' in X.columns:
        odds_data['odds_ratio_to_avg'] = X['odds_ratio_to_avg'].tolist()
        
    # Remove ID columns from prediction data
    X_pred = X.drop(id_cols, axis=1, errors='ignore')
    
    # Create a copy to avoid fragmentation
    X_pred = X_pred.copy()
    
    # Handle datetime columns
    datetime_cols = X_pred.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        # Convert datetime to integer timestamp (days since epoch)
        X_pred[f"{col}_timestamp"] = X_pred[col].astype(np.int64) // 10**9 // 86400
        # Drop the original datetime column
        X_pred = X_pred.drop(col, axis=1)
    
    # Handle Categorical columns - convert to numeric codes
    cat_cols = X_pred.select_dtypes(include=['category']).columns
    for col in cat_cols:
        X_pred[f"{col}_code"] = X_pred[col].cat.codes
        X_pred = X_pred.drop(col, axis=1)
    
    # Fill missing values in numeric columns
    num_cols = X_pred.select_dtypes(include=['number']).columns
    X_pred[num_cols] = X_pred[num_cols].fillna(0)
    
    # Convert to DMatrix
    dtest = xgb.DMatrix(X_pred)
    
    # Get predictions (probabilities for each class)
    predictions = model.predict(dtest)
    
    # Extract probabilities for finishing 1st through 5th
    first_probs = predictions[:, 1]  # Class 1 = finish 1st
    second_probs = predictions[:, 2]  # Class 2 = finish 2nd
    third_probs = predictions[:, 3]   # Class 3 = finish 3rd
    fourth_probs = predictions[:, 4]  # Class 4 = finish 4th
    
    # Create a DataFrame with horses and their probabilities
    results = pd.DataFrame({
        'horse_name': horse_names,
        'first_prob': first_probs,
        'second_prob': second_probs,
        'third_prob': third_probs,
        'fourth_prob': fourth_probs,
        'top4_prob': first_probs + second_probs + third_probs + fourth_probs
    })
    
    # Add odds-related information if available
    if odds_data:
        for key, values in odds_data.items():
            if len(values) == len(results):
                results[key] = values
    
    # Calculate value scores (blend of probabilities and odds)
    if 'dollar_odds' in results.columns:
        # Adjust probabilities based on odds (favor horses with lower odds)
        results['adjusted_first_prob'] = results['first_prob'] * (1 + 1/results['dollar_odds'].clip(1))
        
        # Value score with higher weight to favorites and lower weight to longshots
        results['win_value_score'] = results['first_prob'] * (1/results['dollar_odds'].clip(1))
        
        # Conservative Kelly criterion with safety factor
        results['kelly_value'] = (results['first_prob'] - (1.25 / results['dollar_odds'].clip(0.01))).clip(-1, 0.3)
        
        # Favorite indicator with higher impact
        if 'odds_rank' in results.columns:
            results['favorite_indicator'] = (results['odds_rank'] <= 3).astype(float) * 0.2
            
            # Apply favorite boost to first probability
            results['first_prob'] = results['first_prob'] + results['favorite_indicator']
            
            # Apply longshot penalty
            results['longshot_penalty'] = ((results['odds_rank'] >= results.shape[0] * 0.7) & 
                                          (results['dollar_odds'] > 15)).astype(float) * 0.1
            results['first_prob'] = (results['first_prob'] - results['longshot_penalty']).clip(0, 1)
    
    # Filter out horses with very low probability
    filtered_results = results[results['top4_prob'] >= min_confidence]
    
    if filtered_results.empty:
        print(f"Warning: No horses meet the minimum confidence threshold of {min_confidence}")
        # Fall back to using all horses if none meet the threshold
        filtered_results = results
    
    # Determine the top 5 finishers using a hybrid approach that favors favorites
    top5 = []
    remaining_horses = filtered_results.copy()
    
    # Calculate a combined score that favors horses with better odds (favorites)
    if 'odds_rank' in remaining_horses.columns and 'dollar_odds' in remaining_horses.columns:
        # Create a safety score (higher for favorites, lower for longshots)
        remaining_horses['safety_score'] = (remaining_horses.shape[0] - remaining_horses['odds_rank']) / remaining_horses.shape[0]
        
        # Calculate combined score that balances probability and odds preference
        # Higher weight (0.6) to model probabilities, lower weight (0.4) to odds-based safety
        remaining_horses['win_combined_score'] = (0.6 * remaining_horses['first_prob']) + (0.4 * remaining_horses['safety_score'])
        remaining_horses['place_combined_score'] = (0.6 * remaining_horses['second_prob']) + (0.4 * remaining_horses['safety_score'])
        remaining_horses['show_combined_score'] = (0.6 * remaining_horses['third_prob']) + (0.4 * remaining_horses['safety_score'])
        remaining_horses['fourth_combined_score'] = (0.6 * remaining_horses['fourth_prob']) + (0.4 * remaining_horses['safety_score'])
    else:
        # If no odds data, just use probabilities
        remaining_horses['win_combined_score'] = remaining_horses['first_prob']
        remaining_horses['place_combined_score'] = remaining_horses['second_prob']
        remaining_horses['show_combined_score'] = remaining_horses['third_prob']
        remaining_horses['fourth_combined_score'] = remaining_horses['fourth_prob']
    
    # Find most likely winner (1st place) using combined score
    if not remaining_horses.empty:
        first_idx = remaining_horses['win_combined_score'].argmax()
        first_horse = remaining_horses.iloc[first_idx]['horse_name']
        first_prob = remaining_horses.iloc[first_idx]['first_prob']
        horse_info = {'confidence': float(first_prob)}
        
        # Add odds info if available
        for key in ['dollar_odds', 'odds_rank', 'win_value_score', 'kelly_value', 'value_bet']:
            if key in remaining_horses.columns:
                horse_info[key] = float(remaining_horses.iloc[first_idx][key])
        
        # Add safety score if available
        if 'safety_score' in remaining_horses.columns:
            horse_info['safety_score'] = float(remaining_horses.iloc[first_idx]['safety_score'])
        
        top5.append((first_horse, horse_info))
        remaining_horses = remaining_horses[remaining_horses['horse_name'] != first_horse]
    
    # Find most likely second place finisher using combined score
    if not remaining_horses.empty:
        second_idx = remaining_horses['place_combined_score'].argmax()
        second_horse = remaining_horses.iloc[second_idx]['horse_name']
        second_prob = remaining_horses.iloc[second_idx]['second_prob']
        horse_info = {'confidence': float(second_prob)}
        
        # Add odds info and safety score if available
        for key in ['dollar_odds', 'odds_rank', 'win_value_score', 'kelly_value', 'value_bet', 'safety_score']:
            if key in remaining_horses.columns:
                horse_info[key] = float(remaining_horses.iloc[second_idx][key])
        
        top5.append((second_horse, horse_info))
        remaining_horses = remaining_horses[remaining_horses['horse_name'] != second_horse]
    
    # Find most likely third place finisher using combined score
    if not remaining_horses.empty:
        third_idx = remaining_horses['show_combined_score'].argmax()
        third_horse = remaining_horses.iloc[third_idx]['horse_name']
        third_prob = remaining_horses.iloc[third_idx]['third_prob']
        horse_info = {'confidence': float(third_prob)}
        
        # Add odds info and safety score if available
        for key in ['dollar_odds', 'odds_rank', 'win_value_score', 'kelly_value', 'value_bet', 'safety_score']:
            if key in remaining_horses.columns:
                horse_info[key] = float(remaining_horses.iloc[third_idx][key])
        
        top5.append((third_horse, horse_info))
        remaining_horses = remaining_horses[remaining_horses['horse_name'] != third_horse]
    
    # Find most likely fourth place finisher using combined score
    if not remaining_horses.empty:
        fourth_idx = remaining_horses['fourth_combined_score'].argmax()
        fourth_horse = remaining_horses.iloc[fourth_idx]['horse_name']
        fourth_prob = remaining_horses.iloc[fourth_idx]['fourth_prob']
        horse_info = {'confidence': float(fourth_prob)}
        
        # Add odds info and safety score if available
        for key in ['dollar_odds', 'odds_rank', 'win_value_score', 'kelly_value', 'value_bet', 'safety_score']:
            if key in remaining_horses.columns:
                horse_info[key] = float(remaining_horses.iloc[fourth_idx][key])
        
        top5.append((fourth_horse, horse_info))
        remaining_horses = remaining_horses[remaining_horses['horse_name'] != fourth_horse]
    
    # For fifth place, blend remaining probability with safety factors
    if not remaining_horses.empty:
        # For fifth place, give even higher weight to favorites
        if 'safety_score' in remaining_horses.columns:
            remaining_horses['fifth_combined_score'] = (0.5 * remaining_horses['top4_prob']) + (0.5 * remaining_horses['safety_score'])
            fifth_idx = remaining_horses['fifth_combined_score'].argmax()
        else:
            # Fallback to just using probability
            fifth_idx = remaining_horses['top4_prob'].argmax()
            
        fifth_horse = remaining_horses.iloc[fifth_idx]['horse_name']
        fifth_prob = remaining_horses.iloc[fifth_idx]['top4_prob']
        horse_info = {'confidence': float(fifth_prob)}
        
        # Add odds info and safety score if available
        for key in ['dollar_odds', 'odds_rank', 'win_value_score', 'kelly_value', 'value_bet', 'safety_score']:
            if key in remaining_horses.columns:
                horse_info[key] = float(remaining_horses.iloc[fifth_idx][key])
        
        top5.append((fifth_horse, horse_info))
    
    return top5 