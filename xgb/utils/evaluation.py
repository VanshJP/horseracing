import pandas as pd
import numpy as np
from datetime import datetime

def calculate_score(predictions, actual_results):
    """
    Calculate the score based on the scoring system:
    - +1 point for each correct pick in the top 3 (regardless of order)
    - +1 point for each correct pick in the correct position
    - +1 bonus point if all top 3 picks are in the correct order
    
    Args:
        predictions: List of predicted horses in order (1st, 2nd, 3rd)
        actual_results: List of actual finishers in order (1st, 2nd, 3rd)
        
    Returns:
        Total score (max 7 points)
    """
    score = 0
    
    # Convert inputs to lists if they are not already
    if isinstance(predictions, tuple):
        # Extract just the horse names if predictions are (horse_name, probability) tuples
        predictions = [p[0] if isinstance(p, tuple) else p for p in predictions]
    
    # +1 point for each correct pick in top 3 (regardless of order)
    for horse in predictions:
        if horse in actual_results:
            score += 1
            print(f"✓ {horse} correctly predicted in top 3 (+1)")
    
    # +1 point for each correct pick in correct position
    for i in range(min(len(predictions), len(actual_results))):
        if predictions[i] == actual_results[i]:
            score += 1
            print(f"✓ {predictions[i]} correctly predicted in position {i+1} (+1)")
    
    # +1 bonus point if all top 3 are in correct order
    if len(predictions) >= 3 and len(actual_results) >= 3:
        if predictions[0] == actual_results[0] and predictions[1] == actual_results[1] and predictions[2] == actual_results[2]:
            score += 1
            print("✓ All top 3 in correct order (+1 bonus)")
    
    return score


class PerformanceTracker:
    """
    Track model performance over time and update as new race results come in
    """
    def __init__(self, predictions_file=None):
        """
        Initialize the performance tracker
        
        Args:
            predictions_file: Path to a CSV file with previous predictions
        """
        self.predictions = []
        self.results = []
        self.scores = {}
        
        # Load previous predictions if provided
        if predictions_file:
            self.load_predictions(predictions_file)
    
    def load_predictions(self, file_path):
        """
        Load predictions from a CSV file
        
        Args:
            file_path: Path to the predictions CSV
        """
        try:
            df = pd.read_csv(file_path)
            
            # Group by race identifiers
            race_groups = df.groupby(['race_date', 'track_code', 'race_number'])
            
            # Convert to our prediction format
            for (race_date, track_code, race_num), race_df in race_groups:
                sorted_df = race_df.sort_values('predicted_position')
                predictions = [(row['horse_name'], row['confidence']) 
                              for _, row in sorted_df.iterrows()]
                
                self.predictions.append({
                    'race_date': race_date,
                    'track_code': track_code,
                    'race_number': race_num,
                    'predictions': predictions
                })
            
            print(f"Loaded {len(self.predictions)} race predictions from {file_path}")
        except Exception as e:
            print(f"Error loading predictions: {e}")
    
    def add_result(self, race_date, track_code, race_num, actual_results):
        """
        Add actual race results
        
        Args:
            race_date: Date of the race
            track_code: Track code
            race_num: Race number
            actual_results: List of horses in order of finish (1st, 2nd, 3rd)
        """
        self.results.append({
            'race_date': race_date,
            'track_code': track_code,
            'race_number': race_num,
            'actual_results': actual_results,
            'timestamp': datetime.now()
        })
        
        # Calculate score if we have predictions for this race
        race_key = (race_date, track_code, race_num)
        matching_predictions = [p for p in self.predictions 
                              if (p['race_date'], p['track_code'], p['race_number']) == race_key]
        
        if matching_predictions:
            # Get the predictions (just the horse names)
            pred_horses = [horse for horse, _ in matching_predictions[0]['predictions']]
            
            # Calculate score
            score = calculate_score(pred_horses, actual_results)
            
            # Store the score
            self.scores[race_key] = score
            
            print(f"Score for {race_key}: {score}/7")
            return score
        else:
            print(f"No predictions found for race {race_key}")
            return None
    
    def get_total_score(self):
        """
        Get the total score across all races
        
        Returns:
            Total score and statistics
        """
        if not self.scores:
            return {"total_score": 0, "races_scored": 0, "avg_score": 0}
        
        total_score = sum(self.scores.values())
        avg_score = total_score / len(self.scores)
        
        return {
            "total_score": total_score,
            "races_scored": len(self.scores),
            "avg_score": avg_score
        }
    
    def save_results(self, file_path):
        """
        Save results and scores to a CSV file
        
        Args:
            file_path: Path to save the results CSV
        """
        rows = []
        for result in self.results:
            race_key = (result['race_date'], result['track_code'], result['race_number'])
            score = self.scores.get(race_key, None)
            
            rows.append({
                'race_date': result['race_date'],
                'track_code': result['track_code'],
                'race_number': result['race_number'],
                'first_place': result['actual_results'][0] if len(result['actual_results']) > 0 else None,
                'second_place': result['actual_results'][1] if len(result['actual_results']) > 1 else None,
                'third_place': result['actual_results'][2] if len(result['actual_results']) > 2 else None,
                'score': score,
                'timestamp': result['timestamp']
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(rows)} results to {file_path}")


def track_performance(predictions_file, results_file):
    """
    Convenience function to track performance and update scores
    
    Args:
        predictions_file: Path to predictions CSV
        results_file: Path to results CSV
        
    Returns:
        PerformanceTracker instance
    """
    tracker = PerformanceTracker(predictions_file)
    
    # Load results if the file exists
    try:
        results_df = pd.read_csv(results_file)
        for _, row in results_df.iterrows():
            actual_results = [
                row['first_place'] if not pd.isna(row['first_place']) else None,
                row['second_place'] if not pd.isna(row['second_place']) else None,
                row['third_place'] if not pd.isna(row['third_place']) else None
            ]
            # Filter out Nones
            actual_results = [r for r in actual_results if r is not None]
            
            if actual_results:
                tracker.add_result(
                    row['race_date'],
                    row['track_code'],
                    row['race_number'],
                    actual_results
                )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"No previous results found at {results_file}")
    
    return tracker 