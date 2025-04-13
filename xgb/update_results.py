#!/usr/bin/env python
import argparse
import pandas as pd
from datetime import datetime
from utils.evaluation import PerformanceTracker

def main():
    """
    Script to update the model with new race results and track performance
    """
    parser = argparse.ArgumentParser(description='Update model with race results')
    parser.add_argument('--race_date', required=True, help='Date of the race (YYYY-MM-DD)')
    parser.add_argument('--track_code', required=True, help='Track code')
    parser.add_argument('--race_number', required=True, type=int, help='Race number')
    parser.add_argument('--first', required=True, help='Name of the winning horse')
    parser.add_argument('--second', required=True, help='Name of the second-place horse')
    parser.add_argument('--third', required=True, help='Name of the third-place horse')
    parser.add_argument('--predictions_file', default='xgb/predictions.csv', help='Path to predictions CSV')
    parser.add_argument('--results_file', default='xgb/results.csv', help='Path to results CSV')
    
    args = parser.parse_args()
    
    # Create the performance tracker
    tracker = PerformanceTracker(args.predictions_file)
    
    # Add the new result
    actual_results = [args.first, args.second, args.third]
    score = tracker.add_result(
        args.race_date,
        args.track_code,
        args.race_number,
        actual_results
    )
    
    # Save the updated results
    tracker.save_results(args.results_file)
    
    # Print the total score
    total_stats = tracker.get_total_score()
    print("\nOverall Performance:")
    print(f"Total Score: {total_stats['total_score']}")
    print(f"Races Scored: {total_stats['races_scored']}")
    print(f"Average Score per Race: {total_stats['avg_score']:.2f}")
    
    # Now append this result to a race results CSV for future model training
    try:
        # Try to load the existing race results file
        all_races = pd.read_csv('data/all_tracks_hackathon.csv', low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: Could not find the historical race data file")
        return
    
    # Create new entries for the top 3 horses
    new_rows = []
    
    for i, horse_name in enumerate(actual_results):
        # Find info about this horse from the predictions
        race_preds = pd.read_csv(args.predictions_file)
        horse_info = race_preds[(race_preds['race_date'] == args.race_date) & 
                               (race_preds['track_code'] == args.track_code) &
                               (race_preds['race_number'] == args.race_number) &
                               (race_preds['horse_name'] == horse_name)]
        
        # If we have info about this horse, create a new row
        if not horse_info.empty:
            # Create a template from the most recent row in all_races
            template = all_races.iloc[-1].copy()
            
            # Update with the new race info
            template['race_date'] = args.race_date
            template['track_code'] = args.track_code
            template['race_number'] = args.race_number
            template['horse_name'] = horse_name
            template['finish'] = i + 1  # 1 for first, 2 for second, 3 for third
            
            # We'd need to update other fields like jockey, trainer, etc.
            # but we can't do that without the full race information
            
            new_rows.append(template)
    
    # If we have new rows, append them to the all_races dataframe
    if new_rows:
        new_entries = pd.DataFrame(new_rows)
        updated_races = pd.concat([all_races, new_entries], ignore_index=True)
        
        # Save the updated race data - comment this out for testing
        # updated_races.to_csv('data/all_tracks_hackathon.csv', index=False)
        
        print(f"\nAdded {len(new_rows)} new race entries (NOT saved to main dataset - uncomment to enable)")
    else:
        print("\nCould not create new race entries due to missing information")

if __name__ == "__main__":
    main() 