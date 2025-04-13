#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from utils.evaluation import PerformanceTracker

def plot_feature_importance(importance_file):
    """
    Plot feature importance from a saved CSV
    
    Args:
        importance_file: Path to feature importance CSV
    """
    # Load feature importance data
    importance_df = pd.read_csv(importance_file)
    
    # Sort by importance
    sorted_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x='importance', y='feature', data=sorted_df)
    plt.title('XGBoost Feature Importance (Top 20)', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = 'xgb/feature_importance.png'
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    
    # Display plot
    plt.show()


def plot_score_distribution(results_file):
    """
    Plot the distribution of scores
    
    Args:
        results_file: Path to results CSV
    """
    # Load results data
    results_df = pd.read_csv(results_file)
    
    # Create a count plot of scores
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='score', data=results_df)
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom')
    
    plt.title('Distribution of Prediction Scores', fontsize=16)
    plt.xlabel('Score (out of 7)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(range(0, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    output_path = 'xgb/score_distribution.png'
    plt.savefig(output_path)
    print(f"Score distribution plot saved to {output_path}")
    
    # Display plot
    plt.show()


def plot_confidence_vs_accuracy(predictions_file, results_file):
    """
    Plot the relationship between prediction confidence and actual outcome
    
    Args:
        predictions_file: Path to predictions CSV
        results_file: Path to results CSV
    """
    # Load data
    pred_df = pd.read_csv(predictions_file)
    results_df = pd.read_csv(results_file)
    
    # Merge predictions with results
    merged = pd.merge(
        pred_df, 
        results_df,
        on=['race_date', 'track_code', 'race_number'],
        how='inner'
    )
    
    # Check if predictions were correct
    merged['correct_win'] = (merged['predicted_position'] == 1) & (merged['horse_name'] == merged['first_place'])
    merged['correct_place'] = (merged['predicted_position'] == 2) & (merged['horse_name'] == merged['second_place'])
    merged['correct_show'] = (merged['predicted_position'] == 3) & (merged['horse_name'] == merged['third_place'])
    merged['correct_any'] = ((merged['predicted_position'] == 1) & (merged['horse_name'] == merged['first_place'])) | \
                            ((merged['predicted_position'] == 2) & (merged['horse_name'] == merged['second_place'])) | \
                            ((merged['predicted_position'] == 3) & (merged['horse_name'] == merged['third_place']))
    
    # Create bins for confidence
    merged['confidence_bin'] = pd.cut(merged['confidence'], bins=10)
    
    # Calculate accuracy by confidence bin
    conf_accuracy = merged.groupby('confidence_bin').agg(
        win_accuracy=('correct_win', 'mean'),
        place_accuracy=('correct_place', 'mean'),
        show_accuracy=('correct_show', 'mean'),
        any_accuracy=('correct_any', 'mean'),
        count=('correct_any', 'count')
    ).reset_index()
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(conf_accuracy['confidence_bin'].astype(str), conf_accuracy['win_accuracy'], 
             marker='o', label='Win Accuracy', linewidth=2)
    plt.plot(conf_accuracy['confidence_bin'].astype(str), conf_accuracy['place_accuracy'], 
             marker='s', label='Place Accuracy', linewidth=2)
    plt.plot(conf_accuracy['confidence_bin'].astype(str), conf_accuracy['show_accuracy'], 
             marker='^', label='Show Accuracy', linewidth=2)
    plt.plot(conf_accuracy['confidence_bin'].astype(str), conf_accuracy['any_accuracy'], 
             marker='*', label='Any Position Correct', linewidth=2)
    
    plt.title('Prediction Confidence vs. Accuracy', fontsize=16)
    plt.xlabel('Confidence Bin', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    output_path = 'xgb/confidence_vs_accuracy.png'
    plt.savefig(output_path)
    print(f"Confidence vs. accuracy plot saved to {output_path}")
    
    # Display plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize model performance')
    parser.add_argument('--feature_importance', help='Path to feature importance CSV')
    parser.add_argument('--results', default='xgb/results.csv', help='Path to results CSV')
    parser.add_argument('--predictions', default='xgb/predictions.csv', help='Path to predictions CSV')
    parser.add_argument('--all_plots', action='store_true', help='Generate all plots')
    
    args = parser.parse_args()
    
    # Check if necessary files exist
    results_exists = os.path.exists(args.results)
    predictions_exists = os.path.exists(args.predictions)
    
    if args.feature_importance and (os.path.exists(args.feature_importance) or args.all_plots):
        plot_feature_importance(args.feature_importance)
    
    if results_exists or args.all_plots:
        plot_score_distribution(args.results)
    
    if (results_exists and predictions_exists) or args.all_plots:
        plot_confidence_vs_accuracy(args.predictions, args.results)
    
    if not (results_exists or predictions_exists):
        print("Warning: No results or predictions files found. Run predictions first!")


if __name__ == "__main__":
    main() 