#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation for Full Moon Turning Point Detection System
Evaluates the performance of turning point detection against known turning points.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Import local modules
from src.turning_point_detector import TurningPointDetector
from src.feature_engineering import FeatureEngineer
from src.lunar_phase_calculator import LunarPhaseCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Full Moon Turning Point Detection System")
    
    # Input data
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to market data CSV file")
    parser.add_argument("--turning-points-file", type=str, required=True,
                       help="Path to known turning points CSV file")
    
    # Model options
    parser.add_argument("--model-file", type=str,
                       help="Path to trained model file (optional, uses rule-based if not provided)")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="evaluation",
                       help="Directory to save evaluation results")
    
    # Evaluation options
    parser.add_argument("--tolerance-days", type=int, default=3,
                       help="Number of days tolerance for matching turning points")
    parser.add_argument("--compare-methods", action="store_true",
                       help="Compare ML-based vs rule-based methods")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                       help="Generate and save visualizations")
    
    return parser.parse_args()

def load_data(data_file):
    """Load market data from CSV file."""
    try:
        data = pd.read_csv(data_file)
        
        # Check if Date column exists
        date_col = None
        for col in ['Date', 'date', 'TIME', 'Time', 'time', 'datetime']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col:
            data[date_col] = pd.to_datetime(data[date_col])
            data.set_index(date_col, inplace=True)
        else:
            logger.warning("No date column found in data, assuming index is date")
            
        # Standardize column names
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col_lower
                
        if column_mapping:
            data.rename(columns=column_mapping, inplace=True)
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Data file missing required columns: {missing_cols}")
            sys.exit(1)
            
        # Add volume if missing
        if 'volume' not in data.columns:
            logger.warning("Volume data not found, adding synthetic volume")
            data['volume'] = data['close'] * np.random.randint(1000, 10000, size=len(data))
            
        # Sort by date
        data.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(data)} rows of market data from {data.index.min()} to {data.index.max()}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def load_turning_points(turning_points_file):
    """Load known turning points from CSV file."""
    try:
        tp_df = pd.read_csv(turning_points_file)
        
        # Find date column
        date_col = None
        for col in ['Date', 'date', 'datetime', 'time']:
            if col in tp_df.columns:
                date_col = col
                break
                
        if not date_col:
            logger.error("Turning points file must have a date column")
            sys.exit(1)
            
        # Find type column (top/bottom)
        type_col = None
        for col in ['type', 'Type', 'direction', 'Direction']:
            if col in tp_df.columns:
                type_col = col
                break
                
        if not type_col:
            logger.warning("No type column found in turning points file, assuming all are tops")
            tp_df['type'] = 'top'
            type_col = 'type'
            
        # Convert dates to datetime
        tp_df[date_col] = pd.to_datetime(tp_df[date_col])
            
        # Convert to list of dictionaries
        turning_points = []
        for _, row in tp_df.iterrows():
            turning_points.append({
                'date': row[date_col],
                'type': row[type_col]
            })
            
        logger.info(f"Loaded {len(turning_points)} turning points from file")
        return turning_points
            
    except Exception as e:
        logger.error(f"Error loading turning points: {str(e)}")
        sys.exit(1)

def evaluate_detector(detector, market_data, known_turning_points, tolerance_days=3):
    """
    Evaluate the performance of a turning point detector.
    
    Args:
        detector: TurningPointDetector instance
        market_data: DataFrame with market data
        known_turning_points: List of known turning points
        tolerance_days: Tolerance in days for matching points
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Run detector on data
    detected_turning_points = detector.scan_for_turning_points(market_data)
    
    # Extract dates
    known_dates = [tp['date'] for tp in known_turning_points]
    detected_dates = [tp['date'] for tp in detected_turning_points]
    
    # Match detections to known turning points
    matches = []  # (known_idx, detected_idx, days_diff)
    
    for i, known_date in enumerate(known_dates):
        best_match = None
        min_diff = tolerance_days + 1
        
        for j, detected_date in enumerate(detected_dates):
            days_diff = abs((known_date - detected_date).days)
            
            if days_diff <= tolerance_days and days_diff < min_diff:
                min_diff = days_diff
                best_match = (j, min_diff)
                
        if best_match:
            matches.append((i, best_match[0], best_match[1]))
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(detected_dates) - true_positives
    false_negatives = len(known_dates) - true_positives
    
    # Avoid division by zero
    precision = true_positives / len(detected_dates) if detected_dates else 0
    recall = true_positives / len(known_dates) if known_dates else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average days difference for matches
    avg_days_diff = sum(diff for _, _, diff in matches) / len(matches) if matches else 0
    
    # Direction accuracy (top/bottom)
    direction_correct = 0
    
    for i, j, _ in matches:
        known_type = known_turning_points[i]['type']
        detected_dir = detected_turning_points[j]['direction']
        
        # Normalize types
        if known_type.lower() in ['top', 'high', 'peak']:
            known_type = 'top'
        elif known_type.lower() in ['bottom', 'low', 'trough']:
            known_type = 'bottom'
            
        if known_type == detected_dir:
            direction_correct += 1
            
    direction_accuracy = direction_correct / true_positives if true_positives > 0 else 0
    
    # Collect scores for detected points
    detected_scores = [tp['score'] for tp in detected_turning_points]
    avg_score = sum(detected_scores) / len(detected_scores) if detected_scores else 0
    
    # Results
    results = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_days_diff': avg_days_diff,
        'direction_accuracy': direction_accuracy,
        'avg_score': avg_score,
        'known_count': len(known_dates),
        'detected_count': len(detected_dates),
        'detected_points': detected_turning_points,
        'matches': matches
    }
    
    return results

def visualize_evaluation(market_data, known_turning_points, detected_turning_points, 
                        matches, output_file=None):
    """
    Visualize evaluation results.
    
    Args:
        market_data: DataFrame with market data
        known_turning_points: List of known turning points
        detected_turning_points: List of detected turning points
        matches: List of (known_idx, detected_idx, days_diff) tuples
        output_file: Path to save visualization
    """
    # Create figure with price plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price data
    ax.plot(market_data.index, market_data['close'], label='Close Price')
    
    # Extract dates
    known_dates = [tp['date'] for tp in known_turning_points]
    detected_dates = [tp['date'] for tp in detected_turning_points]
    
    # Create sets of matched indices
    matched_known_indices = set(i for i, _, _ in matches)
    matched_detected_indices = set(j for _, j, _ in matches)
    
    # Plot known turning points
    for i, date in enumerate(known_dates):
        if i in matched_known_indices:
            # Matched known point - green circle
            ax.scatter(date, market_data.loc[date, 'close'] if date in market_data.index else 0, 
                      color='green', marker='o', s=100, zorder=3, alpha=0.7)
        else:
            # Unmatched known point (false negative) - red X
            ax.scatter(date, market_data.loc[date, 'close'] if date in market_data.index else 0, 
                      color='red', marker='x', s=100, zorder=3, alpha=0.7)
    
    # Plot detected turning points
    for j, date in enumerate(detected_dates):
        if j in matched_detected_indices:
            # Matched detection - no need to plot again
            pass
        else:
            # Unmatched detection (false positive) - orange triangle
            ax.scatter(date, market_data.loc[date, 'close'] if date in market_data.index else 0, 
                      color='orange', marker='^', s=100, zorder=3, alpha=0.7)
    
    # Add legend
    ax.scatter([], [], color='green', marker='o', s=100, label='True Positive')
    ax.scatter([], [], color='red', marker='x', s=100, label='False Negative')
    ax.scatter([], [], color='orange', marker='^', s=100, label='False Positive')
    
    # Add lunar phase information
    calculator = LunarPhaseCalculator()
    
    # Generate dates in range
    date_range = pd.date_range(start=market_data.index.min(), end=market_data.index.max(), freq='D')
    
    # Find full moon dates
    full_moon_dates = []
    for date in date_range:
        info = calculator.calculate_lunar_phase(date)
        if info['phase_name'] == 'Full Moon':
            full_moon_dates.append(date)
    
    # Plot full moon dates as vertical lines
    for date in full_moon_dates:
        ax.axvline(x=date, color='blue', alpha=0.3, linestyle='--')
    
    # Add a full moon line to legend
    ax.axvline(x=market_data.index[0], color='blue', alpha=0.3, linestyle='--', label='Full Moon')
    
    # Add legend and labels
    ax.legend()
    ax.set_title('Evaluation of Turning Point Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
    else:
        plt.show()
        
    plt.close()

def visualize_metrics_comparison(ml_results, rule_results, output_file=None):
    """
    Visualize comparison between ML and rule-based approaches.
    
    Args:
        ml_results: Results dictionary for ML-based approach
        rule_results: Results dictionary for rule-based approach
        output_file: Path to save visualization
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Metrics to compare
    metrics = ['precision', 'recall', 'f1_score', 'direction_accuracy']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Direction Accuracy']
    
    # Plot each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Data
        values = [ml_results[metric], rule_results[metric]]
        labels = ['ML-based', 'Rule-based']
        
        # Create bar plot
        bars = ax.bar(labels, values)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Set labels and title
        ax.set_ylabel('Score')
        ax.set_title(label)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
    else:
        plt.show()
        
    plt.close()

def main():
    """Main function to evaluate the detector."""
    # Create directories
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)
    
    # Load market data
    market_data = load_data(args.data_file)
    
    # Load known turning points
    known_turning_points = load_turning_points(args.turning_points_file)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results
    results = {}
    
    # Initialize ML-based detector if model provided
    if args.model_file:
        ml_detector = TurningPointDetector(
            model_path=args.model_file,
            config={
                'confidence_threshold': config['detection']['confidence_threshold'],
                'full_moon_window': config['lunar_parameters']['full_moon_window'],
                'min_volume_ratio': config['feature_engineering']['min_volume_ratio'],
                'min_confirmations': config['detection']['min_confirmations'],
                'use_rule_based': False
            }
        )
        
        # Evaluate ML-based detector
        logger.info("Evaluating ML-based detector...")
        ml_results = evaluate_detector(
            ml_detector, 
            market_data, 
            known_turning_points,
            args.tolerance_days
        )
        
        results['ml_based'] = ml_results
        
        # Save ML results
        ml_results_file = output_dir / f"ml_evaluation_{timestamp}.json"
        with open(ml_results_file, 'w') as f:
            # Convert dates to strings for JSON serialization
            serializable_results = ml_results.copy()
            serializable_results['detected_points'] = [
                {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in tp.items()}
                for tp in ml_results['detected_points']
            ]
            serializable_results.pop('matches', None)  # Remove matches for serialization
            
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved ML-based evaluation results to {ml_results_file}")
    
    # If comparing methods or no model provided, evaluate rule-based detector
    if args.compare_methods or not args.model_file:
        rule_detector = TurningPointDetector(
            config={
                'confidence_threshold': config['detection']['confidence_threshold'],
                'full_moon_window': config['lunar_parameters']['full_moon_window'],
                'min_volume_ratio': config['feature_engineering']['min_volume_ratio'],
                'min_confirmations': config['detection']['min_confirmations'],
                'use_rule_based': True
            }
        )
        
        # Evaluate rule-based detector
        logger.info("Evaluating rule-based detector...")
        rule_results = evaluate_detector(
            rule_detector, 
            market_data, 
            known_turning_points,
            args.tolerance_days
        )
        
        results['rule_based'] = rule_results
        
        # Save rule-based results
        rule_results_file = output_dir / f"rule_evaluation_{timestamp}.json"
        with open(rule_results_file, 'w') as f:
            # Convert dates to strings for JSON serialization
            serializable_results = rule_results.copy()
            serializable_results['detected_points'] = [
                {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in tp.items()}
                for tp in rule_results['detected_points']
            ]
            serializable_results.pop('matches', None)  # Remove matches for serialization
            
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved rule-based evaluation results to {rule_results_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        # Visualize each detector's results
        for method, method_results in results.items():
            viz_file = output_dir / f"{method}_evaluation_viz_{timestamp}.png"
            
            visualize_evaluation(
                market_data,
                known_turning_points,
                method_results['detected_points'],
                method_results['matches'],
                viz_file
            )
        
        # If comparing methods, visualize comparison
        if args.compare_methods and 'ml_based' in results and 'rule_based' in results:
            comp_file = output_dir / f"methods_comparison_{timestamp}.png"
            
            visualize_metrics_comparison(
                results['ml_based'],
                results['rule_based'],
                comp_file
            )
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Data period: {market_data.index.min().date()} to {market_data.index.max().date()}")
    print(f"Known turning points: {len(known_turning_points)}")
    print(f"Tolerance window: ±{args.tolerance_days} days")
    print()
    
    for method, method_results in results.items():
        print(f"=== {method.upper()} DETECTOR ===")
        print(f"Detected turning points: {method_results['detected_count']}")
        print(f"True positives: {method_results['true_positives']}")
        print(f"False positives: {method_results['false_positives']}")
        print(f"False negatives: {method_results['false_negatives']}")
        print(f"Precision: {method_results['precision']:.4f}")
        print(f"Recall: {method_results['recall']:.4f}")
        print(f"F1 Score: {method_results['f1_score']:.4f}")
        print(f"Direction accuracy: {method_results['direction_accuracy']:.4f}")
        print(f"Average days difference: {method_results['avg_days_diff']:.2f}")
        print()
    
    print("="*40)
    
    # Evaluation completed
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
