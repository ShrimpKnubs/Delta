#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Trainer for Full Moon Turning Point Detection
Trains and evaluates machine learning models for market turning point detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
import joblib
import logging
import json
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

# Local imports
from src.feature_engineering import FeatureEngineer
from src.lunar_phase_calculator import LunarPhaseCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TurningPointModelTrainer:
    """
    Trains and evaluates machine learning models for detecting market turning points.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
        """
        # Set default configuration
        self.config = {
            'random_state': 42,
            'test_size': 0.2,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'use_smote': True,
            'smote_ratio': 0.5,  # Target minority class should be 50% of majority class
            'cv_folds': 5,
            'feature_selection': True,
            'use_standard_scaler': True,
            'models_to_try': ['random_forest', 'gradient_boosting'],
            'model_params': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8
                }
            }
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Initialize components
        self.lunar_calculator = LunarPhaseCalculator()
        self.feature_engineer = FeatureEngineer(lunar_calculator=self.lunar_calculator)
        
        # Create output directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger.info("TurningPointModelTrainer initialized")
    
    def prepare_training_data(self, market_data: pd.DataFrame, 
                             turning_points: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for model training.
        
        Args:
            market_data: DataFrame with market data
            turning_points: List of dictionaries with known turning points
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Preparing training data...")
        
        # Generate features
        features_df = self.feature_engineer.create_features(
            market_data, 
            include_target=True,
            known_turning_points=turning_points
        )
        
        # Drop non-feature columns
        X = features_df.drop(['open', 'high', 'low', 'close', 'volume', 
                             'is_turning_point', 'is_top', 'is_bottom'], 
                             axis=1, errors='ignore')
        
        # Target variable
        y = features_df['is_turning_point']
        y = (y > 0).astype(int)
        # Drop date-related columns and non-numeric columns
        X = X.select_dtypes(include=['number'])
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared {X.shape[1]} features and {len(y)} samples")
        logger.info(f"Positive class samples: {sum(y > 0)}, Negative class samples: {sum(y == 0)}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'gradient_boosting') -> Dict:
        """
        Train a model for turning point detection.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train
            
        Returns:
            Dictionary with trained model and performance metrics
        """
        logger.info(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y  # Ensure balanced classes in train/test
        )
        
        # Apply SMOTE if configured
        if self.config['use_smote']:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(
                sampling_strategy=self.config['smote_ratio'],
                random_state=self.config['random_state']
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
            
        logger.info(f"Training data after resampling: {X_train_resampled.shape[0]} samples, "
                  f"Positive: {sum(y_train_resampled > 0)}, Negative: {sum(y_train_resampled == 0)}")
        
        # Create preprocessing pipeline
        preprocessor = []
        
        if self.config['use_standard_scaler']:
            preprocessor.append(('scaler', StandardScaler()))
            
        # Create model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                **self.config['model_params']['random_forest'],
                random_state=self.config['random_state'],
                class_weight=self.config['class_weight'],
                n_jobs=self.config['n_jobs']
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                **self.config['model_params']['gradient_boosting'],
                random_state=self.config['random_state']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        steps = []
        
        # Add preprocessing if needed
        if preprocessor:
            for name, transform in preprocessor:
                steps.append((name, transform))
                
        # Add feature selection if configured
        if self.config['feature_selection']:
            # Use a model to select features
            feature_selector = SelectFromModel(
                GradientBoostingClassifier(
                    n_estimators=50, 
                    random_state=self.config['random_state']
                ),
                threshold='median'
            )
            steps.append(('feature_selection', feature_selector))
            
        # Add model
        steps.append(('model', model))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # Train model
        pipeline.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Performance metrics
        metrics = {
            'accuracy': pipeline.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importances': self._get_feature_importances(pipeline, X.columns)
        }
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            pipeline, X, y, 
            cv=self.config['cv_folds'], 
            scoring='f1'
        )
        metrics['cv_f1_score'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        logger.info(f"Model trained successfully. Test accuracy: {metrics['accuracy']:.4f}, "
                  f"CV F1 score: {metrics['cv_f1_score']:.4f} ± {metrics['cv_f1_std']:.4f}")
        
        return {
            'model': pipeline,
            'metrics': metrics,
            'model_type': model_type
        }
    
    def _get_feature_importances(self, pipeline: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importances from the pipeline."""
        # Get the model from the pipeline
        model = pipeline.named_steps['model']
        
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        # If feature selection was used, get the selected features
        if 'feature_selection' in pipeline.named_steps:
            selector = pipeline.named_steps['feature_selection']
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            importances = model.feature_importances_
            
            # Match importances to original feature names
            result = {
                selected_features[i]: float(importances[i])
                for i in range(len(selected_features))
            }
        else:
            # No feature selection, use all features
            importances = model.feature_importances_
            result = {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }
        
        # Sort by importance
        result = {k: v for k, v in sorted(
            result.items(), key=lambda item: item[1], reverse=True
        )}
        
        return result
    
    def train_and_evaluate_models(self, market_data: pd.DataFrame, 
                               turning_points: List[Dict]) -> Dict:
        """
        Train and evaluate multiple models.
        
        Args:
            market_data: DataFrame with market data
            turning_points: List of dictionaries with known turning points
            
        Returns:
            Dictionary with results for all models
        """
        # Prepare data
        X, y = self.prepare_training_data(market_data, turning_points)
        
        results = {}
        best_model = None
        best_score = -1
        
        # Train each model type
        for model_type in self.config['models_to_try']:
            logger.info(f"Training {model_type} model...")
            
            try:
                model_result = self.train_model(X, y, model_type)
                results[model_type] = model_result
                
                # Track best model
                cv_score = model_result['metrics']['cv_f1_score']
                if cv_score > best_score:
                    best_score = cv_score
                    best_model = model_type
                    
                # Save model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/{model_type}_{timestamp}.joblib"
                joblib.dump(model_result['model'], model_path)
                logger.info(f"Saved model to {model_path}")
                
                # Add path to results
                results[model_type]['model_path'] = model_path
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/model_training_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for model_type, result in results.items():
            if 'error' in result:
                serializable_results[model_type] = {'error': result['error']}
            else:
                serializable_results[model_type] = {
                    'metrics': result['metrics'],
                    'model_type': result['model_type'],
                    'model_path': result['model_path']
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved results to {results_path}")
        
        # Print summary
        logger.info("=== Training Summary ===")
        for model_type, result in results.items():
            if 'error' in result:
                logger.info(f"{model_type}: Error - {result['error']}")
            else:
                metrics = result['metrics']
                logger.info(f"{model_type}: CV F1 = {metrics['cv_f1_score']:.4f}, "
                          f"ROC AUC = {metrics['roc_auc']:.4f}")
                
        if best_model:
            logger.info(f"Best model: {best_model} with F1 score {best_score:.4f}")
            
        return results
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize model training results.
        
        Args:
            results: Dictionary with model results
            save_path: Optional path to save visualizations
        """
        # Ensure save directory exists
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = plt.GridSpec(2, 2)
        
        # Plot ROC curves
        ax1 = fig.add_subplot(gs[0, 0])
        for model_type, result in results.items():
            if 'error' in result:
                continue
                
            metrics = result['metrics']
            if 'roc_curve' in metrics:
                roc = metrics['roc_curve']
                ax1.plot(roc['fpr'], roc['tpr'], 
                        label=f"{model_type} (AUC = {metrics['roc_auc']:.4f})")
                
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot feature importances for best model
        ax2 = fig.add_subplot(gs[0, 1])
        best_model = None
        best_score = -1
        
        for model_type, result in results.items():
            if 'error' in result:
                continue
                
            metrics = result['metrics']
            if metrics['cv_f1_score'] > best_score:
                best_score = metrics['cv_f1_score']
                best_model = model_type
                
        if best_model:
            importances = results[best_model]['metrics']['feature_importances']
            
            # Take top 15 features
            top_features = dict(list(importances.items())[:15])
            
            # Create bar plot
            bars = ax2.barh(
                list(top_features.keys())[::-1], 
                list(top_features.values())[::-1]
            )
            ax2.set_title(f'Top 15 Feature Importances ({best_model})')
            ax2.set_xlabel('Importance')
            # Add values to bars
            for bar in bars:
                width = bar.get_width()
                ax2.text(
                    width + 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{width:.4f}", 
                    ha='left', 
                    va='center'
                )
        
        # Plot F1 scores
        ax3 = fig.add_subplot(gs[1, 0])
        model_types = []
        f1_scores = []
        f1_stds = []
        
        for model_type, result in results.items():
            if 'error' in result:
                continue
                
            metrics = result['metrics']
            model_types.append(model_type)
            f1_scores.append(metrics['cv_f1_score'])
            f1_stds.append(metrics['cv_f1_std'])
            
        # Create bar plot
        bars = ax3.bar(model_types, f1_scores, yerr=f1_stds)
        ax3.set_title('Cross-Validation F1 Scores')
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim([0, 1])
        
        # Add values to bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2, 
                height + 0.01, 
                f"{height:.4f}", 
                ha='center', 
                va='bottom'
            )
            
        # Plot confusion matrix for best model
        ax4 = fig.add_subplot(gs[1, 1])
        if best_model:
            metrics = results[best_model]['metrics']
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                
                # Extract values for confusion matrix
                tn = report['0']['support'] - report['0']['support'] * report['0']['recall']
                fp = report['0']['support'] - tn
                fn = report['1']['support'] - report['1']['support'] * report['1']['recall']
                tp = report['1']['support'] - fn
                
                cm = np.array([[tn, fp], [fn, tp]])
                
                # Plot confusion matrix
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='g', 
                    cmap='Blues',
                    ax=ax4
                )
                ax4.set_title(f'Confusion Matrix ({best_model})')
                ax4.set_xlabel('Predicted')
                ax4.set_ylabel('Actual')
                ax4.set_xticklabels(['No Turn', 'Turn'])
                ax4.set_yticklabels(['No Turn', 'Turn'])
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
    def test_on_unseen_data(self, model_path: str, test_data: pd.DataFrame, 
                           known_turning_points: Optional[List[Dict]] = None) -> Dict:
        """
        Test a trained model on unseen data.
        
        Args:
            model_path: Path to trained model file
            test_data: DataFrame with market data
            known_turning_points: Optional list of known turning points for evaluation
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing model from {model_path} on unseen data...")
        
        # Load model
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {'error': str(e)}
        
        # Generate features
        features_df = self.feature_engineer.create_features(
            test_data, 
            include_target=known_turning_points is not None,
            known_turning_points=known_turning_points
        )
        
        # Prepare features
        X = features_df.drop(['open', 'high', 'low', 'close', 'volume', 
                             'is_turning_point', 'is_top', 'is_bottom'], 
                             axis=1, errors='ignore')
                             
        # Keep only numeric columns
        X = X.select_dtypes(include=['number'])
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get predictions
        y_prob = model.predict_proba(X)[:, 1]
        
        # Add to dataframe
        features_df['turning_point_probability'] = y_prob
        features_df['predicted_turning_point'] = (y_prob >= 0.5).astype(int)
        
        # Evaluate if known turning points provided
        if known_turning_points:
            y_true = features_df['is_turning_point']
            y_pred = features_df['predicted_turning_point']
            
            # Calculate metrics
            metrics = {
                'accuracy': (y_pred == y_true).mean(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            # ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            metrics['roc_auc'] = auc(fpr, tpr)
            
            logger.info(f"Test accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            metrics = None
            
        # Count detected turning points
        detected_points = features_df[features_df['predicted_turning_point'] == 1]
        logger.info(f"Detected {len(detected_points)} turning points in test data")
        
        return {
            'predictions': features_df[['turning_point_probability', 'predicted_turning_point']],
            'metrics': metrics,
            'detected_turning_points': detected_points.index.tolist()
        }


if __name__ == "__main__":
    import pandas as pd
    import yfinance as yf
    from argparse import ArgumentParser
    
    # Command line arguments
    parser = ArgumentParser(description="Train turning point detection models")
    parser.add_argument("--data-file", type=str, help="Path to market data CSV file")
    parser.add_argument("--turning-points-file", type=str, help="Path to known turning points CSV file")
    parser.add_argument("--config-file", type=str, help="Path to configuration JSON file")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to download if data file not provided")
    parser.add_argument("--start-date", type=str, default="2010-01-01", help="Start date for data download")
    parser.add_argument("--end-date", type=str, default="2023-01-01", help="End date for data download")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Create trainer
    trainer = TurningPointModelTrainer(config)
    
    # Load data
    if args.data_file:
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        data.columns = [c.lower() for c in data.columns]
    else:
        # Download data
        data = yf.download(args.symbol, start=args.start_date, end=args.end_date)
        data.columns = [c.lower() for c in data.columns]
    
    # Load known turning points
    if args.turning_points_file:
        turning_points_df = pd.read_csv(args.turning_points_file)
        turning_points = []
        
        for _, row in turning_points_df.iterrows():
            turning_points.append({
                'date': row['date'],
                'type': row['type'] if 'type' in row else 'unknown'
            })
    else:
        # Generate synthetic turning points for demonstration
        logger.warning("No turning points file provided. Generating synthetic turning points...")
        
        # Find potential turning points using rule-based approach
        from turning_point_detector import TurningPointDetector
        detector = TurningPointDetector()
        detected_points = detector.scan_for_turning_points(data)
        
        # Convert to expected format
        turning_points = [
            {
                'date': tp['date'],
                'type': tp['direction']
            }
            for tp in detected_points
        ]
        
        logger.info(f"Generated {len(turning_points)} synthetic turning points")
    
    # Train models
    results = trainer.train_and_evaluate_models(data, turning_points)
    
    # Visualize if requested
    if args.visualize:
        trainer.visualize_results(results, save_path="results/model_comparison.png")
