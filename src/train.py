"""
Main Training Pipeline
Orchestrates the complete training workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent))
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.models.ml_models import RandomForestModel, XGBoostModel, LightGBMModel, EnsembleModel
from src.models.neural_network import NeuralNetworkModel
from src.utils.logger import get_logger, LoggerSetup, PerformanceLogger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


class TrainingPipeline:
    """Complete training pipeline for network attack detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        
        # Models
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        
        logger.info("=" * 80)
        logger.info("AI-Based Network Attack Detection System")
        logger.info("=" * 80)
    
    def run(self, dataset: str = 'cicids2017') -> Dict[str, any]:
        """
        Run complete training pipeline
        
        Args:
            dataset: Dataset to use ('cicids2017' or 'unsw_nb15')
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        logger.info(f"Starting training pipeline with {dataset} dataset")
        
        # Step 1: Load data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)
        df = self._load_data(dataset)
        
        # Step 2: Split data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Splitting Data")
        logger.info("=" * 80)
        train_df, test_df = self.data_loader.split_data(df)
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Testing samples: {len(test_df)}")
        logger.info(f"\nTraining set distribution:")
        for attack, count in self.data_loader.get_attack_distribution(train_df).items():
            logger.info(f"  {attack}: {count}")
        
        # Step 3: Preprocess data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Preprocessing Data")
        logger.info("=" * 80)
        X_train, y_train = self.preprocessor.fit_transform(train_df)
        X_test, y_test = self.preprocessor.transform(test_df)
        
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Testing features shape: {X_test.shape}")
        
        # Step 4: Feature engineering
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Feature Engineering")
        logger.info("=" * 80)
        
        # Create advanced features
        X_train = self.feature_engineer.create_advanced_features(X_train)
        X_test = self.feature_engineer.create_advanced_features(X_test)
        
        # Remove highly correlated features
        X_train = self.feature_engineer.remove_correlated_features(X_train)
        X_test = self.feature_engineer.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_engineer.select_features(X_train, y_train)
        X_test_selected = self.feature_engineer.transform(X_test)
        
        # Get feature importance
        importance = self.feature_engineer.get_feature_importance(X_train_selected, y_train)
        
        logger.info(f"Final feature count: {X_train_selected.shape[1]}")
        
        # Step 5: Train models
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Training Models")
        logger.info("=" * 80)
        
        results = self._train_all_models(X_train_selected, y_train, X_test_selected, y_test)
        
        # Step 6: Create ensemble
        if self.config['models']['ensemble']['enabled']:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: Creating Ensemble Model")
            logger.info("=" * 80)
            ensemble_results = self._create_ensemble(X_test_selected, y_test)
            results['ensemble'] = ensemble_results
        
        # Step 7: Save models and results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Saving Models")
        logger.info("=" * 80)
        self._save_all()
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Best model: {self.best_model}")
        logger.info(f"Best F1 score: {self.best_score:.4f}")
        
        self._print_summary(results)
        
        return results
    
    def _load_data(self, dataset: str) -> pd.DataFrame:
        """Load dataset"""
        if dataset.lower() == 'cicids2017':
            df = self.data_loader.load_cicids2017()
        elif dataset.lower() == 'unsw_nb15' or dataset.lower() == 'unsw-nb15':
            df = self.data_loader.load_unsw_nb15()
        else:
            logger.warning(f"Unknown dataset: {dataset}, using CICIDS2017")
            df = self.data_loader.load_cicids2017()
        
        logger.info(f"Loaded {len(df)} total records")
        logger.info(f"Features: {len(df.columns)}")
        
        return df
    
    def _train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all enabled models"""
        results = {}
        
        models_config = self.config['models']
        
        # Random Forest
        if models_config.get('random_forest', {}).get('enabled', False):
            logger.info("\nTraining Random Forest...")
            rf_model = RandomForestModel()
            rf_model.train(X_train, y_train)
            rf_metrics = rf_model.evaluate(X_test, y_test)
            
            self.models['random_forest'] = rf_model
            results['random_forest'] = rf_metrics
            
            perf_logger.log_metric('f1_score', rf_metrics['f1_score'], 'random_forest')
            
            if rf_metrics['f1_score'] > self.best_score:
                self.best_score = rf_metrics['f1_score']
                self.best_model = 'random_forest'
        
        # XGBoost
        if models_config.get('xgboost', {}).get('enabled', False):
            try:
                logger.info("\nTraining XGBoost...")
                xgb_model = XGBoostModel()
                xgb_model.train(X_train, y_train)
                xgb_metrics = xgb_model.evaluate(X_test, y_test)
                
                self.models['xgboost'] = xgb_model
                results['xgboost'] = xgb_metrics
                
                perf_logger.log_metric('f1_score', xgb_metrics['f1_score'], 'xgboost')
                
                if xgb_metrics['f1_score'] > self.best_score:
                    self.best_score = xgb_metrics['f1_score']
                    self.best_model = 'xgboost'
            except ImportError:
                logger.warning("XGBoost not available, skipping")
        
        # LightGBM
        if models_config.get('lightgbm', {}).get('enabled', False):
            try:
                logger.info("\nTraining LightGBM...")
                lgb_model = LightGBMModel()
                lgb_model.train(X_train, y_train)
                lgb_metrics = lgb_model.evaluate(X_test, y_test)
                
                self.models['lightgbm'] = lgb_model
                results['lightgbm'] = lgb_metrics
                
                perf_logger.log_metric('f1_score', lgb_metrics['f1_score'], 'lightgbm')
                
                if lgb_metrics['f1_score'] > self.best_score:
                    self.best_score = lgb_metrics['f1_score']
                    self.best_model = 'lightgbm'
            except ImportError:
                logger.warning("LightGBM not available, skipping")
        
        # Neural Network
        if models_config.get('neural_network', {}).get('enabled', False):
            try:
                logger.info("\nTraining Neural Network...")
                
                # Create validation split
                val_split = self.config['data']['validation_split']
                split_idx = int(len(X_train) * (1 - val_split))
                
                X_train_nn = X_train.iloc[:split_idx]
                y_train_nn = y_train.iloc[:split_idx]
                X_val_nn = X_train.iloc[split_idx:]
                y_val_nn = y_train.iloc[split_idx:]
                
                nn_model = NeuralNetworkModel()
                nn_model.train(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
                nn_metrics = nn_model.evaluate(X_test, y_test)
                
                self.models['neural_network'] = nn_model
                results['neural_network'] = nn_metrics
                
                perf_logger.log_metric('f1_score', nn_metrics['f1_score'], 'neural_network')
                
                if nn_metrics['f1_score'] > self.best_score:
                    self.best_score = nn_metrics['f1_score']
                    self.best_model = 'neural_network'
            except ImportError:
                logger.warning("TensorFlow not available, skipping neural network")
        
        return results
    
    def _create_ensemble(self, X_test, y_test) -> Dict:
        """Create and evaluate ensemble model"""
        ensemble_config = self.config['models']['ensemble']
        models_to_ensemble = ensemble_config.get('models_to_ensemble', [])
        
        # Get trained models for ensemble
        trained_models = [
            self.models[name] for name in models_to_ensemble 
            if name in self.models and self.models[name].is_trained
        ]
        
        if len(trained_models) < 2:
            logger.warning("Not enough models for ensemble")
            return {}
        
        voting = ensemble_config.get('voting_type', 'soft')
        ensemble = EnsembleModel(trained_models, voting=voting)
        
        ensemble_metrics = ensemble.evaluate(X_test, y_test)
        
        perf_logger.log_metric('f1_score', ensemble_metrics['f1_score'], 'ensemble')
        
        if ensemble_metrics['f1_score'] > self.best_score:
            self.best_score = ensemble_metrics['f1_score']
            self.best_model = 'ensemble'
        
        return ensemble_metrics
    
    def _save_all(self):
        """Save all components"""
        save_dir = self.config['training']['model_save_path']
        
        # Save preprocessor
        self.preprocessor.save(save_dir)
        
        # Save feature engineer
        self.feature_engineer.save(save_dir)
        
        # Save models
        for name, model in self.models.items():
            model.save(save_dir)
    
    def _print_summary(self, results: Dict):
        """Print results summary"""
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Create summary table
        summary_data = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1_score']:.4f}",
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            logger.info(f"\n{summary_df.to_string(index=False)}")


def main():
    """Main entry point"""
    # Setup logging
    log_setup = LoggerSetup(
        name="NetworkAttackDetection",
        log_dir="logs",
        level="INFO",
        console_output=True
    )
    
    # Create and run pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run(dataset='cicids2017')
    
    logger.info("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
