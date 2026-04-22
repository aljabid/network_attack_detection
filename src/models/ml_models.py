"""
Machine Learning Models Module
Implements various ML algorithms for attack detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class MLModel:
    """Base class for machine learning models"""
    
    def __init__(self, model_name: str, config_path: Optional[str] = None):
        """
        Initialize ML model
        
        Args:
            model_name: Name of the model
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.config = get_config(config_path)
        self.model_config = self.config['models'].get(model_name, {})
        self.training_config = self.config['training']
        
        self.model = None
        self.metrics = {}
        self.is_trained = False
        
        logger.info(f"Initialized {model_name} model")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_name} does not support probability prediction")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {self.model_name}")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC AUC for multiclass (one-vs-rest)
        try:
            y_pred_proba = self.predict_proba(X_test)
            self.metrics['roc_auc'] = roc_auc_score(
                y_test, y_pred_proba, 
                multi_class='ovr', 
                average='weighted'
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            self.metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        self.metrics['classification_report'] = report
        
        logger.info(f"{self.model_name} - Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"{self.model_name} - F1 Score: {self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        logger.info(f"Performing {cv}-fold cross-validation on {self.model_name}")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def save(self, save_dir: str = "data/models") -> None:
        """
        Save trained model
        
        Args:
            save_dir: Directory to save model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / f"{self.model_name}_model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save metrics
        metrics_path = save_path / f"{self.model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved {self.model_name} to {model_path}")
    
    def load(self, load_dir: str = "data/models") -> None:
        """
        Load trained model
        
        Args:
            load_dir: Directory to load from
        """
        load_path = Path(load_dir)
        
        model_path = load_path / f"{self.model_name}_model.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.is_trained = True
            logger.info(f"Loaded {self.model_name} from {model_path}")
        
        # Load metrics
        metrics_path = load_path / f"{self.model_name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)


class RandomForestModel(MLModel):
    """Random Forest classifier"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__('random_forest', config_path)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model"""
        logger.info(f"Training {self.model_name} with {len(X_train)} samples")
        
        params = self.model_config.get('params', {})
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info(f"{self.model_name} training completed")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class XGBoostModel(MLModel):
    """XGBoost classifier"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__('xgboost', config_path)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        
        logger.info(f"Training {self.model_name} with {len(X_train)} samples")
        
        params = self.model_config.get('params', {})
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info(f"{self.model_name} training completed")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LightGBMModel(MLModel):
    """LightGBM classifier"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__('lightgbm', config_path)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            raise
        
        logger.info(f"Training {self.model_name} with {len(X_train)} samples")
        
        params = self.model_config.get('params', {})
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info(f"{self.model_name} training completed")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self, models: List[MLModel], voting: str = 'soft'):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            voting: Voting method ('hard' or 'soft')
        """
        self.models = models
        self.voting = voting
        self.ensemble = None
        self.is_trained = False
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def create_ensemble(self) -> None:
        """Create voting ensemble"""
        estimators = [(m.model_name, m.model) for m in self.models if m.is_trained]
        
        if len(estimators) < 2:
            raise ValueError("Need at least 2 trained models for ensemble")
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            n_jobs=-1
        )
        
        logger.info(f"Created {self.voting} voting ensemble")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble"""
        if self.ensemble is None:
            self.create_ensemble()
        
        # Ensemble is already fitted (uses fitted estimators)
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using ensemble"""
        if self.ensemble is None:
            self.create_ensemble()
        
        if self.voting == 'soft':
            return self.ensemble.predict_proba(X)
        else:
            raise AttributeError("Hard voting does not support probability prediction")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Ensemble - F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
