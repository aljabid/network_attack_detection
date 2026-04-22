"""
Feature Engineering Module
Feature extraction, selection, and creation for network traffic data
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif, f_classif, chi2, 
    SelectKBest, RFE
)
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Optional
from pathlib import Path
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class FeatureEngineer:
    """Feature engineering and selection"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature engineer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.feature_config = self.config['features']
        self.selection_config = self.feature_config['selection']
        
        self.selected_features = None
        self.feature_importance = None
        
        logger.info("FeatureEngineer initialized")
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from existing ones
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating advanced features")
        
        df = df.copy()
        
        # Ratio features
        if 'total_fwd_packets' in df.columns and 'total_bwd_packets' in df.columns:
            df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1)
        
        if 'total_length_fwd_packets' in df.columns and 'total_length_bwd_packets' in df.columns:
            df['fwd_bwd_length_ratio'] = df['total_length_fwd_packets'] / (df['total_length_bwd_packets'] + 1)
        
        # Rate features
        if 'flow_duration' in df.columns:
            duration_sec = df['flow_duration'] / 1000000  # Convert to seconds
            duration_sec = duration_sec.replace(0, 0.001)  # Avoid division by zero
            
            if 'total_fwd_packets' in df.columns:
                df['fwd_packets_per_sec'] = df['total_fwd_packets'] / duration_sec
            
            if 'total_bwd_packets' in df.columns:
                df['bwd_packets_per_sec'] = df['total_bwd_packets'] / duration_sec
        
        # Packet size features
        if 'total_length_fwd_packets' in df.columns and 'total_fwd_packets' in df.columns:
            df['avg_fwd_packet_size'] = df['total_length_fwd_packets'] / (df['total_fwd_packets'] + 1)
        
        if 'total_length_bwd_packets' in df.columns and 'total_bwd_packets' in df.columns:
            df['avg_bwd_packet_size'] = df['total_length_bwd_packets'] / (df['total_bwd_packets'] + 1)
        
        # Flag ratios
        flag_cols = [col for col in df.columns if 'flag' in col.lower()]
        if len(flag_cols) > 0:
            df['total_flags'] = df[flag_cols].sum(axis=1)
            
            if 'syn_flag_count' in df.columns:
                df['syn_flag_ratio'] = df['syn_flag_count'] / (df['total_flags'] + 1)
            
            if 'ack_flag_count' in df.columns:
                df['ack_flag_ratio'] = df['ack_flag_count'] / (df['total_flags'] + 1)
        
        # Statistical features
        if 'fwd_packet_length_mean' in df.columns and 'fwd_packet_length_std' in df.columns:
            df['fwd_packet_cv'] = df['fwd_packet_length_std'] / (df['fwd_packet_length_mean'] + 1)
        
        if 'bwd_packet_length_mean' in df.columns and 'bwd_packet_length_std' in df.columns:
            df['bwd_packet_cv'] = df['bwd_packet_length_std'] / (df['bwd_packet_length_mean'] + 1)
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        logger.info(f"Created advanced features. Total features: {len(df.columns)}")
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: Optional[str] = None,
                       n_features: Optional[int] = None) -> pd.DataFrame:
        """
        Select best features
        
        Args:
            X: Features
            y: Target
            method: Selection method
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method is None:
            method = self.selection_config['method']
        
        if n_features is None:
            n_features = self.selection_config['n_features']
        
        # Ensure n_features doesn't exceed available features
        n_features = min(n_features, X.shape[1])
        
        logger.info(f"Selecting {n_features} features using {method}")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X_positive = X - X.min() + 1
            selector = SelectKBest(score_func=chi2, k=n_features)
            X_selected = selector.fit_transform(X_positive, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        else:
            logger.warning(f"Unknown method {method}, using all features")
            self.selected_features = X.columns.tolist()
            return X
        
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {self.selected_features[:10]}...")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def remove_correlated_features(self, X: pd.DataFrame, 
                                   threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            X: Features
            threshold: Correlation threshold
            
        Returns:
            DataFrame with uncorrelated features
        """
        if threshold is None:
            threshold = self.selection_config['correlation_threshold']
        
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        X_reduced = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} correlated features. Remaining: {X_reduced.shape[1]}")
        
        return X_reduced
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                              model_type: str = 'random_forest') -> pd.DataFrame:
        """
        Calculate feature importance
        
        Args:
            X: Features
            y: Target
            model_type: Type of model to use
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating feature importance using {model_type}")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            importances = model.feature_importances_
        
        elif model_type == 'mutual_info':
            importances = mutual_info_classif(X, y, random_state=42)
        
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return pd.DataFrame()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        logger.info("Top 10 important features:")
        logger.info(f"\n{importance_df.head(10)}")
        
        return importance_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using selected features
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if self.selected_features is None:
            logger.warning("No features selected. Using all features.")
            return X
        
        # Keep only selected features
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            logger.warning(
                f"Only {len(available_features)}/{len(self.selected_features)} "
                "selected features available"
            )
        
        return X[available_features]
    
    def save(self, save_dir: str = "data/models"):
        """
        Save feature configuration
        
        Args:
            save_dir: Directory to save to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.selected_features is not None:
            features_path = save_path / "selected_features.pkl"
            joblib.dump(self.selected_features, features_path)
            logger.info(f"Saved selected features to {features_path}")
        
        if self.feature_importance is not None:
            importance_path = save_path / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance to {importance_path}")
    
    def load(self, load_dir: str = "data/models"):
        """
        Load feature configuration
        
        Args:
            load_dir: Directory to load from
        """
        load_path = Path(load_dir)
        
        features_path = load_path / "selected_features.pkl"
        if features_path.exists():
            self.selected_features = joblib.load(features_path)
            logger.info(f"Loaded {len(self.selected_features)} selected features")
        
        importance_path = load_path / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
            logger.info(f"Loaded feature importance scores")
