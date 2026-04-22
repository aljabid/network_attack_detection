"""
Data Preprocessing Module
Handles cleaning, normalization, encoding, and balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Optional, List
from pathlib import Path
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess network traffic data"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.preprocess_config = self.config['preprocessing']
        
        self.scaler = None
        self.label_encoders = {}
        
        logger.info("DataPreprocessor initialized")
    
    def fit_transform(self, train_df: pd.DataFrame, 
                     target_col: str = 'attack_type') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform training data
        
        Args:
            train_df: Training dataframe
            target_col: Target column name
            
        Returns:
            Processed features and target
        """
        logger.info("Fitting and transforming training data")
        
        # Separate features and target
        if target_col in train_df.columns:
            y = train_df[target_col].copy()
            X = train_df.drop(columns=[target_col])
        else:
            y = None
            X = train_df.copy()
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Detect and handle outliers
        '''
        if self.preprocess_config['outlier_detection']:
            X = self._handle_outliers(X)
        if self.preprocess_config['outlier_detection']:
            X = self._handle_outliers(X)
        '''
        if self.preprocess_config['outlier_detection']:
            #X, y = self._handle_outliers(X, y)
            X, y = self._handle_outliers(X, y)

        # Encode categorical features
        X = self._encode_categorical(X, fit=True)
        
        # Scale numerical features
        if self.preprocess_config['scaling']['apply_to_features']:
            X = self._scale_features(X, fit=True)
        
        # Balance classes
        if y is not None and self.preprocess_config['balance_classes']:
            X, y = self._balance_classes(X, y)
        
        logger.info(f"Preprocessed training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def transform(self, test_df: pd.DataFrame, 
                  target_col: str = 'attack_type') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform test data using fitted preprocessor
        
        Args:
            test_df: Test dataframe
            target_col: Target column name
            
        Returns:
            Processed features and target
        """
        logger.info("Transforming test data")
        
        # Separate features and target
        if target_col in test_df.columns:
            y = test_df[target_col].copy()
            X = test_df.drop(columns=[target_col])
        else:
            y = None
            X = test_df.copy()
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=False)
        
        # Encode categorical features
        X = self._encode_categorical(X, fit=False)
        
        # Scale numerical features
        if self.preprocess_config['scaling']['apply_to_features']:
            X = self._scale_features(X, fit=False)
        
        logger.info(f"Preprocessed test data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            fit: Whether to fit strategies
            
        Returns:
            DataFrame with missing values handled
        """
        method = self.preprocess_config['handle_missing']
        
        if method == 'drop':
            # Drop rows with missing values
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before > after:
                logger.info(f"Dropped {before - after} rows with missing values")
        
        elif method == 'mean':
            # Fill with mean for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if fit:
                self.fill_values = df[numerical_cols].mean()
            df[numerical_cols] = df[numerical_cols].fillna(self.fill_values)
        
        elif method == 'median':
            # Fill with median for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if fit:
                self.fill_values = df[numerical_cols].median()
            df[numerical_cols] = df[numerical_cols].fillna(self.fill_values)
        
        elif method == 'mode':
            # Fill with mode
            if fit:
                self.fill_values = df.mode().iloc[0]
            df = df.fillna(self.fill_values)
        
        return df

    def _handle_outliers(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        method = self.preprocess_config['outlier_method']
        contamination = self.preprocess_config['outlier_contamination']
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X, y

        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            predictions = detector.fit_predict(X[numerical_cols].fillna(0))
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
            predictions = detector.fit_predict(X[numerical_cols].fillna(0))

        mask = predictions != -1
        before = len(X)

        # Apply mask to both X and y
        X = X.loc[mask]
        if y is not None:
            y = y.loc[X.index]  # align by index, not by raw mask

        after = len(X)
        logger.info(f"Removed {before - after} outliers using {method}")
        return X, y

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with encoded features
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df
        
        method = self.preprocess_config['encoding']['categorical_method']
        
        if method == 'label':
            for col in categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        elif method == 'onehot':
            if fit:
                df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
                self.encoded_columns = df.columns.tolist()
            else:
                df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
                # Ensure same columns as training
                for col in self.encoded_columns:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.encoded_columns]
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns using {method}")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit scaler
            
        Returns:
            DataFrame with scaled features
        """
        method = self.preprocess_config['scaling']['method']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return df
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            logger.info(f"Fitted {method} scaler on {len(numerical_cols)} features")
        
        else:
            if self.scaler is not None:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def _balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance class distribution
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Balanced features and target
        """
        method = self.preprocess_config['balancing_method']
        
        logger.info(f"Original class distribution:\n{y.value_counts()}")
        
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=5)
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42)
            elif method == 'random_oversample':
                sampler = RandomOverSampler(random_state=42)
            elif method == 'random_undersample':
                sampler = RandomUnderSampler(random_state=42)
            else:
                logger.warning(f"Unknown balancing method: {method}")
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            logger.info(f"Balanced classes using {method}")
            logger.info(f"New class distribution:\n{pd.Series(y_resampled).value_counts()}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        except Exception as e:
            logger.error(f"Error balancing classes: {e}")
            logger.info("Continuing with unbalanced data")
            return X, y
    
    def save(self, save_dir: str = "data/models"):
        """
        Save fitted preprocessor
        
        Args:
            save_dir: Directory to save preprocessor
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = save_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        
        # Save label encoders
        if self.label_encoders:
            encoders_path = save_path / "label_encoders.pkl"
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"Saved label encoders to {encoders_path}")
    
    def load(self, load_dir: str = "data/models"):
        """
        Load fitted preprocessor
        
        Args:
            load_dir: Directory to load from
        """
        load_path = Path(load_dir)
        
        # Load scaler
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load label encoders
        encoders_path = load_path / "label_encoders.pkl"
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)
            logger.info(f"Loaded label encoders from {encoders_path}")
