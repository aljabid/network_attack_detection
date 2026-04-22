"""
Unit Tests for Network Attack Detection System
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.models.ml_models import RandomForestModel
from src.utils.config_loader import get_config


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader is not None
        assert loader.data_config is not None
    
    def test_load_sample_data(self):
        """Test loading sample data"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        assert len(df) > 0
        assert 'attack_type' in df.columns
        assert df['attack_type'].nunique() > 1
    
    def test_split_data(self):
        """Test data splitting"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        train_df, test_df = loader.split_data(df, test_size=0.2)
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(df)
    
    def test_attack_distribution(self):
        """Test attack distribution calculation"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        distribution = loader.get_attack_distribution(df)
        
        assert isinstance(distribution, dict)
        assert len(distribution) > 0
        assert sum(distribution.values()) == len(df)


class TestPreprocessor:
    """Test data preprocessing"""
    
    def test_initialization(self):
        """Test Preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
    
    def test_fit_transform(self):
        """Test fitting and transforming data"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col='attack_type')
        
        assert X is not None
        assert y is not None
        assert len(X) > 0
        assert len(X) == len(y)
        assert not X.isnull().any().any()  # No missing values
    
    def test_transform(self):
        """Test transforming new data"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        train_df, test_df = loader.split_data(df)
        
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df, target_col='attack_type')
        X_test, y_test = preprocessor.transform(test_df, target_col='attack_type')
        
        assert X_test is not None
        assert len(X_test) > 0
        assert X_train.shape[1] == X_test.shape[1]  # Same features


class TestFeatureEngineer:
    """Test feature engineering"""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer is not None
    
    def test_create_advanced_features(self):
        """Test creating advanced features"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col='attack_type')
        
        engineer = FeatureEngineer()
        X_enhanced = engineer.create_advanced_features(X)
        
        assert X_enhanced is not None
        assert X_enhanced.shape[1] >= X.shape[1]  # More or equal features
        assert len(X_enhanced) == len(X)  # Same number of samples
    
    def test_feature_selection(self):
        """Test feature selection"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col='attack_type')
        
        engineer = FeatureEngineer()
        n_features = min(10, X.shape[1])
        X_selected = engineer.select_features(X, y, n_features=n_features)
        
        assert X_selected is not None
        assert X_selected.shape[1] == n_features
        assert len(X_selected) == len(X)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col='attack_type')
        
        engineer = FeatureEngineer()
        importance_df = engineer.get_feature_importance(X, y)
        
        assert importance_df is not None
        assert len(importance_df) == X.shape[1]
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestMLModels:
    """Test machine learning models"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data"""
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        train_df, test_df = loader.split_data(df, test_size=0.3)
        
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df, target_col='attack_type')
        X_test, y_test = preprocessor.transform(test_df, target_col='attack_type')
        
        return X_train, y_train, X_test, y_test
    
    def test_random_forest_training(self, sample_data):
        """Test Random Forest training"""
        X_train, y_train, X_test, y_test = sample_data
        
        model = RandomForestModel()
        model.train(X_train, y_train)
        
        assert model.is_trained
        assert model.model is not None
    
    def test_random_forest_prediction(self, sample_data):
        """Test Random Forest prediction"""
        X_train, y_train, X_test, y_test = sample_data
        
        model = RandomForestModel()
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert predictions.min() >= 0
    
    def test_random_forest_evaluation(self, sample_data):
        """Test Random Forest evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        
        model = RandomForestModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metrics are in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading"""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train and save
        model = RandomForestModel()
        model.train(X_train, y_train)
        model.save(str(tmp_path))
        
        # Load and predict
        loaded_model = RandomForestModel()
        loaded_model.load(str(tmp_path))
        
        assert loaded_model.is_trained
        predictions = loaded_model.predict(X_test)
        assert predictions is not None


class TestConfig:
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = get_config()
        assert config is not None
        assert config.config is not None
    
    def test_config_get(self):
        """Test getting config values"""
        config = get_config()
        
        # Test dot notation access
        value = config.get('data.train_test_split')
        assert value is not None
        
        # Test default value
        value = config.get('nonexistent.key', default='default')
        assert value == 'default'
    
    def test_config_access(self):
        """Test dictionary-style access"""
        config = get_config()
        
        assert 'data' in config.config
        assert 'models' in config.config


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_pipeline(self):
        """Test complete training pipeline"""
        # Load data
        loader = DataLoader()
        df = loader._create_sample_cicids2017()
        train_df, test_df = loader.split_data(df, test_size=0.3)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df, target_col='attack_type')
        X_test, y_test = preprocessor.transform(test_df, target_col='attack_type')
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train = engineer.create_advanced_features(X_train)
        X_test = engineer.create_advanced_features(X_test)
        
        n_features = min(20, X_train.shape[1])
        X_train = engineer.select_features(X_train, y_train, n_features=n_features)
        X_test = engineer.transform(X_test)
        
        # Train model
        model = RandomForestModel()
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Assert pipeline completed successfully
        assert metrics['accuracy'] > 0.5  # Should be better than random
        assert metrics['f1_score'] > 0.3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
