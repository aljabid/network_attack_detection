"""
Deep Learning Models Module
Neural network architectures for attack detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class NeuralNetworkModel:
    """Deep Neural Network for attack classification"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize neural network
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.model_config = self.config['models'].get('neural_network', {})
        self.arch_config = self.model_config.get('architecture', {})
        self.train_config = self.model_config.get('training', {})
        
        self.model = None
        self.history = None
        self.metrics = {}
        self.is_trained = False
        
        logger.info("Initialized Neural Network model")
    
    def build_model(self, input_dim: int, n_classes: int) -> None:
        """
        Build neural network architecture
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, models
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        
        logger.info(f"Building neural network: {input_dim} inputs, {n_classes} outputs")
        
        # Get architecture parameters
        hidden_layers = self.arch_config.get('hidden_layers', [256, 128, 64, 32])
        dropout_rate = self.arch_config.get('dropout_rate', 0.3)
        activation = self.arch_config.get('activation', 'relu')
        output_activation = self.arch_config.get('output_activation', 'softmax')
        use_batch_norm = self.arch_config.get('batch_normalization', True)
        
        # Build model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation=activation, name=f'dense_{i+1}'))
            
            if use_batch_norm:
                model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(n_classes, activation=output_activation, name='output'))
        
        # Compile model
        learning_rate = self.train_config.get('learning_rate', 0.001)
        optimizer_name = self.train_config.get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = 'adam'
        
        loss = self.train_config.get('loss', 'sparse_categorical_crossentropy')
        metrics = self.train_config.get('metrics', ['accuracy'])
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        
        logger.info("Neural network architecture:")
        model.summary(print_fn=logger.info)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> None:
        """
        Train neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import callbacks
        except ImportError:
            logger.error("TensorFlow not installed")
            raise
        
        # Build model if not built
        if self.model is None:
            input_dim = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            self.build_model(input_dim, n_classes)
        
        logger.info(f"Training neural network with {len(X_train)} samples")
        
        # Training parameters
        epochs = self.train_config.get('epochs', 50)
        batch_size = self.train_config.get('batch_size', 256)
        
        # Callbacks
        callback_list = []
        
        # Early stopping
        if self.train_config.get('early_stopping', {}).get('enabled', True):
            early_stop = callbacks.EarlyStopping(
                monitor=self.train_config['early_stopping'].get('monitor', 'val_loss'),
                patience=self.train_config['early_stopping'].get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Reduce learning rate
        if self.train_config.get('reduce_lr', {}).get('enabled', True):
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.train_config['reduce_lr'].get('factor', 0.5),
                patience=self.train_config['reduce_lr'].get('patience', 5),
                min_lr=self.train_config['reduce_lr'].get('min_lr', 0.00001),
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Neural network training completed")
    
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
        
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
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
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        logger.info("Evaluating neural network")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        self.metrics['classification_report'] = report
        
        logger.info(f"Neural Network - Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"Neural Network - F1 Score: {self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def get_training_history(self) -> Dict[str, list]:
        """Get training history"""
        if self.history is None:
            return {}
        
        return {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }
    
    def save(self, save_dir: str = "data/models") -> None:
        """
        Save trained model
        
        Args:
            save_dir: Directory to save model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = save_path / "neural_network_model.h5"
        self.model.save(model_path)
        
        # Save metrics
        metrics_path = save_path / "neural_network_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save training history
        if self.history is not None:
            history_path = save_path / "neural_network_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.get_training_history(), f, indent=2)
        
        logger.info(f"Saved neural network to {model_path}")
    
    def load(self, load_dir: str = "data/models") -> None:
        """
        Load trained model
        
        Args:
            load_dir: Directory to load from
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.error("TensorFlow not installed")
            raise
        
        load_path = Path(load_dir)
        
        model_path = load_path / "neural_network_model.h5"
        if model_path.exists():
            self.model = keras.models.load_model(model_path)
            self.is_trained = True
            logger.info(f"Loaded neural network from {model_path}")
        
        # Load metrics
        metrics_path = load_path / "neural_network_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
