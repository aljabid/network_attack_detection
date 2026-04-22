"""
Real-Time Detection Module
Performs real-time network attack detection on live traffic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from collections import deque
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, AlertLogger
from utils.config_loader import get_config

logger = get_logger(__name__)
alert_logger = AlertLogger()


class RealTimeDetector:
    """Real-time network attack detection system"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained model directory
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.realtime_config = self.config.get('realtime', {})
        
        self.model_path = Path(model_path)
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        
        # Detection parameters
        self.prediction_threshold = self.realtime_config.get('prediction_threshold', 0.7)
        self.alert_threshold = self.realtime_config.get('alert_threshold', 0.9)
        self.buffer_size = self.realtime_config.get('buffer_size', 1000)
        
        # Attack type mapping
        self.attack_types = {v: k for k, v in self.config['data']['attack_types'].items()}
        
        # Buffers
        self.traffic_buffer = deque(maxlen=self.buffer_size)
        self.detection_history = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'attacks_detected': 0,
            'false_positives': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Real-time detector initialized")
    
    def load_models(self) -> None:
        """Load trained models and preprocessors"""
        logger.info(f"Loading models from {self.model_path}")
        
        try:
            # Load main model (use best model from training)
            model_files = list(self.model_path.glob("*_model.pkl"))
            
            if model_files:
                self.model = joblib.load(model_files[0])
                logger.info(f"Loaded model: {model_files[0].name}")
            else:
                # Try loading neural network
                nn_model_path = self.model_path / "neural_network_model.h5"
                if nn_model_path.exists():
                    from models.neural_network import NeuralNetworkModel
                    self.model = NeuralNetworkModel()
                    self.model.load(str(self.model_path))
                    logger.info("Loaded neural network model")
                else:
                    raise FileNotFoundError("No trained model found")
            
            # Load preprocessor
            from preprocessing.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load(str(self.model_path))
            
            # Load feature engineer
            from feature_engineering.feature_engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.load(str(self.model_path))
            
            logger.info("All components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def process_packet(self, packet_features: Dict) -> Tuple[int, float, Dict]:
        """
        Process a single packet and detect attacks
        
        Args:
            packet_features: Dictionary of packet features
            
        Returns:
            Tuple of (attack_type, confidence, alert_info)
        """
        self.stats['total_packets'] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame([packet_features])
        
        # Preprocess
        try:
            X, _ = self.preprocessor.transform(df)
            
            # Feature engineering
            X = self.feature_engineer.create_advanced_features(X)
            X = self.feature_engineer.transform(X)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
            else:
                predicted_class = self.model.predict(X)[0]
                confidence = 1.0
            
            # Determine if this is an attack
            is_attack = predicted_class != 0 and confidence >= self.prediction_threshold
            
            # Create alert info
            alert_info = {
                'timestamp': datetime.now().isoformat(),
                'attack_type': self.attack_types.get(predicted_class, 'unknown'),
                'confidence': float(confidence),
                'source_ip': packet_features.get('source_ip', 'unknown'),
                'destination_ip': packet_features.get('destination_ip', 'unknown'),
                'is_attack': is_attack,
                'severity': self._get_severity(confidence) if is_attack else 'none'
            }
            
            # Log if attack detected
            if is_attack:
                self.stats['attacks_detected'] += 1
                
                if confidence >= self.alert_threshold:
                    alert_logger.log_attack(
                        attack_type=alert_info['attack_type'],
                        confidence=confidence,
                        source_ip=alert_info['source_ip'],
                        details=packet_features
                    )
            
            # Add to history
            self.detection_history.append(alert_info)
            
            return predicted_class, confidence, alert_info
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            return 0, 0.0, {}
    
    def process_batch(self, batch_features: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of packets
        
        Args:
            batch_features: DataFrame with packet features
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Processing batch of {len(batch_features)} packets")
        
        try:
            # Preprocess
            X, _ = self.preprocessor.transform(batch_features)
            
            # Feature engineering
            X = self.feature_engineer.create_advanced_features(X)
            X = self.feature_engineer.transform(X)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                predictions = np.argmax(probabilities, axis=1)
                confidences = np.max(probabilities, axis=1)
            else:
                predictions = self.model.predict(X)
                confidences = np.ones(len(predictions))
            
            # Create results DataFrame
            results = batch_features.copy()
            results['predicted_attack'] = [self.attack_types.get(p, 'unknown') for p in predictions]
            results['confidence'] = confidences
            results['is_attack'] = (predictions != 0) & (confidences >= self.prediction_threshold)
            
            # Update statistics
            self.stats['total_packets'] += len(batch_features)
            self.stats['attacks_detected'] += results['is_attack'].sum()
            
            logger.info(f"Detected {results['is_attack'].sum()} attacks in batch")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return pd.DataFrame()
    
    def _get_severity(self, confidence: float) -> str:
        """
        Determine attack severity based on confidence
        
        Args:
            confidence: Prediction confidence
            
        Returns:
            Severity level
        """
        if confidence >= 0.95:
            return 'critical'
        elif confidence >= 0.85:
            return 'high'
        elif confidence >= 0.75:
            return 'medium'
        else:
            return 'low'
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        stats = self.stats.copy()
        stats['runtime_seconds'] = runtime
        stats['packets_per_second'] = self.stats['total_packets'] / runtime if runtime > 0 else 0
        stats['attack_rate'] = self.stats['attacks_detected'] / self.stats['total_packets'] if self.stats['total_packets'] > 0 else 0
        
        return stats
    
    def get_recent_detections(self, n: int = 10) -> list:
        """
        Get recent detections
        
        Args:
            n: Number of recent detections to return
            
        Returns:
            List of recent detection info
        """
        return list(self.detection_history)[-n:]
    
    def reset_statistics(self) -> None:
        """Reset detection statistics"""
        self.stats = {
            'total_packets': 0,
            'attacks_detected': 0,
            'false_positives': 0,
            'start_time': datetime.now()
        }
        logger.info("Statistics reset")


def create_sample_traffic() -> Dict:
    """
    Create sample network traffic for testing
    
    Returns:
        Dictionary with packet features
    """
    return {
        'flow_duration': np.random.exponential(100000),
        'total_fwd_packets': np.random.poisson(50),
        'total_bwd_packets': np.random.poisson(30),
        'total_length_fwd_packets': np.random.exponential(5000),
        'total_length_bwd_packets': np.random.exponential(3000),
        'fwd_packet_length_max': np.random.exponential(1500),
        'fwd_packet_length_min': np.random.exponential(100),
        'fwd_packet_length_mean': np.random.exponential(500),
        'fwd_packet_length_std': np.random.exponential(200),
        'flow_bytes_per_sec': np.random.exponential(10000),
        'flow_packets_per_sec': np.random.exponential(100),
        'fin_flag_count': np.random.poisson(2),
        'syn_flag_count': np.random.poisson(2),
        'rst_flag_count': np.random.poisson(1),
        'psh_flag_count': np.random.poisson(5),
        'ack_flag_count': np.random.poisson(20),
        'source_ip': f"192.168.1.{np.random.randint(1, 255)}",
        'destination_ip': f"10.0.0.{np.random.randint(1, 255)}"
    }


def main():
    """Test real-time detection"""
    logger.info("Starting real-time detection test")
    
    # Initialize detector
    detector = RealTimeDetector(model_path="data/models")
    detector.load_models()
    
    # Simulate real-time traffic
    logger.info("Simulating network traffic...")
    
    for i in range(100):
        packet = create_sample_traffic()
        attack_type, confidence, alert_info = detector.process_packet(packet)
        
        if alert_info.get('is_attack'):
            logger.info(
                f"⚠️  Attack detected: {alert_info['attack_type']} "
                f"(confidence: {confidence:.2%})"
            )
        
        time.sleep(0.1)  # Simulate packet arrival rate
    
    # Print statistics
    stats = detector.get_statistics()
    logger.info("\n" + "=" * 60)
    logger.info("Detection Statistics:")
    logger.info(f"Total packets processed: {stats['total_packets']}")
    logger.info(f"Attacks detected: {stats['attacks_detected']}")
    logger.info(f"Attack rate: {stats['attack_rate']:.2%}")
    logger.info(f"Processing rate: {stats['packets_per_second']:.2f} packets/sec")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
