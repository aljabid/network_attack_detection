"""
Adversarial Robustness Testing
Tests model robustness against adversarial attacks on network traffic features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdversarialTester:
    """Test model robustness against adversarial perturbations"""
    
    def __init__(self, output_dir: str = "research_results/adversarial"):
        """
        Initialize adversarial tester
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
        logger.info("Adversarial Tester initialized")
    
    def test_robustness(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str = "Model") -> Dict:
        """
        Comprehensive adversarial robustness testing
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Model name
            
        Returns:
            Adversarial test results
        """
        logger.info(f"Testing adversarial robustness for {model_name}")
        
        results = {}
        
        # 1. Random noise perturbations
        noise_results = self.random_noise_test(model, X_test, y_test)
        results['random_noise'] = noise_results
        
        # 2. Feature perturbations
        feature_results = self.feature_perturbation_test(model, X_test, y_test)
        results['feature_perturbation'] = feature_results
        
        # 3. Gradient-based attacks (if model supports)
        if hasattr(model, 'predict_proba'):
            fgsm_results = self.fgsm_attack(model, X_test, y_test)
            results['fgsm'] = fgsm_results
        
        # 4. Evasion attacks
        evasion_results = self.evasion_attack_test(model, X_test, y_test)
        results['evasion'] = evasion_results
        
        # Save results
        self._save_results(results, model_name)
        
        return results
    
    def random_noise_test(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                         noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]) -> Dict:
        """
        Test with random Gaussian noise
        
        Args:
            model: Model to test
            X_test: Clean test data
            y_test: True labels
            noise_levels: List of noise standard deviations
            
        Returns:
            Results for each noise level
        """
        logger.info("Testing random noise robustness")
        
        from sklearn.metrics import accuracy_score
        
        # Baseline accuracy
        y_pred_clean = model.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_clean)
        
        results = {
            'baseline_accuracy': float(baseline_acc),
            'noise_levels': {}
        }
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            X_noisy = X_test.copy()
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_noisy + noise
            
            # Predict
            y_pred_noisy = model.predict(X_noisy)
            noisy_acc = accuracy_score(y_test, y_pred_noisy)
            
            # Calculate accuracy drop
            acc_drop = baseline_acc - noisy_acc
            
            results['noise_levels'][str(noise_level)] = {
                'accuracy': float(noisy_acc),
                'accuracy_drop': float(acc_drop),
                'relative_drop_pct': float((acc_drop / baseline_acc) * 100)
            }
            
            logger.info(f"Noise {noise_level}: Accuracy {noisy_acc:.4f} "
                       f"(drop: {acc_drop:.4f})")
        
        return results
    
    def feature_perturbation_test(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                  perturbation_pct: float = 0.1) -> Dict:
        """
        Test by perturbing individual features
        
        Args:
            model: Model to test
            X_test: Test data
            y_test: True labels
            perturbation_pct: Percentage to perturb each feature
            
        Returns:
            Feature sensitivity results
        """
        logger.info("Testing feature perturbation sensitivity")
        
        from sklearn.metrics import accuracy_score
        
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        
        feature_sensitivity = {}
        
        for feature in X_test.columns:
            X_perturbed = X_test.copy()
            
            # Perturb this feature
            feature_std = X_test[feature].std()
            perturbation = np.random.normal(0, feature_std * perturbation_pct, len(X_test))
            X_perturbed[feature] = X_perturbed[feature] + perturbation
            
            # Measure accuracy drop
            perturbed_acc = accuracy_score(y_test, model.predict(X_perturbed))
            acc_drop = baseline_acc - perturbed_acc
            
            feature_sensitivity[feature] = {
                'accuracy': float(perturbed_acc),
                'accuracy_drop': float(acc_drop),
                'sensitivity_score': float(abs(acc_drop))
            }
        
        # Rank features by sensitivity
        ranked = sorted(feature_sensitivity.items(), 
                       key=lambda x: x[1]['sensitivity_score'], 
                       reverse=True)
        
        results = {
            'baseline_accuracy': float(baseline_acc),
            'perturbation_percentage': perturbation_pct,
            'feature_sensitivity': feature_sensitivity,
            'most_sensitive_features': [f[0] for f in ranked[:10]],
            'least_sensitive_features': [f[0] for f in ranked[-10:]]
        }
        
        logger.info(f"Most sensitive features: {results['most_sensitive_features'][:5]}")
        
        return results
    
    def fgsm_attack(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                   epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            model: Model to attack
            X_test: Test data
            y_test: True labels
            epsilon_values: Attack strengths
            
        Returns:
            FGSM attack results
        """
        logger.info("Testing FGSM adversarial attacks")
        
        from sklearn.metrics import accuracy_score
        
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        
        results = {
            'baseline_accuracy': float(baseline_acc),
            'epsilon_values': {}
        }
        
        for epsilon in epsilon_values:
            # Simplified FGSM-like perturbation
            # Calculate gradients approximately using feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.ones(X_test.shape[1])
            
            # Create adversarial examples
            X_adv = X_test.copy()
            perturbation = epsilon * np.sign(np.random.randn(*X_test.shape)) * importance
            X_adv = X_adv + perturbation
            
            # Evaluate
            adv_acc = accuracy_score(y_test, model.predict(X_adv))
            acc_drop = baseline_acc - adv_acc
            
            results['epsilon_values'][str(epsilon)] = {
                'accuracy': float(adv_acc),
                'accuracy_drop': float(acc_drop),
                'attack_success_rate': float((acc_drop / baseline_acc) * 100)
            }
            
            logger.info(f"FGSM epsilon={epsilon}: Accuracy {adv_acc:.4f}")
        
        return results
    
    def evasion_attack_test(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                           n_samples: int = 100) -> Dict:
        """
        Test evasion attacks - mimicking normal traffic
        
        Args:
            model: Model to test
            X_test: Test data
            y_test: True labels
            n_samples: Number of samples to test
            
        Returns:
            Evasion attack results
        """
        logger.info("Testing evasion attacks")
        
        from sklearn.metrics import accuracy_score
        
        # Select attack samples
        attack_mask = y_test != 0
        attack_samples = X_test[attack_mask].head(n_samples)
        attack_labels = y_test[attack_mask].head(n_samples)
        
        # Select normal samples to mimic
        normal_samples = X_test[y_test == 0].head(n_samples)
        
        # Create evasion: blend attack with normal traffic
        blend_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {
            'n_samples': n_samples,
            'blend_ratios': {}
        }
        
        for ratio in blend_ratios:
            # Blend attack samples with normal samples
            X_evaded = attack_samples * (1 - ratio) + normal_samples.values * ratio
            
            # Predict
            y_pred = model.predict(X_evaded)
            
            # Calculate evasion success (predicted as normal when actually attack)
            evasion_success = np.sum(y_pred == 0) / len(y_pred)
            detection_rate = 1 - evasion_success
            
            results['blend_ratios'][str(ratio)] = {
                'evasion_success_rate': float(evasion_success),
                'detection_rate': float(detection_rate)
            }
            
            logger.info(f"Blend ratio {ratio}: Detection rate {detection_rate:.4f}")
        
        return results
    
    def _save_results(self, results: Dict, model_name: str):
        """Save adversarial test results"""
        output_file = self.output_dir / f"{model_name}_adversarial.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Adversarial results saved to {output_file}")
