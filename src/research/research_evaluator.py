"""
Advanced Evaluation Framework for Research
Comprehensive evaluation including statistical tests, cross-dataset validation,
and adversarial robustness testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class ResearchEvaluator:
    """Advanced evaluation framework for research paper"""
    
    def __init__(self, output_dir: str = "research_results"):
        """
        Initialize research evaluator
        
        Args:
            output_dir: Directory for research outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'cross_validation': {},
            'statistical_tests': {},
            'cross_dataset': {},
            'adversarial': {},
            'timing': {}
        }
        
        logger.info("Research Evaluator initialized")
    
    def comprehensive_evaluation(self, model, X_train, y_train, X_test, y_test,
                                model_name: str = "Model") -> Dict:
        """
        Perform comprehensive evaluation for research paper
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Starting comprehensive evaluation for {model_name}")
        
        results = {}
        
        # 1. Cross-validation with confidence intervals
        cv_results = self.cross_validation_evaluation(model, X_train, y_train, model_name)
        results['cross_validation'] = cv_results
        
        # 2. Test set evaluation
        test_results = self.test_set_evaluation(model, X_test, y_test, model_name)
        results['test_set'] = test_results
        
        # 3. Statistical significance testing
        stat_results = self.statistical_significance_tests(model, X_train, y_train, model_name)
        results['statistical'] = stat_results
        
        # 4. Timing analysis
        timing_results = self.timing_analysis(model, X_test, model_name)
        results['timing'] = timing_results
        
        # 5. Per-class performance
        class_results = self.per_class_analysis(model, X_test, y_test, model_name)
        results['per_class'] = class_results
        
        # Save results
        self._save_results(results, model_name)
        
        return results
    
    def cross_validation_evaluation(self, model, X, y, model_name: str,
                                   n_folds: int = 10) -> Dict:
        """
        Perform k-fold cross-validation with confidence intervals
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            model_name: Model name
            n_folds: Number of CV folds
            
        Returns:
            CV results with confidence intervals
        """
        logger.info(f"Performing {n_folds}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted',
            'roc_auc_ovr_weighted': 'roc_auc_ovr_weighted'
        }
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Calculate mean and confidence intervals
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # 95% confidence interval
            test_ci = stats.t.interval(
                0.95,
                len(test_scores) - 1,
                loc=np.mean(test_scores),
                scale=stats.sem(test_scores)
            )
            
            results[metric] = {
                'test_mean': float(np.mean(test_scores)),
                'test_std': float(np.std(test_scores)),
                'test_ci_lower': float(test_ci[0]),
                'test_ci_upper': float(test_ci[1]),
                'train_mean': float(np.mean(train_scores)),
                'train_std': float(np.std(train_scores)),
                'all_folds': test_scores.tolist()
            }
        
        logger.info(f"Cross-validation complete. Accuracy: {results['accuracy']['test_mean']:.4f} "
                   f"± {results['accuracy']['test_std']:.4f}")
        
        return results
    
    def test_set_evaluation(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Detailed test set evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Model name
            
        Returns:
            Test set metrics
        """
        logger.info("Performing test set evaluation")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Additional metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # ROC AUC
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, 
                                       multi_class='ovr', average='weighted')
            except:
                roc_auc = None
        else:
            roc_auc = None
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'matthews_corr_coef': float(mcc),
            'cohen_kappa': float(kappa),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'n_samples': len(y_test)
        }
        
        logger.info(f"Test set accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
        
        return results
    
    def statistical_significance_tests(self, model, X, y, model_name: str,
                                       n_iterations: int = 30) -> Dict:
        """
        Statistical significance testing using repeated holdout
        
        Args:
            model: Model to test
            X: Features
            y: Labels
            model_name: Model name
            n_iterations: Number of random splits
            
        Returns:
            Statistical test results
        """
        logger.info(f"Running statistical significance tests ({n_iterations} iterations)")
        
        scores = []
        
        for i in range(n_iterations):
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i, stratify=y
            )
            
            # Clone and train model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_clone.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Statistical tests
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(scores)
        
        # T-test against baseline (e.g., 0.9)
        baseline = 0.90
        t_stat, t_p = stats.ttest_1samp(scores, baseline)
        
        results = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'min_accuracy': float(np.min(scores)),
            'max_accuracy': float(np.max(scores)),
            'median_accuracy': float(np.median(scores)),
            'ci_95_lower': float(np.percentile(scores, 2.5)),
            'ci_95_upper': float(np.percentile(scores, 97.5)),
            'normality_test': {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            },
            't_test_vs_baseline': {
                'baseline': baseline,
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significantly_better': (t_p < 0.05) and (np.mean(scores) > baseline)
            },
            'all_scores': scores.tolist()
        }
        
        logger.info(f"Statistical tests complete. Mean: {np.mean(scores):.4f} "
                   f"[{results['ci_95_lower']:.4f}, {results['ci_95_upper']:.4f}]")
        
        return results
    
    def timing_analysis(self, model, X_test, model_name: str,
                       n_runs: int = 100) -> Dict:
        """
        Analyze inference timing
        
        Args:
            model: Trained model
            X_test: Test features
            model_name: Model name
            n_runs: Number of timing runs
            
        Returns:
            Timing statistics
        """
        import time
        
        logger.info(f"Analyzing inference timing ({n_runs} runs)")
        
        # Warm up
        for _ in range(10):
            _ = model.predict(X_test[:100])
        
        # Batch timing
        batch_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X_test)
            end = time.perf_counter()
            batch_times.append(end - start)
        
        # Single sample timing
        single_times = []
        for _ in range(n_runs):
            sample = X_test.iloc[[0]]
            start = time.perf_counter()
            _ = model.predict(sample)
            end = time.perf_counter()
            single_times.append(end - start)
        
        batch_times = np.array(batch_times)
        single_times = np.array(single_times)
        
        results = {
            'batch_inference': {
                'mean_ms': float(np.mean(batch_times) * 1000),
                'std_ms': float(np.std(batch_times) * 1000),
                'median_ms': float(np.median(batch_times) * 1000),
                'samples_per_sec': float(len(X_test) / np.mean(batch_times))
            },
            'single_sample': {
                'mean_ms': float(np.mean(single_times) * 1000),
                'std_ms': float(np.std(single_times) * 1000),
                'median_ms': float(np.median(single_times) * 1000)
            },
            'n_samples': len(X_test),
            'n_runs': n_runs
        }
        
        logger.info(f"Timing: {results['single_sample']['mean_ms']:.2f}ms per sample, "
                   f"{results['batch_inference']['samples_per_sec']:.0f} samples/sec")
        
        return results
    
    def per_class_analysis(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Per-class performance analysis
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Model name
            
        Returns:
            Per-class metrics
        """
        logger.info("Analyzing per-class performance")
        
        y_pred = model.predict(X_test)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        classes = np.unique(y_test)
        
        results = {}
        for i, cls in enumerate(classes):
            results[f'class_{cls}'] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return results
    
    def _save_results(self, results: Dict, model_name: str):
        """Save results to JSON"""
        output_file = self.output_dir / f"{model_name}_evaluation.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_latex_table(self, all_results: Dict[str, Dict]) -> str:
        """
        Generate LaTeX table for paper
        
        Args:
            all_results: Results for all models
            
        Returns:
            LaTeX table string
        """
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\toprule\n"
        latex += "Model & Accuracy & Precision & Recall & F1 Score \\\\\n"
        latex += "\\midrule\n"
        
        for model_name, results in all_results.items():
            test_results = results.get('test_set', {})
            acc = test_results.get('accuracy', 0)
            prec = test_results.get('precision', 0)
            rec = test_results.get('recall', 0)
            f1 = test_results.get('f1_score', 0)
            
            latex += f"{model_name} & {acc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
