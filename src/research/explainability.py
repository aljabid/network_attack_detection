"""
Model Explainability Module
Provides interpretability using SHAP values and feature importance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityAnalyzer:
    """Analyze and explain model predictions"""
    
    def __init__(self, output_dir: str = "research_results/explainability"):
        """
        Initialize explainability analyzer
        
        Args:
            output_dir: Output directory for explanations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.explanations = {}
        
        logger.info("Explainability Analyzer initialized")
    
    def analyze_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     model_name: str = "Model", n_samples: int = 100) -> Dict:
        """
        Comprehensive explainability analysis
        
        Args:
            model: Trained model
            X_train: Training data (for SHAP background)
            X_test: Test data to explain
            model_name: Model name
            n_samples: Number of samples for SHAP
            
        Returns:
            Explanation results
        """
        logger.info(f"Analyzing explainability for {model_name}")
        
        results = {}
        
        # 1. Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance_results = self.analyze_feature_importance(
                model, X_train.columns, model_name
            )
            results['feature_importance'] = importance_results
        
        # 2. SHAP analysis (simplified version)
        shap_results = self.shap_analysis_simplified(
            model, X_train, X_test, n_samples, model_name
        )
        results['shap'] = shap_results
        
        # 3. Decision path analysis
        if hasattr(model, 'estimators_'):  # For tree-based models
            path_results = self.analyze_decision_paths(model, X_test, n_samples)
            results['decision_paths'] = path_results
        
        # Save results
        self._save_results(results, model_name)
        
        return results
    
    def analyze_feature_importance(self, model, feature_names: List[str],
                                   model_name: str) -> Dict:
        """
        Analyze and visualize feature importance
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            model_name: Model name
            
        Returns:
            Feature importance analysis
        """
        logger.info("Analyzing feature importance")
        
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        # Find features for 90% cumulative importance
        features_for_90 = len(importance_df[importance_df['cumulative_importance'] <= 0.9])
        
        results = {
            'top_10_features': importance_df.head(10).to_dict('records'),
            'features_for_90_pct': features_for_90,
            'total_features': len(feature_names),
            'importance_concentration': {
                'top_5_pct': float(importance_df.head(5)['importance'].sum()),
                'top_10_pct': float(importance_df.head(10)['importance'].sum()),
                'top_20_pct': float(importance_df.head(20)['importance'].sum())
            }
        }
        
        # Plot
        self._plot_feature_importance(importance_df, model_name)
        
        logger.info(f"Top 5 features account for {results['importance_concentration']['top_5_pct']:.2%} "
                   f"of total importance")
        
        return results
    
    def shap_analysis_simplified(self, model, X_train: pd.DataFrame, 
                                 X_test: pd.DataFrame, n_samples: int,
                                 model_name: str) -> Dict:
        """
        Simplified SHAP-like analysis without requiring shap library
        
        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            n_samples: Number of samples to analyze
            model_name: Model name
            
        Returns:
            SHAP-like explanation results
        """
        logger.info(f"Performing SHAP-like analysis on {n_samples} samples")
        
        # Sample data
        X_sample = X_test.head(n_samples)
        
        # Calculate baseline prediction (mean of training predictions)
        baseline_pred = model.predict(X_train).mean()
        
        # For each sample, calculate feature contributions
        feature_contributions = []
        
        for idx in range(min(10, len(X_sample))):  # Analyze first 10 samples
            sample = X_sample.iloc[[idx]]
            prediction = model.predict(sample)[0]
            
            # Calculate approximate contributions by perturbation
            contributions = {}
            for feature in X_sample.columns:
                # Zero out this feature
                sample_perturbed = sample.copy()
                sample_perturbed[feature] = X_train[feature].mean()
                
                # Measure prediction change
                pred_perturbed = model.predict(sample_perturbed)[0]
                contribution = prediction - pred_perturbed
                
                contributions[feature] = float(contribution)
            
            feature_contributions.append({
                'sample_idx': idx,
                'prediction': int(prediction),
                'contributions': contributions
            })
        
        # Aggregate contributions across samples
        avg_abs_contributions = {}
        for feature in X_sample.columns:
            contribs = [abs(fc['contributions'][feature]) for fc in feature_contributions]
            avg_abs_contributions[feature] = float(np.mean(contribs))
        
        # Rank features by average absolute contribution
        ranked_features = sorted(avg_abs_contributions.items(), 
                               key=lambda x: x[1], reverse=True)
        
        results = {
            'n_samples_analyzed': min(10, len(X_sample)),
            'baseline_prediction': float(baseline_pred),
            'top_contributing_features': [
                {'feature': f, 'avg_abs_contribution': c} 
                for f, c in ranked_features[:15]
            ],
            'sample_explanations': feature_contributions[:3]  # Save first 3
        }
        
        logger.info(f"Top contributing features: {[f for f, _ in ranked_features[:5]]}")
        
        return results
    
    def analyze_decision_paths(self, model, X_test: pd.DataFrame, 
                               n_samples: int) -> Dict:
        """
        Analyze decision paths for tree-based models
        
        Args:
            model: Tree-based model
            X_test: Test data
            n_samples: Number of samples
            
        Returns:
            Decision path statistics
        """
        logger.info("Analyzing decision paths")
        
        X_sample = X_test.head(n_samples)
        
        # For Random Forest, analyze tree depth and leaf statistics
        if hasattr(model, 'estimators_'):
            tree_depths = []
            leaf_counts = []
            
            for estimator in model.estimators_[:10]:  # First 10 trees
                tree = estimator.tree_
                tree_depths.append(tree.max_depth)
                leaf_counts.append(tree.n_leaves)
            
            results = {
                'avg_tree_depth': float(np.mean(tree_depths)),
                'std_tree_depth': float(np.std(tree_depths)),
                'avg_leaf_count': float(np.mean(leaf_counts)),
                'std_leaf_count': float(np.std(leaf_counts)),
                'n_trees_analyzed': len(tree_depths)
            }
            
            logger.info(f"Average tree depth: {results['avg_tree_depth']:.1f}")
            
            return results
        
        return {}
    
    def explain_prediction(self, model, sample: pd.DataFrame, 
                          feature_names: List[str]) -> Dict:
        """
        Explain a single prediction
        
        Args:
            model: Trained model
            sample: Single sample to explain
            feature_names: Feature names
            
        Returns:
            Prediction explanation
        """
        prediction = model.predict(sample)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(sample)[0]
        else:
            probabilities = None
        
        # Get feature values
        feature_values = sample.iloc[0].to_dict()
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
            
            # Rank features by (importance * value) for this sample
            feature_scores = {
                f: abs(importance[f] * feature_values[f]) 
                for f in feature_names
            }
            top_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        explanation = {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'top_features': [
                {'feature': f, 'value': feature_values[f], 'score': s}
                for f, s in top_features
            ]
        }
        
        return explanation
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        
        top_20 = importance_df.head(20)
        
        plt.barh(range(len(top_20)), top_20['importance'])
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Importance Score', fontweight='bold')
        plt.ylabel('Features', fontweight='bold')
        plt.title(f'Top 20 Feature Importance - {model_name}', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_file = self.output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_file}")
    
    def _save_results(self, results: Dict, model_name: str):
        """Save explainability results"""
        output_file = self.output_dir / f"{model_name}_explainability.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Explainability results saved to {output_file}")
