"""
Visualization Module
Creates plots and dashboards for model performance and attack analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Create visualizations for attack detection system"""
    
    def __init__(self, output_dir: str = "dashboards"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Visualizer initialized")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             model_name: str = "Model") -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
        """
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def plot_roc_curves(self, models_data: Dict[str, Dict], n_classes: int) -> None:
        """
        Plot ROC curves for multiple models
        
        Args:
            models_data: Dictionary with model names and their data
            n_classes: Number of classes
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (model_name, data) in enumerate(models_data.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Plot ROC curve for each class
            for i in range(min(n_classes, 5)):  # Plot max 5 classes
                if 'y_true' in data and 'y_pred_proba' in data:
                    fpr, tpr, _ = roc_curve(
                        (data['y_true'] == i).astype(int),
                        data['y_pred_proba'][:, i] if len(data['y_pred_proba'].shape) > 1 else data['y_pred_proba']
                    )
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves - {model_name}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "roc_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curves to {save_path}")
    
    def plot_model_comparison(self, results: Dict[str, Dict]) -> None:
        """
        Plot comparison of model metrics
        
        Args:
            results: Dictionary with model results
        """
        metrics_data = []
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                metrics_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1_score']
                })
        
        if not metrics_data:
            logger.warning("No metrics data to plot")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.2
        
        ax.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, df['F1 Score'], width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison to {save_path}")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20,
                               model_name: str = "Model") -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            model_name: Name of the model
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance to {save_path}")
    
    def plot_attack_distribution(self, attack_counts: Dict[str, int]) -> None:
        """
        Plot distribution of attack types
        
        Args:
            attack_counts: Dictionary with attack type counts
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        attacks = list(attack_counts.keys())
        counts = list(attack_counts.values())
        
        # Bar plot
        ax1.bar(range(len(attacks)), counts, alpha=0.8)
        ax1.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Attack Type Distribution (Bar)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(attacks)))
        ax1.set_xticklabels(attacks, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(counts, labels=attacks, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Attack Type Distribution (Pie)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / "attack_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attack distribution to {save_path}")
    
    def plot_training_history(self, history: Dict[str, List], model_name: str = "Model") -> None:
        """
        Plot training history for neural networks
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
        """
        if not history:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if 'loss' in history:
            ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'Model Loss - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'accuracy' in history:
            ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Model Accuracy - {model_name}', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{model_name}_training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history to {save_path}")
    
    def create_summary_report(self, results: Dict, output_path: Optional[str] = None) -> None:
        """
        Create HTML summary report
        
        Args:
            results: Training results
            output_path: Path to save report
        """
        if output_path is None:
            output_path = self.output_dir / "summary_report.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Network Attack Detection - Training Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #27ae60; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛡️ AI-Based Network Attack Detection System</h1>
                <h2>Training Summary Report</h2>
        """
        
        # Add results table
        html_content += """
                <h2>Model Performance</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
        """
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                html_content += f"""
                    <tr>
                        <td>{model_name.replace('_', ' ').title()}</td>
                        <td class="metric">{metrics['accuracy']:.4f}</td>
                        <td class="metric">{metrics['precision']:.4f}</td>
                        <td class="metric">{metrics['recall']:.4f}</td>
                        <td class="metric">{metrics['f1_score']:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Saved summary report to {output_path}")
