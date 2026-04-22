"""
Comprehensive Research Training Script
Generates all results, tables, and figures for the SynSec 2026 paper
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import json
from typing import Dict

sys.path.append(str(Path(__file__).parent))
from preprocessing.data_loader import DataLoader
from preprocessing.preprocessor import DataPreprocessor
from feature_engineering.feature_engineer import FeatureEngineer
from models.ml_models import RandomForestModel, XGBoostModel, LightGBMModel
from research.research_evaluator import ResearchEvaluator
from research.adversarial_tester import AdversarialTester
from research.explainability import ExplainabilityAnalyzer
from utils.logger import get_logger, LoggerSetup
from utils.config_loader import get_config

logger = get_logger(__name__)


class ResearchPipeline:
    """Complete research pipeline for SynSec 2026 submission"""
    
    #def __init__(self, config_path: Optional [str] = None):
    def __init__(self, config_path: [str] = None):
        """Initialize research pipeline"""
        self.config = get_config(config_path)
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        
        # Research components
        self.evaluator = ResearchEvaluator()
        self.adversarial_tester = AdversarialTester()
        self.explainability = ExplainabilityAnalyzer()
        
        # Results storage
        self.all_results = {}

        logger.info("=" * 80)
        logger.info("SynSec 2026 Research Pipeline")
        logger.info("AI-Based Network Attack Detection with Ensemble Learning")
        logger.info("=" * 80)
    
    def run_complete_research(self, dataset: str = 'cicids2017') -> Dict:
        """
        Run complete research pipeline
        
        Args:
            dataset: Dataset to use
            
        Returns:
            Complete research results
        """
        start_time = time.time()
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA PREPARATION")
        logger.info("=" * 80)
        
        # Load and prepare data
        df = self._load_data(dataset)
        train_df, test_df = self.data_loader.split_data(df, test_size=0.2)
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Preprocess
        logger.info("\nPreprocessing data...")
        X_train, y_train = self.preprocessor.fit_transform(train_df)
        X_test, y_test = self.preprocessor.transform(test_df)
        
        # Feature engineering
        logger.info("\nEngineering features...")
        X_train = self.feature_engineer.create_advanced_features(X_train)
        X_test = self.feature_engineer.create_advanced_features(X_test)
        
        X_train = self.feature_engineer.remove_correlated_features(X_train)
        X_test = self.feature_engineer.transform(X_test)
        
        X_train_selected = self.feature_engineer.select_features(X_train, y_train, n_features=50)
        X_test_selected = self.feature_engineer.transform(X_test)
        
        logger.info(f"Final features: {X_train_selected.shape[1]}")
        
        # PHASE 2: MODEL TRAINING
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: MODEL TRAINING & EVALUATION")
        logger.info("=" * 80)
        
        models_to_test = ['random_forest', 'xgboost', 'lightgbm']
        trained_models = {}
        
        for model_type in models_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.upper()}")
            logger.info(f"{'='*60}")
            
            model = self._train_model(model_type, X_train_selected, y_train)
            if model is not None:
                trained_models[model_type] = model
                
                # Comprehensive evaluation
                logger.info(f"\nComprehensive evaluation of {model_type}...")
                eval_results = self.evaluator.comprehensive_evaluation(
                    model.model, X_train_selected, y_train, X_test_selected, y_test,
                    model_name=model_type
                )
                self.all_results[model_type] = {'evaluation': eval_results}
                
                # Adversarial testing
                logger.info(f"\nAdversarial testing of {model_type}...")
                adv_results = self.adversarial_tester.test_robustness(
                    model.model, X_test_selected, y_test,
                    model_name=model_type
                )
                self.all_results[model_type]['adversarial'] = adv_results
                
                # Explainability
                logger.info(f"\nExplainability analysis of {model_type}...")
                explain_results = self.explainability.analyze_model(
                    model.model, X_train_selected, X_test_selected,
                    model_name=model_type
                )
                self.all_results[model_type]['explainability'] = explain_results
        
        # PHASE 3: COMPARATIVE ANALYSIS
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: COMPARATIVE ANALYSIS")
        logger.info("=" * 80)
        
        self._generate_comparative_analysis()
        
        # PHASE 4: GENERATE PAPER ARTIFACTS
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: GENERATING PAPER ARTIFACTS")
        logger.info("=" * 80)
        
        self._generate_paper_artifacts()
        
        # Save all results
        self._save_all_results()
        
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("RESEARCH PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Results saved to: research_results/")
        logger.info("=" * 80)
        
        return self.all_results
    
    def _load_data(self, dataset: str) -> pd.DataFrame:
        """Load dataset"""
        if dataset.lower() == 'cicids2017':
            df = self.data_loader.load_cicids2017()
        elif dataset.lower() == 'unsw_nb15':
            df = self.data_loader.load_unsw_nb15()
        else:
            logger.warning(f"Unknown dataset: {dataset}, using CICIDS2017")
            df = self.data_loader.load_cicids2017()
        
        logger.info(f"Loaded {len(df)} records")
        
        # Get attack distribution
        distribution = self.data_loader.get_attack_distribution(df)
        logger.info("\nAttack distribution:")
        for attack, count in distribution.items():
            logger.info(f"  {attack:20s}: {count:6d} ({count/len(df)*100:.2f}%)")
        
        return df
    
    def _train_model(self, model_type: str, X_train, y_train):
        """Train a specific model"""
        try:
            if model_type == 'random_forest':
                model = RandomForestModel()
            elif model_type == 'xgboost':
                model = XGBoostModel()
            elif model_type == 'lightgbm':
                model = LightGBMModel()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            model.train(X_train, y_train)
            logger.info(f"{model_type} training complete")
            
            return model
            
        except ImportError as e:
            logger.warning(f"{model_type} not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            return None
    
    def _generate_comparative_analysis(self):
        """Generate comparative analysis across models"""
        logger.info("\nGenerating comparative analysis...")
        
        comparison = {
            'model_comparison': {},
            'best_model': None,
            'best_f1_score': 0.0
        }
        
        for model_name, results in self.all_results.items():
            eval_results = results['evaluation']
            test_results = eval_results['test_set']
            cv_results = eval_results['cross_validation']
            
            comparison['model_comparison'][model_name] = {
                'test_accuracy': test_results['accuracy'],
                'test_f1': test_results['f1_score'],
                'cv_accuracy_mean': cv_results['accuracy']['test_mean'],
                'cv_accuracy_ci': [
                    cv_results['accuracy']['test_ci_lower'],
                    cv_results['accuracy']['test_ci_upper']
                ]
            }
            
            # Track best model
            if test_results['f1_score'] > comparison['best_f1_score']:
                comparison['best_f1_score'] = test_results['f1_score']
                comparison['best_model'] = model_name
        
        logger.info(f"\nBest model: {comparison['best_model']}")
        logger.info(f"Best F1 score: {comparison['best_f1_score']:.4f}")
        
        # Save comparison
        output_file = Path("research_results/model_comparison.json")
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        self.all_results['comparison'] = comparison
    
    def _generate_paper_artifacts(self):
        """Generate LaTeX tables and figures for paper"""
        logger.info("\nGenerating LaTeX tables...")
        
        # Performance comparison table
        latex_table = self.evaluator.generate_latex_table(
            {name: res['evaluation'] for name, res in self.all_results.items() 
             if 'evaluation' in res}
        )
        
        # Save LaTeX table
        table_file = Path("research_results/performance_table.tex")
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"LaTeX table saved to: {table_file}")
        
        # Generate summary for paper
        summary = self._generate_paper_summary()
        summary_file = Path("research_results/paper_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Paper summary saved to: {summary_file}")
    
    def _generate_paper_summary(self) -> str:
        """Generate text summary for paper"""
        summary = []
        summary.append("=" * 80)
        summary.append("RESEARCH SUMMARY FOR PAPER")
        summary.append("=" * 80)
        summary.append("")
        
        # Dataset info
        summary.append("DATASET:")
        summary.append("- CICIDS2017: Realistic network traffic with labeled attacks")
        summary.append("- 10,000 samples (sample dataset for demonstration)")
        summary.append("- 9 attack categories + normal traffic")
        summary.append("")
        
        # Model results
        summary.append("MODEL PERFORMANCE:")
        for model_name, results in self.all_results.items():
            if 'evaluation' not in results:
                continue
            
            eval_res = results['evaluation']
            test_res = eval_res['test_set']
            cv_res = eval_res['cross_validation']
            
            summary.append(f"\n{model_name.upper()}:")
            summary.append(f"  Test Accuracy: {test_res['accuracy']:.4f}")
            summary.append(f"  Test F1 Score: {test_res['f1_score']:.4f}")
            summary.append(f"  CV Accuracy: {cv_res['accuracy']['test_mean']:.4f} "
                         f"± {cv_res['accuracy']['test_std']:.4f}")
            summary.append(f"  95% CI: [{cv_res['accuracy']['test_ci_lower']:.4f}, "
                         f"{cv_res['accuracy']['test_ci_upper']:.4f}]")
        
        # Best model
        if 'comparison' in self.all_results:
            comp = self.all_results['comparison']
            summary.append("")
            summary.append(f"BEST MODEL: {comp['best_model']}")
            summary.append(f"BEST F1 SCORE: {comp['best_f1_score']:.4f}")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def _save_all_results(self):
        """Save all results to JSON"""
        output_file = Path("research_results/complete_results.json")
        
        # Convert to JSON-serializable format
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=convert)
        
        logger.info(f"\nComplete results saved to: {output_file}")


def main():
    """Main entry point"""
    # Setup logging
    log_setup = LoggerSetup(
        name="ResearchPipeline",
        log_dir="logs",
        level="INFO",
        console_output=True
    )
    
    # Create and run research pipeline
    pipeline = ResearchPipeline()
    results = pipeline.run_complete_research(dataset='cicids2017')
    
    logger.info("\n✓ Research pipeline completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review results in research_results/")
    logger.info("2. Check performance_table.tex for LaTeX table")
    logger.info("3. Review paper_summary.txt for key findings")
    logger.info("4. Use complete_results.json for detailed analysis")


if __name__ == "__main__":
    main()
