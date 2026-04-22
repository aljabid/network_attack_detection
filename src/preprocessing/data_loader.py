"""
Data Loader Module
Handles loading and initial processing of network traffic datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import glob
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class DataLoader:
    """Load and manage network traffic datasets"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.data_config = self.config['data']
        self.attack_mapping = self.data_config['attack_types']
        
        logger.info("DataLoader initialized")
    
    def load_cicids2017(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset
        
        Args:
            data_path: Path to CICIDS2017 data directory
            
        Returns:
            Combined DataFrame
        """
        if data_path is None:
            data_path = self.data_config['datasets'][0]['path']
        
        data_path = Path(data_path)
        logger.info(f"Loading CICIDS2017 dataset from {data_path}")
        
        # CICIDS2017 typically comes as multiple CSV files
        csv_files = glob.glob(str(data_path / "*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {data_path}")
            return self._create_sample_cicids2017()
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding='utf-8', low_memory=False)
                dataframes.append(df)
                logger.info(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Total CICIDS2017 records: {len(combined_df)}")
            return self._preprocess_cicids2017(combined_df)
        else:
            return self._create_sample_cicids2017()
    
    def load_unsw_nb15(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset
        
        Args:
            data_path: Path to UNSW-NB15 data directory
            
        Returns:
            Combined DataFrame
        """
        if data_path is None:
            for dataset in self.data_config['datasets']:
                if dataset['name'] == 'UNSW-NB15':
                    data_path = dataset['path']
                    break
        
        data_path = Path(data_path)
        logger.info(f"Loading UNSW-NB15 dataset from {data_path}")
        
        # UNSW-NB15 typically has training and testing files
        csv_files = glob.glob(str(data_path / "*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {data_path}")
            return self._create_sample_unsw_nb15()
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding='utf-8', low_memory=False)
                dataframes.append(df)
                logger.info(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Total UNSW-NB15 records: {len(combined_df)}")
            return self._preprocess_unsw_nb15(combined_df)
        else:
            return self._create_sample_unsw_nb15()
    
    def _preprocess_cicids2017(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess CICIDS2017 dataset
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Preprocessing CICIDS2017 dataset")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map attack labels
        if 'label' in df.columns:
            df['attack_type'] = df['label'].apply(self._map_cicids_attack)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Preprocessed CICIDS2017: {len(df)} records")
        return df
    
    def _preprocess_unsw_nb15(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess UNSW-NB15 dataset
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Preprocessing UNSW-NB15 dataset")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map attack labels
        if 'attack_cat' in df.columns:
            df['attack_type'] = df['attack_cat'].apply(self._map_unsw_attack)
        elif 'label' in df.columns:
            df['attack_type'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Preprocessed UNSW-NB15: {len(df)} records")
        return df
    
    def _map_cicids_attack(self, label: str) -> int:
        """
        Map CICIDS2017 attack labels to numeric codes
        
        Args:
            label: Attack label string
            
        Returns:
            Attack type code
        """
        label = str(label).lower().strip()
        
        if 'benign' in label or label == 'normal':
            return self.attack_mapping['normal']
        elif 'ddos' in label:
            return self.attack_mapping['ddos']
        elif 'dos' in label:
            return self.attack_mapping['dos']
        elif 'brute' in label or 'ftp' in label or 'ssh' in label:
            return self.attack_mapping['brute_force']
        elif 'port' in label or 'scan' in label:
            return self.attack_mapping['port_scan']
        elif 'bot' in label:
            return self.attack_mapping['botnet']
        elif 'infiltr' in label:
            return self.attack_mapping['infiltration']
        elif 'web' in label or 'sql' in label or 'xss' in label:
            return self.attack_mapping['web_attack']
        else:
            return self.attack_mapping['exploit']
    
    def _map_unsw_attack(self, label: str) -> int:
        """
        Map UNSW-NB15 attack labels to numeric codes
        
        Args:
            label: Attack label string
            
        Returns:
            Attack type code
        """
        label = str(label).lower().strip()
        
        if label == 'normal' or label == 'nan' or pd.isna(label):
            return self.attack_mapping['normal']
        elif label == 'dos':
            return self.attack_mapping['dos']
        elif label == 'exploits':
            return self.attack_mapping['exploit']
        elif label == 'backdoor':
            return self.attack_mapping['backdoor']
        elif label == 'reconnaissance' or label == 'analysis':
            return self.attack_mapping['port_scan']
        else:
            return self.attack_mapping['exploit']
    
    def _create_sample_cicids2017(self) -> pd.DataFrame:
        """
        Create sample CICIDS2017-like data for demonstration
        
        Returns:
            Sample dataframe
        """
        logger.warning("Creating sample CICIDS2017 data for demonstration")
        
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'flow_duration': np.random.exponential(100000, n_samples),
            'total_fwd_packets': np.random.poisson(50, n_samples),
            'total_bwd_packets': np.random.poisson(30, n_samples),
            'total_length_fwd_packets': np.random.exponential(5000, n_samples),
            'total_length_bwd_packets': np.random.exponential(3000, n_samples),
            'fwd_packet_length_max': np.random.exponential(1500, n_samples),
            'fwd_packet_length_min': np.random.exponential(100, n_samples),
            'fwd_packet_length_mean': np.random.exponential(500, n_samples),
            'fwd_packet_length_std': np.random.exponential(200, n_samples),
            'flow_bytes_per_sec': np.random.exponential(10000, n_samples),
            'flow_packets_per_sec': np.random.exponential(100, n_samples),
            'fin_flag_count': np.random.poisson(2, n_samples),
            'syn_flag_count': np.random.poisson(2, n_samples),
            'rst_flag_count': np.random.poisson(1, n_samples),
            'psh_flag_count': np.random.poisson(5, n_samples),
            'ack_flag_count': np.random.poisson(20, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate labels (70% normal, 30% attacks)
        attack_labels = np.random.choice(
            list(self.attack_mapping.values()),
            size=n_samples,
            p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.03, 0.02, 0.00]
        )
        df['attack_type'] = attack_labels
        
        # Add some attack-specific patterns
        attack_mask = df['attack_type'] != 0
        df.loc[attack_mask, 'flow_packets_per_sec'] *= 3
        
        logger.info(f"Created sample data: {len(df)} records")
        return df
    
    def _create_sample_unsw_nb15(self) -> pd.DataFrame:
        """Create sample UNSW-NB15-like data"""
        return self._create_sample_cicids2017()
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input dataframe
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Train and test dataframes
        """
        if test_size is None:
            test_size = 1 - self.data_config['train_test_split']
        
        if random_state is None:
            random_state = self.data_config['random_state']
        
        # Stratified split based on attack type
        if 'attack_type' in df.columns:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df['attack_type']
            )
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
        
        logger.info(f"Data split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    def get_attack_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get distribution of attack types
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of attack counts
        """
        if 'attack_type' not in df.columns:
            return {}
        
        distribution = df['attack_type'].value_counts().to_dict()
        
        # Map numeric codes to names
        reverse_mapping = {v: k for k, v in self.attack_mapping.items()}
        named_distribution = {
            reverse_mapping.get(k, f"unknown_{k}"): v 
            for k, v in distribution.items()
        }
        
        return named_distribution
