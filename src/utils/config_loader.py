"""
Configuration Loader and Manager
Handles loading and accessing configuration settings
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file. If None, uses default path.
        """
        if config_path is None:
            # Get project root directory
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.train_test_split')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save config. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


# Global configuration instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        Config instance
    """
    global _config
    
    if _config is None:
        _config = Config(config_path)
    
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        New Config instance
    """
    global _config
    _config = Config(config_path)
    return _config
