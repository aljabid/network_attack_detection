"""
Logging Configuration and Utilities
Provides centralized logging for the entire system
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """Setup and manage application logging"""
    
    def __init__(self, name: str = "NetworkAttackDetection", 
                 log_dir: str = "logs",
                 log_file: str = "system.log",
                 level: str = "INFO",
                 console_output: bool = True):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_file: Log filename
            level: Logging level
            console_output: Whether to output to console
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.level = getattr(logging, level.upper())
        self.console_output = console_output
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation
        log_path = self.log_dir / self.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Return configured logger"""
        return self.logger


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (if None, uses module name)
        level: Logging level
        
    Returns:
        Logger instance
    """
    if name is None:
        # Use caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals['__name__']
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Setup basic configuration if not already configured
        logger.setLevel(getattr(logging, level.upper()))
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class AlertLogger:
    """Special logger for security alerts"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize alert logger
        
        Args:
            log_dir: Directory for alert logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_alert_logger()
    
    def _setup_alert_logger(self) -> logging.Logger:
        """Configure alert logger"""
        logger = logging.getLogger("SecurityAlerts")
        logger.setLevel(logging.WARNING)
        logger.handlers.clear()
        
        # Alert file handler
        alert_path = self.log_dir / "alerts.log"
        alert_handler = logging.handlers.RotatingFileHandler(
            alert_path,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        
        alert_formatter = logging.Formatter(
            '%(asctime)s - SECURITY ALERT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        alert_handler.setFormatter(alert_formatter)
        logger.addHandler(alert_handler)
        
        # Also output critical alerts to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_formatter = logging.Formatter(
            '🚨 CRITICAL ALERT: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_attack(self, attack_type: str, confidence: float, 
                   source_ip: str, details: dict):
        """
        Log detected attack
        
        Args:
            attack_type: Type of attack detected
            confidence: Confidence score
            source_ip: Source IP address
            details: Additional attack details
        """
        message = (
            f"Attack Detected: {attack_type} | "
            f"Confidence: {confidence:.2%} | "
            f"Source: {source_ip} | "
            f"Details: {details}"
        )
        
        if confidence >= 0.95:
            self.logger.critical(message)
        elif confidence >= 0.85:
            self.logger.error(message)
        else:
            self.logger.warning(message)
    
    def get_logger(self) -> logging.Logger:
        """Return alert logger"""
        return self.logger


# Performance logger for tracking system metrics
class PerformanceLogger:
    """Logger for tracking model and system performance"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize performance logger
        
        Args:
            log_dir: Directory for performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_performance_logger()
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Configure performance logger"""
        logger = logging.getLogger("Performance")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Performance log file
        perf_path = self.log_dir / "performance.log"
        perf_handler = logging.FileHandler(perf_path)
        
        perf_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)
        
        return logger
    
    def log_metric(self, metric_name: str, value: float, 
                   model_name: Optional[str] = None):
        """
        Log performance metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            model_name: Model name (optional)
        """
        if model_name:
            message = f"Model: {model_name} | Metric: {metric_name} | Value: {value:.4f}"
        else:
            message = f"Metric: {metric_name} | Value: {value:.4f}"
        
        self.logger.info(message)
    
    def get_logger(self) -> logging.Logger:
        """Return performance logger"""
        return self.logger
