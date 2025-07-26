"""Logging utilities for FSRA project."""

import os
import logging
import sys
from datetime import datetime


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class TensorBoardLogger:
    """Simple TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Logging to console only.")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image."""
        if self.writer:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


class MetricLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, name: str = 'metrics'):
        self.log_dir = log_dir
        self.name = name
        self.metrics = {}
        
        # Create log file
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{name}_{timestamp}.csv')
        
        # Write header
        self.header_written = False
    
    def log(self, epoch: int, **kwargs):
        """Log metrics for an epoch."""
        # Add epoch to metrics
        metrics = {'epoch': epoch, **kwargs}
        
        # Write header if first time
        if not self.header_written:
            with open(self.log_file, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
            self.header_written = True
        
        # Write metrics
        with open(self.log_file, 'a') as f:
            values = [str(metrics[key]) for key in metrics.keys()]
            f.write(','.join(values) + '\n')
        
        # Store in memory
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'max'):
        """Get best value for a metric."""
        if metric not in self.metrics:
            return None
        
        values = self.metrics[metric]
        if mode == 'max':
            return max(values)
        elif mode == 'min':
            return min(values)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_latest(self, metric: str):
        """Get latest value for a metric."""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        return self.metrics[metric][-1]
