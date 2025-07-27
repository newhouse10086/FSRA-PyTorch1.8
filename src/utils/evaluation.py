"""
Evaluation metrics and utilities for model performance assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


class MetricsCalculator:
    """Calculate various classification metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC calculation)
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # AUC metrics (if probabilities are provided)
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc_roc_macro'] = roc_auc_score(y_true, y_prob, 
                                                           multi_class='ovr', average='macro')
                    metrics['auc_roc_weighted'] = roc_auc_score(y_true, y_prob, 
                                                              multi_class='ovr', average='weighted')
            except ValueError as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Calculate per-class metrics."""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        df = pd.DataFrame({
            'class_name': self.class_names[:len(precision)],
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        return df


class EvaluationTracker:
    """Track and save evaluation metrics during training."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics_history = []
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        
        # Initialize CSV file
        self.csv_path = os.path.join(save_dir, f'{experiment_name}_metrics.csv')
        
    def add_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], learning_rate: float = None):
        """Add metrics for an epoch."""
        epoch_data = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            epoch_data[f'train_{key}'] = value
            
        # Add validation metrics with prefix
        for key, value in val_metrics.items():
            epoch_data[f'val_{key}'] = value
            
        self.metrics_history.append(epoch_data)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.csv_path, index=False)
        
    def plot_metrics(self, metrics_to_plot: List[str] = None):
        """Plot training and validation metrics."""
        if not self.metrics_history:
            return
            
        df = pd.DataFrame(self.metrics_history)
        
        if metrics_to_plot is None:
            # Default metrics to plot
            metrics_to_plot = ['loss', 'accuracy', 'auc_roc', 'f1_macro']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics_to_plot:
            if f'train_{metric}' in df.columns and f'val_{metric}' in df.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
            
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):  # Plot up to 4 metrics
            ax = axes[i]
            
            # Plot training and validation curves
            ax.plot(df['epoch'], df[f'train_{metric}'], 'b-', label=f'Train {metric.title()}', linewidth=2)
            ax.plot(df['epoch'], df[f'val_{metric}'], 'r-', label=f'Validation {metric.title()}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(len(available_metrics), 4):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', f'{self.experiment_name}_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], epoch: int):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names[:cm.shape[1]], 
                   yticklabels=class_names[:cm.shape[0]])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 
                                f'{self.experiment_name}_confusion_matrix_epoch_{epoch}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       class_names: List[str], epoch: int):
        """Plot ROC curves for multi-class classification."""
        n_classes = y_prob.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        # Binarize the output for multi-class ROC
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Plot ROC curve for each class
        for i in range(min(n_classes, 10)):  # Plot up to 10 classes
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_names[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Epoch {epoch}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 
                                f'{self.experiment_name}_roc_curves_epoch_{epoch}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 class_names: List[str], epoch: int):
        """Save detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names[:max(y_true)+1], 
                                     output_dict=True)
        
        # Convert to DataFrame for better formatting
        df = pd.DataFrame(report).transpose()
        
        # Save to CSV
        report_path = os.path.join(self.save_dir, 
                                  f'{self.experiment_name}_classification_report_epoch_{epoch}.csv')
        df.to_csv(report_path)
        
        return df


def evaluate_model(model, dataloader, device, num_classes: int, 
                  class_names: List[str] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: List of class names
        
    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_prob)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                # Handle two-view data
                (inputs_s, labels_s), (inputs_d, labels_d) = batch_data
                inputs = inputs_s.to(device)
                labels = labels_s.to(device)
            else:
                # Handle single-view data
                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            # Forward pass
            if hasattr(model, 'model_1'):
                # Two-view model
                outputs, _ = model(inputs, None)
                if isinstance(outputs, list):
                    outputs = outputs[0]  # Use first output (global classifier)
            else:
                # Single-view model
                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[0]  # Use first output
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes, class_names)
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
    
    return metrics, y_true, y_pred, y_prob
