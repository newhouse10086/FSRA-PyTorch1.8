#!/usr/bin/env python3
"""
Test script for evaluation functionality.
This script tests the evaluation metrics calculation and visualization without requiring actual data.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.evaluation import MetricsCalculator, EvaluationTracker


def create_dummy_data(num_samples=1000, num_classes=10):
    """Create dummy prediction data for testing."""
    # Generate random true labels
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Generate predictions with some correlation to true labels
    y_pred = y_true.copy()
    # Add some noise (wrong predictions)
    noise_indices = np.random.choice(num_samples, size=int(0.2 * num_samples), replace=False)
    y_pred[noise_indices] = np.random.randint(0, num_classes, len(noise_indices))
    
    # Generate probability scores
    y_prob = np.random.rand(num_samples, num_classes)
    # Make probabilities more realistic
    for i in range(num_samples):
        y_prob[i, y_true[i]] += 0.5  # Boost true class probability
    
    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    return y_true, y_pred, y_prob


def test_metrics_calculator():
    """Test the MetricsCalculator class."""
    print("Testing MetricsCalculator...")
    
    num_classes = 10
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Create calculator
    calculator = MetricsCalculator(num_classes, class_names)
    
    # Generate test data
    y_true, y_pred, y_prob = create_dummy_data(1000, num_classes)
    
    # Calculate metrics
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
    
    print("Calculated metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Calculate per-class metrics
    per_class_metrics = calculator.calculate_per_class_metrics(y_true, y_pred)
    print("\nPer-class metrics:")
    print(per_class_metrics.head())
    
    print("✓ MetricsCalculator test passed!")
    return metrics


def test_evaluation_tracker():
    """Test the EvaluationTracker class."""
    print("\nTesting EvaluationTracker...")
    
    # Create temporary directory for testing
    save_dir = "test_evaluation_output"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create tracker
    tracker = EvaluationTracker(save_dir, "test_experiment")
    
    num_classes = 5  # Use fewer classes for clearer visualization
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Simulate training for several epochs
    for epoch in range(10):
        # Generate dummy training metrics
        train_metrics = {
            'loss': 2.0 - epoch * 0.15 + np.random.normal(0, 0.05),
            'accuracy': 20 + epoch * 7 + np.random.normal(0, 2),
            'f1_macro': 0.2 + epoch * 0.07 + np.random.normal(0, 0.02),
        }
        
        # Generate dummy validation data
        y_true, y_pred, y_prob = create_dummy_data(200, num_classes)
        
        # Calculate validation metrics
        calculator = MetricsCalculator(num_classes, class_names)
        val_metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Add to tracker
        tracker.add_epoch_metrics(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=0.01 * (0.9 ** (epoch // 3))
        )
        
        # Generate visualizations every few epochs
        if epoch % 3 == 0:
            tracker.plot_confusion_matrix(y_true, y_pred, class_names, epoch)
            tracker.plot_roc_curves(y_true, y_prob, class_names, epoch)
            tracker.save_classification_report(y_true, y_pred, class_names, epoch)
    
    # Generate final plots
    tracker.plot_metrics(['loss', 'accuracy', 'f1_macro', 'auc_roc_macro'])
    
    print(f"✓ EvaluationTracker test passed! Check output in '{save_dir}' directory")
    print(f"  - CSV file: {tracker.csv_path}")
    print(f"  - Plots directory: {os.path.join(save_dir, 'plots')}")
    
    return tracker


def test_model_evaluation():
    """Test model evaluation with a dummy model."""
    print("\nTesting model evaluation...")
    
    # Create a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc = nn.Linear(10, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create dummy data loader
    class DummyDataLoader:
        def __init__(self, num_samples=100, num_classes=5):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.batch_size = 16
        
        def __iter__(self):
            for i in range(0, self.num_samples, self.batch_size):
                batch_size = min(self.batch_size, self.num_samples - i)
                inputs = torch.randn(batch_size, 10)
                labels = torch.randint(0, self.num_classes, (batch_size,))
                yield inputs, labels
        
        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    # Test setup
    num_classes = 5
    class_names = [f"Class_{i}" for i in range(num_classes)]
    device = torch.device('cpu')
    
    model = DummyModel(num_classes)
    dataloader = DummyDataLoader(100, num_classes)
    
    # Import evaluation function
    from src.utils.evaluation import evaluate_model
    
    # Evaluate model
    metrics, y_true, y_pred, y_prob = evaluate_model(
        model, dataloader, device, num_classes, class_names
    )
    
    print("Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("✓ Model evaluation test passed!")
    return metrics


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING EVALUATION FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test 1: Metrics Calculator
        metrics = test_metrics_calculator()
        
        # Test 2: Evaluation Tracker
        tracker = test_evaluation_tracker()
        
        # Test 3: Model Evaluation
        model_metrics = test_model_evaluation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe evaluation system is ready for use in training scripts.")
        print("Key features tested:")
        print("  ✓ Comprehensive metrics calculation (ACC, AUC, Precision, Recall, F1)")
        print("  ✓ CSV logging of training progress")
        print("  ✓ Automatic visualization generation")
        print("  ✓ Confusion matrix plotting")
        print("  ✓ ROC curve visualization")
        print("  ✓ Classification report generation")
        print("  ✓ Model evaluation integration")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
