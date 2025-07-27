# Evaluation System Guide

## Overview

The FSRA project now includes a comprehensive evaluation system that automatically calculates performance metrics, generates visualizations, and saves detailed reports during training.

## Features

### 🎯 Automatic Metrics Calculation
- **Accuracy**: Overall classification accuracy
- **AUC-ROC**: Area Under ROC Curve (macro/weighted for multi-class)
- **Precision**: Macro and micro averaged precision
- **Recall**: Macro and micro averaged recall  
- **F1-Score**: Macro and micro averaged F1-score
- **Per-class metrics**: Individual class performance

### 📊 Automatic Visualizations
- **Training Curves**: Loss, accuracy, AUC, F1-score over epochs
- **Confusion Matrix**: Visual confusion matrix every 10 epochs
- **ROC Curves**: Multi-class ROC curves (for ≤20 classes)
- **Classification Reports**: Detailed per-class performance

### 💾 Data Logging
- **CSV Files**: Complete metrics history with timestamps
- **TensorBoard**: Real-time training monitoring
- **Classification Reports**: Detailed performance breakdowns

## Usage

### 1. Training with Evaluation

The evaluation system is automatically integrated into both training scripts:

#### Traditional FSRA Model
```bash
# Train FSRA with automatic evaluation
python scripts/train.py \
    --config config/default_config.yaml \
    --model FSRA \
    --data_dir /path/to/University-1652 \
    --batch_size 16 \
    --num_epochs 120
```

#### New ViT Model
```bash
# Train New ViT with automatic evaluation
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --data_dir /path/to/University-1652 \
    --batch_size 16 \
    --num_epochs 150
```

### 2. Console Output

During training, you'll see detailed metrics output:

```
Epoch: 10 [100/500] Loss: 1.2345 Acc: 78.45%
Evaluating on test set - Epoch 10
Test Results - Epoch 10:
  Accuracy: 0.7845
  AUC-ROC: 0.8923
  Precision (Macro): 0.7756
  Recall (Macro): 0.7689
  F1-Score (Macro): 0.7722
```

### 3. Output Files Structure

```
logs/
├── {experiment_name}_metrics.csv          # Complete metrics history
├── plots/                                 # Visualization directory
│   ├── {experiment_name}_metrics.png      # Training curves
│   ├── {experiment_name}_confusion_matrix_epoch_10.png
│   ├── {experiment_name}_roc_curves_epoch_10.png
│   └── ...
├── {experiment_name}_classification_report_epoch_10.csv
├── tensorboard/                           # TensorBoard logs
└── train.log                             # Training logs
```

### 4. CSV Metrics File

The CSV file contains comprehensive training history:

| epoch | learning_rate | timestamp | train_loss | train_accuracy | val_accuracy | val_auc_roc_macro | val_precision_macro | val_recall_macro | val_f1_macro |
|-------|---------------|-----------|------------|----------------|--------------|-------------------|-------------------|------------------|--------------|
| 0 | 0.01 | 2024-01-01T10:00:00 | 2.1234 | 45.67 | 42.34 | 0.6789 | 0.4123 | 0.4234 | 0.4178 |
| 1 | 0.01 | 2024-01-01T10:05:00 | 1.8765 | 52.34 | 48.91 | 0.7234 | 0.4789 | 0.4891 | 0.4839 |

### 5. TensorBoard Monitoring

View real-time training progress:

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006
```

## Testing the Evaluation System

### Run Evaluation Tests

```bash
# Test the evaluation functionality
python scripts/test_evaluation.py
```

This will:
- Test metrics calculation with dummy data
- Generate sample visualizations
- Verify CSV logging functionality
- Create test output in `test_evaluation_output/`

### Expected Test Output

```
Testing MetricsCalculator...
Calculated metrics:
  accuracy: 0.8120
  precision_macro: 0.8156
  recall_macro: 0.8120
  f1_macro: 0.8098
  auc_roc_macro: 0.9234
✓ MetricsCalculator test passed!

Testing EvaluationTracker...
✓ EvaluationTracker test passed! Check output in 'test_evaluation_output' directory

Testing model evaluation...
✓ Model evaluation test passed!

ALL TESTS PASSED SUCCESSFULLY!
```

## Configuration

### Evaluation Settings

You can customize evaluation behavior in the config file:

```yaml
training:
  # Evaluation frequency
  eval_interval: 1  # Evaluate every N epochs
  
  # Visualization frequency  
  plot_interval: 10  # Generate plots every N epochs
  
  # Metrics to track
  track_metrics: ['accuracy', 'auc_roc', 'f1_macro', 'precision_macro', 'recall_macro']

system:
  # Output directories
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  
  # Logging settings
  log_interval: 100  # Log every N batches
```

### Custom Experiment Names

The evaluation tracker automatically generates experiment names based on:
- Model type (FSRA/NewViT)
- Training mode (pretrained/from_scratch)
- Timestamp

Example: `new_vit_resnet18_pretrained_vit_scratch_20240101_120000`

## Advanced Usage

### Custom Metrics

You can extend the evaluation system with custom metrics:

```python
from src.utils.evaluation import MetricsCalculator

class CustomMetricsCalculator(MetricsCalculator):
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        metrics = super().calculate_metrics(y_true, y_pred, y_prob)
        
        # Add custom metrics
        metrics['custom_metric'] = self.calculate_custom_metric(y_true, y_pred)
        
        return metrics
    
    def calculate_custom_metric(self, y_true, y_pred):
        # Your custom metric calculation
        return custom_value
```

### Custom Visualizations

Add custom plots to the evaluation tracker:

```python
def plot_custom_visualization(self, data, epoch):
    plt.figure(figsize=(10, 6))
    # Your custom plotting code
    plt.savefig(os.path.join(self.save_dir, 'plots', f'custom_plot_epoch_{epoch}.png'))
    plt.close()
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install matplotlib seaborn scikit-learn networkx
   ```

2. **Memory Issues with Large Datasets**
   - Reduce batch size
   - Evaluate less frequently (increase eval_interval)

3. **Too Many Classes for ROC Curves**
   - ROC curves are automatically disabled for >20 classes
   - Adjust threshold in evaluation.py if needed

4. **CSV File Corruption**
   - Check disk space
   - Ensure write permissions in log directory

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Impact

The evaluation system is designed to be lightweight:

- **Metrics calculation**: ~0.1-0.5 seconds per epoch
- **Visualization generation**: ~1-3 seconds every 10 epochs  
- **CSV logging**: Negligible overhead
- **Memory usage**: Minimal additional memory required

## Best Practices

1. **Regular Monitoring**: Check metrics every few epochs during training
2. **Early Stopping**: Use validation metrics to implement early stopping
3. **Hyperparameter Tuning**: Use logged metrics to optimize hyperparameters
4. **Model Comparison**: Compare different models using saved CSV files
5. **Reproducibility**: Save experiment configurations with results

## Integration with Other Tools

### Weights & Biases Integration

```python
import wandb

# Log metrics to W&B
wandb.log(metrics, step=epoch)
```

### MLflow Integration

```python
import mlflow

# Log metrics to MLflow
for metric_name, metric_value in metrics.items():
    mlflow.log_metric(metric_name, metric_value, step=epoch)
```

The evaluation system provides a solid foundation for monitoring and analyzing your model's performance throughout the training process!
