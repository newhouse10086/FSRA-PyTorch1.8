#!/usr/bin/env python3
"""
Training script specifically for New ViT model with community clustering.
Compatible with PyTorch 1.8, CUDA 10.2, cuDNN 7, Ubuntu 18.04.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from config import load_config, Config, merge_config_with_args
from src.models.new_vit import make_new_vit_model
from src.models.model_factory import create_cross_attention_model, print_model_info
from src.datasets import make_dataloader
from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.metrics import AverageMeter, accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='New ViT Training')
    
    # Config file
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to config file')
    
    # Override config options
    parser.add_argument('--data_dir', type=str, help='Training data directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use')
    
    # New ViT specific options
    parser.add_argument('--use_pretrained_resnet', action='store_true', default=True,
                       help='Use pretrained ResNet18')
    parser.add_argument('--use_pretrained_vit', action='store_true', default=False,
                       help='Use pretrained ViT weights')
    parser.add_argument('--num_final_clusters', type=int, default=3,
                       help='Number of final clusters')
    parser.add_argument('--from_scratch', action='store_true',
                       help='Train completely from scratch')
    
    # Training options
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_vit', type=str, help='Path to pretrained ViT model')
    
    return parser.parse_args()


def setup_device(config):
    """Setup device and distributed training."""
    if config.system.use_gpu and torch.cuda.is_available():
        # Parse GPU IDs
        gpu_ids = [int(x) for x in config.system.gpu_ids.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        
        # Set default GPU
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
        
        print(f"Using GPU(s): {gpu_ids}")
        return device, gpu_ids
    else:
        device = torch.device('cpu')
        print("Using CPU")
        return device, []


def create_new_vit_model(config, use_pretrained_resnet=True, use_pretrained_vit=False, 
                        num_final_clusters=3, pretrained_vit_path=None):
    """Create New ViT model with specified settings."""
    model = make_new_vit_model(
        num_classes=config.model.num_classes,
        use_pretrained_resnet=use_pretrained_resnet,
        use_pretrained_vit=use_pretrained_vit,
        num_final_clusters=num_final_clusters,
        return_f=config.model.return_features,
        views=config.data.views,
        share_weights=config.model.share_weights
    )
    
    # Load pretrained ViT weights if specified
    if pretrained_vit_path and os.path.exists(pretrained_vit_path):
        print(f"Loading pretrained ViT weights from: {pretrained_vit_path}")
        if hasattr(model, 'model_1'):
            # Two-view model
            model.model_1.load_pretrained_vit(pretrained_vit_path)
            if not config.model.share_weights and hasattr(model, 'model_2'):
                model.model_2.load_pretrained_vit(pretrained_vit_path)
        else:
            # Single-view model
            model.load_pretrained_vit(pretrained_vit_path)
    
    return model


def create_optimizers(model, cross_attention, config):
    """Create optimizers and schedulers."""
    # Main model optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )
    
    # Cross attention optimizer
    cross_optimizer = torch.optim.SGD(
        cross_attention.parameters(),
        lr=0.001,
        momentum=0.9
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.training.lr_scheduler_steps,
        gamma=config.training.lr_scheduler_gamma
    )
    
    return optimizer, cross_optimizer, scheduler


def create_loss_functions(config):
    """Create loss functions."""
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # Triplet loss (if needed)
    triplet_loss = None
    if config.training.triplet_loss_weight > 0:
        from src.losses.triplet_loss import TripletLoss
        triplet_loss = TripletLoss(margin=config.training.triplet_loss_weight)
    
    return criterion, mse_loss, triplet_loss


def compute_alignment_loss(features_s, features_d, mse_loss):
    """Compute feature alignment loss between satellite and drone features."""
    if len(features_s) != len(features_d):
        return torch.tensor(0.0, device=features_s[0].device)
    
    alignment_loss = 0.0
    for feat_s, feat_d in zip(features_s, features_d):
        alignment_loss += mse_loss(feat_s, feat_d)
    
    return alignment_loss / len(features_s)


def train_epoch(model, cross_attention, dataloader, optimizer, cross_optimizer, 
                criterion, mse_loss, triplet_loss, config, epoch, logger, writer):
    """Train for one epoch."""
    model.train()
    cross_attention.train()
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    alignment_losses = AverageMeter()
    accuracies = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, (data_s, data_d) in enumerate(dataloader):
        # Unpack data
        inputs_s, labels_s = data_s
        inputs_d, labels_d = data_d
        
        # Move to device
        if torch.cuda.is_available():
            inputs_s = inputs_s.cuda()
            inputs_d = inputs_d.cuda()
            labels_s = labels_s.cuda()
            labels_d = labels_d.cuda()
        
        batch_size = inputs_s.size(0)
        if batch_size < config.data.batch_size:
            continue
        
        # Zero gradients
        optimizer.zero_grad()
        cross_optimizer.zero_grad()
        
        # Forward pass
        outputs_s, features_s = model(inputs_s)
        outputs_d, features_d = model(inputs_d)
        
        # Classification loss
        cls_loss = 0.0
        for out_s, out_d in zip(outputs_s, outputs_d):
            cls_loss += criterion(out_s, labels_s) + criterion(out_d, labels_d)
        cls_loss /= len(outputs_s)
        
        # Feature alignment loss
        alignment_loss = compute_alignment_loss(features_s, features_d, mse_loss)
        
        # Total loss
        total_loss = cls_loss + 0.1 * alignment_loss  # Weight alignment loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        cross_optimizer.step()
        
        # Update metrics
        acc = accuracy(outputs_s[0], labels_s)[0]
        losses.update(total_loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        alignment_losses.update(alignment_loss.item(), batch_size)
        accuracies.update(acc.item(), batch_size)
        
        # Log progress
        if batch_idx % config.system.log_interval == 0:
            logger.info(f'Epoch: {epoch} [{batch_idx}/{len(dataloader)}] '
                       f'Loss: {losses.avg:.4f} '
                       f'Cls: {cls_losses.avg:.4f} '
                       f'Align: {alignment_losses.avg:.4f} '
                       f'Acc: {accuracies.avg:.2f}%')
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Train/Loss', losses.avg, epoch)
        writer.add_scalar('Train/ClassificationLoss', cls_losses.avg, epoch)
        writer.add_scalar('Train/AlignmentLoss', alignment_losses.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracies.avg, epoch)
    
    epoch_time = time.time() - start_time
    logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s')
    
    return losses.avg


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = Config()
    
    # Override model name to NewViT
    config.model.name = "NewViT"
    
    # Merge with command line arguments
    config = merge_config_with_args(config, vars(args))
    
    # Determine training mode
    if args.from_scratch:
        use_pretrained_resnet = False
        use_pretrained_vit = False
        training_mode = "completely from scratch"
    else:
        use_pretrained_resnet = args.use_pretrained_resnet
        use_pretrained_vit = args.use_pretrained_vit
        training_mode = f"ResNet18: {'pretrained' if use_pretrained_resnet else 'scratch'}, ViT: {'pretrained' if use_pretrained_vit else 'scratch'}"
    
    # Setup logging
    logger = setup_logger('train_new_vit', config.system.log_dir)
    logger.info(f"Starting New ViT training with config: {args.config}")
    logger.info(f"Training mode: {training_mode}")
    
    # Setup device
    device, gpu_ids = setup_device(config)
    
    # Create data loader
    try:
        dataloader, class_names, dataset_sizes = make_dataloader(config, mode='train')
        config.model.num_classes = len(class_names)
        logger.info(f"Dataset loaded: {len(class_names)} classes, {dataset_sizes['total']} samples")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        logger.info("This is expected if the dataset is not available. The model structure is still valid.")
        # Create dummy values for testing
        class_names = [f"class_{i}" for i in range(config.model.num_classes)]
        dataset_sizes = {'total': 1000}
        dataloader = None
    
    # Create models
    model = create_new_vit_model(
        config, 
        use_pretrained_resnet=use_pretrained_resnet,
        use_pretrained_vit=use_pretrained_vit,
        num_final_clusters=args.num_final_clusters,
        pretrained_vit_path=args.pretrained_vit
    )
    cross_attention = create_cross_attention_model(config)
    
    # Move to device
    model = model.to(device)
    cross_attention = cross_attention.to(device)
    
    # Print model information
    print_model_info(model, "New ViT Model")
    print_model_info(cross_attention, "Cross Attention Model")
    
    logger.info("New ViT models created successfully")
    
    # Create optimizers
    optimizer, cross_optimizer, scheduler = create_optimizers(model, cross_attention, config)
    
    # Create loss functions
    criterion, mse_loss, triplet_loss = create_loss_functions(config)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.system.log_dir, 'tensorboard'))
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    if dataloader is not None:
        logger.info("Starting training...")
        for epoch in range(start_epoch, config.training.num_epochs):
            # Train one epoch
            train_loss = train_epoch(
                model, cross_attention, dataloader, optimizer, cross_optimizer,
                criterion, mse_loss, triplet_loss, config, epoch, logger, writer
            )
            
            # Update learning rate
            scheduler.step()
            
            # Save checkpoint
            if epoch % config.system.save_interval == 0:
                save_checkpoint(model, optimizer, epoch, train_loss, 
                              os.path.join(config.system.checkpoint_dir, f'new_vit_checkpoint_epoch_{epoch}.pth'))
        
        logger.info("Training completed!")
    else:
        logger.info("No dataloader available. Training skipped.")
    
    # Close tensorboard writer
    writer.close()


if __name__ == '__main__':
    main()
