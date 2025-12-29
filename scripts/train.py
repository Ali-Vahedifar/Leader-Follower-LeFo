#!/usr/bin/env python3
"""
Training Script for Leader-Follower Signal Prediction.

This script trains the LeFo model on haptic teleoperation data using
the game-theoretic MiniMax optimization framework.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Usage:
    python train.py --data-dir ./data --output-dir ./outputs --epochs 100

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lefo.models import LeaderFollowerModel, create_model
from src.lefo.train import Trainer, TrainingConfig, EarlyStopping, ModelCheckpoint
from src.lefo.data import HapticDataLoader, create_synthetic_dataset
from src.lefo.game import MiniMaxOptimizer, KLDivergenceLoss, MutualInformationEstimator
from src.lefo.utils import set_seed, setup_logging, save_model
from src.lefo.config import ModelConfig, DataConfig, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Leader-Follower signal prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing haptic dataset CSV files"
    )
    data_group.add_argument(
        "--dataset-type", type=str, default="drag",
        choices=["drag", "horizontal_fast", "horizontal_slow",
                 "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"],
        help="Type of haptic dataset to use"
    )
    data_group.add_argument(
        "--use-synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    data_group.add_argument(
        "--sequence-length", type=int, default=10,
        help="Input sequence length for prediction"
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--human-layers", type=int, default=12,
        help="Number of layers in human (Leader) network"
    )
    model_group.add_argument(
        "--robot-layers", type=int, default=8,
        help="Number of layers in robot (Follower) network"
    )
    model_group.add_argument(
        "--hidden-dim", type=int, default=100,
        help="Hidden dimension size for all layers"
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability during training"
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    train_group.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate"
    )
    train_group.add_argument(
        "--momentum", type=float, default=0.9,
        help="SGD momentum"
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="L2 regularization weight decay"
    )
    train_group.add_argument(
        "--lr-decay-epochs", type=int, nargs="+", default=[50, 75],
        help="Epochs at which to decay learning rate"
    )
    train_group.add_argument(
        "--lr-decay-factor", type=float, default=0.1,
        help="Factor to multiply learning rate at decay epochs"
    )
    train_group.add_argument(
        "--gradient-clip", type=float, default=1.0,
        help="Maximum gradient norm for clipping"
    )
    train_group.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Early stopping patience (0 to disable)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory to save model checkpoints and logs"
    )
    output_group.add_argument(
        "--experiment-name", type=str, default=None,
        help="Name for this experiment (auto-generated if not provided)"
    )
    output_group.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs"
    )
    
    # Hardware arguments
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    hw_group.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers"
    )
    hw_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    log_group.add_argument(
        "--wandb-project", type=str, default="lefo-tactile-internet",
        help="W&B project name"
    )
    log_group.add_argument(
        "--tensorboard", action="store_true",
        help="Enable TensorBoard logging"
    )
    log_group.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity level"
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides command line args)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.verbose)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("LeFo: Leader-Follower Signal Prediction Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"lefo_{args.dataset_type}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Load or create data
    if args.use_synthetic:
        print("Creating synthetic dataset for testing...")
        train_loader, val_loader, test_loader = create_synthetic_dataset(
            n_samples=5000,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size
        )
        input_dim = 18  # 9 features * 2 (human + robot)
        output_dim = 9
    else:
        print(f"Loading dataset from {args.data_dir}...")
        data_loader = HapticDataLoader(
            data_dir=args.data_dir,
            dataset_type=args.dataset_type,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_workers=args.num_workers
        )
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader
        test_loader = data_loader.test_loader
        input_dim = data_loader.input_dim
        output_dim = data_loader.output_dim
    
    # Create model
    print("Creating model...")
    model = LeaderFollowerModel(
        input_dim=input_dim,
        output_dim=output_dim,
        human_layers=args.human_layers,
        robot_layers=args.robot_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"Model architecture:")
    print(f"  Human Network: {args.human_layers} layers")
    print(f"  Robot Network: {args.robot_layers} layers")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_decay_epochs=args.lr_decay_epochs,
        lr_decay_factor=args.lr_decay_factor,
        gradient_clip=args.gradient_clip,
        dropout=args.dropout
    )
    
    # Setup callbacks
    callbacks = []
    
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            patience=args.early_stopping_patience,
            monitor='val_loss',
            mode='min'
        ))
    
    callbacks.append(ModelCheckpoint(
        filepath=str(output_dir / "best_model.pt"),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ))
    
    # Initialize W&B if enabled
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )
            wandb.watch(model)
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
    
    # Initialize TensorBoard if enabled
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        except ImportError:
            print("Warning: tensorboard not installed, skipping TB logging")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        callbacks=callbacks
    )
    
    # Train
    print("\nStarting training...")
    print("-"*60)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] if isinstance(vals, (list, np.ndarray)) else vals
            for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    save_model(
        model=model,
        path=final_model_path,
        config=vars(args),
        history=history
    )
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    
    test_loss = 0.0
    human_correct = 0
    robot_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            human_input = batch['human_input'].to(device)
            robot_input = batch['robot_input'].to(device)
            human_target = batch['human_target'].to(device)
            robot_target = batch['robot_target'].to(device)
            
            human_pred, robot_pred = model(human_input, robot_input)
            
            # Calculate accuracy (prediction within threshold)
            threshold = 0.1
            human_correct += ((human_pred - human_target).abs() < threshold).float().mean().item()
            robot_correct += ((robot_pred - robot_target).abs() < threshold).float().mean().item()
            total += 1
    
    human_accuracy = 100 * human_correct / total
    robot_accuracy = 100 * robot_correct / total
    
    print("-"*60)
    print("TEST RESULTS")
    print("-"*60)
    print(f"Human Prediction Accuracy: {human_accuracy:.2f}%")
    print(f"Robot Prediction Accuracy: {robot_accuracy:.2f}%")
    print("-"*60)
    
    # Save test results
    test_results = {
        'human_accuracy': human_accuracy,
        'robot_accuracy': robot_accuracy,
        'dataset_type': args.dataset_type,
        'epochs_trained': len(history['train_loss'])
    }
    
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Log to W&B
    if args.wandb:
        wandb.log({
            'test/human_accuracy': human_accuracy,
            'test/robot_accuracy': robot_accuracy
        })
        wandb.finish()
    
    # Close TensorBoard
    if tb_writer:
        tb_writer.close()
    
    print(f"\nTraining complete!")
    print(f"All outputs saved to: {output_dir}")
    
    return history


if __name__ == "__main__":
    main()
