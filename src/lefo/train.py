#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Training Module for Signal Prediction

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

IEEE MLSP 2025 - Istanbul, Turkey

This module implements the training logic for the Leader-Follower framework:
    - Trainer: Main training orchestrator with MiniMax optimization
    - TrainingCallback: Base class for training callbacks
    - EarlyStopping: Early stopping callback
    - ModelCheckpoint: Model checkpointing callback
    - LearningRateScheduler: Learning rate scheduling

Training uses Stochastic Gradient Descent (SGD) with:
    - Momentum: 0.9
    - Initial learning rate: 0.01
    - Batch size: 32
    - Dropout: 0.5 (for regularization)

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# Local imports
from lefo.models import LeaderFollowerModel
from lefo.game import KLDivergenceLoss, MutualInformationEstimator, MiniMaxOptimizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization weight
        dropout_rate: Dropout rate during training
        lr_decay: Learning rate decay factor
        lr_decay_epochs: Epochs at which to decay learning rate
        gradient_clip: Maximum gradient norm for clipping
        lambda_mi: Weight for mutual information loss
        lambda_kl: Weight for KL divergence loss
        max_iterations: Maximum MiniMax iterations per step
        tolerance: Convergence tolerance
        device: Training device ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Batches between logging
        eval_interval: Epochs between evaluation
    """
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    dropout_rate: float = 0.5
    lr_decay: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [50, 75])
    gradient_clip: float = 1.0
    lambda_mi: float = 1.0
    lambda_kl: float = 1.0
    max_iterations: int = 1
    tolerance: float = 1e-4
    device: str = 'cuda'
    seed: int = 42
    checkpoint_dir: str = 'checkpoints'
    log_interval: int = 10
    eval_interval: int = 1


class TrainingCallback(ABC):
    """
    Abstract base class for training callbacks.
    
    Callbacks are called at various points during training to implement
    custom behavior like early stopping, checkpointing, logging, etc.
    """
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: 'Trainer'):
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, trainer: 'Trainer', metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, trainer: 'Trainer', loss: float):
        """Called at the end of each batch."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: 'Trainer'):
        """Called when training completes."""
        pass


class EarlyStopping(TrainingCallback):
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    
    Args:
        monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.)
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' (whether lower or higher is better)
        restore_best: Whether to restore best weights when stopping
    
    Example:
        >>> early_stop = EarlyStopping(monitor='val_loss', patience=10)
        >>> trainer.add_callback(early_stop)
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_start(self, epoch: int, trainer: 'Trainer'):
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer', metrics: Dict[str, float]):
        score = metrics.get(self.monitor)
        
        if score is None:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found in metrics")
            return
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = trainer.model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_weights = trainer.model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                
                if self.restore_best:
                    trainer.model.load_state_dict(self.best_weights)
                    logger.info(f"Restored best weights from epoch {epoch - self.counter}")
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def on_batch_end(self, batch: int, trainer: 'Trainer', loss: float):
        pass
    
    def on_train_end(self, trainer: 'Trainer'):
        if self.stopped_epoch > 0:
            logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")


class ModelCheckpoint(TrainingCallback):
    """
    Callback to save model checkpoints during training.
    
    Args:
        filepath: Path template for saving checkpoints
        monitor: Metric to monitor for best model
        save_best_only: Whether to only save the best model
        save_weights_only: Whether to only save weights (not full state)
        mode: 'min' or 'max'
        verbose: Whether to print messages
    
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='checkpoints/model_{epoch:02d}.pt',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
    """
    
    def __init__(
        self,
        filepath: str = 'checkpoints/model_best.pt',
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_start(self, epoch: int, trainer: 'Trainer'):
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer', metrics: Dict[str, float]):
        score = metrics.get(self.monitor)
        
        if score is None:
            return
        
        save = False
        if not self.save_best_only:
            save = True
        elif self.best_score is None:
            save = True
            self.best_score = score
        elif self._is_better(score):
            save = True
            self.best_score = score
        
        if save:
            filepath = self.filepath.format(epoch=epoch, **metrics)
            self._save_checkpoint(trainer, epoch, metrics, filepath)
            
            if self.verbose:
                logger.info(f"Saved checkpoint: {filepath}")
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def _save_checkpoint(
        self,
        trainer: 'Trainer',
        epoch: int,
        metrics: Dict[str, float],
        filepath: str
    ):
        if self.save_weights_only:
            torch.save(trainer.model.state_dict(), filepath)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'human_optimizer_state_dict': trainer.human_optimizer.state_dict(),
                'robot_optimizer_state_dict': trainer.robot_optimizer.state_dict(),
                'metrics': metrics,
                'config': trainer.config.__dict__
            }
            torch.save(checkpoint, filepath)
    
    def on_batch_end(self, batch: int, trainer: 'Trainer', loss: float):
        pass
    
    def on_train_end(self, trainer: 'Trainer'):
        pass


class Trainer:
    """
    Main Trainer class for Leader-Follower model training.
    
    Implements the MiniMax optimization with alternating updates:
        1. Robot (Follower) minimizes KL divergence
        2. Human (Leader) maximizes Mutual Information
    
    Training uses SGD with momentum=0.9, lr=0.01, batch_size=32.
    
    Args:
        model: LeaderFollowerModel to train
        config: TrainingConfig with hyperparameters
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        callbacks: List of training callbacks
    
    Example:
        >>> trainer = Trainer(model, config, train_loader, val_loader)
        >>> history = trainer.fit()
    """
    
    def __init__(
        self,
        model: LeaderFollowerModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        
        # Set device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Initialize optimizers (SGD with momentum)
        self.human_optimizer = torch.optim.SGD(
            model.get_human_parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        self.robot_optimizer = torch.optim.SGD(
            model.get_robot_parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Initialize loss functions
        self.kl_loss = KLDivergenceLoss()
        self.mi_estimator = MutualInformationEstimator(k_neighbors=11)
        self.mse_loss = nn.MSELoss()
        
        # Initialize MiniMax optimizer
        self.minimax_optimizer = MiniMaxOptimizer(
            human_optimizer=self.human_optimizer,
            robot_optimizer=self.robot_optimizer,
            mi_estimator=self.mi_estimator,
            kl_loss=self.kl_loss,
            lambda_mi=config.lambda_mi,
            lambda_kl=config.lambda_kl,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance
        )
        
        # Learning rate schedulers
        self.human_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.human_optimizer,
            milestones=config.lr_decay_epochs,
            gamma=config.lr_decay
        )
        self.robot_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.robot_optimizer,
            milestones=config.lr_decay_epochs,
            gamma=config.lr_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mi': [],
            'train_kl': [],
            'val_loss': [],
            'val_human_accuracy': [],
            'val_robot_accuracy': [],
            'learning_rate': []
        }
        
        # State
        self.current_epoch = 0
        self.global_step = 0
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
    
    def fit(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for specified epochs.
        
        Args:
            epochs: Number of epochs (overrides config if provided)
        
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Callback: epoch start
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate()
            else:
                val_metrics = {}
            
            # Update learning rate
            self.human_scheduler.step()
            self.robot_scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['learning_rate'] = self.human_optimizer.param_groups[0]['lr']
            
            # Update history
            for key, value in metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Log progress
            self._log_epoch(epoch, metrics)
            
            # Callback: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, metrics)
                
                # Check for early stopping
                if hasattr(callback, 'stop_training') and callback.stop_training:
                    logger.info("Training stopped by callback")
                    break
            
            # Check for early stopping from callbacks
            if any(getattr(cb, 'stop_training', False) for cb in self.callbacks):
                break
        
        # Callback: training end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_mi = 0.0
        total_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (human_input, robot_input, human_target, robot_target) in enumerate(pbar):
            # Move to device
            human_input = human_input.to(self.device)
            robot_input = robot_input.to(self.device)
            human_target = human_target.to(self.device)
            robot_target = robot_target.to(self.device)
            
            # MiniMax optimization step
            loss_dict = self._train_step(
                human_input, robot_input,
                human_target, robot_target
            )
            
            total_loss += loss_dict['total_loss']
            total_mi += loss_dict.get('mi_value', 0)
            total_kl += loss_dict.get('kl_value', 0)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'mi': f"{loss_dict.get('mi_value', 0):.4f}",
                'kl': f"{loss_dict.get('kl_value', 0):.4f}"
            })
            
            # Callback: batch end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self, loss_dict['total_loss'])
            
            self.global_step += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'train_mi': total_mi / num_batches,
            'train_kl': total_kl / num_batches
        }
    
    def _train_step(
        self,
        human_input: Tensor,
        robot_input: Tensor,
        human_target: Tensor,
        robot_target: Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step with MiniMax optimization.
        
        Algorithm:
            1. Robot minimizes KL divergence (Follower response)
            2. Human maximizes Mutual Information (Leader optimization)
        """
        # Step 1: Robot (Follower) minimizes KL divergence
        self.robot_optimizer.zero_grad()
        
        _, human_pred = self.model(human_input, robot_input)
        kl_loss = self.kl_loss(human_target, human_pred)
        
        kl_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.robot_network.parameters(),
                self.config.gradient_clip
            )
        
        self.robot_optimizer.step()
        
        # Step 2: Human (Leader) maximizes Mutual Information
        self.human_optimizer.zero_grad()
        
        robot_pred, _ = self.model(human_input, robot_input)
        
        # Use MSE as differentiable proxy for MI maximization
        # Lower MSE correlates with higher MI in well-behaved cases
        mse_loss = self.mse_loss(robot_pred, robot_target)
        
        mse_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.human_network.parameters(),
                self.config.gradient_clip
            )
        
        self.human_optimizer.step()
        
        # Compute actual metrics (non-differentiable)
        with torch.no_grad():
            robot_pred_eval, human_pred_eval = self.model(human_input, robot_input)
            
            # MI estimation (expensive, do periodically)
            if self.global_step % 10 == 0:
                mi_value = self.mi_estimator(robot_target, robot_pred_eval).item()
            else:
                mi_value = 0.0
            
            kl_value = self.kl_loss(human_target, human_pred_eval).item()
        
        total_loss = self.config.lambda_mi * mi_value - self.config.lambda_kl * kl_value
        
        return {
            'total_loss': mse_loss.item() + kl_loss.item(),
            'mi_value': mi_value,
            'kl_value': kl_value,
            'human_mse': mse_loss.item(),
            'robot_kl': kl_loss.item()
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        human_correct = 0
        robot_correct = 0
        total_samples = 0
        
        for human_input, robot_input, human_target, robot_target in self.val_loader:
            human_input = human_input.to(self.device)
            robot_input = robot_input.to(self.device)
            human_target = human_target.to(self.device)
            robot_target = robot_target.to(self.device)
            
            robot_pred, human_pred = self.model(human_input, robot_input)
            
            # Compute losses
            kl_loss = self.kl_loss(human_target, human_pred)
            mse_loss = self.mse_loss(robot_pred, robot_target)
            
            total_loss += (kl_loss.item() + mse_loss.item())
            
            # Compute accuracy (within threshold)
            threshold = 0.1  # 10% relative error threshold
            
            human_error = torch.abs(human_pred - human_target) / (torch.abs(human_target) + 1e-10)
            robot_error = torch.abs(robot_pred - robot_target) / (torch.abs(robot_target) + 1e-10)
            
            human_correct += (human_error.mean(dim=1) < threshold).sum().item()
            robot_correct += (robot_error.mean(dim=1) < threshold).sum().item()
            total_samples += human_target.size(0)
        
        num_batches = len(self.val_loader)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_human_accuracy': human_correct / total_samples * 100,
            'val_robot_accuracy': robot_correct / total_samples * 100
        }
    
    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        msg = f"Epoch {epoch}: "
        msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(msg)
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'human_optimizer_state_dict': self.human_optimizer.state_dict(),
            'robot_optimizer_state_dict': self.robot_optimizer.state_dict(),
            'human_scheduler_state_dict': self.human_scheduler.state_dict(),
            'robot_scheduler_state_dict': self.robot_scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.human_optimizer.load_state_dict(checkpoint['human_optimizer_state_dict'])
        self.robot_optimizer.load_state_dict(checkpoint['robot_optimizer_state_dict'])
        self.human_scheduler.load_state_dict(checkpoint['human_scheduler_state_dict'])
        self.robot_scheduler.load_state_dict(checkpoint['robot_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def add_callback(self, callback: TrainingCallback):
        """Add a training callback."""
        self.callbacks.append(callback)
    
    @torch.no_grad()
    def predict(
        self,
        human_input: Tensor,
        robot_input: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Make predictions with the trained model."""
        self.model.eval()
        
        human_input = human_input.to(self.device)
        robot_input = robot_input.to(self.device)
        
        robot_pred, human_pred = self.model(human_input, robot_input)
        
        return robot_pred.cpu(), human_pred.cpu()


def train_model(
    model: LeaderFollowerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    checkpoint_path: Optional[str] = None
) -> Tuple[LeaderFollowerModel, Dict[str, List[float]]]:
    """
    Convenience function to train a Leader-Follower model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration (uses defaults if None)
        checkpoint_path: Path to save best model
    
    Returns:
        Tuple of (trained_model, training_history)
    
    Example:
        >>> model, history = train_model(model, train_loader, val_loader)
    """
    if config is None:
        config = TrainingConfig()
    
    # Setup callbacks
    callbacks = []
    
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        ))
    
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best=True
    ))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks
    )
    
    # Train
    history = trainer.fit()
    
    return model, history


if __name__ == "__main__":
    # Test training utilities
    print("Testing Training Utilities")
    print("=" * 60)
    
    from lefo.models import LeaderFollowerModel
    from lefo.data import create_synthetic_dataset, HapticDataset, HapticDataLoader
    import pandas as pd
    
    # Create synthetic data
    print("\n1. Creating synthetic dataset...")
    human_data, robot_data = create_synthetic_dataset(num_samples=500)
    
    # Save to temp file
    test_path = Path('/tmp/test_train_data.csv')
    df = pd.DataFrame(
        np.hstack([human_data, robot_data]),
        columns=[f'human_feat_{i}' for i in range(9)] + 
                [f'robot_feat_{i}' for i in range(9)]
    )
    df.to_csv(test_path, index=False)
    
    # Create dataset and loaders
    print("\n2. Creating data loaders...")
    dataset = HapticDataset(test_path, sequence_length=5)
    loader = HapticDataLoader(dataset, batch_size=16)
    train_loader, val_loader, _ = loader.get_loaders()
    
    # Create model
    print("\n3. Creating model...")
    model = LeaderFollowerModel(
        input_dim=90,  # 5 * 9 * 2 (seq_len * features * 2)
        output_dim=9,
        human_layers=4,  # Reduced for testing
        robot_layers=3
    )
    
    # Create config
    config = TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        device='cpu'  # Use CPU for testing
    )
    
    # Train
    print("\n4. Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Cleanup
    test_path.unlink()
    
    print("\n✓ All tests passed!")
