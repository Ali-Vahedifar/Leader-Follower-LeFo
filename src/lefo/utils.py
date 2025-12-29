"""
Utility Functions for Leader-Follower Signal Prediction.

This module provides utility functions for:
    - Reproducibility (seed setting)
    - Device management
    - Logging configuration
    - Model checkpointing
    - Configuration management
    - File I/O operations

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducibility.
    
    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).
    
    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        gpu_id: Specific GPU ID to use. If None, uses first available GPU.
        
    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id < torch.cuda.device_count():
                return torch.device(f'cuda:{gpu_id}')
            else:
                logging.warning(f"GPU {gpu_id} not available. Using GPU 0.")
                return torch.device('cuda:0')
        return torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information.
    
    Returns:
        Dictionary with device information.
    """
    info = {
        'device_type': 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'num_gpus': 0,
        'gpu_names': [],
        'gpu_memory': []
    }
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['num_gpus'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        for i in range(info['num_gpus']):
            props = torch.cuda.get_device_properties(i)
            info['gpu_names'].append(props.name)
            info['gpu_memory'].append(props.total_memory / (1024**3))  # GB
    
    elif info['mps_available']:
        info['device_type'] = 'mps'
    
    return info


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files.
        log_level: Logging level.
        log_file: Specific log file name. If None, uses timestamp.
        console: Whether to also log to console.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger('lefo')
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'lefo_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch number.
        loss: Current loss value.
        path: Path to save checkpoint.
        **kwargs: Additional items to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        path: Path to checkpoint file.
        model: Model to load weights into. If None, returns state dict.
        optimizer: Optimizer to load state into.
        device: Device to load to.
        
    Returns:
        Checkpoint dictionary.
    """
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count number of parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
        
    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Generate a summary of model architecture.
    
    Args:
        model: PyTorch model.
        input_shape: Shape of input tensor (excluding batch dimension).
        
    Returns:
        Summary string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    lines.append(f"{'Layer':<40} {'Output Shape':<20} {'Params':>15}")
    lines.append("-" * 80)
    
    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params, trainable_params
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(lines) - 3
            
            # Get output shape
            if isinstance(output, (list, tuple)):
                output_shape = str([list(o.shape) for o in output])
            else:
                output_shape = str(list(output.shape))
            
            # Count parameters
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if params > 0:
                total_params += params
                trainable_params += trainable
                lines.append(f"{class_name:<40} {output_shape:<20} {params:>15,}")
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           module != model:
            hooks.append(module.register_forward_hook(hook))
    
    hooks = []
    model.apply(register_hook)
    
    # Forward pass
    device = next(model.parameters()).device
    x = torch.zeros(1, *input_shape).to(device)
    model.eval()
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    lines.append("=" * 80)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append(f"Non-trainable params: {total_params - trainable_params:,}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Data to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        path: Input file path.
        
    Returns:
        Loaded dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def get_experiment_name(
    prefix: str = 'exp',
    include_timestamp: bool = True,
    suffix: Optional[str] = None
) -> str:
    """Generate experiment name.
    
    Args:
        prefix: Name prefix.
        include_timestamp: Whether to include timestamp.
        suffix: Optional suffix.
        
    Returns:
        Experiment name string.
    """
    parts = [prefix]
    
    if include_timestamp:
        parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    if suffix:
        parts.append(suffix)
    
    return '_'.join(parts)


class AverageMeter:
    """Compute and store the average and current value.
    
    Useful for tracking loss and accuracy during training.
    """
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update with new value.
        
        Args:
            val: New value.
            n: Number of items (for weighted average).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value.
            
        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def move_to_device(
    data: Union[torch.Tensor, Dict, List, Tuple],
    device: torch.device
) -> Union[torch.Tensor, Dict, List, Tuple]:
    """Recursively move data to device.
    
    Args:
        data: Data to move (tensor, dict, list, or tuple).
        device: Target device.
        
    Returns:
        Data on target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data


def gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm of model parameters.
    
    Args:
        model: PyTorch model.
        
    Returns:
        L2 norm of all gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def clip_gradient_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """Clip gradients by norm.
    
    Args:
        model: PyTorch model.
        max_norm: Maximum gradient norm.
        norm_type: Norm type (default: L2).
        
    Returns:
        Original total gradient norm.
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type
    ).item()


def print_model_size(model: nn.Module) -> None:
    """Print model size in MB.
    
    Args:
        model: PyTorch model.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_mb:.2f} MB")
