#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Signal Prediction for Loss Mitigation in Tactile Internet
A Leader-Follower Game-Theoretic Approach

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

This paper has been accepted to IEEE MLSP 2025 (Istanbul, Turkey)

Copyright (c) 2025 Mohammad Ali Vahedifar

This module provides the main entry point for the LeFo package, implementing
a cooperative Stackelberg game-based approach for signal prediction in Tactile Internet
systems. The framework enables bidirectional predictive modeling between human operators
and remote robots to overcome packet loss and relax latency constraints.

Key Components:
    - LeaderFollowerModel: Main model class combining human and robot networks
    - HumanNetwork: 12-layer fully connected network for human side prediction
    - RobotNetwork: 8-layer fully connected network for robot side prediction
    - HapticDataset: Dataset class for loading haptic interaction data
    - Trainer: Training orchestrator with MiniMax optimization
    - MutualInformationEstimator: KNN-based MI estimation
    - KLDivergenceLoss: KL divergence loss for robot utility

Example:
    >>> from lefo import LeaderFollowerModel, HapticDataset, Trainer
    >>> from lefo.config import Config
    >>> 
    >>> config = Config.from_yaml('configs/default.yaml')
    >>> dataset = HapticDataset('data/haptic_traces.csv')
    >>> model = LeaderFollowerModel(human_layers=12, robot_layers=8)
    >>> trainer = Trainer(model, config)
    >>> trainer.fit(dataset)

References:
    [1] Vahedifar, M. A., & Zhang, Q. (2025). Signal Prediction for Loss Mitigation
        in Tactile Internet: A Leader-Follower Game-Theoretic Approach. 
        IEEE MLSP 2025, Istanbul, Turkey.
"""

__version__ = "1.0.0"
__author__ = "Mohammad Ali Vahedifar"
__email__ = "av@ece.au.dk"
__license__ = "MIT"

# Core model imports
from lefo.models import (
    LeaderFollowerModel,
    HumanNetwork,
    RobotNetwork,
    FullyConnectedBlock,
)

# Data handling imports
from lefo.data import (
    HapticDataset,
    HapticDataLoader,
    DeadbandProcessor,
    DataNormalizer,
)

# Game theory components
from lefo.game import (
    MutualInformationEstimator,
    KLDivergenceLoss,
    MiniMaxOptimizer,
    StackelbergEquilibrium,
)

# Training utilities
from lefo.train import (
    Trainer,
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
)

# Evaluation metrics
from lefo.evaluate import (
    PredictionAccuracy,
    InferenceTimer,
    UpperBoundVerifier,
)

# Configuration
from lefo.config import Config

# Visualization
from lefo.visualization import (
    plot_prediction_accuracy,
    plot_inference_times,
    plot_upper_bound_verification,
    plot_parameter_space,
)

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Models
    "LeaderFollowerModel",
    "HumanNetwork", 
    "RobotNetwork",
    "FullyConnectedBlock",
    
    # Data
    "HapticDataset",
    "HapticDataLoader",
    "DeadbandProcessor",
    "DataNormalizer",
    
    # Game Theory
    "MutualInformationEstimator",
    "KLDivergenceLoss",
    "MiniMaxOptimizer",
    "StackelbergEquilibrium",
    
    # Training
    "Trainer",
    "TrainingCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    
    # Evaluation
    "PredictionAccuracy",
    "InferenceTimer",
    "UpperBoundVerifier",
    
    # Config
    "Config",
    
    # Visualization
    "plot_prediction_accuracy",
    "plot_inference_times",
    "plot_upper_bound_verification",
    "plot_parameter_space",
]


def get_version():
    """Return the current version of the LeFo package."""
    return __version__


def get_device():
    """Get the best available device (CUDA or CPU)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
