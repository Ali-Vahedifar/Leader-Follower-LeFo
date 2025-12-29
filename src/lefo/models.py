#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Neural Network Architectures for Signal Prediction

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

IEEE MLSP 2025 - Istanbul, Turkey

This module implements the neural network architectures for the Leader-Follower framework:
    - HumanNetwork: 12-layer fully connected network for human side (Leader)
    - RobotNetwork: 8-layer fully connected network for robot side (Follower)
    - LeaderFollowerModel: Combined model for bidirectional prediction

The human operator benefits from a deeper network to handle the complex task of
predicting and generating high-dimensional haptic signals while accounting for
latency constraints. The robot operates effectively with a shallower network
that prioritizes and minimizes computational overhead.

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import math
from typing import Tuple, Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FullyConnectedBlock(nn.Module):
    """
    A fully connected block with optional batch normalization and dropout.
    
    This building block consists of:
        Linear -> (BatchNorm) -> ReLU -> (Dropout)
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        dropout_rate: Dropout probability (0 to disable)
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
    
    Example:
        >>> block = FullyConnectedBlock(100, 100, dropout_rate=0.5)
        >>> x = torch.randn(32, 100)
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([32, 100])
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super(FullyConnectedBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Select activation function
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
        }
        self.activation = activations.get(activation, nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using He initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He (Kaiming) initialization."""
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the block."""
        x = self.linear(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        return x


class HumanNetwork(nn.Module):
    """
    Human Side Network (Leader) - 12-layer Fully Connected Network
    
    The human operator benefits from a deeper network to handle the complex task
    of predicting and generating high-dimensional haptic signals while accounting
    for latency constraints. This network predicts the robot's next state based
    on historical human and robot data.
    
    Architecture:
        Input -> [FC Block x 12] -> Output Layer
        
        Each FC Block: Linear(100) -> ReLU -> Dropout(0.5)
    
    Args:
        input_dim: Dimension of input features (default: 18 for 3D pos, vel, force x 2)
        output_dim: Dimension of output features (default: 9 for 3D pos, vel, force)
        hidden_units: Number of hidden units per layer (default: 100)
        num_layers: Number of hidden layers (default: 12)
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: False)
    
    Example:
        >>> model = HumanNetwork(input_dim=18, output_dim=9)
        >>> x = torch.randn(32, 18)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 9])
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 9,
        hidden_units: int = 100,
        num_layers: int = 12,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super(HumanNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(
            FullyConnectedBlock(
                input_dim, 
                hidden_units,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                FullyConnectedBlock(
                    hidden_units,
                    hidden_units,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer (no dropout or activation)
        self.output_layer = nn.Linear(hidden_units, output_dim)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to predict robot's next state.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains historical human and robot signals
        
        Returns:
            Predicted robot signal of shape (batch_size, output_dim)
        """
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output
    
    def get_features(self, x: Tensor) -> Tensor:
        """Extract intermediate features for analysis."""
        return self.feature_extractor(x)


class RobotNetwork(nn.Module):
    """
    Robot Side Network (Follower) - 8-layer Fully Connected Network
    
    The robot operates effectively with a shallower network that prioritizes
    and minimizes computational overhead. This network predicts the human's
    next action based on received human commands and robot state.
    
    Architecture:
        Input -> [FC Block x 8] -> Output Layer
        
        Each FC Block: Linear(100) -> ReLU -> Dropout(0.5)
    
    Args:
        input_dim: Dimension of input features (default: 18)
        output_dim: Dimension of output features (default: 9)
        hidden_units: Number of hidden units per layer (default: 100)
        num_layers: Number of hidden layers (default: 8)
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: False)
    
    Example:
        >>> model = RobotNetwork(input_dim=18, output_dim=9)
        >>> x = torch.randn(32, 18)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 9])
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 9,
        hidden_units: int = 100,
        num_layers: int = 8,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super(RobotNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(
            FullyConnectedBlock(
                input_dim,
                hidden_units,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                FullyConnectedBlock(
                    hidden_units,
                    hidden_units,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_units, output_dim)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to predict human's next action.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted human signal of shape (batch_size, output_dim)
        """
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output
    
    def get_features(self, x: Tensor) -> Tensor:
        """Extract intermediate features for analysis."""
        return self.feature_extractor(x)


class LeaderFollowerModel(nn.Module):
    """
    Combined Leader-Follower Model for Bidirectional Signal Prediction
    
    This model combines both the Human (Leader) and Robot (Follower) networks
    for bidirectional signal prediction in Tactile Internet applications.
    The interaction is formulated as a MiniMax optimization problem:
    
        min_R max_H L(H,R,θ) = E[max I(S_R, Ŝ_R)] - E[min KL(S_H || Ŝ_H)]
    
    where:
        - I: Mutual Information between actual and predicted robot signals
        - KL: KL divergence between actual and predicted human signals
    
    Args:
        input_dim: Dimension of input features (default: 18)
        output_dim: Dimension of output per agent (default: 9)
        human_layers: Number of layers in human network (default: 12)
        robot_layers: Number of layers in robot network (default: 8)
        hidden_units: Hidden units per layer (default: 100)
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Use batch normalization (default: False)
    
    Example:
        >>> model = LeaderFollowerModel()
        >>> human_input = torch.randn(32, 18)
        >>> robot_input = torch.randn(32, 18)
        >>> robot_pred, human_pred = model(human_input, robot_input)
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 9,
        human_layers: int = 12,
        robot_layers: int = 8,
        hidden_units: int = 100,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super(LeaderFollowerModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Human network (Leader) - predicts robot's next state
        self.human_network = HumanNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            num_layers=human_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation
        )
        
        # Robot network (Follower) - predicts human's next action
        self.robot_network = RobotNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            num_layers=robot_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation
        )
    
    def forward(
        self,
        human_input: Tensor,
        robot_input: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for bidirectional prediction.
        
        Args:
            human_input: Input to human network (historical data for robot prediction)
            robot_input: Input to robot network (historical data for human prediction)
        
        Returns:
            Tuple of (robot_prediction, human_prediction)
        """
        # Human (Leader) predicts robot's next state
        robot_prediction = self.human_network(human_input)
        
        # Robot (Follower) predicts human's next action
        human_prediction = self.robot_network(robot_input)
        
        return robot_prediction, human_prediction
    
    def predict_robot(self, x: Tensor) -> Tensor:
        """Predict robot's next state (Human side)."""
        return self.human_network(x)
    
    def predict_human(self, x: Tensor) -> Tensor:
        """Predict human's next action (Robot side)."""
        return self.robot_network(x)
    
    def get_human_parameters(self):
        """Get parameters of the human network."""
        return self.human_network.parameters()
    
    def get_robot_parameters(self):
        """Get parameters of the robot network."""
        return self.robot_network.parameters()
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in each network."""
        human_params = sum(p.numel() for p in self.human_network.parameters() if p.requires_grad)
        robot_params = sum(p.numel() for p in self.robot_network.parameters() if p.requires_grad)
        
        return {
            'human_network': human_params,
            'robot_network': robot_params,
            'total': human_params + robot_params
        }
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get detailed information about model layers."""
        return {
            'human_network': {
                'num_layers': self.human_network.num_layers,
                'hidden_units': self.human_network.hidden_units,
                'input_dim': self.human_network.input_dim,
                'output_dim': self.human_network.output_dim
            },
            'robot_network': {
                'num_layers': self.robot_network.num_layers,
                'hidden_units': self.robot_network.hidden_units,
                'input_dim': self.robot_network.input_dim,
                'output_dim': self.robot_network.output_dim
            }
        }


class SequencePredictor(nn.Module):
    """
    Sequence-to-sequence predictor using the ARMA-style formulation.
    
    Implements the signal prediction equation:
        Ŝ_A(n+1) = Σ Ω_i S_A(n+1-i) + Σ λ_j ε_{n+1-j}
    
    where:
        - Ω_i: Auto-regressive coefficients
        - λ_j: Moving average coefficients
        - ε_{n+1-j}: Error terms
    
    Args:
        input_dim: Dimension of input signal
        output_dim: Dimension of output signal
        sequence_length: Length of input sequence (N)
        hidden_dim: Hidden dimension for feature extraction
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 9,
        sequence_length: int = 10,
        hidden_dim: int = 64
    ):
        super(SequencePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # Auto-regressive weights (Ω)
        self.ar_weights = nn.Parameter(torch.randn(sequence_length, input_dim))
        
        # Moving average weights (λ)
        self.ma_weights = nn.Parameter(torch.randn(sequence_length, input_dim))
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(sequence_length * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.ar_weights)
        nn.init.xavier_uniform_(self.ma_weights)
    
    def forward(
        self,
        sequence: Tensor,
        errors: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for sequence prediction.
        
        Args:
            sequence: Input sequence of shape (batch, seq_len, input_dim)
            errors: Error terms of shape (batch, seq_len, input_dim), optional
        
        Returns:
            Predicted next sample of shape (batch, output_dim)
        """
        batch_size = sequence.size(0)
        
        # Auto-regressive component
        ar_component = torch.sum(
            sequence * self.ar_weights.unsqueeze(0),
            dim=1
        )
        
        # Moving average component
        if errors is not None:
            ma_component = torch.sum(
                errors * self.ma_weights.unsqueeze(0),
                dim=1
            )
        else:
            ma_component = torch.zeros_like(ar_component)
        
        # Combined prediction
        combined = ar_component + ma_component
        
        # Feature extraction for non-linear mapping
        flat_sequence = sequence.view(batch_size, -1)
        features = self.feature_net(flat_sequence)
        
        # Final prediction
        prediction = combined + features
        
        return prediction


def create_model(config: Dict[str, Any]) -> LeaderFollowerModel:
    """
    Factory function to create a LeaderFollowerModel from configuration.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Initialized LeaderFollowerModel
    
    Example:
        >>> config = {
        ...     'input_dim': 18,
        ...     'output_dim': 9,
        ...     'human_layers': 12,
        ...     'robot_layers': 8,
        ...     'hidden_units': 100,
        ...     'dropout_rate': 0.5
        ... }
        >>> model = create_model(config)
    """
    return LeaderFollowerModel(
        input_dim=config.get('input_dim', 18),
        output_dim=config.get('output_dim', 9),
        human_layers=config.get('human_layers', 12),
        robot_layers=config.get('robot_layers', 8),
        hidden_units=config.get('hidden_units', 100),
        dropout_rate=config.get('dropout_rate', 0.5),
        use_batch_norm=config.get('use_batch_norm', False),
        activation=config.get('activation', 'relu')
    )


if __name__ == "__main__":
    # Test the models
    print("Testing Leader-Follower Neural Network Models")
    print("=" * 60)
    
    # Create model
    model = LeaderFollowerModel()
    
    # Print model info
    print(f"\nModel Architecture:")
    print(model.get_layer_info())
    
    # Print parameter counts
    params = model.count_parameters()
    print(f"\nParameter Counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 32
    human_input = torch.randn(batch_size, 18)
    robot_input = torch.randn(batch_size, 18)
    
    robot_pred, human_pred = model(human_input, robot_input)
    
    print(f"\nForward Pass Test:")
    print(f"  Input shape: {human_input.shape}")
    print(f"  Robot prediction shape: {robot_pred.shape}")
    print(f"  Human prediction shape: {human_pred.shape}")
    
    print("\n✓ All tests passed!")
