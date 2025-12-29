"""
Configuration Management for Leader-Follower Signal Prediction.

This module provides configuration classes and utilities for managing
experiment parameters, model hyperparameters, and training settings.

Supports:
    - YAML configuration files
    - Command-line overrides
    - Default configurations
    - Configuration validation

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class ModelConfig:
    """Configuration for neural network models.
    
    Attributes:
        human_layers: Number of layers in human (Leader) network.
        robot_layers: Number of layers in robot (Follower) network.
        hidden_size: Hidden layer size for both networks.
        input_size: Input feature dimension.
        output_size: Output feature dimension.
        dropout: Dropout probability.
        activation: Activation function ('relu', 'gelu', 'tanh').
        use_batch_norm: Whether to use batch normalization.
    """
    human_layers: int = 12
    robot_layers: int = 8
    hidden_size: int = 100
    input_size: int = 18  # 9 features * 2 (human + robot)
    output_size: int = 9   # 9 features (position, velocity, force)
    dropout: float = 0.5
    activation: str = 'relu'
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularization weight.
        lr_decay_factor: Learning rate decay factor.
        lr_decay_epochs: Epochs at which to decay learning rate.
        gradient_clip: Maximum gradient norm.
        early_stopping_patience: Patience for early stopping.
        min_delta: Minimum improvement for early stopping.
    """
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [50, 75])
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    min_delta: float = 1e-4


@dataclass
class DataConfig:
    """Configuration for data loading and processing.
    
    Attributes:
        data_dir: Directory containing datasets.
        dataset_type: Type of haptic dataset.
        train_split: Proportion for training.
        val_split: Proportion for validation.
        sequence_length: Input sequence length.
        velocity_threshold: Deadband threshold for velocity.
        force_threshold: Deadband threshold for force.
        normalize: Normalization method ('standard', 'minmax', 'none').
        num_workers: Number of data loader workers.
    """
    data_dir: str = './data'
    dataset_type: str = 'drag_max_stiffness_y'
    train_split: float = 0.7
    val_split: float = 0.15
    sequence_length: int = 10
    velocity_threshold: float = 0.1
    force_threshold: float = 0.1
    normalize: str = 'standard'
    num_workers: int = 4


@dataclass
class GameConfig:
    """Configuration for game theory components.
    
    Attributes:
        kl_method: KL divergence method ('softmax', 'gaussian').
        mi_k_neighbors: Number of neighbors for MI estimation.
        alternating_steps: Steps per network in alternating optimization.
        temperature: Temperature for softmax in KL divergence.
    """
    kl_method: str = 'softmax'
    mi_k_neighbors: int = 11
    alternating_steps: int = 1
    temperature: float = 1.0


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpoints.
    
    Attributes:
        log_dir: Directory for logs.
        checkpoint_dir: Directory for model checkpoints.
        save_every: Save checkpoint every N epochs.
        log_interval: Log training progress every N batches.
        use_tensorboard: Whether to use TensorBoard.
        use_wandb: Whether to use Weights & Biases.
        project_name: W&B project name.
        experiment_name: Experiment name.
    """
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10
    log_interval: int = 50
    use_tensorboard: bool = True
    use_wandb: bool = False
    project_name: str = 'lefo-tactile'
    experiment_name: str = 'default'


@dataclass
class Config:
    """Complete configuration for Leader-Follower experiments.
    
    Combines all configuration components into a single object.
    
    Attributes:
        model: Model configuration.
        training: Training configuration.
        data: Data configuration.
        game: Game theory configuration.
        logging: Logging configuration.
        seed: Random seed for reproducibility.
        device: Device to use ('cuda', 'cpu', 'auto').
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    game: GameConfig = field(default_factory=GameConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = 'auto'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'game': asdict(self.game),
            'logging': asdict(self.logging),
            'seed': self.seed,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            Config instance.
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            game=GameConfig(**config_dict.get('game', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto')
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            path: Input file path.
            
        Returns:
            Config instance.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, updates: Dict[str, Any]) -> 'Config':
        """Update configuration with new values.
        
        Supports nested updates using dot notation:
            config.update({'model.hidden_size': 128})
        
        Args:
            updates: Dictionary of updates.
            
        Returns:
            Updated Config instance.
        """
        config_dict = self.to_dict()
        
        for key, value in updates.items():
            if '.' in key:
                # Nested update
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return Config.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Config with default values.
    """
    return Config()


def get_paper_config() -> Config:
    """Get configuration matching paper specifications.
    
    Returns paper-accurate configuration for reproducing results.
    
    Returns:
        Config matching IEEE MLSP 2025 paper.
    """
    return Config(
        model=ModelConfig(
            human_layers=12,
            robot_layers=8,
            hidden_size=100,
            input_size=18,
            output_size=9,
            dropout=0.5,
            activation='relu',
            use_batch_norm=True
        ),
        training=TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0,
            lr_decay_factor=0.1,
            lr_decay_epochs=[50, 75],
            gradient_clip=1.0,
            early_stopping_patience=10
        ),
        data=DataConfig(
            train_split=0.7,
            val_split=0.15,
            sequence_length=10,
            velocity_threshold=0.1,
            force_threshold=0.1,
            normalize='standard'
        ),
        game=GameConfig(
            kl_method='softmax',
            mi_k_neighbors=11,
            alternating_steps=1
        ),
        seed=42
    )


def get_dataset_configs() -> Dict[str, DataConfig]:
    """Get configurations for each dataset type.
    
    Returns:
        Dictionary mapping dataset names to DataConfig.
    """
    base_config = DataConfig()
    
    datasets = {
        'drag_max_stiffness_y': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'drag_max_stiffness_y'}
        ),
        'horizontal_movement_fast': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'horizontal_movement_fast'}
        ),
        'horizontal_movement_slow': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'horizontal_movement_slow'}
        ),
        'tap_hold_z_fast': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'tap_hold_z_fast'}
        ),
        'tap_hold_z_slow': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'tap_hold_z_slow'}
        ),
        'tapping_max_yz': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'tapping_max_yz'}
        ),
        'tapping_max_z': DataConfig(
            **{**asdict(base_config), 'dataset_type': 'tapping_max_z'}
        )
    }
    
    return datasets


def validate_config(config: Config) -> List[str]:
    """Validate configuration values.
    
    Args:
        config: Configuration to validate.
        
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    # Model validation
    if config.model.human_layers < 1:
        errors.append("human_layers must be >= 1")
    if config.model.robot_layers < 1:
        errors.append("robot_layers must be >= 1")
    if config.model.hidden_size < 1:
        errors.append("hidden_size must be >= 1")
    if not 0 <= config.model.dropout <= 1:
        errors.append("dropout must be in [0, 1]")
    
    # Training validation
    if config.training.epochs < 1:
        errors.append("epochs must be >= 1")
    if config.training.batch_size < 1:
        errors.append("batch_size must be >= 1")
    if config.training.learning_rate <= 0:
        errors.append("learning_rate must be > 0")
    if not 0 <= config.training.momentum <= 1:
        errors.append("momentum must be in [0, 1]")
    
    # Data validation
    if not 0 < config.data.train_split < 1:
        errors.append("train_split must be in (0, 1)")
    if not 0 < config.data.val_split < 1:
        errors.append("val_split must be in (0, 1)")
    if config.data.train_split + config.data.val_split >= 1:
        errors.append("train_split + val_split must be < 1")
    
    # Game validation
    if config.game.mi_k_neighbors < 1:
        errors.append("mi_k_neighbors must be >= 1")
    
    return errors


def merge_configs(
    base_config: Config,
    override_config: Optional[Dict[str, Any]] = None
) -> Config:
    """Merge base configuration with overrides.
    
    Args:
        base_config: Base configuration.
        override_config: Dictionary of overrides.
        
    Returns:
        Merged configuration.
    """
    if override_config is None:
        return base_config
    
    return base_config.update(override_config)


def config_from_args(args) -> Config:
    """Create configuration from command-line arguments.
    
    Args:
        args: Parsed argparse arguments.
        
    Returns:
        Config instance.
    """
    # Start with paper config
    config = get_paper_config()
    
    # Override with command-line arguments
    updates = {}
    
    if hasattr(args, 'epochs') and args.epochs is not None:
        updates['training.epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        updates['training.batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        updates['training.learning_rate'] = args.lr
    if hasattr(args, 'hidden_size') and args.hidden_size is not None:
        updates['model.hidden_size'] = args.hidden_size
    if hasattr(args, 'dropout') and args.dropout is not None:
        updates['model.dropout'] = args.dropout
    if hasattr(args, 'seed') and args.seed is not None:
        updates['seed'] = args.seed
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        updates['data.data_dir'] = args.data_dir
    if hasattr(args, 'dataset') and args.dataset is not None:
        updates['data.dataset_type'] = args.dataset
    
    return config.update(updates)


# Environment variable configuration
def config_from_env() -> Dict[str, Any]:
    """Get configuration overrides from environment variables.
    
    Environment variables:
        LEFO_SEED: Random seed
        LEFO_EPOCHS: Number of epochs
        LEFO_BATCH_SIZE: Batch size
        LEFO_LR: Learning rate
        LEFO_DATA_DIR: Data directory
        LEFO_LOG_DIR: Log directory
        LEFO_DEVICE: Device to use
    
    Returns:
        Dictionary of configuration overrides.
    """
    overrides = {}
    
    env_mapping = {
        'LEFO_SEED': ('seed', int),
        'LEFO_EPOCHS': ('training.epochs', int),
        'LEFO_BATCH_SIZE': ('training.batch_size', int),
        'LEFO_LR': ('training.learning_rate', float),
        'LEFO_DATA_DIR': ('data.data_dir', str),
        'LEFO_LOG_DIR': ('logging.log_dir', str),
        'LEFO_DEVICE': ('device', str)
    }
    
    for env_var, (config_key, type_fn) in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            overrides[config_key] = type_fn(value)
    
    return overrides
