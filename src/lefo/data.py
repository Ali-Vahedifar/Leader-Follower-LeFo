#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Data Loading and Preprocessing for Haptic Signals

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

IEEE MLSP 2025 - Istanbul, Turkey

This module implements data loading and preprocessing utilities for haptic signals:
    - HapticDataset: PyTorch Dataset for haptic interaction data
    - HapticDataLoader: Custom DataLoader with haptic-specific features
    - DeadbandProcessor: Perceptual deadband filtering
    - DataNormalizer: Normalization utilities

The datasets capture kinaesthetic interactions recorded using a Novint Falcon
haptic device within a Chai3D virtual environment, including 3D position,
velocity, and force measurements.

Dataset Types:
    1. Tapping
    2. Tap and Hold
    3. Holding
    4. Free Air Movement (Horizontal Movement)
    5. Shearing (Diagonal Contact with Ground)

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Feature column definitions
POSITION_COLS = ['position_x', 'position_y', 'position_z']
VELOCITY_COLS = ['velocity_x', 'velocity_y', 'velocity_z']
FORCE_COLS = ['force_x', 'force_y', 'force_z']

HUMAN_COLS = [f'human_{c}' for c in POSITION_COLS + VELOCITY_COLS + FORCE_COLS]
ROBOT_COLS = [f'robot_{c}' for c in POSITION_COLS + VELOCITY_COLS + FORCE_COLS]


class DeadbandProcessor:
    """
    Perceptual Deadband Processor for Haptic Signals
    
    Implements deadband filtering to enhance perceptual relevance by filtering
    out minor velocity and force variations imperceptible to human users.
    
    The deadband mechanism filters signals where:
        |signal[t] - signal[t-1]| < threshold * max(|signal|)
    
    Args:
        velocity_threshold: Deadband threshold for velocity (default: 0.1 = 10%)
        force_threshold: Deadband threshold for force (default: 0.1 = 10%)
        position_threshold: Deadband threshold for position (default: 0.05 = 5%)
    
    Example:
        >>> processor = DeadbandProcessor(velocity_threshold=0.1, force_threshold=0.1)
        >>> filtered_data = processor.apply(raw_data)
    
    Reference:
        Rodriguez Guevara et al. (2025). Zenodo dataset (DOI: 10.5281/zenodo.14924062)
    """
    
    def __init__(
        self,
        velocity_threshold: float = 0.1,
        force_threshold: float = 0.1,
        position_threshold: float = 0.05
    ):
        self.velocity_threshold = velocity_threshold
        self.force_threshold = force_threshold
        self.position_threshold = position_threshold
    
    def apply(
        self,
        data: np.ndarray,
        feature_indices: Optional[Dict[str, List[int]]] = None
    ) -> np.ndarray:
        """
        Apply deadband filtering to haptic signals.
        
        Args:
            data: Input data of shape (N, features)
            feature_indices: Dictionary mapping feature types to column indices
                           Default: {'position': [0,1,2], 'velocity': [3,4,5], 'force': [6,7,8]}
        
        Returns:
            Filtered data with same shape
        """
        if feature_indices is None:
            feature_indices = {
                'position': [0, 1, 2],
                'velocity': [3, 4, 5],
                'force': [6, 7, 8]
            }
        
        filtered = data.copy()
        
        # Apply deadband for each feature type
        thresholds = {
            'position': self.position_threshold,
            'velocity': self.velocity_threshold,
            'force': self.force_threshold
        }
        
        for feature_type, indices in feature_indices.items():
            threshold = thresholds.get(feature_type, 0.1)
            
            for idx in indices:
                signal = data[:, idx]
                max_val = np.max(np.abs(signal)) + 1e-10
                
                # Compute differences
                diff = np.abs(np.diff(signal, prepend=signal[0]))
                
                # Apply deadband: set small changes to previous value
                mask = diff < threshold * max_val
                
                for i in range(1, len(signal)):
                    if mask[i]:
                        filtered[i, idx] = filtered[i-1, idx]
        
        return filtered
    
    def compute_reduction_ratio(
        self,
        original: np.ndarray,
        filtered: np.ndarray
    ) -> float:
        """Compute the data reduction ratio after deadband filtering."""
        original_changes = np.sum(np.abs(np.diff(original, axis=0)) > 0)
        filtered_changes = np.sum(np.abs(np.diff(filtered, axis=0)) > 0)
        
        if original_changes == 0:
            return 0.0
        
        return 1.0 - (filtered_changes / original_changes)


class DataNormalizer:
    """
    Data Normalizer for Haptic Signals
    
    Supports multiple normalization strategies:
        - Standard (z-score) normalization
        - Min-Max normalization
        - Per-feature normalization
        - Global normalization
    
    Args:
        method: Normalization method ('standard', 'minmax', 'none')
        per_feature: Whether to normalize each feature independently
        clip_outliers: Whether to clip outliers before normalization
        outlier_std: Number of standard deviations for outlier clipping
    
    Example:
        >>> normalizer = DataNormalizer(method='standard')
        >>> normalizer.fit(train_data)
        >>> normalized_data = normalizer.transform(data)
    """
    
    def __init__(
        self,
        method: str = 'standard',
        per_feature: bool = True,
        clip_outliers: bool = True,
        outlier_std: float = 3.0
    ):
        self.method = method
        self.per_feature = per_feature
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        
        self.scaler = None
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """
        Fit the normalizer to the data.
        
        Args:
            data: Training data of shape (N, features)
        
        Returns:
            Self for chaining
        """
        if self.clip_outliers:
            data = self._clip_outliers(data)
        
        if self.method == 'standard':
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0) + 1e-10
        elif self.method == 'minmax':
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform
        
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if self.method == 'standard':
            return (data - self.mean_) / self.std_
        elif self.method == 'minmax':
            return (data - self.min_) / (self.max_ - self.min_ + 1e-10)
        else:
            return data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        if self.method == 'standard':
            return data * self.std_ + self.mean_
        elif self.method == 'minmax':
            return data * (self.max_ - self.min_) + self.min_
        else:
            return data
    
    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """Clip outliers based on standard deviation."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        lower = mean - self.outlier_std * std
        upper = mean + self.outlier_std * std
        
        return np.clip(data, lower, upper)
    
    def get_params(self) -> Dict[str, Any]:
        """Get normalizer parameters for saving."""
        return {
            'method': self.method,
            'mean': self.mean_,
            'std': self.std_,
            'min': self.min_,
            'max': self.max_
        }
    
    def load_params(self, params: Dict[str, Any]):
        """Load normalizer parameters."""
        self.method = params['method']
        self.mean_ = params['mean']
        self.std_ = params['std']
        self.min_ = params['min']
        self.max_ = params['max']
        self.is_fitted = True


class HapticDataset(Dataset):
    """
    PyTorch Dataset for Haptic Interaction Data
    
    Loads and preprocesses haptic data traces for Leader-Follower learning.
    Supports multiple dataset types from the Novint Falcon / Chai3D environment.
    
    Features:
        - Position (X, Y, Z): 3D position in workspace
        - Velocity (X, Y, Z): 3D velocity components
        - Force (X, Y, Z): 3D force feedback
    
    Args:
        data_path: Path to CSV data file or directory
        sequence_length: Number of past samples to use for prediction
        prediction_horizon: Number of future samples to predict
        features: List of features to use ('position', 'velocity', 'force', 'all')
        normalize: Whether to normalize data
        normalization_method: Normalization method ('standard', 'minmax')
        apply_deadband: Whether to apply deadband filtering
        deadband_threshold: Threshold for deadband (default: 0.1)
        train: Whether this is training data (affects some processing)
        transform: Optional transform to apply to samples
    
    Example:
        >>> dataset = HapticDataset(
        ...     'data/drag_max_stiffness_y.csv',
        ...     sequence_length=10,
        ...     features='all'
        ... )
        >>> human_input, robot_input, human_target, robot_target = dataset[0]
    """
    
    DATASET_TYPES = [
        'drag_max_stiffness_y',
        'horizontal_movement_fast',
        'horizontal_movement_slow',
        'tap_hold_max_stiffness_z_fast',
        'tap_hold_max_stiffness_z_slow',
        'tapping_max_stiffness_yz',
        'tapping_max_stiffness_z'
    ]
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        features: Union[str, List[str]] = 'all',
        normalize: bool = True,
        normalization_method: str = 'standard',
        apply_deadband: bool = True,
        deadband_threshold: float = 0.1,
        train: bool = True,
        transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.normalize = normalize
        self.apply_deadband = apply_deadband
        self.train = train
        self.transform = transform
        
        # Initialize processors
        self.deadband_processor = DeadbandProcessor(
            velocity_threshold=deadband_threshold,
            force_threshold=deadband_threshold
        ) if apply_deadband else None
        
        self.normalizer = DataNormalizer(
            method=normalization_method
        ) if normalize else None
        
        # Load and preprocess data
        self.human_data, self.robot_data = self._load_data()
        
        # Store original data before normalization
        self.human_data_raw = self.human_data.copy()
        self.robot_data_raw = self.robot_data.copy()
        
        # Apply preprocessing
        if self.apply_deadband:
            self.human_data = self.deadband_processor.apply(self.human_data)
            self.robot_data = self.deadband_processor.apply(self.robot_data)
        
        if self.normalize:
            # Fit on combined data
            combined = np.vstack([self.human_data, self.robot_data])
            self.normalizer.fit(combined)
            self.human_data = self.normalizer.transform(self.human_data)
            self.robot_data = self.normalizer.transform(self.robot_data)
        
        # Compute valid indices for sequence sampling
        self.valid_length = len(self.human_data) - sequence_length - prediction_horizon
        
        logger.info(f"Loaded dataset with {len(self.human_data)} samples, "
                   f"{self.valid_length} valid sequences")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file."""
        if self.data_path.is_file():
            return self._load_csv(self.data_path)
        elif self.data_path.is_dir():
            return self._load_directory(self.data_path)
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
    
    def _load_csv(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file."""
        df = pd.read_csv(path)
        
        # Try to identify column structure
        if 'human_position_x' in df.columns:
            human_cols = [c for c in df.columns if c.startswith('human_')]
            robot_cols = [c for c in df.columns if c.startswith('robot_')]
        elif 'h_pos_x' in df.columns:
            human_cols = [c for c in df.columns if c.startswith('h_')]
            robot_cols = [c for c in df.columns if c.startswith('r_')]
        else:
            # Assume first 9 columns are human, next 9 are robot
            num_cols = df.shape[1]
            human_cols = df.columns[:num_cols // 2].tolist()
            robot_cols = df.columns[num_cols // 2:].tolist()
        
        # Select features
        if self.features == 'all':
            pass  # Use all columns
        elif isinstance(self.features, list):
            # Filter columns based on feature names
            human_cols = [c for c in human_cols if any(f in c for f in self.features)]
            robot_cols = [c for c in robot_cols if any(f in c for f in self.features)]
        
        human_data = df[human_cols].values.astype(np.float32)
        robot_data = df[robot_cols].values.astype(np.float32)
        
        return human_data, robot_data
    
    def _load_directory(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from directory with multiple files."""
        human_list = []
        robot_list = []
        
        for file_path in sorted(path.glob('*.csv')):
            h, r = self._load_csv(file_path)
            human_list.append(h)
            robot_list.append(r)
        
        return np.vstack(human_list), np.vstack(robot_list)
    
    def __len__(self) -> int:
        """Return number of valid sequences."""
        return max(0, self.valid_length)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single sample for training.
        
        Returns:
            Tuple of:
                - human_input: Historical human+robot data for predicting robot (seq_len * 2, features)
                - robot_input: Historical human+robot data for predicting human (seq_len * 2, features)
                - human_target: Ground truth human signal at n+1
                - robot_target: Ground truth robot signal at n+1
        """
        # Extract sequence
        start_idx = idx
        end_idx = idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1
        
        # Historical data
        human_seq = self.human_data[start_idx:end_idx]
        robot_seq = self.robot_data[start_idx:end_idx]
        
        # Targets (next sample)
        human_target = self.human_data[target_idx]
        robot_target = self.robot_data[target_idx]
        
        # Concatenate human and robot history for input
        # Shape: (seq_len, features * 2) -> flatten to (seq_len * features * 2)
        combined_seq = np.concatenate([human_seq, robot_seq], axis=1)
        human_input = combined_seq.flatten()
        robot_input = combined_seq.flatten()  # Same input for simplicity
        
        # Convert to tensors
        human_input = torch.from_numpy(human_input).float()
        robot_input = torch.from_numpy(robot_input).float()
        human_target = torch.from_numpy(human_target).float()
        robot_target = torch.from_numpy(robot_target).float()
        
        # Apply transform if any
        if self.transform:
            human_input = self.transform(human_input)
            robot_input = self.transform(robot_input)
        
        return human_input, robot_input, human_target, robot_target
    
    def get_feature_names(self) -> List[str]:
        """Get names of features in the dataset."""
        base_names = ['position_x', 'position_y', 'position_z',
                     'velocity_x', 'velocity_y', 'velocity_z',
                     'force_x', 'force_y', 'force_z']
        return base_names
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_samples': len(self.human_data),
            'num_sequences': len(self),
            'sequence_length': self.sequence_length,
            'human_mean': np.mean(self.human_data_raw, axis=0).tolist(),
            'human_std': np.std(self.human_data_raw, axis=0).tolist(),
            'robot_mean': np.mean(self.robot_data_raw, axis=0).tolist(),
            'robot_std': np.std(self.robot_data_raw, axis=0).tolist(),
        }


class HapticDataLoader:
    """
    Custom DataLoader for Haptic Data with Additional Utilities
    
    Wraps PyTorch DataLoader with haptic-specific functionality:
        - Automatic train/validation/test splitting
        - Sequence-aware batching
        - Data augmentation options
    
    Args:
        dataset: HapticDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        seed: Random seed for reproducibility
    
    Example:
        >>> loader = HapticDataLoader(dataset, batch_size=32)
        >>> train_loader, val_loader, test_loader = loader.get_loaders()
    """
    
    def __init__(
        self,
        dataset: HapticDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits."""
        total_len = len(self.dataset)
        train_len = int(total_len * self.train_ratio)
        val_len = int(total_len * self.val_ratio)
        test_len = total_len - train_len - val_len
        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_len, val_len, test_len],
            generator=generator
        )
        
        logger.info(f"Split sizes - Train: {train_len}, Val: {val_len}, Test: {test_len}")
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test DataLoaders."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader only."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_synthetic_dataset(
    num_samples: int = 10000,
    feature_dim: int = 9,
    noise_level: float = 0.1,
    correlation: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic haptic dataset for testing.
    
    Generates correlated human-robot signals with realistic dynamics.
    
    Args:
        num_samples: Number of samples to generate
        feature_dim: Number of features (default: 9 for 3D pos, vel, force)
        noise_level: Level of noise to add
        correlation: Correlation between human and robot signals
        seed: Random seed
    
    Returns:
        Tuple of (human_data, robot_data)
    """
    np.random.seed(seed)
    
    # Generate base signals with smooth dynamics
    t = np.linspace(0, 10 * np.pi, num_samples)
    
    human_data = np.zeros((num_samples, feature_dim))
    robot_data = np.zeros((num_samples, feature_dim))
    
    for i in range(feature_dim):
        # Different frequency for each feature
        freq = 0.5 + i * 0.2
        phase = np.random.rand() * 2 * np.pi
        
        # Human signal: smooth sinusoidal with harmonics
        human_signal = np.sin(freq * t + phase) + \
                      0.3 * np.sin(2 * freq * t) + \
                      0.1 * np.sin(3 * freq * t)
        
        # Robot signal: correlated but with delay and response characteristics
        delay = int(5 + np.random.rand() * 10)
        robot_signal = correlation * np.roll(human_signal, delay) + \
                      (1 - correlation) * np.random.randn(num_samples)
        
        # Add noise
        human_data[:, i] = human_signal + noise_level * np.random.randn(num_samples)
        robot_data[:, i] = robot_signal + noise_level * np.random.randn(num_samples)
    
    return human_data.astype(np.float32), robot_data.astype(np.float32)


def download_dataset(output_dir: str = 'data/', dataset_name: str = 'all'):
    """
    Download haptic dataset from Zenodo.
    
    Args:
        output_dir: Directory to save downloaded data
        dataset_name: Name of specific dataset or 'all'
    
    Note:
        Downloads from https://zenodo.org/record/14924062
    """
    import urllib.request
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zenodo_url = "https://zenodo.org/record/14924062/files/"
    
    datasets = HapticDataset.DATASET_TYPES if dataset_name == 'all' else [dataset_name]
    
    for ds_name in datasets:
        filename = f"{ds_name}.csv"
        url = zenodo_url + filename
        output_file = output_path / filename
        
        if output_file.exists():
            logger.info(f"File already exists: {output_file}")
            continue
        
        logger.info(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_file)
            logger.info(f"Downloaded: {output_file}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    # Test data loading utilities
    print("Testing Data Loading Utilities")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    human_data, robot_data = create_synthetic_dataset(num_samples=1000)
    print(f"   Human data shape: {human_data.shape}")
    print(f"   Robot data shape: {robot_data.shape}")
    
    # Save synthetic data for testing
    test_data_path = Path('/tmp/test_haptic_data.csv')
    df = pd.DataFrame(
        np.hstack([human_data, robot_data]),
        columns=[f'human_feat_{i}' for i in range(9)] + 
                [f'robot_feat_{i}' for i in range(9)]
    )
    df.to_csv(test_data_path, index=False)
    print(f"   Saved test data to {test_data_path}")
    
    # Test HapticDataset
    print("\n2. Testing HapticDataset...")
    dataset = HapticDataset(
        test_data_path,
        sequence_length=10,
        normalize=True,
        apply_deadband=True
    )
    print(f"   Dataset length: {len(dataset)}")
    print(f"   Statistics: {dataset.get_statistics()}")
    
    # Get sample
    human_input, robot_input, human_target, robot_target = dataset[0]
    print(f"   Sample shapes:")
    print(f"     Human input: {human_input.shape}")
    print(f"     Robot input: {robot_input.shape}")
    print(f"     Human target: {human_target.shape}")
    print(f"     Robot target: {robot_target.shape}")
    
    # Test DataLoader
    print("\n3. Testing HapticDataLoader...")
    loader = HapticDataLoader(dataset, batch_size=32)
    train_loader, val_loader, test_loader = loader.get_loaders()
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test deadband processor
    print("\n4. Testing DeadbandProcessor...")
    processor = DeadbandProcessor(velocity_threshold=0.1)
    filtered = processor.apply(human_data)
    reduction = processor.compute_reduction_ratio(human_data, filtered)
    print(f"   Data reduction ratio: {reduction:.2%}")
    
    # Clean up
    test_data_path.unlink()
    
    print("\n✓ All tests passed!")
