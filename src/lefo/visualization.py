"""
Visualization Module for Leader-Follower Signal Prediction.

This module provides plotting functions for generating paper figures:
    - Prediction accuracy bar charts
    - Inference time comparisons
    - Training curves
    - Signal prediction visualizations
    - Upper bound verification plots
    - Network architecture diagrams

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Paper-style configuration
PAPER_STYLE = {
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
}

# Color scheme
COLORS = {
    'human': '#2E86AB',       # Blue for human/leader
    'robot': '#A23B72',       # Magenta for robot/follower
    'position': '#F18F01',    # Orange for position
    'velocity': '#C73E1D',    # Red for velocity
    'force': '#3B1F2B',       # Dark for force
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'background': '#F5F5F5'
}


def set_paper_style():
    """Set matplotlib style for paper figures."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    
    plt.rcParams.update(PAPER_STYLE)
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
        sns.set_palette([COLORS['human'], COLORS['robot'], COLORS['accent']])


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Prediction Accuracy Comparison',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot prediction accuracy comparison across datasets.
    
    Args:
        results: Dictionary mapping dataset names to accuracy results.
                 Each result should have 'human_accuracy' and 'robot_accuracy'.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    datasets = list(results.keys())
    human_acc = [results[d].get('human_accuracy', 0) for d in datasets]
    robot_acc = [results[d].get('robot_accuracy', 0) for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, human_acc, width, label='Human (Leader)',
                   color=COLORS['human'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, robot_acc, width, label='Robot (Follower)',
                   color=COLORS['robot'], edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_inference_times(
    results: Dict[str, Dict[str, float]],
    feature_types: List[str] = ['position', 'velocity', 'force'],
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Inference Time by Feature Type',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot inference times across datasets and feature types.
    
    Args:
        results: Dictionary mapping dataset names to timing results.
        feature_types: List of feature types to plot.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    datasets = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [COLORS['position'], COLORS['velocity'], COLORS['force']]
    
    for i, feature in enumerate(feature_types):
        times = [results[d].get(f'{feature}_time_ms', 0) for d in datasets]
        offset = (i - len(feature_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=feature.capitalize(),
                     color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Training Curves',
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Plot training loss and accuracy curves.
    
    Args:
        history: Dictionary with training history.
                 Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train Loss',
                color=COLORS['human'], linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Val Loss',
                color=COLORS['robot'], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Mutual Information plot
    ax = axes[1]
    if 'train_mi' in history:
        ax.plot(epochs, history['train_mi'], label='Train MI',
                color=COLORS['accent'], linewidth=2)
    if 'val_mi' in history:
        ax.plot(epochs, history['val_mi'], label='Val MI',
                color=COLORS['force'], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual Information')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # KL Divergence plot
    ax = axes[2]
    if 'train_kl' in history:
        ax.plot(epochs, history['train_kl'], label='Train KL',
                color=COLORS['velocity'], linewidth=2)
    if 'val_kl' in history:
        ax.plot(epochs, history['val_kl'], label='Val KL',
                color=COLORS['position'], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_signal_prediction(
    actual: np.ndarray,
    predicted: np.ndarray,
    feature_names: Optional[List[str]] = None,
    time_axis: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Signal Prediction',
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot actual vs predicted signals.
    
    Args:
        actual: Actual signal values, shape (time_steps, features).
        predicted: Predicted signal values, shape (time_steps, features).
        feature_names: Names for each feature.
        time_axis: Time values for x-axis.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    n_features = actual.shape[1] if len(actual.shape) > 1 else 1
    if n_features == 1:
        actual = actual.reshape(-1, 1)
        predicted = predicted.reshape(-1, 1)
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    if time_axis is None:
        time_axis = np.arange(len(actual))
    
    # Determine subplot layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        ax.plot(time_axis, actual[:, i], label='Actual',
                color=COLORS['human'], linewidth=1.5, alpha=0.8)
        ax.plot(time_axis, predicted[:, i], label='Predicted',
                color=COLORS['robot'], linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Signal Value')
        ax.set_title(feature_names[i])
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_upper_bound_verification(
    actual_diffs: List[float],
    upper_bounds: List[float],
    epochs: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Upper Bound Verification',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot upper bound verification results.
    
    Args:
        actual_diffs: Actual loss differences.
        upper_bounds: Computed upper bounds.
        epochs: Epoch numbers for x-axis.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    if epochs is None:
        epochs = list(range(1, len(actual_diffs) + 1))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, actual_diffs, label='Actual Difference',
            color=COLORS['human'], linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, upper_bounds, label='Upper Bound',
            color=COLORS['robot'], linewidth=2, linestyle='--', marker='s', markersize=4)
    
    # Fill between to show bound satisfaction
    ax.fill_between(epochs, actual_diffs, upper_bounds,
                   where=[a <= b for a, b in zip(actual_diffs, upper_bounds)],
                   alpha=0.3, color='green', label='Bound Satisfied')
    ax.fill_between(epochs, actual_diffs, upper_bounds,
                   where=[a > b for a, b in zip(actual_diffs, upper_bounds)],
                   alpha=0.3, color='red', label='Bound Violated')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Difference')
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_parameter_space(
    param_values: Dict[str, np.ndarray],
    accuracies: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Parameter Space Analysis',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot parameter space heatmap.
    
    Args:
        param_values: Dictionary with parameter names and values.
        accuracies: 2D array of accuracies for parameter combinations.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    param_names = list(param_values.keys())
    if len(param_names) != 2:
        raise ValueError("Exactly 2 parameters required for heatmap")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(accuracies, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(param_values[param_names[1]])))
    ax.set_xticklabels([f'{v:.3g}' for v in param_values[param_names[1]]])
    ax.set_yticks(range(len(param_values[param_names[0]])))
    ax.set_yticklabels([f'{v:.3g}' for v in param_values[param_names[0]]])
    
    ax.set_xlabel(param_names[1])
    ax.set_ylabel(param_names[0])
    ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)')
    
    # Add text annotations
    for i in range(len(param_values[param_names[0]])):
        for j in range(len(param_values[param_names[1]])):
            text = ax.text(j, i, f'{accuracies[i, j]:.1f}',
                          ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_game_equilibrium(
    leader_utilities: List[float],
    follower_utilities: List[float],
    iterations: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Game Equilibrium Convergence',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot convergence to Stackelberg equilibrium.
    
    Args:
        leader_utilities: Leader (human) utility values.
        follower_utilities: Follower (robot) utility values.
        iterations: Iteration numbers for x-axis.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    if iterations is None:
        iterations = list(range(1, len(leader_utilities) + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Utility evolution
    ax = axes[0]
    ax.plot(iterations, leader_utilities, label='Leader (Human)',
            color=COLORS['human'], linewidth=2)
    ax.plot(iterations, follower_utilities, label='Follower (Robot)',
            color=COLORS['robot'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Utility')
    ax.set_title('Utility Evolution')
    ax.legend()
    
    # Phase space
    ax = axes[1]
    ax.plot(leader_utilities, follower_utilities, color=COLORS['accent'],
            linewidth=1.5, alpha=0.7)
    ax.scatter(leader_utilities[0], follower_utilities[0], color='green',
              s=100, zorder=5, label='Start', marker='o')
    ax.scatter(leader_utilities[-1], follower_utilities[-1], color='red',
              s=100, zorder=5, label='End', marker='*')
    ax.set_xlabel('Leader Utility')
    ax.set_ylabel('Follower Utility')
    ax.set_title('Phase Space')
    ax.legend()
    
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_network_architecture(
    layer_sizes: List[int],
    network_name: str = 'Neural Network',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot neural network architecture diagram.
    
    Args:
        layer_sizes: List of layer sizes.
        network_name: Name of the network.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib Figure object.
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    layer_positions = np.linspace(0.1, 0.9, n_layers)
    
    for layer_idx, (x_pos, n_neurons) in enumerate(zip(layer_positions, layer_sizes)):
        # Limit displayed neurons for readability
        display_neurons = min(n_neurons, 10)
        
        if n_neurons <= 10:
            y_positions = np.linspace(0.2, 0.8, n_neurons)
        else:
            y_positions = np.linspace(0.2, 0.8, 10)
        
        for i, y_pos in enumerate(y_positions):
            # Draw neuron
            color = COLORS['human'] if layer_idx == 0 else \
                   COLORS['robot'] if layer_idx == n_layers - 1 else \
                   COLORS['accent']
            
            circle = plt.Circle((x_pos, y_pos), 0.015, color=color,
                                ec='black', linewidth=0.5)
            ax.add_patch(circle)
            
            # Draw connections to next layer
            if layer_idx < n_layers - 1:
                next_display = min(layer_sizes[layer_idx + 1], 10)
                next_y = np.linspace(0.2, 0.8, next_display)
                for ny in next_y:
                    ax.plot([x_pos + 0.015, layer_positions[layer_idx + 1] - 0.015],
                           [y_pos, ny], color='gray', alpha=0.1, linewidth=0.3)
        
        # Layer label
        ax.text(x_pos, 0.1, f'{n_neurons}', ha='center', va='top', fontsize=10)
        
        if layer_idx == 0:
            ax.text(x_pos, 0.92, 'Input', ha='center', va='bottom', fontsize=11)
        elif layer_idx == n_layers - 1:
            ax.text(x_pos, 0.92, 'Output', ha='center', va='bottom', fontsize=11)
        else:
            ax.text(x_pos, 0.92, f'Hidden {layer_idx}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(network_name, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def create_all_paper_figures(
    results: Dict[str, Dict],
    history: Dict[str, List[float]],
    output_dir: Union[str, Path],
    format: str = 'pdf'
) -> None:
    """Create all figures for the paper.
    
    Args:
        results: Dictionary with evaluation results per dataset.
        history: Training history dictionary.
        output_dir: Output directory for figures.
        format: Output format ('pdf', 'png', 'svg').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Accuracy comparison
    plot_accuracy_comparison(
        results,
        save_path=output_dir / f'fig1_accuracy_comparison.{format}',
        title='Prediction Accuracy Across Datasets'
    )
    
    # Figure 2: Inference times
    plot_inference_times(
        results,
        save_path=output_dir / f'fig2_inference_times.{format}',
        title='Inference Time by Feature Type'
    )
    
    # Figure 3: Training curves
    plot_training_curves(
        history,
        save_path=output_dir / f'fig3_training_curves.{format}',
        title='Training Progress'
    )
    
    # Figure 4: Network architectures
    plot_network_architecture(
        [18] + [100] * 12 + [9],
        network_name='Human (Leader) Network - 12 Layers',
        save_path=output_dir / f'fig4a_human_network.{format}'
    )
    
    plot_network_architecture(
        [18] + [100] * 8 + [9],
        network_name='Robot (Follower) Network - 8 Layers',
        save_path=output_dir / f'fig4b_robot_network.{format}'
    )
    
    print(f"All figures saved to {output_dir}")
