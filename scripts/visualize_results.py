#!/usr/bin/env python3
"""
Visualization Script for LeFo Results.

Generates publication-quality figures for the IEEE MLSP 2025 paper
on Leader-Follower signal prediction.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Usage:
    python visualize_results.py --results-dir ./results --output-dir ./figures

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# IEEE paper style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),  # IEEE single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'human': '#2ecc71',  # Green
    'robot': '#3498db',  # Blue
    'mi': '#e74c3c',     # Red
    'kl': '#9b59b6',     # Purple
    'bound': '#f39c12',  # Orange
}

# Dataset display names
DATASET_NAMES = {
    'drag': 'Drag Max Y',
    'horizontal_fast': 'Horiz. Fast',
    'horizontal_slow': 'Horiz. Slow',
    'tap_hold_fast': 'Tap Hold Fast',
    'tap_hold_slow': 'Tap Hold Slow',
    'tapping_yz': 'Tapping Y-Z',
    'tapping_z': 'Tapping Z'
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualization figures for LeFo results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing evaluation results JSON files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./figures",
        help="Directory to save generated figures"
    )
    parser.add_argument(
        "--format", type=str, default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output format for figures"
    )
    parser.add_argument(
        "--history-file", type=str, default=None,
        help="Training history JSON file for loss curves"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all available figures"
    )
    
    return parser.parse_args()


def load_results(results_dir):
    """Load all evaluation results from a directory."""
    results_dir = Path(results_dir)
    results = {}
    
    for results_file in results_dir.glob("evaluation_*.json"):
        dataset_name = results_file.stem.replace("evaluation_", "")
        with open(results_file) as f:
            results[dataset_name] = json.load(f)
    
    return results


def plot_prediction_accuracy(results, output_path):
    """
    Plot prediction accuracy comparison across datasets.
    
    Creates a grouped bar chart comparing human and robot prediction
    accuracy for each dataset type.
    """
    datasets = list(results.keys())
    human_acc = [results[d]['human_accuracy'] for d in datasets]
    robot_acc = [results[d]['robot_accuracy'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    bars1 = ax.bar(x - width/2, human_acc, width, label='Human (Leader)',
                   color=COLORS['human'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, robot_acc, width, label='Robot (Follower)',
                   color=COLORS['robot'], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([60, 100])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_inference_times(results, output_path):
    """
    Plot inference time comparison across datasets.
    
    Creates a bar chart with error bars showing inference time
    statistics for each dataset.
    """
    datasets = list(results.keys())
    times = [results[d]['inference_time_ms'] for d in datasets]
    stds = [results[d].get('inference_time_std_ms', 0) for d in datasets]
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    x = np.arange(len(datasets))
    bars = ax.bar(x, times, yerr=stds, capsize=3,
                  color='#3498db', edgecolor='black', linewidth=0.5,
                  error_kw={'linewidth': 0.5})
    
    ax.set_ylabel('Inference Time (ms)')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets], rotation=45, ha='right')
    
    # Add Tactile Internet latency threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='TI Threshold (1 ms)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_history(history, output_path):
    """
    Plot training loss curves.
    
    Creates a multi-panel figure showing training and validation
    loss, accuracy, MI, and KL divergence over epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', color=COLORS['human'])
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Val', color=COLORS['robot'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # Accuracy
    ax = axes[0, 1]
    if 'train_accuracy' in history:
        ax.plot(epochs, history['train_accuracy'], label='Train', color=COLORS['human'])
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], label='Val', color=COLORS['robot'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Prediction Accuracy')
    ax.legend()
    
    # Mutual Information
    ax = axes[1, 0]
    if 'train_mi' in history:
        ax.plot(epochs, history['train_mi'], color=COLORS['mi'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MI (nats)')
    ax.set_title('Mutual Information')
    
    # KL Divergence
    ax = axes[1, 1]
    if 'train_kl' in history:
        ax.plot(epochs, history['train_kl'], color=COLORS['kl'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_upper_bound_verification(results, output_path):
    """
    Plot upper bound verification results.
    
    Creates a scatter plot comparing actual loss difference to
    the Taylor expansion upper bound.
    """
    datasets = []
    bounds = []
    actuals = []
    
    for dataset, data in results.items():
        if 'upper_bound_value' in data:
            datasets.append(dataset)
            bounds.append(data['upper_bound_value'])
            actuals.append(data['actual_loss_diff'])
    
    if not datasets:
        print("No upper bound data available")
        return
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, bounds, width, label='Upper Bound',
           color=COLORS['bound'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, actuals, width, label='Actual Diff',
           color=COLORS['mi'], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Loss Difference')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets], rotation=45, ha='right')
    ax.legend()
    ax.set_title('Upper Bound Verification')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_game_theory_visualization(output_path):
    """
    Create visualization of the Stackelberg game structure.
    
    Illustrates the Leader-Follower relationship in the
    game-theoretic framework.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Draw nodes
    human_pos = (0.25, 0.7)
    robot_pos = (0.75, 0.7)
    game_pos = (0.5, 0.3)
    
    circle_radius = 0.12
    
    # Human (Leader) node
    human_circle = plt.Circle(human_pos, circle_radius, color=COLORS['human'],
                              ec='black', linewidth=1.5)
    ax.add_patch(human_circle)
    ax.text(human_pos[0], human_pos[1], 'Human\n(Leader)', ha='center', va='center',
            fontweight='bold', fontsize=10)
    
    # Robot (Follower) node
    robot_circle = plt.Circle(robot_pos, circle_radius, color=COLORS['robot'],
                              ec='black', linewidth=1.5)
    ax.add_patch(robot_circle)
    ax.text(robot_pos[0], robot_pos[1], 'Robot\n(Follower)', ha='center', va='center',
            fontweight='bold', fontsize=10)
    
    # Game equilibrium node
    game_circle = plt.Circle(game_pos, circle_radius, color=COLORS['bound'],
                             ec='black', linewidth=1.5)
    ax.add_patch(game_circle)
    ax.text(game_pos[0], game_pos[1], 'Stackelberg\nEquilibrium', ha='center', va='center',
            fontweight='bold', fontsize=9)
    
    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)
    
    # Human to Game
    ax.annotate('', xy=(game_pos[0] - 0.08, game_pos[1] + circle_radius),
                xytext=(human_pos[0] + 0.05, human_pos[1] - circle_radius),
                arrowprops=arrow_style)
    ax.text(0.28, 0.5, 'max MI', fontsize=9, color=COLORS['mi'])
    
    # Robot to Game
    ax.annotate('', xy=(game_pos[0] + 0.08, game_pos[1] + circle_radius),
                xytext=(robot_pos[0] - 0.05, robot_pos[1] - circle_radius),
                arrowprops=arrow_style)
    ax.text(0.68, 0.5, 'min KL', fontsize=9, color=COLORS['kl'])
    
    # Human to Robot (Leader moves first)
    ax.annotate('', xy=(robot_pos[0] - circle_radius - 0.02, robot_pos[1]),
                xytext=(human_pos[0] + circle_radius + 0.02, human_pos[1]),
                arrowprops=dict(arrowstyle='->', color='gray', linewidth=2,
                               connectionstyle='arc3,rad=0.2'))
    ax.text(0.5, 0.85, 'Leader moves first', ha='center', fontsize=9, color='gray')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stackelberg Game Framework', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_network_architecture(output_path):
    """
    Create visualization of the neural network architecture.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    
    # Human network (12 layers)
    ax = axes[0]
    layers = 12
    for i in range(layers):
        y = 1 - (i + 0.5) / layers
        rect = plt.Rectangle((0.3, y - 0.03), 0.4, 0.06,
                              facecolor=COLORS['human'], edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.5, y, f'FC-100', ha='center', va='center', fontsize=7)
    
    ax.text(0.5, 1.08, 'Human Network', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 1.02, '(Leader, 12 layers)', ha='center', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.axis('off')
    
    # Robot network (8 layers)
    ax = axes[1]
    layers = 8
    for i in range(layers):
        y = 1 - (i + 0.5) / layers * 0.67 - 0.165
        rect = plt.Rectangle((0.3, y - 0.03), 0.4, 0.06,
                              facecolor=COLORS['robot'], edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.5, y, f'FC-100', ha='center', va='center', fontsize=7)
    
    ax.text(0.5, 1.08, 'Robot Network', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 1.02, '(Follower, 8 layers)', ha='center', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_signal_prediction_example(output_path):
    """
    Create example signal prediction visualization.
    """
    # Generate synthetic example data
    np.random.seed(42)
    t = np.linspace(0, 2, 200)
    
    # True signals
    human_true = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
    robot_true = 0.8 * np.sin(2 * np.pi * t + 0.2) + 0.2 * np.sin(6 * np.pi * t + 0.1)
    
    # Predictions (with some error)
    human_pred = human_true + np.random.normal(0, 0.1, len(t))
    robot_pred = robot_true + np.random.normal(0, 0.15, len(t))
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    
    # Human signal
    ax = axes[0]
    ax.plot(t, human_true, 'k-', label='Ground Truth', linewidth=1.5)
    ax.plot(t, human_pred, '--', color=COLORS['human'], label='Prediction', linewidth=1)
    ax.set_ylabel('Human Signal')
    ax.legend(loc='upper right')
    ax.set_title('Signal Prediction Example')
    
    # Robot signal
    ax = axes[1]
    ax.plot(t, robot_true, 'k-', label='Ground Truth', linewidth=1.5)
    ax.plot(t, robot_pred, '--', color=COLORS['robot'], label='Prediction', linewidth=1)
    ax.set_ylabel('Robot Signal')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main visualization function."""
    args = parse_args()
    
    print("="*60)
    print("LeFo Results Visualization")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        print("Generating example figures with synthetic data...")
    
    fmt = args.format
    
    # Generate figures
    print("\nGenerating figures...")
    
    if results:
        plot_prediction_accuracy(results, output_dir / f"prediction_accuracy.{fmt}")
        plot_inference_times(results, output_dir / f"inference_times.{fmt}")
        plot_upper_bound_verification(results, output_dir / f"upper_bound.{fmt}")
    
    # Training history
    if args.history_file:
        with open(args.history_file) as f:
            history = json.load(f)
        plot_training_history(history, output_dir / f"training_history.{fmt}")
    
    # Always generate these
    plot_game_theory_visualization(output_dir / f"game_theory.{fmt}")
    plot_network_architecture(output_dir / f"network_architecture.{fmt}")
    plot_signal_prediction_example(output_dir / f"signal_prediction.{fmt}")
    
    print("\n" + "="*60)
    print(f"All figures saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
