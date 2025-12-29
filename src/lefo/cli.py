#!/usr/bin/env python3
"""
Command-Line Interface for LeFo Framework.

This module provides CLI entry points for training, evaluation, and inference
of Leader-Follower signal prediction models.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch


def train_cli() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train Leader-Follower signal prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing haptic dataset CSV files"
    )
    parser.add_argument(
        "--dataset-type", type=str, default="drag",
        choices=["drag", "horizontal_fast", "horizontal_slow",
                 "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"],
        help="Type of haptic dataset to use"
    )
    
    # Model arguments
    parser.add_argument(
        "--human-layers", type=int, default=12,
        help="Number of layers in human (Leader) network"
    )
    parser.add_argument(
        "--robot-layers", type=int, default=8,
        help="Number of layers in robot (Follower) network"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=100,
        help="Hidden dimension size for all layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability during training"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="SGD momentum"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="L2 regularization weight decay"
    )
    parser.add_argument(
        "--lr-decay-epochs", type=int, nargs="+", default=[50, 75],
        help="Epochs at which to decay learning rate"
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.1,
        help="Factor to multiply learning rate at decay epochs"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory to save model checkpoints and logs"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="lefo_experiment",
        help="Name for this experiment (used in output filenames)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity level"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup for --help
    from .config import TrainingConfig, ModelConfig, DataConfig
    from .train import train_model
    from .data import HapticDataLoader
    from .models import create_model
    from .utils import set_seed, setup_logging
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.verbose)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create configurations
    model_config = ModelConfig(
        human_layers=args.human_layers,
        robot_layers=args.robot_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    data_config = DataConfig(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_decay_epochs=args.lr_decay_epochs,
        lr_decay_factor=args.lr_decay_factor
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading dataset from {args.data_dir}...")
    data_loader = HapticDataLoader(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Train
    print("Starting training...")
    history = train_model(
        model=model,
        data_loader=data_loader,
        config=training_config,
        device=device,
        output_dir=output_dir,
        experiment_name=args.experiment_name
    )
    
    print(f"Training complete. Results saved to {output_dir}")


def evaluate_cli() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Leader-Follower model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing test dataset"
    )
    parser.add_argument(
        "--dataset-type", type=str, default="drag",
        choices=["drag", "horizontal_fast", "horizontal_slow",
                 "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"],
        help="Type of haptic dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )
    
    args = parser.parse_args()
    
    from .evaluate import evaluate_model, PredictionAccuracy, InferenceTimer
    from .models import LeaderFollowerModel
    from .data import HapticDataLoader
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model = LeaderFollowerModel(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim'],
        human_layers=checkpoint.get('human_layers', 12),
        robot_layers=checkpoint.get('robot_layers', 8),
        hidden_dim=checkpoint.get('hidden_dim', 100)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading test data from {args.data_dir}...")
    data_loader = HapticDataLoader(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size
    )
    
    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(model, data_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Human Prediction Accuracy: {results['human_accuracy']:.2f}%")
    print(f"Robot Prediction Accuracy: {results['robot_accuracy']:.2f}%")
    print(f"Average Inference Time: {results['inference_time_ms']:.2f} ms")
    print(f"Upper Bound Satisfied: {results['upper_bound_satisfied']}")
    print("="*50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results_file = output_dir / f"evaluation_{args.dataset_type}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")


def inference_cli() -> None:
    """CLI entry point for real-time inference."""
    parser = argparse.ArgumentParser(
        description="Run real-time inference with trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input-file", type=str,
        help="Input CSV file for batch inference"
    )
    parser.add_argument(
        "--output-file", type=str, default="predictions.csv",
        help="Output file for predictions"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode for real-time input"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    from .models import LeaderFollowerModel
    from .utils import load_model
    import numpy as np
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model_path, device)
    model.eval()
    
    if args.interactive:
        print("Interactive mode. Enter comma-separated values (9 features).")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("Input > ").strip()
                if user_input.lower() == 'quit':
                    break
                
                values = [float(x) for x in user_input.split(',')]
                if len(values) != 9:
                    print("Error: Expected 9 values (position, velocity, force)")
                    continue
                
                x = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    human_pred, robot_pred = model(x, x)
                
                print(f"Human prediction: {human_pred.cpu().numpy().flatten()}")
                print(f"Robot prediction: {robot_pred.cpu().numpy().flatten()}\n")
                
            except ValueError as e:
                print(f"Error parsing input: {e}")
            except KeyboardInterrupt:
                break
        
        print("Exiting.")
    
    elif args.input_file:
        import pandas as pd
        
        print(f"Processing {args.input_file}...")
        df = pd.read_csv(args.input_file)
        
        # Assume standard column format
        human_cols = [c for c in df.columns if 'human' in c.lower()]
        robot_cols = [c for c in df.columns if 'robot' in c.lower()]
        
        human_data = torch.tensor(df[human_cols].values, dtype=torch.float32).to(device)
        robot_data = torch.tensor(df[robot_cols].values, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            human_pred, robot_pred = model(human_data, robot_data)
        
        # Save predictions
        results = pd.DataFrame({
            **{f'human_pred_{i}': human_pred[:, i].cpu().numpy() 
               for i in range(human_pred.shape[1])},
            **{f'robot_pred_{i}': robot_pred[:, i].cpu().numpy() 
               for i in range(robot_pred.shape[1])}
        })
        results.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    
    else:
        print("Error: Specify --input-file or --interactive mode")
        sys.exit(1)


def download_cli() -> None:
    """CLI entry point for dataset download."""
    parser = argparse.ArgumentParser(
        description="Download haptic datasets from Zenodo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="Directory to save downloaded datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "drag", "horizontal_fast", "horizontal_slow",
                 "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"],
        help="Which dataset to download"
    )
    
    args = parser.parse_args()
    
    from .data import download_dataset
    
    print(f"Downloading datasets to {args.output_dir}...")
    download_dataset(args.output_dir, args.dataset)
    print("Download complete!")


def visualize_cli() -> None:
    """CLI entry point for result visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize training results and predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./figures",
        help="Directory to save generated figures"
    )
    parser.add_argument(
        "--format", type=str, default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for figures"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for raster formats"
    )
    
    args = parser.parse_args()
    
    from .visualization import (
        plot_training_history,
        plot_prediction_accuracy,
        plot_inference_times,
        plot_upper_bound_verification
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating figures in {output_dir}...")
    
    # Load and visualize results
    import json
    results_files = list(Path(args.results_dir).glob("*.json"))
    
    for results_file in results_files:
        with open(results_file) as f:
            results = json.load(f)
        
        dataset_name = results_file.stem.replace("evaluation_", "")
        
        # Generate figures
        plot_prediction_accuracy(
            results,
            save_path=output_dir / f"accuracy_{dataset_name}.{args.format}",
            dpi=args.dpi
        )
    
    print(f"Figures saved to {output_dir}")


def main() -> None:
    """Main entry point that dispatches to subcommands."""
    parser = argparse.ArgumentParser(
        description="LeFo: Leader-Follower Signal Prediction Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lefo train --data-dir ./data --epochs 100
  lefo evaluate --model-path ./model.pt --data-dir ./data
  lefo inference --model-path ./model.pt --interactive
  lefo download --output-dir ./data
  lefo visualize --results-dir ./results

For more information, visit: https://github.com/mavahedifar/lefo-tactile-internet
        """
    )
    
    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.set_defaults(func=train_cli)
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.set_defaults(func=evaluate_cli)
    
    # Inference subcommand
    infer_parser = subparsers.add_parser("inference", help="Run inference")
    infer_parser.set_defaults(func=inference_cli)
    
    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.set_defaults(func=download_cli)
    
    # Visualize subcommand
    viz_parser = subparsers.add_parser("visualize", help="Visualize results")
    viz_parser.set_defaults(func=visualize_cli)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to appropriate function
    if args.command == "train":
        train_cli()
    elif args.command == "evaluate":
        evaluate_cli()
    elif args.command == "inference":
        inference_cli()
    elif args.command == "download":
        download_cli()
    elif args.command == "visualize":
        visualize_cli()


if __name__ == "__main__":
    main()
