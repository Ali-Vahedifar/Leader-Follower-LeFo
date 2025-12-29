#!/usr/bin/env python3
"""
Real-Time Inference Script for Leader-Follower Signal Prediction.

This script performs real-time inference with trained LeFo models,
supporting both batch processing and interactive modes.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Usage:
    python inference.py --model-path ./model.pt --input-file data.csv
    python inference.py --model-path ./model.pt --interactive

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lefo.models import LeaderFollowerModel
from src.lefo.data import DataNormalizer, DeadbandProcessor
from src.lefo.utils import load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained Leader-Follower model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
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
        "--stream", action="store_true",
        help="Stream predictions in real-time (requires --input-file)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--apply-deadband", action="store_true",
        help="Apply perceptual deadband filtering"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Apply normalization to inputs"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference benchmark"
    )
    parser.add_argument(
        "--num-benchmark-runs", type=int, default=1000,
        help="Number of runs for benchmarking"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity level"
    )
    
    return parser.parse_args()


def interactive_mode(model, device, args):
    """Run inference in interactive mode."""
    print("\n" + "="*60)
    print("LeFo Interactive Inference Mode")
    print("="*60)
    print("Enter haptic signals as comma-separated values.")
    print("Format: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'help' for more information.")
    print("="*60 + "\n")
    
    # Initialize processors if needed
    normalizer = DataNormalizer() if args.normalize else None
    deadband = DeadbandProcessor() if args.apply_deadband else None
    
    while True:
        try:
            user_input = input("Human signal > ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break
            
            if user_input.lower() == 'help':
                print("\nHelp:")
                print("  - Enter 9 comma-separated values for human signal")
                print("  - Values: position (x,y,z), velocity (x,y,z), force (x,y,z)")
                print("  - Example: 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 1.0, 1.1, 1.2")
                print("  - Commands: 'quit', 'exit', 'help', 'benchmark'\n")
                continue
            
            if user_input.lower() == 'benchmark':
                run_quick_benchmark(model, device)
                continue
            
            # Parse input
            try:
                human_values = [float(x.strip()) for x in user_input.split(',')]
            except ValueError as e:
                print(f"Error parsing input: {e}")
                print("Please enter 9 comma-separated numbers.")
                continue
            
            if len(human_values) != 9:
                print(f"Error: Expected 9 values, got {len(human_values)}")
                continue
            
            # Get robot signal
            robot_input = input("Robot signal > ").strip()
            try:
                robot_values = [float(x.strip()) for x in robot_input.split(',')]
            except ValueError as e:
                print(f"Error parsing robot input: {e}")
                continue
            
            if len(robot_values) != 9:
                print(f"Error: Expected 9 values, got {len(robot_values)}")
                continue
            
            # Convert to tensors
            human_tensor = torch.tensor(human_values, dtype=torch.float32).unsqueeze(0)
            robot_tensor = torch.tensor(robot_values, dtype=torch.float32).unsqueeze(0)
            
            # Apply preprocessing if enabled
            if deadband:
                human_tensor = deadband(human_tensor)
                robot_tensor = deadband(robot_tensor)
            
            if normalizer:
                human_tensor = normalizer.transform(human_tensor)
                robot_tensor = normalizer.transform(robot_tensor)
            
            # Move to device
            human_tensor = human_tensor.to(device)
            robot_tensor = robot_tensor.to(device)
            
            # Inference
            start_time = time.perf_counter()
            with torch.no_grad():
                human_pred, robot_pred = model(human_tensor, robot_tensor)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Display results
            human_pred_np = human_pred.cpu().numpy().flatten()
            robot_pred_np = robot_pred.cpu().numpy().flatten()
            
            print(f"\nPredictions (inference time: {inference_time:.2f} ms):")
            print(f"  Human (Leader) prediction:")
            print(f"    Position: [{human_pred_np[0]:.4f}, {human_pred_np[1]:.4f}, {human_pred_np[2]:.4f}]")
            print(f"    Velocity: [{human_pred_np[3]:.4f}, {human_pred_np[4]:.4f}, {human_pred_np[5]:.4f}]")
            print(f"    Force:    [{human_pred_np[6]:.4f}, {human_pred_np[7]:.4f}, {human_pred_np[8]:.4f}]")
            print(f"  Robot (Follower) prediction:")
            print(f"    Position: [{robot_pred_np[0]:.4f}, {robot_pred_np[1]:.4f}, {robot_pred_np[2]:.4f}]")
            print(f"    Velocity: [{robot_pred_np[3]:.4f}, {robot_pred_np[4]:.4f}, {robot_pred_np[5]:.4f}]")
            print(f"    Force:    [{robot_pred_np[6]:.4f}, {robot_pred_np[7]:.4f}, {robot_pred_np[8]:.4f}]")
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(model, device, args):
    """Run batch inference on a file."""
    print(f"Processing {args.input_file}...")
    
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Identify columns
    human_cols = [c for c in df.columns if 'human' in c.lower()]
    robot_cols = [c for c in df.columns if 'robot' in c.lower()]
    
    if not human_cols:
        # Try default column names
        human_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'force_x', 'force_y', 'force_z']
        robot_cols = human_cols  # Same features for robot
        
        if not all(c in df.columns for c in human_cols):
            print("Error: Could not identify data columns.")
            print(f"Available columns: {df.columns.tolist()}")
            return
    
    # Prepare data
    human_data = torch.tensor(df[human_cols].values, dtype=torch.float32)
    robot_data = torch.tensor(df[robot_cols].values, dtype=torch.float32)
    
    # Initialize processors if needed
    if args.apply_deadband:
        deadband = DeadbandProcessor()
        human_data = deadband(human_data)
        robot_data = deadband(robot_data)
    
    if args.normalize:
        normalizer = DataNormalizer()
        normalizer.fit(human_data)
        human_data = normalizer.transform(human_data)
        robot_data = normalizer.transform(robot_data)
    
    # Move to device
    human_data = human_data.to(device)
    robot_data = robot_data.to(device)
    
    # Inference
    print(f"Running inference on {len(human_data)} samples...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        human_pred, robot_pred = model(human_data, robot_data)
    
    total_time = (time.perf_counter() - start_time) * 1000
    avg_time = total_time / len(human_data)
    
    print(f"Total inference time: {total_time:.2f} ms")
    print(f"Average per sample: {avg_time:.4f} ms")
    
    # Denormalize if needed
    if args.normalize:
        human_pred = normalizer.inverse_transform(human_pred.cpu())
        robot_pred = normalizer.inverse_transform(robot_pred.cpu())
    else:
        human_pred = human_pred.cpu()
        robot_pred = robot_pred.cpu()
    
    # Save predictions
    human_pred_np = human_pred.numpy()
    robot_pred_np = robot_pred.numpy()
    
    # Create output dataframe
    output_df = pd.DataFrame()
    
    for i, col in enumerate(['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'force_x', 'force_y', 'force_z']):
        output_df[f'human_pred_{col}'] = human_pred_np[:, i]
        output_df[f'robot_pred_{col}'] = robot_pred_np[:, i]
    
    output_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")


def stream_inference(model, device, args):
    """Stream predictions in real-time."""
    print(f"Streaming predictions from {args.input_file}...")
    print("Press Ctrl+C to stop.\n")
    
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Identify columns (simplified)
    n_features = 9
    
    try:
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Extract features (assume first 9 are human, next 9 are robot)
            human_values = row.values[:n_features].astype(np.float32)
            robot_values = row.values[n_features:2*n_features].astype(np.float32) if len(row) >= 2*n_features else human_values
            
            # Convert to tensors
            human_tensor = torch.tensor(human_values).unsqueeze(0).to(device)
            robot_tensor = torch.tensor(robot_values).unsqueeze(0).to(device)
            
            # Inference
            start_time = time.perf_counter()
            with torch.no_grad():
                human_pred, robot_pred = model(human_tensor, robot_tensor)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Display
            print(f"[{idx:5d}] Human: {human_pred.cpu().numpy().flatten()[:3]} | "
                  f"Robot: {robot_pred.cpu().numpy().flatten()[:3]} | "
                  f"Time: {inference_time:.2f}ms")
            
            # Simulate real-time (adjust delay as needed)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStreaming stopped.")


def run_benchmark(model, device, args):
    """Run inference benchmark."""
    print("\n" + "="*60)
    print("LeFo Inference Benchmark")
    print("="*60)
    
    # Create random input data
    batch_sizes = [1, 8, 16, 32, 64, 128]
    input_dim = 9
    
    results = []
    
    for batch_size in batch_sizes:
        human_data = torch.randn(batch_size, input_dim).to(device)
        robot_data = torch.randn(batch_size, input_dim).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(human_data, robot_data)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(args.num_benchmark_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(human_data, robot_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        
        result = {
            'batch_size': batch_size,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput': batch_size / (np.mean(times) / 1000)
        }
        results.append(result)
        
        print(f"Batch {batch_size:4d}: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms "
              f"({result['throughput']:.0f} samples/sec)")
    
    print("="*60)
    
    # Save results
    benchmark_file = Path(args.output_file).parent / "benchmark_results.json"
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to {benchmark_file}")
    
    return results


def run_quick_benchmark(model, device, n_runs=100):
    """Run a quick benchmark."""
    human_data = torch.randn(1, 9).to(device)
    robot_data = torch.randn(1, 9).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(human_data, robot_data)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(human_data, robot_data)
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    print(f"\nQuick Benchmark ({n_runs} runs):")
    print(f"  Mean: {np.mean(times):.3f} ms")
    print(f"  Std:  {np.std(times):.3f} ms")
    print(f"  Min:  {np.min(times):.3f} ms")
    print(f"  Max:  {np.max(times):.3f} ms\n")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("LeFo: Leader-Follower Signal Prediction Inference")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print("="*60)
    
    # Load model
    print(f"Loading model...")
    model, config = load_model(args.model_path, device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Run appropriate mode
    if args.benchmark:
        run_benchmark(model, device, args)
    elif args.interactive:
        interactive_mode(model, device, args)
    elif args.stream and args.input_file:
        stream_inference(model, device, args)
    elif args.input_file:
        batch_inference(model, device, args)
    else:
        print("\nError: Specify --input-file, --interactive, or --benchmark mode")
        print("Run with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
