#!/usr/bin/env python3
"""
Evaluation Script for Leader-Follower Signal Prediction.

This script evaluates trained LeFo models on haptic teleoperation data,
computing prediction accuracy, inference times, and upper bound verification.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Usage:
    python evaluate.py --model-path ./outputs/best_model.pt --data-dir ./data

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lefo.models import LeaderFollowerModel
from src.lefo.data import HapticDataLoader
from src.lefo.evaluate import (
    PredictionAccuracy,
    InferenceTimer,
    UpperBoundVerifier,
    compute_metrics
)
from src.lefo.game import MutualInformationEstimator, KLDivergenceLoss
from src.lefo.utils import load_model


def parse_args():
    """Parse command line arguments."""
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
        "--dataset-type", type=str, default="all",
        choices=["all", "drag", "horizontal_fast", "horizontal_slow",
                 "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"],
        help="Type of haptic dataset (use 'all' to evaluate on all datasets)"
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
    parser.add_argument(
        "--num-inference-runs", type=int, default=100,
        help="Number of inference runs for timing"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Prediction accuracy threshold"
    )
    parser.add_argument(
        "--verify-upper-bound", action="store_true",
        help="Verify Taylor expansion upper bound"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save raw predictions to file"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity level"
    )
    
    return parser.parse_args()


def evaluate_single_dataset(
    model: LeaderFollowerModel,
    data_loader: HapticDataLoader,
    device: torch.device,
    args
) -> dict:
    """
    Evaluate model on a single dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Computation device
        args: Command line arguments
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    accuracy_metric = PredictionAccuracy(threshold=args.threshold)
    timer = InferenceTimer()
    mi_estimator = MutualInformationEstimator()
    kl_loss = KLDivergenceLoss()
    
    all_human_preds = []
    all_robot_preds = []
    all_human_targets = []
    all_robot_targets = []
    
    # Evaluation loop
    test_loader = data_loader.test_loader
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", disable=args.verbose < 1):
            human_input = batch['human_input'].to(device)
            robot_input = batch['robot_input'].to(device)
            human_target = batch['human_target'].to(device)
            robot_target = batch['robot_target'].to(device)
            
            # Time inference
            timer.start()
            human_pred, robot_pred = model(human_input, robot_input)
            timer.stop()
            
            # Collect predictions
            all_human_preds.append(human_pred.cpu())
            all_robot_preds.append(robot_pred.cpu())
            all_human_targets.append(human_target.cpu())
            all_robot_targets.append(robot_target.cpu())
    
    # Concatenate all predictions
    human_preds = torch.cat(all_human_preds, dim=0)
    robot_preds = torch.cat(all_robot_preds, dim=0)
    human_targets = torch.cat(all_human_targets, dim=0)
    robot_targets = torch.cat(all_robot_targets, dim=0)
    
    # Compute accuracy
    human_accuracy = accuracy_metric(human_preds, human_targets)
    robot_accuracy = accuracy_metric(robot_preds, robot_targets)
    
    # Compute MI and KL divergence
    mi_value = mi_estimator(robot_targets.numpy(), robot_preds.numpy())
    kl_value = kl_loss(human_preds, human_targets).item()
    
    # Compute detailed metrics
    metrics = compute_metrics(
        human_preds.numpy(),
        human_targets.numpy(),
        robot_preds.numpy(),
        robot_targets.numpy()
    )
    
    # Inference timing statistics
    timing_stats = timer.get_statistics()
    
    # Compile results
    results = {
        'human_accuracy': float(human_accuracy),
        'robot_accuracy': float(robot_accuracy),
        'mutual_information': float(mi_value),
        'kl_divergence': float(kl_value),
        'inference_time_ms': timing_stats['mean_ms'],
        'inference_time_std_ms': timing_stats['std_ms'],
        'inference_time_min_ms': timing_stats['min_ms'],
        'inference_time_max_ms': timing_stats['max_ms'],
        'num_samples': len(human_preds),
        **metrics
    }
    
    # Upper bound verification
    if args.verify_upper_bound:
        verifier = UpperBoundVerifier()
        bound_results = verifier.verify(model, test_loader, device)
        results.update({
            'upper_bound_satisfied': bound_results['satisfied'],
            'upper_bound_value': bound_results['bound'],
            'actual_loss_diff': bound_results['actual_diff'],
            'lambda_max': bound_results['lambda_max']
        })
    
    # Save predictions if requested
    if args.save_predictions:
        results['predictions'] = {
            'human_pred': human_preds.numpy().tolist(),
            'robot_pred': robot_preds.numpy().tolist(),
            'human_target': human_targets.numpy().tolist(),
            'robot_target': robot_targets.numpy().tolist()
        }
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("LeFo: Leader-Follower Model Evaluation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print("="*60)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets to evaluate
    if args.dataset_type == "all":
        dataset_types = [
            "drag", "horizontal_fast", "horizontal_slow",
            "tap_hold_fast", "tap_hold_slow", "tapping_yz", "tapping_z"
        ]
    else:
        dataset_types = [args.dataset_type]
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_type in dataset_types:
        print(f"\nEvaluating on {dataset_type}...")
        print("-"*40)
        
        try:
            # Load data
            data_loader = HapticDataLoader(
                data_dir=args.data_dir,
                dataset_type=dataset_type,
                batch_size=args.batch_size
            )
            
            # Evaluate
            results = evaluate_single_dataset(
                model=model,
                data_loader=data_loader,
                device=device,
                args=args
            )
            
            # Print results
            print(f"Human Prediction Accuracy: {results['human_accuracy']:.2f}%")
            print(f"Robot Prediction Accuracy: {results['robot_accuracy']:.2f}%")
            print(f"Mutual Information: {results['mutual_information']:.4f}")
            print(f"KL Divergence: {results['kl_divergence']:.4f}")
            print(f"Inference Time: {results['inference_time_ms']:.2f} ± {results['inference_time_std_ms']:.2f} ms")
            
            if args.verify_upper_bound:
                status = "✓" if results['upper_bound_satisfied'] else "✗"
                print(f"Upper Bound: {status} (bound={results['upper_bound_value']:.4f}, actual={results['actual_loss_diff']:.4f})")
            
            all_results[dataset_type] = results
            
            # Save individual results
            results_file = output_dir / f"evaluation_{dataset_type}.json"
            with open(results_file, 'w') as f:
                # Remove predictions for JSON if they exist (too large)
                save_results = {k: v for k, v in results.items() if k != 'predictions'}
                json.dump(save_results, f, indent=2)
            
            # Save predictions separately if requested
            if args.save_predictions and 'predictions' in results:
                pred_file = output_dir / f"predictions_{dataset_type}.json"
                with open(pred_file, 'w') as f:
                    json.dump(results['predictions'], f)
                print(f"Predictions saved to {pred_file}")
                
        except Exception as e:
            print(f"Error evaluating {dataset_type}: {e}")
            all_results[dataset_type] = {'error': str(e)}
    
    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Dataset':<20} {'Human Acc':<12} {'Robot Acc':<12} {'Time (ms)':<12}")
    print("-"*60)
    
    for dataset_type, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_type:<20} ERROR: {results['error']}")
        else:
            print(f"{dataset_type:<20} {results['human_accuracy']:>10.2f}% {results['robot_accuracy']:>10.2f}% {results['inference_time_ms']:>10.2f}")
    
    print("="*60)
    
    # Save combined results
    combined_file = output_dir / "evaluation_summary.json"
    with open(combined_file, 'w') as f:
        # Remove predictions from combined results
        clean_results = {
            k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
            for k, v in all_results.items()
        }
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
