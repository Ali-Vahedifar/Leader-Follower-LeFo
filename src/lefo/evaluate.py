"""
Evaluation Metrics for Leader-Follower Signal Prediction.

This module implements evaluation metrics for the Leader-Follower game-theoretic
framework as described in the IEEE MLSP 2025 paper.

Metrics include:
    - Prediction Accuracy (Accuracy_n = 1 - |Ŝ_n - S_n| / max(S_n))
    - Inference Time measurement
    - Upper Bound Verification for Taylor expansion
    - Mutual Information and KL Divergence evaluation

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark
Conference: IEEE MLSP 2025, Istanbul, Turkey

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .game import KLDivergenceLoss, MutualInformationEstimator, TaylorExpansionUpperBound
from .models import LeaderFollowerModel


@dataclass
class EvaluationResults:
    """Container for evaluation results.
    
    Attributes:
        human_accuracy: Prediction accuracy for human (Leader) network.
        robot_accuracy: Prediction accuracy for robot (Follower) network.
        human_inference_time_ms: Inference time for human network in milliseconds.
        robot_inference_time_ms: Inference time for robot network in milliseconds.
        mutual_information: Estimated mutual information I(S_R, Ŝ_R).
        kl_divergence: KL divergence KL(S_H || Ŝ_H).
        upper_bound_satisfied: Whether Taylor expansion upper bound is satisfied.
        upper_bound_ratio: Ratio of actual loss difference to upper bound.
        per_feature_accuracy: Dictionary of accuracy per feature type.
        detailed_metrics: Additional detailed metrics.
    """
    human_accuracy: float = 0.0
    robot_accuracy: float = 0.0
    human_inference_time_ms: float = 0.0
    robot_inference_time_ms: float = 0.0
    mutual_information: float = 0.0
    kl_divergence: float = 0.0
    upper_bound_satisfied: bool = True
    upper_bound_ratio: float = 0.0
    per_feature_accuracy: Dict[str, float] = field(default_factory=dict)
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Union[float, bool, Dict]]:
        """Convert results to dictionary format."""
        return {
            'human_accuracy': self.human_accuracy,
            'robot_accuracy': self.robot_accuracy,
            'human_inference_time_ms': self.human_inference_time_ms,
            'robot_inference_time_ms': self.robot_inference_time_ms,
            'mutual_information': self.mutual_information,
            'kl_divergence': self.kl_divergence,
            'upper_bound_satisfied': self.upper_bound_satisfied,
            'upper_bound_ratio': self.upper_bound_ratio,
            'per_feature_accuracy': self.per_feature_accuracy,
            'detailed_metrics': self.detailed_metrics
        }
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            "=" * 60,
            "Evaluation Results",
            "=" * 60,
            f"Human (Leader) Accuracy:     {self.human_accuracy:.2f}%",
            f"Robot (Follower) Accuracy:   {self.robot_accuracy:.2f}%",
            f"Human Inference Time:        {self.human_inference_time_ms:.2f} ms",
            f"Robot Inference Time:        {self.robot_inference_time_ms:.2f} ms",
            f"Mutual Information:          {self.mutual_information:.4f}",
            f"KL Divergence:               {self.kl_divergence:.4f}",
            f"Upper Bound Satisfied:       {self.upper_bound_satisfied}",
            f"Upper Bound Ratio:           {self.upper_bound_ratio:.4f}",
        ]
        
        if self.per_feature_accuracy:
            lines.append("-" * 60)
            lines.append("Per-Feature Accuracy:")
            for feature, acc in self.per_feature_accuracy.items():
                lines.append(f"  {feature}: {acc:.2f}%")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class PredictionAccuracy:
    """Compute prediction accuracy as defined in the paper.
    
    Accuracy_n = 1 - |Ŝ_n - S_n| / max(S_n)
    
    Where:
        - Ŝ_n is the predicted signal at step n
        - S_n is the actual signal at step n
        - max(S_n) is the maximum absolute value for normalization
    
    Args:
        epsilon: Small value to prevent division by zero.
        reduction: Reduction method ('mean', 'none', 'sum').
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):
        self.epsilon = epsilon
        self.reduction = reduction
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        per_feature: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Compute prediction accuracy.
        
        Args:
            predictions: Predicted signals, shape (batch_size, features).
            targets: Actual signals, shape (batch_size, features).
            per_feature: If True, also return per-feature accuracy.
            
        Returns:
            If per_feature is False: Overall accuracy percentage.
            If per_feature is True: Tuple of (overall accuracy, per-feature dict).
        """
        # Ensure tensors are on same device
        if predictions.device != targets.device:
            predictions = predictions.to(targets.device)
        
        # Compute absolute error
        abs_error = torch.abs(predictions - targets)
        
        # Compute normalization factor (max of targets along batch)
        max_targets = torch.max(torch.abs(targets), dim=0)[0]
        max_targets = torch.clamp(max_targets, min=self.epsilon)
        
        # Compute normalized error
        normalized_error = abs_error / max_targets.unsqueeze(0)
        
        # Compute accuracy
        accuracy = 1.0 - normalized_error
        
        # Clamp to [0, 1] range
        accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
        
        if per_feature:
            # Per-feature accuracy
            feature_accuracy = accuracy.mean(dim=0) * 100
            feature_names = ['pos_x', 'pos_y', 'pos_z', 
                           'vel_x', 'vel_y', 'vel_z',
                           'force_x', 'force_y', 'force_z']
            
            per_feature_dict = {}
            for i, name in enumerate(feature_names[:feature_accuracy.shape[0]]):
                per_feature_dict[name] = feature_accuracy[i].item()
            
            # Aggregate by type
            if feature_accuracy.shape[0] >= 9:
                per_feature_dict['position'] = feature_accuracy[:3].mean().item()
                per_feature_dict['velocity'] = feature_accuracy[3:6].mean().item()
                per_feature_dict['force'] = feature_accuracy[6:9].mean().item()
        
        # Overall accuracy
        if self.reduction == 'mean':
            overall = accuracy.mean().item() * 100
        elif self.reduction == 'sum':
            overall = accuracy.sum().item() * 100
        else:
            overall = accuracy * 100
        
        if per_feature:
            return overall, per_feature_dict
        return overall


class InferenceTimer:
    """Measure inference time for neural network models.
    
    Provides accurate timing measurements with GPU synchronization
    and warmup iterations for stable results.
    
    Args:
        warmup_iterations: Number of warmup iterations before timing.
        num_iterations: Number of iterations to average over.
        sync_cuda: Whether to synchronize CUDA before/after timing.
    """
    
    def __init__(
        self,
        warmup_iterations: int = 10,
        num_iterations: int = 100,
        sync_cuda: bool = True
    ):
        self.warmup_iterations = warmup_iterations
        self.num_iterations = num_iterations
        self.sync_cuda = sync_cuda
    
    def _sync(self):
        """Synchronize CUDA if available and enabled."""
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def time_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Time a single model forward pass.
        
        Args:
            model: PyTorch model to time.
            input_tensor: Input tensor for the model.
            
        Returns:
            Tuple of (mean_time_ms, std_time_ms, min_time_ms).
        """
        model.eval()
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = model(input_tensor)
            
            # Actual timing
            for _ in range(self.num_iterations):
                self._sync()
                start = time.perf_counter()
                _ = model(input_tensor)
                self._sync()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        return times.mean(), times.std(), times.min()
    
    def time_leader_follower(
        self,
        model: LeaderFollowerModel,
        input_tensor: torch.Tensor
    ) -> Dict[str, Tuple[float, float, float]]:
        """Time both human and robot networks.
        
        Args:
            model: LeaderFollowerModel containing both networks.
            input_tensor: Input tensor for the models.
            
        Returns:
            Dictionary with timing results for 'human' and 'robot' networks.
        """
        return {
            'human': self.time_model(model.human_network, input_tensor),
            'robot': self.time_model(model.robot_network, input_tensor)
        }


class UpperBoundVerifier:
    """Verify the Taylor expansion upper bound from the paper.
    
    Verifies: L_{n-1}(θ_n) - L_{n-1}(θ_{n-1}) ≤ (1/2) λ_max^{n-1} ||Δθ||²
    
    Where:
        - L is the loss function
        - θ_n and θ_{n-1} are parameters at consecutive steps
        - λ_max is the maximum eigenvalue of the Hessian
        - Δθ = θ_n - θ_{n-1}
    
    Args:
        tolerance: Tolerance for bound satisfaction.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.taylor_bound = TaylorExpansionUpperBound()
    
    def verify(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        old_params: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor]
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Verify the upper bound for a parameter update.
        
        Args:
            model: Neural network model.
            loss_fn: Loss function.
            inputs: Input tensor.
            targets: Target tensor.
            old_params: Parameters before update.
            new_params: Parameters after update.
            
        Returns:
            Tuple of:
                - satisfied: Whether bound is satisfied
                - ratio: Ratio of actual to bound (< 1.0 means satisfied)
                - details: Dictionary with detailed values
        """
        # Load old parameters and compute loss
        for name, param in model.named_parameters():
            if name in old_params:
                param.data.copy_(old_params[name])
        
        model.eval()
        with torch.no_grad():
            outputs_old = model(inputs)
            loss_old = loss_fn(outputs_old, targets).item()
        
        # Load new parameters and compute loss
        for name, param in model.named_parameters():
            if name in new_params:
                param.data.copy_(new_params[name])
        
        with torch.no_grad():
            outputs_new = model(inputs)
            loss_new = loss_fn(outputs_new, targets).item()
        
        # Compute actual loss difference
        actual_diff = loss_new - loss_old
        
        # Compute parameter difference norm squared
        delta_theta_sq = 0.0
        for name in old_params:
            if name in new_params:
                diff = new_params[name] - old_params[name]
                delta_theta_sq += (diff ** 2).sum().item()
        
        # Estimate maximum eigenvalue using Taylor bound class
        lambda_max = self.taylor_bound.estimate_max_eigenvalue(
            model, loss_fn, inputs, targets
        )
        
        # Compute upper bound
        upper_bound = 0.5 * lambda_max * delta_theta_sq
        
        # Check satisfaction
        satisfied = actual_diff <= upper_bound + self.tolerance
        ratio = actual_diff / (upper_bound + 1e-10) if upper_bound > 0 else 0.0
        
        details = {
            'loss_old': loss_old,
            'loss_new': loss_new,
            'actual_diff': actual_diff,
            'upper_bound': upper_bound,
            'lambda_max': lambda_max,
            'delta_theta_squared': delta_theta_sq
        }
        
        return satisfied, ratio, details


class Evaluator:
    """Complete evaluation pipeline for Leader-Follower models.
    
    Combines all evaluation metrics into a single comprehensive evaluation.
    
    Args:
        model: LeaderFollowerModel to evaluate.
        device: Device for computation.
        num_timing_iterations: Iterations for timing measurements.
    """
    
    def __init__(
        self,
        model: LeaderFollowerModel,
        device: Optional[torch.device] = None,
        num_timing_iterations: int = 100
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize metric computers
        self.accuracy_metric = PredictionAccuracy()
        self.timer = InferenceTimer(num_iterations=num_timing_iterations)
        self.mi_estimator = MutualInformationEstimator()
        self.kl_loss = KLDivergenceLoss()
        self.bound_verifier = UpperBoundVerifier()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> EvaluationResults:
        """Run complete evaluation on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data.
            verbose: Whether to print progress.
            
        Returns:
            EvaluationResults with all computed metrics.
        """
        self.model.eval()
        
        # Accumulators
        human_predictions = []
        robot_predictions = []
        human_targets = []
        robot_targets = []
        
        # Collect all predictions
        for batch in dataloader:
            inputs = batch['input'].to(self.device)
            human_target = batch['human_target'].to(self.device)
            robot_target = batch['robot_target'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            human_predictions.append(outputs['robot_predicted'].cpu())
            robot_predictions.append(outputs['human_predicted'].cpu())
            human_targets.append(human_target.cpu())
            robot_targets.append(robot_target.cpu())
        
        # Concatenate all results
        human_predictions = torch.cat(human_predictions, dim=0)
        robot_predictions = torch.cat(robot_predictions, dim=0)
        human_targets = torch.cat(human_targets, dim=0)
        robot_targets = torch.cat(robot_targets, dim=0)
        
        # Compute prediction accuracy
        human_accuracy, human_per_feature = self.accuracy_metric(
            human_predictions, human_targets, per_feature=True
        )
        robot_accuracy, robot_per_feature = self.accuracy_metric(
            robot_predictions, robot_targets, per_feature=True
        )
        
        # Compute timing
        sample_input = next(iter(dataloader))['input'].to(self.device)
        timing = self.timer.time_leader_follower(self.model, sample_input)
        
        # Compute MI and KL
        mi_value = self.mi_estimator(
            robot_targets.numpy(), 
            robot_predictions.numpy()
        )
        kl_value = self.kl_loss(
            robot_predictions.to(self.device),
            human_targets.to(self.device)
        ).item()
        
        # Combine per-feature accuracy
        per_feature_accuracy = {
            f'human_{k}': v for k, v in human_per_feature.items()
        }
        per_feature_accuracy.update({
            f'robot_{k}': v for k, v in robot_per_feature.items()
        })
        
        results = EvaluationResults(
            human_accuracy=human_accuracy,
            robot_accuracy=robot_accuracy,
            human_inference_time_ms=timing['human'][0],
            robot_inference_time_ms=timing['robot'][0],
            mutual_information=mi_value,
            kl_divergence=kl_value,
            per_feature_accuracy=per_feature_accuracy,
            detailed_metrics={
                'human_timing_std': timing['human'][1],
                'robot_timing_std': timing['robot'][1],
                'human_timing_min': timing['human'][2],
                'robot_timing_min': timing['robot'][2],
                'num_samples': len(human_predictions)
            }
        )
        
        if verbose:
            print(results)
        
        return results
    
    def evaluate_per_dataset(
        self,
        dataloaders: Dict[str, DataLoader],
        verbose: bool = True
    ) -> Dict[str, EvaluationResults]:
        """Evaluate on multiple datasets.
        
        Args:
            dataloaders: Dictionary mapping dataset names to DataLoaders.
            verbose: Whether to print progress.
            
        Returns:
            Dictionary mapping dataset names to EvaluationResults.
        """
        results = {}
        
        for name, dataloader in dataloaders.items():
            if verbose:
                print(f"\nEvaluating on {name}...")
            results[name] = self.evaluate(dataloader, verbose=verbose)
        
        return results


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """Convenience function for computing prediction accuracy.
    
    Args:
        predictions: Predicted signals.
        targets: Actual signals.
        
    Returns:
        Accuracy percentage.
    """
    metric = PredictionAccuracy()
    return metric(predictions, targets)


def time_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    num_iterations: int = 100
) -> float:
    """Convenience function for timing model inference.
    
    Args:
        model: PyTorch model.
        input_shape: Shape of input tensor (excluding batch dimension).
        device: Device for computation.
        num_iterations: Number of iterations to average.
        
    Returns:
        Mean inference time in milliseconds.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    input_tensor = torch.randn(1, *input_shape).to(device)
    timer = InferenceTimer(num_iterations=num_iterations)
    
    mean_time, _, _ = timer.time_model(model, input_tensor)
    return mean_time


def generate_results_table(
    results: Dict[str, EvaluationResults],
    output_format: str = 'latex'
) -> str:
    """Generate formatted results table.
    
    Args:
        results: Dictionary of dataset names to EvaluationResults.
        output_format: 'latex' or 'markdown'.
        
    Returns:
        Formatted table string.
    """
    if output_format == 'latex':
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Prediction Accuracy and Inference Time Results}",
            r"\label{tab:results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Dataset & Human Acc. (\%) & Robot Acc. (\%) & Human Time (ms) & Robot Time (ms) \\",
            r"\midrule"
        ]
        
        for name, res in results.items():
            lines.append(
                f"{name} & {res.human_accuracy:.2f} & {res.robot_accuracy:.2f} & "
                f"{res.human_inference_time_ms:.2f} & {res.robot_inference_time_ms:.2f} \\\\"
            )
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
    else:  # markdown
        lines = [
            "| Dataset | Human Acc. (%) | Robot Acc. (%) | Human Time (ms) | Robot Time (ms) |",
            "|---------|----------------|----------------|-----------------|-----------------|"
        ]
        
        for name, res in results.items():
            lines.append(
                f"| {name} | {res.human_accuracy:.2f} | {res.robot_accuracy:.2f} | "
                f"{res.human_inference_time_ms:.2f} | {res.robot_inference_time_ms:.2f} |"
            )
    
    return "\n".join(lines)
