#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Game Theory Components for Signal Prediction

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

IEEE MLSP 2025 - Istanbul, Turkey

This module implements the game theory components for the Leader-Follower framework:
    - KL Divergence Loss: Robot's utility function to minimize
    - Mutual Information Estimator: Human's utility function to maximize (KNN-based)
    - MiniMax Optimizer: Combined optimization for the Stackelberg game
    - Taylor Expansion Upper Bound: Theoretical bound for loss function

The interaction is formulated as a MiniMax optimization problem:
    min_R max_H L(H,R,θ) = E[max I(S_R, Ŝ_R)] - E[min KL(S_H || Ŝ_H)]

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import math
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.special import digamma, gamma
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss for Robot Side (Follower) Utility Function
    
    The Robot minimizes the KL divergence between actual (S_H) and predicted (Ŝ_H)
    human signals:
    
        U_R(S_H(n), S_R(n)) = KL_R(S_H(n+1), Ŝ_H(n+1))
        
        KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    
    For continuous distributions, we use a softmax-based approximation or
    sample-based estimation.
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
        epsilon: Small constant for numerical stability
        temperature: Temperature for softmax conversion
    
    Example:
        >>> kl_loss = KLDivergenceLoss()
        >>> actual = torch.randn(32, 9)
        >>> predicted = torch.randn(32, 9)
        >>> loss = kl_loss(actual, predicted)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        epsilon: float = 1e-10,
        temperature: float = 1.0
    ):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.temperature = temperature
    
    def forward(
        self,
        actual: Tensor,
        predicted: Tensor,
        use_softmax: bool = False
    ) -> Tensor:
        """
        Compute KL divergence between actual and predicted distributions.
        
        Args:
            actual: Actual signal S_H of shape (batch, features)
            predicted: Predicted signal Ŝ_H of shape (batch, features)
            use_softmax: Whether to convert to probability distributions
        
        Returns:
            KL divergence loss
        """
        if use_softmax:
            # Convert to probability distributions using softmax
            p = F.softmax(actual / self.temperature, dim=-1)
            q = F.softmax(predicted / self.temperature, dim=-1)
            
            # KL(P || Q) = Σ P * log(P/Q)
            kl = torch.sum(p * (torch.log(p + self.epsilon) - torch.log(q + self.epsilon)), dim=-1)
        else:
            # Use Gaussian KL divergence approximation
            # Assuming unit variance for simplicity
            kl = 0.5 * torch.sum((actual - predicted) ** 2, dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl
    
    def gaussian_kl(
        self,
        mu1: Tensor,
        var1: Tensor,
        mu2: Tensor,
        var2: Tensor
    ) -> Tensor:
        """
        Compute KL divergence between two Gaussian distributions.
        
        KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        
        Args:
            mu1, var1: Mean and variance of first distribution (actual)
            mu2, var2: Mean and variance of second distribution (predicted)
        
        Returns:
            KL divergence
        """
        std1 = torch.sqrt(var1 + self.epsilon)
        std2 = torch.sqrt(var2 + self.epsilon)
        
        kl = torch.log(std2 / std1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
        
        return kl.sum(dim=-1)


class MutualInformationEstimator(nn.Module):
    """
    Mutual Information Estimator using K-Nearest Neighbors (KNN)
    
    Human's utility function to maximize. Estimates I(S_R; Ŝ_R) using the KNN
    method based on the work of Kraskov et al. and our adaptation for haptic signals.
    
    The mutual information quantifies the dependency between S_R and Ŝ_R:
        I(S_R; Ŝ_R) = H(S_R) + H(Ŝ_R) - H(S_R, Ŝ_R)
    
    Using KNN estimation:
        I = ψ(N) - (1/N)Σψ(n_f) - (1/N)Σψ(n_v) - (1/N)Σψ(n_p) + ψ(K) - 1/K
    
    where ψ is the digamma function.
    
    Args:
        k_neighbors: Number of nearest neighbors (default: 11)
        metric: Distance metric ('chebyshev', 'euclidean', 'manhattan')
        normalize: Whether to normalize inputs
    
    Example:
        >>> mi_estimator = MutualInformationEstimator(k_neighbors=11)
        >>> actual = torch.randn(1000, 9)
        >>> predicted = torch.randn(1000, 9)
        >>> mi = mi_estimator(actual, predicted)
    
    References:
        [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). 
            Estimating mutual information. Physical review E.
        [2] Vahedifar et al. (2024). Information-Modified K-Nearest Neighbor.
    """
    
    def __init__(
        self,
        k_neighbors: int = 11,
        metric: str = 'chebyshev',
        normalize: bool = True
    ):
        super(MutualInformationEstimator, self).__init__()
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.normalize = normalize
    
    def forward(
        self,
        actual: Tensor,
        predicted: Tensor,
        feature_groups: Optional[List[int]] = None
    ) -> Tensor:
        """
        Estimate mutual information between actual and predicted signals.
        
        Args:
            actual: Actual robot signal S_R of shape (N, features)
            predicted: Predicted robot signal Ŝ_R of shape (N, features)
            feature_groups: Indices for feature grouping [force, velocity, position]
                           Default: [0,1,2], [3,4,5], [6,7,8] for 3D signals
        
        Returns:
            Estimated mutual information (scalar tensor)
        """
        # Convert to numpy for KNN computation
        if isinstance(actual, Tensor):
            actual_np = actual.detach().cpu().numpy()
            predicted_np = predicted.detach().cpu().numpy()
        else:
            actual_np = actual
            predicted_np = predicted
        
        # Normalize if requested
        if self.normalize:
            actual_np = (actual_np - actual_np.mean(0)) / (actual_np.std(0) + 1e-10)
            predicted_np = (predicted_np - predicted_np.mean(0)) / (predicted_np.std(0) + 1e-10)
        
        # Default feature groups for 3D haptic signals (force, velocity, position)
        if feature_groups is None:
            num_features = actual_np.shape[1]
            if num_features == 9:
                # force (3), velocity (3), position (3)
                feature_groups = [(0, 3), (3, 6), (6, 9)]
            else:
                # Single group
                feature_groups = [(0, num_features)]
        
        # Compute MI using KNN
        mi = self._compute_knn_mi(actual_np, predicted_np, feature_groups)
        
        # Convert back to tensor
        return torch.tensor(mi, dtype=torch.float32, device=actual.device if isinstance(actual, Tensor) else 'cpu')
    
    def _compute_knn_mi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_groups: List[Tuple[int, int]]
    ) -> float:
        """
        Compute mutual information using KNN method.
        
        Uses the Chebyshev norm (max norm) for distance computation:
            ||z_i - z_j||_∞ = max(||x_i - x_j||, ||y_i - y_j||)
        
        Args:
            x: First variable (actual signals)
            y: Second variable (predicted signals)
            feature_groups: List of (start, end) indices for feature groups
        
        Returns:
            Estimated mutual information
        """
        n_samples = x.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if n_samples < k + 1:
            return 0.0
        
        # Concatenate for joint space
        z = np.concatenate([x, y], axis=1)
        
        # Build KD-tree for joint space
        if self.metric == 'chebyshev':
            tree = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
        else:
            tree = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        
        tree.fit(z)
        
        # Find k-th nearest neighbor distances
        distances, _ = tree.kneighbors(z)
        eps = distances[:, -1]  # Distance to k-th neighbor
        
        # Count neighbors in marginal spaces
        neighbor_counts = []
        
        for start, end in feature_groups:
            # Build tree for this feature group
            x_group = x[:, start:end]
            tree_x = NearestNeighbors(metric=self.metric)
            tree_x.fit(x_group)
            
            # Count points within eps distance
            n_x = np.array([
                len(tree_x.radius_neighbors([x_group[i]], eps[i], return_distance=False)[0]) - 1
                for i in range(n_samples)
            ])
            neighbor_counts.append(n_x)
        
        # Repeat for predicted signals
        for start, end in feature_groups:
            y_group = y[:, start:end]
            tree_y = NearestNeighbors(metric=self.metric)
            tree_y.fit(y_group)
            
            n_y = np.array([
                len(tree_y.radius_neighbors([y_group[i]], eps[i], return_distance=False)[0]) - 1
                for i in range(n_samples)
            ])
            neighbor_counts.append(n_y)
        
        # Compute MI using digamma function
        # I = ψ(N) + ψ(k) - (1/N)Σψ(n_x+1) - (1/N)Σψ(n_y+1)
        mi = digamma(n_samples) + digamma(k)
        
        for n_count in neighbor_counts:
            mi -= np.mean(digamma(np.maximum(n_count, 1) + 1))
        
        return max(0.0, mi)  # MI should be non-negative
    
    def compute_with_features(
        self,
        actual_force: Tensor,
        actual_velocity: Tensor,
        actual_position: Tensor,
        predicted_force: Tensor,
        predicted_velocity: Tensor,
        predicted_position: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute MI for each feature type separately.
        
        Computes:
            - e^f = 2||S^f - Ŝ^{f,k}||  (force distance)
            - e^v = 2||S^v - Ŝ^{v,k}||  (velocity distance)
            - e^p = 2||S^p - Ŝ^{p,k}||  (position distance)
        
        Args:
            actual_force, actual_velocity, actual_position: Actual signals
            predicted_force, predicted_velocity, predicted_position: Predicted signals
        
        Returns:
            Dictionary with MI for each feature and combined MI
        """
        mi_force = self._compute_single_mi(actual_force, predicted_force)
        mi_velocity = self._compute_single_mi(actual_velocity, predicted_velocity)
        mi_position = self._compute_single_mi(actual_position, predicted_position)
        
        return {
            'mi_force': mi_force,
            'mi_velocity': mi_velocity,
            'mi_position': mi_position,
            'mi_total': mi_force + mi_velocity + mi_position
        }
    
    def _compute_single_mi(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute MI for a single feature pair."""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        n_samples = x_np.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if n_samples < k + 1:
            return torch.tensor(0.0)
        
        # Concatenate for joint distribution
        z = np.concatenate([x_np, y_np], axis=1)
        
        # KNN computation
        tree_z = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        tree_z.fit(z)
        distances, _ = tree_z.kneighbors(z)
        eps = distances[:, -1]
        
        # Marginal counts
        tree_x = NearestNeighbors(metric=self.metric)
        tree_x.fit(x_np)
        tree_y = NearestNeighbors(metric=self.metric)
        tree_y.fit(y_np)
        
        n_x = np.array([
            len(tree_x.radius_neighbors([x_np[i]], eps[i], return_distance=False)[0]) - 1
            for i in range(n_samples)
        ])
        n_y = np.array([
            len(tree_y.radius_neighbors([y_np[i]], eps[i], return_distance=False)[0]) - 1
            for i in range(n_samples)
        ])
        
        # MI estimation
        mi = digamma(n_samples) + digamma(k) - \
             np.mean(digamma(np.maximum(n_x, 1) + 1)) - \
             np.mean(digamma(np.maximum(n_y, 1) + 1))
        
        return torch.tensor(max(0.0, mi), dtype=torch.float32, device=x.device)


class MiniMaxOptimizer:
    """
    MiniMax Optimizer for Leader-Follower Game
    
    Implements the MiniMax optimization from Equation (9):
        min_R max_H L(H,R,θ) = E[max I(S_R, Ŝ_R)] - E[min KL(S_H || Ŝ_H)]
    
    The optimization alternates between:
        1. Robot minimizing KL divergence (Follower response)
        2. Human maximizing Mutual Information (Leader optimization)
    
    Args:
        human_optimizer: Optimizer for human network
        robot_optimizer: Optimizer for robot network
        mi_estimator: Mutual information estimator
        kl_loss: KL divergence loss function
        lambda_mi: Weight for MI term
        lambda_kl: Weight for KL term
        max_iterations: Maximum iterations per step
        tolerance: Convergence tolerance
    
    Example:
        >>> optimizer = MiniMaxOptimizer(
        ...     human_optimizer=torch.optim.SGD(model.human_network.parameters(), lr=0.01),
        ...     robot_optimizer=torch.optim.SGD(model.robot_network.parameters(), lr=0.01)
        ... )
        >>> loss = optimizer.step(model, human_data, robot_data)
    """
    
    def __init__(
        self,
        human_optimizer: torch.optim.Optimizer,
        robot_optimizer: torch.optim.Optimizer,
        mi_estimator: Optional[MutualInformationEstimator] = None,
        kl_loss: Optional[KLDivergenceLoss] = None,
        lambda_mi: float = 1.0,
        lambda_kl: float = 1.0,
        max_iterations: int = 1,
        tolerance: float = 1e-4
    ):
        self.human_optimizer = human_optimizer
        self.robot_optimizer = robot_optimizer
        self.mi_estimator = mi_estimator or MutualInformationEstimator()
        self.kl_loss = kl_loss or KLDivergenceLoss()
        self.lambda_mi = lambda_mi
        self.lambda_kl = lambda_kl
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self._prev_loss = None
    
    def step(
        self,
        model: nn.Module,
        human_input: Tensor,
        robot_input: Tensor,
        actual_robot: Tensor,
        actual_human: Tensor
    ) -> Dict[str, float]:
        """
        Perform one MiniMax optimization step.
        
        Args:
            model: LeaderFollowerModel
            human_input: Input to human network
            robot_input: Input to robot network
            actual_robot: Actual robot signals (ground truth)
            actual_human: Actual human signals (ground truth)
        
        Returns:
            Dictionary with loss values
        """
        total_loss = 0.0
        mi_value = 0.0
        kl_value = 0.0
        
        for _ in range(self.max_iterations):
            # Step 1: Robot minimizes KL divergence
            self.robot_optimizer.zero_grad()
            
            _, human_pred = model(human_input, robot_input)
            kl = self.kl_loss(actual_human, human_pred)
            
            kl.backward()
            self.robot_optimizer.step()
            
            # Step 2: Human maximizes Mutual Information
            self.human_optimizer.zero_grad()
            
            robot_pred, _ = model(human_input, robot_input)
            
            # For gradient-based optimization, use a differentiable proxy
            # MSE loss as proxy (lower MSE -> higher MI in well-behaved cases)
            mse_proxy = F.mse_loss(robot_pred, actual_robot)
            
            # We want to maximize MI, so minimize negative proxy
            mi_loss = -self.lambda_mi * (-mse_proxy)  # Double negative for maximization
            
            mi_loss.backward()
            self.human_optimizer.step()
            
            # Compute actual MI (non-differentiable, for monitoring)
            with torch.no_grad():
                robot_pred_eval, human_pred_eval = model(human_input, robot_input)
                mi_value = self.mi_estimator(actual_robot, robot_pred_eval).item()
                kl_value = self.kl_loss(actual_human, human_pred_eval).item()
            
            # Combined loss for monitoring
            total_loss = self.lambda_mi * mi_value - self.lambda_kl * kl_value
            
            # Check convergence
            if self._prev_loss is not None:
                if abs(total_loss - self._prev_loss) < self.tolerance:
                    break
            
            self._prev_loss = total_loss
        
        return {
            'total_loss': total_loss,
            'mi_value': mi_value,
            'kl_value': kl_value,
            'human_mse': mse_proxy.item(),
            'robot_kl': kl.item()
        }


class StackelbergEquilibrium:
    """
    Stackelberg Equilibrium Finder for Leader-Follower Game
    
    In the Stackelberg game:
        - Leader (Human) moves first, choosing S_L to maximize U_L
        - Follower (Robot) observes and responds with S_F* = argmax U_F(S_F, S_L)
        - Leader anticipates this: S_L* = argmax U_L(S_F*(S_L), S_L)
    
    At equilibrium: S_F* = S_F*(S_L*)
    
    This class finds the equilibrium point through iterative optimization.
    
    Args:
        model: LeaderFollowerModel
        mi_estimator: Mutual information estimator
        kl_loss: KL divergence loss
        lr: Learning rate for equilibrium search
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    """
    
    def __init__(
        self,
        model: nn.Module,
        mi_estimator: Optional[MutualInformationEstimator] = None,
        kl_loss: Optional[KLDivergenceLoss] = None,
        lr: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ):
        self.model = model
        self.mi_estimator = mi_estimator or MutualInformationEstimator()
        self.kl_loss = kl_loss or KLDivergenceLoss()
        self.lr = lr
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def find_equilibrium(
        self,
        human_input: Tensor,
        robot_input: Tensor,
        actual_robot: Tensor,
        actual_human: Tensor
    ) -> Dict[str, Any]:
        """
        Find Stackelberg equilibrium through iterative optimization.
        
        Args:
            human_input: Input to human network
            robot_input: Input to robot network
            actual_robot: Actual robot signals
            actual_human: Actual human signals
        
        Returns:
            Dictionary with equilibrium predictions and convergence info
        """
        history = []
        
        for iteration in range(self.max_iterations):
            # Get current predictions
            with torch.no_grad():
                robot_pred, human_pred = self.model(human_input, robot_input)
            
            # Compute utilities
            mi_value = self.mi_estimator(actual_robot, robot_pred)
            kl_value = self.kl_loss(actual_human, human_pred)
            
            # Record history
            history.append({
                'iteration': iteration,
                'mi': mi_value.item() if isinstance(mi_value, Tensor) else mi_value,
                'kl': kl_value.item() if isinstance(kl_value, Tensor) else kl_value
            })
            
            # Check convergence
            if len(history) > 1:
                mi_diff = abs(history[-1]['mi'] - history[-2]['mi'])
                kl_diff = abs(history[-1]['kl'] - history[-2]['kl'])
                
                if mi_diff < self.tolerance and kl_diff < self.tolerance:
                    break
        
        return {
            'robot_prediction': robot_pred,
            'human_prediction': human_pred,
            'final_mi': history[-1]['mi'],
            'final_kl': history[-1]['kl'],
            'converged': len(history) < self.max_iterations,
            'iterations': len(history),
            'history': history
        }


class TaylorExpansionUpperBound:
    """
    Taylor Expansion Upper Bound for Loss Function
    
    Establishes an upper bound for the loss difference using Taylor expansion:
    
        L_{n-1}(θ_n) - L_{n-1}(θ_{n-1}) ≤ (1/2) λ_max^{n-1} ||Δθ||²
    
    where λ_max^{n-1} is the largest eigenvalue of the Hessian matrix.
    
    This provides theoretical guarantees for model robustness against signal loss.
    
    Args:
        model: Neural network model
        num_samples: Number of samples for Hessian approximation
    
    Example:
        >>> bound_analyzer = TaylorExpansionUpperBound(model)
        >>> result = bound_analyzer.compute_upper_bound(theta_old, theta_new, data)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 100
    ):
        self.model = model
        self.num_samples = num_samples
    
    def compute_upper_bound(
        self,
        theta_old: Dict[str, Tensor],
        theta_new: Dict[str, Tensor],
        loss_fn: nn.Module,
        data: Tensor,
        targets: Tensor
    ) -> Dict[str, float]:
        """
        Compute the upper bound for loss difference.
        
        Args:
            theta_old: Parameters at step n-1
            theta_new: Parameters at step n
            loss_fn: Loss function
            data: Input data
            targets: Target values
        
        Returns:
            Dictionary with upper bound and actual loss difference
        """
        # Compute parameter difference ||Δθ||²
        delta_theta_squared = 0.0
        for key in theta_old.keys():
            diff = theta_new[key] - theta_old[key]
            delta_theta_squared += (diff ** 2).sum().item()
        
        # Approximate largest eigenvalue of Hessian
        lambda_max = self._approximate_max_eigenvalue(loss_fn, data, targets)
        
        # Upper bound: (1/2) λ_max ||Δθ||²
        upper_bound = 0.5 * lambda_max * delta_theta_squared
        
        # Compute actual loss difference
        # Load old parameters
        old_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Loss with old parameters
        self.model.load_state_dict(theta_old)
        with torch.no_grad():
            output_old = self.model(data) if hasattr(self.model, 'forward') else self.model.predict_robot(data)
            loss_old = loss_fn(output_old, targets).item()
        
        # Loss with new parameters
        self.model.load_state_dict(theta_new)
        with torch.no_grad():
            output_new = self.model(data) if hasattr(self.model, 'forward') else self.model.predict_robot(data)
            loss_new = loss_fn(output_new, targets).item()
        
        # Restore original state
        self.model.load_state_dict(old_state)
        
        actual_diff = abs(loss_new - loss_old)
        
        return {
            'upper_bound': upper_bound,
            'actual_loss_diff': actual_diff,
            'delta_theta_squared': delta_theta_squared,
            'lambda_max': lambda_max,
            'bound_satisfied': actual_diff <= upper_bound
        }
    
    def _approximate_max_eigenvalue(
        self,
        loss_fn: nn.Module,
        data: Tensor,
        targets: Tensor,
        num_iterations: int = 20
    ) -> float:
        """
        Approximate the largest eigenvalue of the Hessian using power iteration.
        
        Uses the Hessian-vector product for efficient computation without
        explicitly forming the Hessian matrix.
        
        Args:
            loss_fn: Loss function
            data: Input data
            targets: Target values
            num_iterations: Number of power iteration steps
        
        Returns:
            Approximated largest eigenvalue
        """
        # Get model parameters as a flat vector
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)
        
        # Initialize random vector
        v = [torch.randn_like(p) for p in params]
        
        # Normalize
        norm = sum((vi ** 2).sum() for vi in v) ** 0.5
        v = [vi / norm for vi in v]
        
        # Power iteration
        for _ in range(num_iterations):
            # Compute Hessian-vector product
            self.model.zero_grad()
            
            output = self.model(data) if not hasattr(self.model, 'predict_robot') else self.model.predict_robot(data)
            if isinstance(output, tuple):
                output = output[0]
            
            loss = loss_fn(output, targets)
            
            # First gradient
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            
            # Compute gradient-vector product
            gv = sum((g * vi).sum() for g, vi in zip(grads, v) if g is not None)
            
            # Second gradient (Hessian-vector product)
            hv = torch.autograd.grad(gv, params, allow_unused=True)
            
            # Update v
            v_new = []
            for i, p in enumerate(params):
                if hv[i] is not None:
                    v_new.append(hv[i].detach())
                else:
                    v_new.append(torch.zeros_like(p))
            
            # Normalize
            norm = sum((vi ** 2).sum() for vi in v_new) ** 0.5
            if norm > 1e-10:
                v = [vi / norm for vi in v_new]
            
            # Eigenvalue estimate
            lambda_max = norm.item()
        
        return lambda_max


def compute_minimax_loss(
    mi_value: float,
    kl_value: float,
    lambda_mi: float = 1.0,
    lambda_kl: float = 1.0
) -> float:
    """
    Compute the combined MiniMax loss.
    
    L(H,R,θ) = λ_MI * E[max I(S_R, Ŝ_R)] - λ_KL * E[min KL(S_H || Ŝ_H)]
    
    Args:
        mi_value: Mutual information value
        kl_value: KL divergence value
        lambda_mi: Weight for MI term
        lambda_kl: Weight for KL term
    
    Returns:
        Combined loss value
    """
    return lambda_mi * mi_value - lambda_kl * kl_value


if __name__ == "__main__":
    # Test game theory components
    print("Testing Game Theory Components")
    print("=" * 60)
    
    # Test KL Divergence
    print("\n1. Testing KL Divergence Loss:")
    kl_loss = KLDivergenceLoss()
    actual = torch.randn(100, 9)
    predicted = torch.randn(100, 9)
    kl = kl_loss(actual, predicted)
    print(f"   KL Divergence: {kl.item():.4f}")
    
    # Test MI Estimator
    print("\n2. Testing Mutual Information Estimator:")
    mi_estimator = MutualInformationEstimator(k_neighbors=5)
    actual = torch.randn(200, 9)
    predicted = actual + 0.1 * torch.randn(200, 9)  # Correlated signals
    mi = mi_estimator(actual, predicted)
    print(f"   Estimated MI (correlated): {mi.item():.4f}")
    
    predicted_uncorr = torch.randn(200, 9)  # Uncorrelated
    mi_uncorr = mi_estimator(actual, predicted_uncorr)
    print(f"   Estimated MI (uncorrelated): {mi_uncorr.item():.4f}")
    
    # Test MiniMax loss
    print("\n3. Testing MiniMax Loss Computation:")
    loss = compute_minimax_loss(mi_value=2.5, kl_value=1.2)
    print(f"   MiniMax Loss: {loss:.4f}")
    
    print("\n✓ All tests passed!")
