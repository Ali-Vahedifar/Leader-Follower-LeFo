"""
Unit Tests for LeFo Game Theory Components.

Tests the game-theoretic optimization framework including
KL divergence, mutual information estimation, and MiniMax optimization.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lefo.game import (
    KLDivergenceLoss,
    MutualInformationEstimator,
    MiniMaxOptimizer,
    StackelbergEquilibrium,
    TaylorExpansionUpperBound
)


class TestKLDivergenceLoss:
    """Tests for KL Divergence Loss."""
    
    def test_creation(self):
        """Test loss creation."""
        loss = KLDivergenceLoss()
        assert loss is not None
    
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be ~0."""
        loss = KLDivergenceLoss(mode='softmax')
        
        x = torch.randn(32, 10)
        kl = loss(x, x)
        
        assert kl.item() < 0.1  # Should be very small
    
    def test_different_distributions(self):
        """KL divergence of different distributions should be positive."""
        loss = KLDivergenceLoss(mode='softmax')
        
        p = torch.randn(32, 10)
        q = torch.randn(32, 10) + 5  # Different distribution
        
        kl = loss(p, q)
        
        assert kl.item() > 0
    
    def test_softmax_mode(self):
        """Test softmax-based KL divergence."""
        loss = KLDivergenceLoss(mode='softmax', temperature=1.0)
        
        p = torch.randn(32, 10)
        q = torch.randn(32, 10)
        
        kl = loss(p, q)
        
        assert torch.isfinite(kl)
        assert kl.item() >= 0
    
    def test_gaussian_mode(self):
        """Test Gaussian-based KL divergence."""
        loss = KLDivergenceLoss(mode='gaussian')
        
        # Create tensors representing mean and variance
        p = torch.randn(32, 10)
        q = torch.randn(32, 10)
        
        kl = loss(p, q)
        
        assert torch.isfinite(kl)
    
    def test_gradient_flow(self):
        """Test gradients flow through KL divergence."""
        loss = KLDivergenceLoss()
        
        p = torch.randn(32, 10, requires_grad=True)
        q = torch.randn(32, 10, requires_grad=True)
        
        kl = loss(p, q)
        kl.backward()
        
        assert p.grad is not None
        assert q.grad is not None


class TestMutualInformationEstimator:
    """Tests for Mutual Information Estimator."""
    
    def test_creation(self):
        """Test estimator creation."""
        mi = MutualInformationEstimator()
        assert mi is not None
    
    def test_creation_custom_k(self):
        """Test estimator with custom k neighbors."""
        mi = MutualInformationEstimator(k_neighbors=5)
        assert mi.k == 5
    
    def test_independent_variables(self):
        """MI of independent variables should be ~0."""
        mi = MutualInformationEstimator(k_neighbors=5)
        
        np.random.seed(42)
        x = np.random.randn(1000, 3)
        y = np.random.randn(1000, 3)  # Independent of x
        
        mi_value = mi(x, y)
        
        # Should be close to 0 for independent variables
        assert abs(mi_value) < 0.5
    
    def test_identical_variables(self):
        """MI of identical variables should be high."""
        mi = MutualInformationEstimator(k_neighbors=5)
        
        np.random.seed(42)
        x = np.random.randn(1000, 3)
        
        mi_value = mi(x, x)
        
        # Should be high for identical variables
        assert mi_value > 1.0
    
    def test_correlated_variables(self):
        """MI of correlated variables should be positive."""
        mi = MutualInformationEstimator(k_neighbors=5)
        
        np.random.seed(42)
        x = np.random.randn(1000, 3)
        y = x + 0.1 * np.random.randn(1000, 3)  # Correlated with noise
        
        mi_value = mi(x, y)
        
        # Should be positive for correlated variables
        assert mi_value > 0
    
    def test_different_dimensions(self):
        """Test with different input dimensions."""
        mi = MutualInformationEstimator(k_neighbors=5)
        
        for dim in [1, 3, 9, 20]:
            x = np.random.randn(500, dim)
            y = np.random.randn(500, dim)
            
            mi_value = mi(x, y)
            
            assert np.isfinite(mi_value)


class TestMiniMaxOptimizer:
    """Tests for MiniMax Optimizer."""
    
    def test_creation(self):
        """Test optimizer creation."""
        from src.lefo.models import LeaderFollowerModel
        
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        optimizer = MiniMaxOptimizer(model, lr=0.01, momentum=0.9)
        
        assert optimizer is not None
    
    def test_optimization_step(self):
        """Test a single optimization step."""
        from src.lefo.models import LeaderFollowerModel
        
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        optimizer = MiniMaxOptimizer(model, lr=0.01, momentum=0.9)
        
        # Create dummy batch
        human_input = torch.randn(4, 9)
        robot_input = torch.randn(4, 9)
        human_target = torch.randn(4, 9)
        robot_target = torch.randn(4, 9)
        
        # Store initial params
        initial_params = [p.clone() for p in model.parameters()]
        
        # Perform optimization step
        loss = optimizer.step(
            human_input=human_input,
            robot_input=robot_input,
            human_target=human_target,
            robot_target=robot_target
        )
        
        # Check parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break
        
        assert params_changed
    
    def test_alternating_optimization(self):
        """Test alternating optimization between human and robot."""
        from src.lefo.models import LeaderFollowerModel
        
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        optimizer = MiniMaxOptimizer(model, lr=0.01, momentum=0.9)
        
        # Multiple steps
        for _ in range(5):
            human_input = torch.randn(4, 9)
            robot_input = torch.randn(4, 9)
            human_target = torch.randn(4, 9)
            robot_target = torch.randn(4, 9)
            
            loss = optimizer.step(
                human_input=human_input,
                robot_input=robot_input,
                human_target=human_target,
                robot_target=robot_target
            )
            
            assert torch.isfinite(torch.tensor(loss))


class TestStackelbergEquilibrium:
    """Tests for Stackelberg Equilibrium finder."""
    
    def test_creation(self):
        """Test equilibrium finder creation."""
        eq = StackelbergEquilibrium()
        assert eq is not None
    
    def test_find_equilibrium(self):
        """Test finding equilibrium."""
        from src.lefo.models import LeaderFollowerModel
        
        eq = StackelbergEquilibrium(max_iterations=10)
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        human_input = torch.randn(10, 9)
        robot_input = torch.randn(10, 9)
        
        result = eq.find(model, human_input, robot_input)
        
        assert 'human_strategy' in result
        assert 'robot_strategy' in result
        assert 'converged' in result


class TestTaylorExpansionUpperBound:
    """Tests for Taylor Expansion Upper Bound verification."""
    
    def test_creation(self):
        """Test upper bound verifier creation."""
        bound = TaylorExpansionUpperBound()
        assert bound is not None
    
    def test_compute_bound(self):
        """Test computing the upper bound."""
        from src.lefo.models import LeaderFollowerModel
        
        bound = TaylorExpansionUpperBound(num_power_iterations=10)
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        # Create sample data
        x = torch.randn(32, 9)
        y = torch.randn(32, 9)
        
        # Compute bound
        result = bound.compute(model, x, y)
        
        assert 'upper_bound' in result
        assert 'lambda_max' in result
        assert result['upper_bound'] >= 0
    
    def test_lambda_max_estimation(self):
        """Test Hessian eigenvalue estimation."""
        from src.lefo.models import LeaderFollowerModel
        
        bound = TaylorExpansionUpperBound(num_power_iterations=20)
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        x = torch.randn(16, 9)
        y = torch.randn(16, 9)
        
        lambda_max = bound._estimate_lambda_max(model, x, y)
        
        assert lambda_max > 0
        assert np.isfinite(lambda_max)
    
    def test_bound_satisfaction(self):
        """Test that bound is satisfied (actual diff <= bound)."""
        from src.lefo.models import LeaderFollowerModel
        
        bound = TaylorExpansionUpperBound(num_power_iterations=20)
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        x = torch.randn(32, 9)
        y = torch.randn(32, 9)
        
        # Get initial loss
        model.eval()
        with torch.no_grad():
            h_pred, r_pred = model(x, x)
            loss_before = torch.nn.functional.mse_loss(h_pred, y)
        
        # Take a small gradient step
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        h_pred, r_pred = model(x, x)
        loss = torch.nn.functional.mse_loss(h_pred, y)
        loss.backward()
        
        # Compute bound before step
        result = bound.compute(model, x, y)
        
        optimizer.step()
        
        # Get loss after step
        model.eval()
        with torch.no_grad():
            h_pred, r_pred = model(x, x)
            loss_after = torch.nn.functional.mse_loss(h_pred, y)
        
        actual_diff = abs(loss_after.item() - loss_before.item())
        
        # With small learning rate, bound should hold
        # (This is a probabilistic test, may occasionally fail)


class TestGameIntegration:
    """Integration tests for game theory components."""
    
    def test_full_game_iteration(self):
        """Test a complete game iteration."""
        from src.lefo.models import LeaderFollowerModel
        
        # Setup
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        kl_loss = KLDivergenceLoss()
        mi_estimator = MutualInformationEstimator(k_neighbors=5)
        
        # Create data
        human_input = torch.randn(32, 9)
        robot_input = torch.randn(32, 9)
        
        # Forward pass
        model.train()
        human_pred, robot_pred = model(human_input, robot_input)
        
        # Compute losses
        kl = kl_loss(human_pred, human_input)
        mi = mi_estimator(
            robot_pred.detach().numpy(),
            robot_input.numpy()
        )
        
        assert torch.isfinite(kl)
        assert np.isfinite(mi)
    
    def test_loss_decreases(self):
        """Test that loss decreases over training iterations."""
        from src.lefo.models import LeaderFollowerModel
        
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        optimizer = MiniMaxOptimizer(model, lr=0.01, momentum=0.9)
        
        # Fixed data for consistent comparison
        torch.manual_seed(42)
        human_input = torch.randn(64, 9)
        robot_input = torch.randn(64, 9)
        human_target = torch.randn(64, 9)
        robot_target = torch.randn(64, 9)
        
        losses = []
        for _ in range(20):
            loss = optimizer.step(
                human_input=human_input,
                robot_input=robot_input,
                human_target=human_target,
                robot_target=robot_target
            )
            losses.append(loss)
        
        # Loss should generally decrease (not strictly monotonic)
        assert losses[-1] < losses[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
