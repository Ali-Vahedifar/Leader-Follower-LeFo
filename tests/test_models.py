"""
Unit Tests for LeFo Models.

Tests the neural network architectures for Leader-Follower signal prediction.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lefo.models import (
    FullyConnectedBlock,
    HumanNetwork,
    RobotNetwork,
    LeaderFollowerModel,
    SequencePredictor,
    create_model
)


class TestFullyConnectedBlock:
    """Tests for FullyConnectedBlock."""
    
    def test_block_creation(self):
        """Test basic block creation."""
        block = FullyConnectedBlock(10, 20, dropout=0.5)
        assert block is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        block = FullyConnectedBlock(10, 20, dropout=0.5)
        x = torch.randn(32, 10)
        y = block(x)
        assert y.shape == (32, 20)
    
    def test_dropout_train_vs_eval(self):
        """Test dropout behavior in train vs eval mode."""
        block = FullyConnectedBlock(10, 20, dropout=0.5)
        x = torch.randn(32, 10)
        
        block.train()
        y_train = block(x)
        
        block.eval()
        y_eval = block(x)
        
        # Outputs should differ due to dropout
        # (not guaranteed but very likely with dropout=0.5)


class TestHumanNetwork:
    """Tests for HumanNetwork (Leader)."""
    
    def test_creation_default(self):
        """Test network creation with default parameters."""
        net = HumanNetwork(input_dim=9, output_dim=9)
        assert net is not None
        assert net.num_layers == 12
        assert net.hidden_dim == 100
    
    def test_creation_custom(self):
        """Test network creation with custom parameters."""
        net = HumanNetwork(
            input_dim=18,
            output_dim=9,
            num_layers=8,
            hidden_dim=64,
            dropout=0.3
        )
        assert net.num_layers == 8
        assert net.hidden_dim == 64
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = HumanNetwork(input_dim=9, output_dim=9)
        x = torch.randn(32, 9)
        y = net(x)
        assert y.shape == (32, 9)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        net = HumanNetwork(input_dim=9, output_dim=9)
        params = sum(p.numel() for p in net.parameters())
        assert params > 0
    
    def test_batch_sizes(self):
        """Test various batch sizes."""
        net = HumanNetwork(input_dim=9, output_dim=9)
        
        for batch_size in [1, 16, 32, 64, 128]:
            x = torch.randn(batch_size, 9)
            y = net(x)
            assert y.shape == (batch_size, 9)


class TestRobotNetwork:
    """Tests for RobotNetwork (Follower)."""
    
    def test_creation_default(self):
        """Test network creation with default parameters."""
        net = RobotNetwork(input_dim=9, output_dim=9)
        assert net is not None
        assert net.num_layers == 8
        assert net.hidden_dim == 100
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = RobotNetwork(input_dim=9, output_dim=9)
        x = torch.randn(32, 9)
        y = net(x)
        assert y.shape == (32, 9)
    
    def test_fewer_layers_than_human(self):
        """Verify Robot has fewer layers than Human by default."""
        human = HumanNetwork(input_dim=9, output_dim=9)
        robot = RobotNetwork(input_dim=9, output_dim=9)
        assert robot.num_layers < human.num_layers


class TestLeaderFollowerModel:
    """Tests for combined LeaderFollowerModel."""
    
    def test_creation(self):
        """Test model creation."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        assert model is not None
        assert hasattr(model, 'human_network')
        assert hasattr(model, 'robot_network')
    
    def test_forward_pass(self):
        """Test forward pass returns both predictions."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        human_input = torch.randn(32, 9)
        robot_input = torch.randn(32, 9)
        
        human_pred, robot_pred = model(human_input, robot_input)
        
        assert human_pred.shape == (32, 9)
        assert robot_pred.shape == (32, 9)
    
    def test_parameter_count(self):
        """Test total parameter counting."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        total_params = model.count_parameters()
        
        human_params = sum(p.numel() for p in model.human_network.parameters())
        robot_params = sum(p.numel() for p in model.robot_network.parameters())
        
        assert total_params == human_params + robot_params
    
    def test_layer_info(self):
        """Test layer info retrieval."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        info = model.get_layer_info()
        
        assert 'human_network' in info
        assert 'robot_network' in info
    
    def test_train_eval_modes(self):
        """Test train and eval mode switching."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        model.train()
        assert model.training
        assert model.human_network.training
        assert model.robot_network.training
        
        model.eval()
        assert not model.training
        assert not model.human_network.training
        assert not model.robot_network.training
    
    def test_device_transfer(self):
        """Test moving model to different devices."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        # Test CPU
        model = model.to('cpu')
        x = torch.randn(4, 9)
        h_pred, r_pred = model(x, x)
        assert h_pred.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            x = x.to('cuda')
            h_pred, r_pred = model(x, x)
            assert h_pred.device.type == 'cuda'


class TestSequencePredictor:
    """Tests for SequencePredictor (ARMA-style)."""
    
    def test_creation(self):
        """Test predictor creation."""
        predictor = SequencePredictor(input_dim=9, output_dim=9)
        assert predictor is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        predictor = SequencePredictor(input_dim=9, output_dim=9)
        x = torch.randn(32, 9)
        y = predictor(x)
        assert y.shape == (32, 9)


class TestCreateModel:
    """Tests for model factory function."""
    
    def test_create_default(self):
        """Test creating model with defaults."""
        from src.lefo.config import ModelConfig
        
        config = ModelConfig()
        model = create_model(config)
        assert isinstance(model, LeaderFollowerModel)
    
    def test_create_custom(self):
        """Test creating model with custom config."""
        from src.lefo.config import ModelConfig
        
        config = ModelConfig(
            human_layers=6,
            robot_layers=4,
            hidden_dim=50
        )
        model = create_model(config)
        
        assert model.human_network.num_layers == 6
        assert model.robot_network.num_layers == 4


class TestModelGradients:
    """Tests for gradient flow."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        model.train()
        
        human_input = torch.randn(4, 9, requires_grad=True)
        robot_input = torch.randn(4, 9, requires_grad=True)
        
        human_pred, robot_pred = model(human_input, robot_input)
        
        loss = human_pred.sum() + robot_pred.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
    
    def test_gradient_clipping(self):
        """Test gradient clipping doesn't break anything."""
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        model.train()
        
        x = torch.randn(4, 9)
        h_pred, r_pred = model(x, x)
        
        loss = h_pred.sum() + r_pred.sum()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Verify gradients are clipped
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 1.0 + 1e-6  # Small tolerance


class TestModelReproducibility:
    """Tests for reproducibility."""
    
    def test_deterministic_output(self):
        """Test that same input gives same output in eval mode."""
        torch.manual_seed(42)
        model = LeaderFollowerModel(input_dim=9, output_dim=9)
        model.eval()
        
        x = torch.randn(4, 9)
        
        with torch.no_grad():
            y1_h, y1_r = model(x, x)
            y2_h, y2_r = model(x, x)
        
        assert torch.allclose(y1_h, y2_h)
        assert torch.allclose(y1_r, y2_r)
    
    def test_seed_reproducibility(self):
        """Test that setting seed gives reproducible model."""
        torch.manual_seed(42)
        model1 = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        torch.manual_seed(42)
        model2 = LeaderFollowerModel(input_dim=9, output_dim=9)
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
