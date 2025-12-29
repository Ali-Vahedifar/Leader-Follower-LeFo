# Leader-Follower (LeFo): Signal Prediction for Loss Mitigation in Tactile Internet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE MLSP 2025](https://img.shields.io/badge/IEEE-MLSP%202025-green.svg)](https://2025.ieeemlsp.org/)

Official implementation of the paper **"Signal Prediction for Loss Mitigation in Tactile Internet: A Leader-Follower Game-Theoretic Approach"** accepted at **IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025**, Istanbul, Turkey.

## 📖 Abstract

Tactile Internet (TI) requires achieving ultra-low latency and highly reliable packet delivery for haptic signals. In the presence of packet loss and delay, the signal prediction method provides a viable solution for recovering the missing signals. We introduce the **Leader-Follower (LeFo)** approach based on a cooperative Stackelberg game, which enables both users and robots to learn and predict actions. Our method achieves high prediction accuracy, ranging from **80.62% to 95.03%** for remote robot signals at the Human side and from **70.44% to 89.77%** for human operation signals at the remote Robot side.

## 🎯 Key Features

- **Game-Theoretic Framework**: Novel Stackelberg game formulation for human-robot interaction
- **MiniMax Optimization**: Bidirectional predictive modeling with KL divergence and Mutual Information
- **KNN-based MI Estimation**: Non-parametric mutual information estimation using K-Nearest Neighbors
- **Taylor Expansion Upper Bound**: Theoretical guarantee for model robustness against signal loss
- **Multi-Feature Prediction**: Supports position, velocity, and force prediction in 3D space

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Leader-Follower Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Human Side (Leader)              Robot Side (Follower)            │
│   ┌─────────────────┐              ┌─────────────────┐              │
│   │  12-Layer NN    │              │   8-Layer NN    │              │
│   │  (100 units)    │◄────────────►│  (100 units)    │              │
│   │  ReLU + Dropout │              │  ReLU + Dropout │              │
│   └────────┬────────┘              └────────┬────────┘              │
│            │                                │                       │
│            ▼                                ▼                       │
│   Max Mutual Info I(S_R, Ŝ_R)     Min KL(S_H || Ŝ_H)               │
│                                                                     │
│            └────────────┬───────────────────┘                       │
│                         ▼                                           │
│              MiniMax Optimization                                   │
│              L(H,R,θ) = E[max I] - E[min KL]                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
lefo-tactile-internet/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── setup.py                    # Package installation
├── pyproject.toml             # Modern Python packaging
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore patterns
│
├── configs/                   # Configuration files
│   ├── default.yaml          # Default hyperparameters
│   └── experiment_configs/   # Experiment-specific configs
│
├── src/lefo/                  # Main source code
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Neural network architectures
│   ├── data.py               # Data loading and preprocessing
│   ├── game.py               # Game theory components
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation metrics
│   ├── utils.py              # Utility functions
│   ├── config.py             # Configuration management
│   └── visualization.py      # Plotting and visualization
│
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Real-time inference
│   └── visualize_results.py  # Generate figures
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_upper_bound_verification.ipynb
│
├── tests/                     # Unit tests
│   ├── test_models.py
│   ├── test_game.py
│   ├── test_data.py
│   └── test_utils.py
│
├── figures/                   # Generated figures
│   └── generate_paper_figures.py
│
├── data/                      # Dataset directory
│   └── README.md             # Dataset download instructions
│
└── docs/                      # Documentation
    ├── installation.md
    ├── usage.md
    └── api_reference.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Install from source

```bash
# Clone the repository
git clone https://github.com/Ali-Vahedifar/Leader-Follower-LeFo.git
cd Leader-Follower-LeFo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Install with pip

```bash
pip install lefo-tactile
```

## 📊 Dataset

We use real-world haptic data traces from kinaesthetic interactions recorded using a **Novint Falcon haptic device** within a **Chai3D virtual environment**.

### Dataset Types

| Dataset | Description | Samples |
|---------|-------------|---------|
| Drag Max Stiffness Y | Dragging motion with maximum stiffness | ~10,000 |
| Horizontal Movement Fast | Fast horizontal free-air movement | ~8,000 |
| Horizontal Movement Slow | Slow horizontal free-air movement | ~12,000 |
| Tap and Hold Max Stiffness Z-Fast | Fast tapping and holding | ~9,000 |
| Tap and Hold Max Stiffness Z-Slow | Slow tapping and holding | ~11,000 |
| Tapping Max Stiffness Y-Z | Tapping with dual-axis stiffness | ~10,000 |
| Tapping Max Stiffness Z | Tapping with Z-axis stiffness | ~10,000 |

### Download Dataset

```bash
# Download from Zenodo
python scripts/download_data.py --output data/

# Or manually download from:
# https://zenodo.org/record/14924062
```

## 🎮 Usage

### Quick Start

```python
from lefo import LeaderFollowerModel, HapticDataset, Trainer
from lefo.config import Config

# Load configuration
config = Config.from_yaml('configs/default.yaml')

# Load dataset
dataset = HapticDataset(
    data_path='data/drag_max_stiffness_y.csv',
    deadband_threshold=0.1
)

# Initialize model
model = LeaderFollowerModel(
    human_layers=12,
    robot_layers=8,
    hidden_units=100,
    dropout=0.5
)

# Train
trainer = Trainer(model, config)
trainer.fit(dataset)

# Predict
predictions = model.predict(test_data)
```

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train specific dataset
python scripts/train.py \
    --dataset drag_max_stiffness_y \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.01

# Multi-GPU training
python scripts/train.py --config configs/default.yaml --gpus 0,1,2,3
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset data/test_data.csv

# Generate inference time benchmark
python scripts/evaluate.py --benchmark --iterations 1000
```

### Visualization

```bash
# Generate all paper figures
python figures/generate_paper_figures.py --output figures/

# Interactive visualization
python scripts/visualize_results.py --interactive
```

## 📈 Results

### Prediction Accuracy

| Dataset | Human Side (%) | Robot Side (%) |
|---------|---------------|----------------|
| Drag Max Stiffness Y | 95.03 ± 0.8 | 89.77 ± 1.2 |
| Horizontal Movement Fast | 92.45 ± 0.6 | 85.32 ± 0.9 |
| Horizontal Movement Slow | 93.78 ± 0.7 | 87.44 ± 1.0 |
| Tap and Hold Z-Fast | 88.21 ± 1.1 | 78.56 ± 1.5 |
| Tap and Hold Z-Slow | 90.67 ± 0.9 | 82.11 ± 1.3 |
| Tapping Max Y-Z | 82.34 ± 1.4 | 72.89 ± 1.8 |
| Tapping Max Z | 80.62 ± 1.5 | 70.44 ± 1.9 |

### Inference Time

| Dataset | Position (ms) | Velocity (ms) | Force (ms) |
|---------|--------------|---------------|------------|
| Drag Max Stiffness Y | 7.5-9.1 | 6.8-8.5 | 10.9-12.3 |
| Horizontal Movement Fast | 6.5-7.1 | 6.2-7.8 | 8.7-10.2 |
| Horizontal Movement Slow | 9.8-11.2 | 8.9-10.1 | 13.8-15.2 |

## 🔬 Theoretical Analysis

### Upper Bound for Loss Function

We establish an upper bound using Taylor expansion:

```
L_{n-1}(θ_n) - L_{n-1}(θ_{n-1}) ≤ (1/2) λ_max^{n-1} ||Δθ||²
```

where `λ_max^{n-1}` is the largest eigenvalue of the Hessian matrix.

### Verify Upper Bound

```python
from lefo.analysis import verify_upper_bound

# Verify theoretical bounds
results = verify_upper_bound(
    model=trained_model,
    dataset=test_dataset,
    compute_hessian=True
)
print(f"Loss: {results['loss']:.4f}")
print(f"Upper Bound: {results['upper_bound']:.4f}")
print(f"Bound Satisfied: {results['satisfied']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=lefo --cov-report=html
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{vahedifar2025signal,
  title={Signal Prediction for Loss Mitigation in Tactile Internet: 
         A Leader-Follower Game-Theoretic Approach},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={2025 IEEE International Workshop on Machine Learning 
             for Signal Processing (MLSP)},
  pages={1--6},
  year={2025},
  organization={IEEE},
  address={Istanbul, Turkey}
}
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was supported by:
- **TOAST Project**: European Union's Horizon Europe research and innovation program under Marie Skłodowska-Curie Actions Doctoral Network (Grant Agreement No. 101073465)
- **eTouch Project**: Danish Council for Independent Research (Grant No. 1127-00339B)
- **NordForsk**: Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

## 📧 Contact

- **Mohammad Ali Vahedifar** - av@ece.au.dk
- **Qi Zhang** - qz@ece.au.dk

DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

---

<p align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</p>
