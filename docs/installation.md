# Installation Guide

This guide covers installation of the LeFo (Leader-Follower) signal prediction framework.

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU** (optional): NVIDIA GPU with CUDA support for accelerated training

## Quick Installation

### From PyPI (Recommended)

```bash
pip install lefo-tactile-internet
```

### From Source

```bash
git clone https://github.com/mavahedifar/lefo-tactile-internet.git
cd lefo-tactile-internet
pip install -e .
```

## Detailed Installation

### 1. Create Virtual Environment (Recommended)

Using venv:
```bash
python -m venv lefo-env
source lefo-env/bin/activate  # Linux/macOS
# or
lefo-env\Scripts\activate  # Windows
```

Using conda:
```bash
conda create -n lefo python=3.10
conda activate lefo
```

### 2. Install PyTorch

Install PyTorch according to your system configuration:

**CPU only:**
```bash
pip install torch torchvision
```

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for more options.

### 3. Install LeFo

**Standard installation:**
```bash
pip install lefo-tactile-internet
```

**Development installation (includes test dependencies):**
```bash
pip install lefo-tactile-internet[dev]
```

**Full installation (includes all optional dependencies):**
```bash
pip install lefo-tactile-internet[all]
```

### 4. Verify Installation

```python
import lefo
print(lefo.__version__)

# Test model creation
from lefo.models import LeaderFollowerModel
model = LeaderFollowerModel(input_dim=9, output_dim=9)
print(f"Model parameters: {model.count_parameters():,}")
```

Or from command line:
```bash
lefo --version
```

## Download Datasets

Download the haptic datasets from Zenodo:

```bash
lefo download --output-dir ./data --dataset all
```

Or download specific datasets:
```bash
lefo download --output-dir ./data --dataset drag
```

## GPU Setup

### Verify CUDA Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Common GPU Issues

**1. CUDA not found:**
- Ensure NVIDIA drivers are installed
- Install CUDA toolkit matching your PyTorch version
- Set `CUDA_HOME` environment variable

**2. Out of memory:**
- Reduce batch size
- Use gradient checkpointing
- Use mixed precision training

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Version Conflicts

If you have version conflicts, create a fresh virtual environment:
```bash
python -m venv fresh-env
source fresh-env/bin/activate
pip install lefo-tactile-internet
```

### Permission Errors

On Linux/macOS, you may need to use `--user` flag or `sudo`:
```bash
pip install --user lefo-tactile-internet
```

## Docker Installation

A Docker image is available for reproducible environments:

```bash
# Pull the image
docker pull mavahedifar/lefo-tactile-internet:latest

# Run with GPU support
docker run --gpus all -it mavahedifar/lefo-tactile-internet:latest

# Run with mounted data directory
docker run --gpus all -v $(pwd)/data:/app/data -it mavahedifar/lefo-tactile-internet:latest
```

## Next Steps

After installation:

1. [Quick Start Guide](usage.md) - Get started with basic usage
2. [API Reference](api_reference.md) - Detailed API documentation
3. [Examples](../notebooks/) - Jupyter notebook tutorials

## Citation

If you use this software, please cite:

```bibtex
@inproceedings{vahedifar2025lefo,
  title={Leader-Follower Signal Prediction for Tactile Internet},
  author={Vahedifar, Mohammad Ali and Zhang, Qi},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025},
  organization={IEEE}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/mavahedifar/lefo-tactile-internet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mavahedifar/lefo-tactile-internet/discussions)
- **Email**: av@ece.au.dk
