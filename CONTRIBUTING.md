# Contributing to LeFo

Thank you for your interest in contributing to the LeFo (Leader-Follower) signal prediction framework! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lefo-tactile-internet.git
   cd lefo-tactile-internet
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/mavahedifar/lefo-tactile-internet.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Verify Installation

Run the test suite to verify everything is set up correctly:
```bash
pytest tests/ -v
```

## Making Contributions

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize code for speed or memory
- **Examples**: Add usage examples or tutorials

### Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Pull Request Process

1. **Update documentation** if your changes affect the API
2. **Add tests** for new functionality
3. **Ensure all tests pass** before submitting
4. **Update the CHANGELOG** if applicable
5. **Request review** from maintainers

### PR Title Format

Use conventional commit style:
- `feat: Add new feature`
- `fix: Fix bug in ...`
- `docs: Update documentation`
- `test: Add tests for ...`
- `refactor: Refactor ...`
- `perf: Improve performance of ...`

## Coding Standards

### Python Style

We follow PEP 8 with the following specifics:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import sorting
- **Formatting**: Use `black` for code formatting
- **Type hints**: Use type hints for all public functions

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.

    Example:
        >>> function_name(1, "test")
        True
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src/lefo --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for >80% code coverage

### Test Categories

Use pytest markers for test categories:

```python
@pytest.mark.slow
def test_full_training():
    """Test that takes >10 seconds."""
    pass

@pytest.mark.gpu
def test_gpu_operations():
    """Test that requires GPU."""
    pass
```

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Keep API documentation up-to-date
- Add docstrings to all public functions

## Questions?

If you have questions or need help:

1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Contact the maintainers

## Attribution

Contributors will be acknowledged in the project's CONTRIBUTORS file and release notes.

---

Thank you for contributing to LeFo! Your efforts help advance research in tactile internet and haptic teleoperation.
