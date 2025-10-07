# 🤝 Contributing to Intelligent Cloud Scaling System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

---

## 📜 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

---

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs
Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Screenshots** if applicable

**Bug Report Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9]
- PyTorch Version: [e.g., 1.9.0]
```

### 💡 Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. Include:

- **Clear title and detailed description**
- **Use case** - why is this enhancement useful?
- **Possible implementation** approach
- **Alternatives considered**

### 🔧 Code Contributions
We welcome code contributions! Here are areas where you can help:

#### Model Improvements
- [ ] Implement GRU or Transformer models
- [ ] Add more feature engineering techniques
- [ ] Improve hyperparameter tuning
- [ ] Implement transfer learning

#### System Enhancements
- [ ] Add support for multiple AWS regions
- [ ] Implement cost optimization features
- [ ] Add anomaly detection
- [ ] Create mobile app for dashboard

#### Documentation
- [ ] Improve API documentation
- [ ] Add video tutorials
- [ ] Create troubleshooting guide
- [ ] Translate documentation

#### Testing
- [ ] Add unit tests
- [ ] Create integration tests
- [ ] Add performance benchmarks
- [ ] Implement CI/CD pipeline

---

## 🛠️ Development Setup

### 1. Fork & Clone
```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Intelligent-Cloud-Scaling-System-Architecture.git
cd Intelligent-Cloud-Scaling-System-Architecture
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

### 4. Configure AWS (Optional)
```bash
aws configure
# Enter your AWS credentials
```

### 5. Create Feature Branch
```bash
git checkout -b feature/YourFeatureName
```

---

## 🔄 Pull Request Process

### 1. Before Submitting
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Ensure all tests pass
- [ ] Follow coding standards
- [ ] Update CHANGELOG.md

### 2. PR Title Format
Use conventional commits format:
```
feat: Add GRU model implementation
fix: Correct scaling threshold logic
docs: Update README with new examples
test: Add unit tests for DataCollector
refactor: Optimize LSTM training loop
```

### 3. PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing
```

### 4. Review Process
1. Submit PR
2. Automated checks run
3. Code review by maintainers
4. Address feedback
5. Approval & merge

---

## 📝 Coding Standards

### Python Style Guide
Follow PEP 8 with these specifics:

#### Formatting
```python
# Use 4 spaces for indentation
def train_model(data, epochs=50):
    """
    Train LSTM model.
    
    Args:
        data (pd.DataFrame): Training data
        epochs (int): Number of training epochs
        
    Returns:
        torch.nn.Module: Trained model
    """
    pass
```

#### Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`

```python
# Good
def calculate_cpu_prediction(metrics):
    MAX_THRESHOLD = 70
    model = LSTMModel()
    _hidden_state = None
```

#### Imports
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from train_model import create_sequences
from utils import preprocess_data
```

#### Docstrings
Use Google style:
```python
def predict_load(model, sequence, scaler):
    """
    Predict CPU load for next 5 minutes.
    
    Args:
        model (nn.Module): Trained LSTM model
        sequence (np.array): Input sequence of shape (1, 24, 4)
        scaler (MinMaxScaler): Fitted scaler object
        
    Returns:
        float: Predicted CPU utilization (0-100)
        
    Raises:
        ValueError: If sequence shape is invalid
        
    Example:
        >>> predicted_cpu = predict_load(model, last_24_hours, scaler)
        >>> print(f"Predicted CPU: {predicted_cpu}%")
    """
    pass
```

### Code Quality Tools
```bash
# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py

# Sort imports
isort *.py
```

---

## 🧪 Testing Guidelines

### Unit Tests
```python
import unittest
import torch
from train_model import create_sequences

class TestSequenceCreation(unittest.TestCase):
    def test_sequence_length(self):
        """Test sequence creation with correct length"""
        data = np.random.rand(100, 4)
        X, y = create_sequences(data, sequence_length=12)
        self.assertEqual(X.shape[1], 12)
        
    def test_output_shape(self):
        """Test output array shape"""
        data = np.random.rand(100, 4)
        X, y = create_sequences(data, sequence_length=12)
        self.assertEqual(len(X), len(y))

if __name__ == '__main__':
    unittest.main()
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_training.py

# Run with coverage
python -m pytest --cov=.

# Run specific test
python -m pytest tests/test_training.py::test_model_accuracy
```

### Integration Tests
Test end-to-end workflows:
```python
def test_full_training_pipeline():
    """Test complete training workflow"""
    # 1. Generate data
    data = generate_synthetic_data()
    
    # 2. Train model
    model = train_model(data)
    
    # 3. Evaluate
    accuracy = evaluate_model(model)
    
    # 4. Assert
    assert accuracy > 0.75
```

---

## 📊 Performance Guidelines

### Model Training
- Document training time and resource usage
- Optimize for both CPU and GPU
- Use batch processing efficiently
- Implement early stopping

### Code Optimization
```python
# Use vectorization
# Bad
for i in range(len(data)):
    result[i] = data[i] * 2

# Good
result = data * 2

# Use list comprehensions
# Bad
squares = []
for x in range(10):
    squares.append(x**2)

# Good
squares = [x**2 for x in range(10)]
```

---

## 📚 Documentation

### Code Comments
```python
# Bad
x = x * 2  # Multiply by 2

# Good
# Scale features to range [0, 1] for neural network input
normalized_features = scaler.transform(raw_features)
```

### README Updates
When adding features, update:
- [ ] Table of Contents
- [ ] Features section
- [ ] Installation steps (if changed)
- [ ] Usage examples
- [ ] API documentation

---

## 🚀 Release Process

### Version Numbering
Follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes (v2.0.0)
- **MINOR**: New features (v1.1.0)
- **PATCH**: Bug fixes (v1.0.1)

### Changelog Format
```markdown
## [1.1.0] - 2025-01-15
### Added
- GRU model implementation
- Multi-region support

### Changed
- Improved LSTM accuracy to 79%

### Fixed
- Scaling threshold bug

### Deprecated
- Old training script (use train_model_v2.py)
```

---

## 🏷️ Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Docs improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - Will not be worked on

---

## 💬 Communication

### Where to Ask Questions
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Email**: Direct contact (see README)

### Response Times
- Issues: Within 48 hours
- PRs: Within 1 week
- Security issues: Within 24 hours

---

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

---

## 📞 Contact

- **Maintainer**: Gannoji Sathvik
- **Email**: your.email@example.com
- **GitHub**: [@GannojiSathvik](https://github.com/GannojiSathvik)

---

Thank you for contributing! 🎉

**Remember**: Every contribution, no matter how small, is valuable! 🌟
