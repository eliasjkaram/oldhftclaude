# Contributing to Alpaca-MCP Trading System

Thank you for your interest in contributing to the Alpaca-MCP Trading System! This document provides guidelines for contributing to this project.

## 🔒 Security First

**CRITICAL**: This is a trading system with access to real financial accounts. Security is paramount.

### Before Contributing

1. **Never commit credentials** or sensitive data
2. **Always run security audit** before commits: `python security_audit.py`
3. **Use paper trading** for all development and testing
4. **Review the security checklist** in PRE_PUSH_CHECKLIST.md

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Help maintain a secure codebase
- Report security issues privately
- Test thoroughly before submitting

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. Git with pre-commit hooks
3. Virtual environment tool (venv, conda)
4. Alpaca paper trading account
5. Understanding of trading risks

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/alpaca-mcp.git
cd alpaca-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your PAPER trading credentials only

# Run security audit
python security_audit.py
```

## Project Structure

The project should be organized as follows:

```
alpaca-mcp/
├── src/                    # Core source code
│   ├── algorithms/         # Trading algorithms
│   │   ├── __init__.py
│   │   ├── base.py        # Base algorithm class
│   │   ├── arbitrage/     # Arbitrage strategies
│   │   ├── options/       # Options strategies
│   │   └── ml/            # ML-based strategies
│   ├── ai/                # AI/ML components
│   │   ├── models/        # Trained models
│   │   ├── training/      # Training pipelines
│   │   └── inference/     # Inference engines
│   ├── data/              # Data processing
│   │   ├── collectors/    # Data collection
│   │   ├── processors/    # Data preprocessing
│   │   └── storage/       # Data storage interfaces
│   ├── execution/         # Order execution
│   │   ├── brokers/       # Broker interfaces
│   │   ├── orders/        # Order management
│   │   └── positions/     # Position tracking
│   ├── risk/              # Risk management
│   │   ├── limits/        # Position limits
│   │   ├── metrics/       # Risk metrics
│   │   └── alerts/        # Risk alerts
│   ├── utils/             # Utilities
│   │   ├── config.py      # Configuration
│   │   ├── logging.py     # Logging setup
│   │   └── helpers.py     # Helper functions
│   └── main.py           # Main entry point
├── config/                # Configuration files
│   ├── trading/          # Trading configs
│   ├── risk/             # Risk parameters
│   └── system/           # System settings
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
├── scripts/              # Utility scripts
│   ├── deployment/       # Deployment scripts
│   ├── analysis/         # Analysis tools
│   └── maintenance/      # Maintenance scripts
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── guides/           # User guides
│   └── architecture/     # System architecture
├── examples/             # Example implementations
├── monitoring/           # Monitoring configs
│   ├── prometheus/       # Prometheus configs
│   └── grafana/          # Grafana dashboards
└── deployment/           # Deployment files
    ├── docker/           # Docker files
    └── kubernetes/       # K8s manifests
```

### File Organization Guidelines

1. **Move files to appropriate directories** based on functionality
2. **Remove duplicate files** - keep only the best implementation
3. **Rename files** to be descriptive and follow conventions
4. **Group related functionality** together
5. **Separate concerns** - one file, one purpose

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Changes

- Write clean, documented code
- Follow the coding standards
- Add appropriate tests
- Update documentation

### 3. Run Quality Checks

```bash
# Run security audit
python security_audit.py

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
isort src/ --check

# Run pre-commit hooks manually
pre-commit run --all-files
```

### 4. Commit Changes

```bash
# Stage changes selectively
git add -p

# Commit with descriptive message
git commit -m "feat: add new arbitrage detection algorithm

- Implement cross-exchange arbitrage detection
- Add unit tests for arbitrage calculator
- Update documentation with examples"
```

#### Commit Message Format

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks
- `perf:` Performance improvements
- `security:` Security improvements

### 5. Push and Create PR

```bash
# Run final validation
./pre_push_validate.sh

# Push to your fork
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

1. **Follow PEP 8** with these modifications:
   - Line length: 100 characters
   - Use Black for formatting
   - Use isort for imports

2. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private methods: `_leading_underscore`

3. **Documentation**:
   ```python
   def calculate_position_size(
       capital: float,
       risk_percentage: float,
       stop_loss_price: float,
       entry_price: float
   ) -> float:
       """
       Calculate optimal position size based on risk parameters.
       
       Args:
           capital: Total available capital
           risk_percentage: Percentage of capital to risk (0-100)
           stop_loss_price: Stop loss price level
           entry_price: Entry price level
           
       Returns:
           float: Number of shares to trade
           
       Raises:
           ValueError: If risk_percentage is invalid
           
       Example:
           >>> size = calculate_position_size(10000, 2, 95, 100)
           >>> print(f"Position size: {size} shares")
       """
   ```

4. **Type Hints**:
   - Use type hints for all functions
   - Use `typing` module for complex types
   - Consider using `mypy` for type checking

5. **Error Handling**:
   ```python
   try:
       result = risky_operation()
   except SpecificException as e:
       logger.error(f"Operation failed: {e}")
       # Handle gracefully
       raise
   finally:
       # Cleanup resources
       cleanup()
   ```

### Security Guidelines

1. **Never hardcode credentials**
2. **Use environment variables** for configuration
3. **Validate all inputs** especially from external sources
4. **Log security events** but not sensitive data
5. **Use secure communication** (HTTPS, SSL)
6. **Implement rate limiting** for API calls
7. **Handle errors gracefully** without exposing internals

## Testing Guidelines

### Test Structure

```python
# tests/unit/test_arbitrage.py
import pytest
from src.algorithms.arbitrage import ArbitrageDetector

class TestArbitrageDetector:
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return ArbitrageDetector(threshold=0.01)
    
    def test_detects_simple_arbitrage(self, detector):
        """Test detection of simple arbitrage opportunity."""
        # Arrange
        prices = {'EXCHANGE_A': 100.00, 'EXCHANGE_B': 101.50}
        
        # Act
        opportunities = detector.detect(prices)
        
        # Assert
        assert len(opportunities) == 1
        assert opportunities[0].profit_percentage > 1.0
```

### Testing Requirements

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Edge Cases**: Test boundary conditions
4. **Error Cases**: Test error handling
5. **Performance Tests**: Test under load
6. **Security Tests**: Test security measures

### Test Coverage

- Aim for >80% code coverage
- Critical paths must have 100% coverage
- Run coverage report: `pytest --cov=src tests/`

## Documentation

### Code Documentation

- All public APIs must be documented
- Include docstrings with examples
- Keep comments up-to-date
- Document complex algorithms

### Project Documentation

- Update README.md for major changes
- Add guides for new features
- Document configuration options
- Include troubleshooting tips

### API Documentation

- Use consistent format
- Include request/response examples
- Document error codes
- Version your APIs

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**
2. **Run security audit**
3. **Update documentation**
4. **Fill out PR template**
5. **Link related issues**
6. **Request review from maintainers**

### PR Title Format

Use the same format as commit messages:
- `feat: add multi-exchange arbitrage detection`
- `fix: resolve memory leak in data collector`
- `docs: update options trading guide`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Security
- [ ] Security audit passed
- [ ] No credentials exposed
- [ ] Input validation added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Release Process

1. **Version Bumping**: Follow semantic versioning
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Full regression testing
4. **Security**: Final security audit
5. **Documentation**: Ensure docs are current
6. **Deployment**: Follow deployment guide

## Getting Help

- **Issues**: Use GitHub issues for bugs/features
- **Discussions**: Use GitHub discussions for questions
- **Security**: Email security@yourdomain.com for security issues
- **Documentation**: Check docs/ directory first

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for helping make Alpaca-MCP better and more secure!