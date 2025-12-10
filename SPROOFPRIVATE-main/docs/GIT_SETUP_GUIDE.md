# 🚀 Git Setup Guide for Alpaca-MCP Trading System

## 📋 Quick Setup

### 1. Initialize Git Repository
```bash
cd /home/harry/alpaca-mcp
git init
```

### 2. Create `.gitignore`
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Environment & Secrets
.env
.env.*
*.env
api_keys.py
*_credentials.json
*_config_secret.py
real_alpaca_config.py

# Data & Logs
*.db
*.sqlite
*.log
logs/
backtest_results/
minio_cache/
pipeline_output/
multi_year_results/
*.csv
*.parquet

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Build
build/
dist/
*.egg-info/
EOF
```

### 3. Create Initial README
```bash
cat > README.md << 'EOF'
# Alpaca-MCP Trading System

A comprehensive algorithmic trading platform with AI/ML capabilities, institutional-grade features, and multiple trading strategies.

## 🚀 Features

- **250+ Components** for modular trading system design
- **70+ Trading Algorithms** including ML-based strategies
- **25+ Options Strategies** with Greeks calculation
- **GPU Acceleration** for high-performance computing
- **Multi-Asset Support** (Stocks, Options, Futures, Crypto)
- **Real-time & Historical Data** integration
- **Paper & Live Trading** capabilities

## 📊 Main Systems

### Ultimate Trading Systems
- `enhanced_ultimate_engine.py` - Institutional-grade trading engine
- `FINAL_ULTIMATE_COMPLETE_SYSTEM.py` - Complete integrated system with GUI
- `enhanced_trading_gui.py` - Professional trading interface

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run system check
python FINAL_100_LAUNCHER.py

# Launch trading GUI
python enhanced_trading_gui.py
```

## 📁 Project Structure

```
alpaca-mcp/
├── core/              # Core infrastructure components
├── advanced/          # Advanced trading components
├── src/               # Source modules
├── tests/             # Test suites
├── docs/              # Documentation
└── systems/           # Complete trading systems
```

## 📝 Documentation

- [System Architecture](SYSTEM_ARCHITECTURE_GUIDE.md)
- [Complete Hierarchy](COMPLETE_SYSTEM_HIERARCHY.md)
- [Setup Guide](GIT_SETUP_GUIDE.md)

## ⚠️ Disclaimer

This software is for educational purposes. Trading involves risk. Past performance does not guarantee future results.

## 📄 License

[Your License Here]
EOF
```

### 4. Create requirements.txt
```bash
cat > requirements.txt << 'EOF'
# Core Dependencies
alpaca-py>=0.13.0
alpaca-trade-api>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
asyncio-throttle>=1.0.0

# Data & API
yfinance>=0.2.28
requests>=2.31.0
websocket-client>=1.6.0
aiohttp>=3.8.0
minio>=7.1.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
torch>=2.0.0
tensorflow>=2.13.0

# Technical Analysis
pandas-ta>=0.3.14
ta-lib>=0.4.28

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# GUI
tkinter  # Usually comes with Python

# Options Pricing
scipy>=1.11.0
statsmodels>=0.14.0
quantlib>=1.30  # Optional

# Utilities
python-dotenv>=1.0.0
pytest>=7.4.0
black>=23.0.0
pylint>=2.17.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# GPU (Optional)
cupy-cuda12x>=12.0.0  # For CUDA 12.x
# cupy-cuda11x>=12.0.0  # For CUDA 11.x
EOF
```

### 5. Create .env.example
```bash
cat > .env.example << 'EOF'
# Alpaca API Keys
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER_API_KEY=your_paper_api_key_here
ALPACA_PAPER_SECRET_KEY=your_paper_secret_key_here

# OpenRouter API (for AI features)
OPENROUTER_API_KEY=your_openrouter_key_here

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_minio_access_key
MINIO_SECRET_KEY=your_minio_secret_key
MINIO_BUCKET=trading-data

# Trading Configuration
TRADING_MODE=paper  # paper or live
MAX_POSITION_SIZE=0.02  # 2% of portfolio
RISK_LIMIT=0.10  # 10% max drawdown

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db
EOF
```

### 6. Initial Commit
```bash
# Add core files
git add .gitignore README.md requirements.txt .env.example
git add COMPLETE_SYSTEM_HIERARCHY.md GIT_SETUP_GUIDE.md
git add SYSTEM_ARCHITECTURE_GUIDE.md CLAUDE.md

# Add main systems
git add enhanced_ultimate_engine.py
git add FINAL_ULTIMATE_COMPLETE_SYSTEM.py
git add enhanced_trading_gui.py
git add FINAL_100_LAUNCHER.py

# Add core infrastructure
git add core/*.py
git add advanced/*.py
git add src/*.py

# Commit
git commit -m "Initial commit: Alpaca-MCP Trading System

- Core infrastructure with 250+ components
- Ultimate trading systems (institutional-grade)
- 70+ trading algorithms
- Complete documentation
- Git setup and configuration"
```

### 7. Create GitHub Repository
```bash
# Create repo on GitHub, then:
git remote add origin https://github.com/yourusername/alpaca-mcp.git
git branch -M main
git push -u origin main
```

## 📂 Recommended Git Workflow

### Branch Structure
```
main
├── develop
├── feature/new-algorithm
├── feature/ui-improvements
├── bugfix/data-validation
└── release/v1.0
```

### Commit Message Format
```
type(scope): subject

body (optional)

footer (optional)
```

Examples:
```bash
git commit -m "feat(core): add market regime detection"
git commit -m "fix(gui): resolve unicode display issues"
git commit -m "docs: update system architecture guide"
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

## 🔒 Security Notes

1. **Never commit**:
   - API keys
   - Passwords
   - `.env` files
   - Real trading configurations

2. **Use environment variables** for all sensitive data

3. **Review commits** before pushing:
   ```bash
   git diff --staged
   ```

## 🏷️ Versioning

Use semantic versioning: `MAJOR.MINOR.PATCH`

```bash
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0
```

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

Ready to version control your trading system! 🚀