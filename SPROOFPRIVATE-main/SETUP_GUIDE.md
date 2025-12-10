# Complete Setup Guide for Real Implementations

## Step 1: Install Dependencies

### Core Requirements
```bash
# Create and activate virtual environment
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate

# Install core requirements
pip install -r requirements_real.txt
```

### Optional High-Performance Libraries
```bash
# For American options pricing
pip install QuantLib-Python

# For GPU acceleration (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Set Environment Variables

### Linux/Mac
```bash
# Add to ~/.bashrc or ~/.zshrc
export ALPACA_API_KEY="YOUR_PAPER_API_KEY"
export ALPACA_SECRET_KEY="YOUR_PAPER_SECRET_KEY"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# Reload shell
source ~/.bashrc
```

### Windows
```powershell
# Set permanently in PowerShell
[Environment]::SetEnvironmentVariable("ALPACA_API_KEY", "YOUR_KEY", "User")
[Environment]::SetEnvironmentVariable("ALPACA_SECRET_KEY", "YOUR_SECRET", "User")
[Environment]::SetEnvironmentVariable("ALPACA_BASE_URL", "https://paper-api.alpaca.markets", "User")
```

### Using .env file (Recommended)
Create a `.env` file in the project root:
```env
ALPACA_API_KEY=YOUR_PAPER_API_KEY
ALPACA_SECRET_KEY=YOUR_PAPER_SECRET_KEY
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Step 3: Fix Syntax Errors

```bash
# Run the syntax fixer
python fix_syntax_errors.py

# Check the report
cat SYNTAX_FIX_REPORT.md
```

## Step 4: Integrate Real Implementations

```bash
# Run the integration script
python integrate_real_implementations.py

# This will:
# 1. Create a backup of your current code
# 2. Copy real implementations to proper locations
# 3. Update all imports
# 4. Generate an integration report
```

## Step 5: Test with Paper Trading

```bash
# Run comprehensive tests
python setup_and_test_paper_trading.py

# This will test:
# - Data provider connectivity
# - Options Greeks calculations
# - Execution algorithms
# - ML components
```

## Step 6: Verify Installation

### Quick Verification Script
Create `verify_setup.py`:

```python
#!/usr/bin/env python3
"""Quick verification of setup"""

import sys
import importlib

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

print("Checking required modules...")
modules = [
    'alpaca.trading.client',
    'pandas_ta',
    'py_vollib.black_scholes',
    'stable_baselines3',
    'transformers',
    'xgboost',
    'lightgbm'
]

all_good = all(check_module(m) for m in modules)

if all_good:
    print("\n✅ All modules installed correctly!")
else:
    print("\n❌ Some modules are missing. Run: pip install -r requirements_real.txt")
    
# Check environment variables
import os
if os.environ.get('ALPACA_API_KEY'):
    print("✅ ALPACA_API_KEY is set")
else:
    print("❌ ALPACA_API_KEY not set")
```

## Step 7: Start Paper Trading

### Simple Trading Script
```python
from src.real_implementations.advanced_data_provider import AdvancedDataProvider
from src.real_implementations.advanced_execution_algorithms import SmartOrderRouter
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

# Initialize
provider = AdvancedDataProvider()
trading_client = TradingClient(
    os.environ['ALPACA_API_KEY'],
    os.environ['ALPACA_SECRET_KEY'],
    paper=True
)

# Get market data
price = provider.get_current_price('AAPL')
print(f"AAPL Price: ${price}")

# Execute a small test trade
router = SmartOrderRouter(trading_client, data_stream)
metrics = await router.route_order(
    symbol='AAPL',
    quantity=1,  # Start with 1 share
    side=OrderSide.BUY,
    urgency=0.5
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Ensure you're in the virtual environment
   which python  # Should show trading_env/bin/python
   
   # Reinstall requirements
   pip install --upgrade -r requirements_real.txt
   ```

2. **API Connection Errors**
   ```python
   # Test API credentials
   from alpaca.trading.client import TradingClient
   client = TradingClient(api_key, secret_key, paper=True)
   account = client.get_account()
   print(f"Account status: {account.status}")
   ```

3. **Syntax Errors Persist**
   ```bash
   # Use black formatter
   pip install black
   black --line-length 100 src/
   ```

4. **Missing Dependencies**
   ```bash
   # Install specific missing module
   pip install module_name
   
   # Or use conda for complex dependencies
   conda install -c conda-forge quantlib
   ```

## Best Practices

1. **Start Small**
   - Test with 1-share orders first
   - Gradually increase position sizes
   - Monitor execution quality

2. **Use Paper Trading**
   - Run for at least 1-2 weeks
   - Test all strategies
   - Validate data accuracy

3. **Monitor Performance**
   - Check execution metrics
   - Compare with benchmarks
   - Log all activities

4. **Risk Management**
   - Set position limits
   - Use stop losses
   - Monitor account balance

## Next Steps

1. **Customize Strategies**
   - Modify execution parameters
   - Add new ML models
   - Create custom indicators

2. **Backtest Thoroughly**
   - Use historical data
   - Validate strategy performance
   - Check for overfitting

3. **Scale Gradually**
   - Increase data sources
   - Add more symbols
   - Enhance ML models

4. **Production Deployment**
   - Set up monitoring
   - Implement alerts
   - Create backup systems

## Support Resources

- **Alpaca Documentation**: https://alpaca.markets/docs/
- **pandas-ta**: https://github.com/twopirllc/pandas-ta
- **stable-baselines3**: https://stable-baselines3.readthedocs.io/
- **py_vollib**: https://github.com/vollib/py_vollib

## Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Environment variables set
- [ ] Syntax errors fixed
- [ ] Real implementations integrated
- [ ] Paper trading tests passed
- [ ] Simple trades executed successfully
- [ ] Monitoring set up
- [ ] Backup created

Once all items are checked, your system is ready for paper trading!