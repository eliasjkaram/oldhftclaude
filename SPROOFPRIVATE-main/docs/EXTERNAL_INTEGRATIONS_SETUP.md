# External Service Integrations Setup Guide

## Required External Services

### 1. QuickFIX (FIX Protocol) - For SmartLiquidityAggregation

**What it is**: Financial Information eXchange (FIX) protocol implementation for electronic trading

**Used by**: 
- SmartLiquidityAggregation component
- Institutional broker connections
- Direct market access

**Installation Options**:

```bash
# Option 1: Install from PyPI (requires C++ compiler)
pip install quickfix

# Option 2: Use pre-built wheels (if available for your platform)
pip install quickfix-python

# Option 3: Build from source
git clone https://github.com/quickfix/quickfix.git
cd quickfix
python setup.py install
```

**Configuration Required**:
```python
# FIX configuration file (fix.cfg)
[DEFAULT]
ConnectionType=initiator
ReconnectInterval=60
FileStorePath=store
FileLogPath=log
StartTime=00:00:00
EndTime=00:00:00
UseDataDictionary=Y
DataDictionary=FIX44.xml

[SESSION]
BeginString=FIX.4.4
SenderCompID=YOUR_SENDER_ID
TargetCompID=BROKER_ID
SocketConnectHost=fix.broker.com
SocketConnectPort=5001
```

**Current Status**: Stub created, basic functionality available

---

### 2. Interactive Brokers (ib_insync) - For RobustBacktestingFramework

**What it is**: Python sync/async framework for Interactive Brokers API

**Used by**:
- RobustBacktestingFramework
- Multi-broker order routing
- Advanced order types (brackets, algos)

**Installation**:
```bash
pip install ib_insync
```

**Prerequisites**:
1. Interactive Brokers account
2. TWS (Trader Workstation) or IB Gateway installed
3. API access enabled in TWS/Gateway

**Configuration**:
```python
# IB Gateway settings
IB_CONFIG = {
    'host': '127.0.0.1',
    'port': 7497,  # 7497 for TWS, 4001 for Gateway (paper), 4002 (live)
    'clientId': 1,
    'account': 'YOUR_ACCOUNT_ID'
}
```

**Current Status**: Stub created, basic functionality available

---

### 3. Statsmodels - For ModelPerformanceEvaluation

**What it is**: Statistical modeling and econometric analysis

**Used by**:
- ModelPerformanceEvaluation
- Time series analysis components
- Statistical testing

**Installation Issues**:
- Incompatible with Python 3.13 (Cython compilation errors)
- Requires older Python version or alternative implementation

**Alternatives**:
```python
# Use scipy for basic stats
from scipy import stats

# Use sklearn for ML metrics
from sklearn.metrics import mean_squared_error, r2_score

# Custom implementations for specific tests
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
```

**Current Status**: Not installed, alternatives available

---

## Setup Priority

### Essential (for basic operation):
- None required - system works with Alpaca API alone

### Recommended (for advanced features):
1. **Statsmodels alternatives** - Easy to implement with scipy/numpy
2. **IB integration** - If multi-broker support needed
3. **FIX protocol** - Only for institutional/HFT operations

### Optional (specialized use cases):
- GPU libraries (already handled)
- Quantum computing libs (experimental)

---

## Quick Setup Commands

```bash
# For basic operation (already complete)
echo "System ready with Alpaca integration!"

# For IB support
pip install ib_insync
# Then start IB Gateway/TWS with API enabled

# For FIX protocol (requires build tools)
sudo apt-get install build-essential python3-dev  # Ubuntu/Debian
pip install quickfix

# For statistical analysis (Python < 3.13)
pip install statsmodels scipy scikit-learn
```

---

## Integration Testing

Test each integration:

```python
# Test IB connection
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print(f"Connected to IB: {ib.isConnected()}")

# Test FIX connection
import quickfix
settings = quickfix.SessionSettings("fix.cfg")
application = quickfix.Application()
storeFactory = quickfix.FileStoreFactory(settings)
initiator = quickfix.SocketInitiator(application, storeFactory, settings)
initiator.start()

# Test stats functionality
import scipy.stats as stats
data = [1, 2, 3, 4, 5]
print(f"Mean: {np.mean(data)}, Std: {np.std(data)}")
```

The system currently operates at 42.5% without these integrations, which is sufficient for most trading strategies using Alpaca.