# AI Trading System - Installation & Setup Guide
## Complete Setup Instructions for Real Data Integration

### 📋 Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Network**: Internet connection for API access

---

## 🚀 Quick Start (5 Minutes)

### 1. Clone and Navigate
```bash
cd /home/harry/alpaca-mcp/
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv trading_env

# Activate virtual environment
source trading_env/bin/activate  # Linux/macOS
# OR
trading_env\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
# Core trading packages
pip install yfinance alpaca-trade-api pandas-datareader

# Scientific computing
pip install numpy pandas scipy matplotlib

# AI/ML packages (optional for advanced features)
pip install scikit-learn requests asyncio

# Alternative installation if pip fails
pip install --break-system-packages yfinance alpaca-trade-api pandas-datareader numpy pandas scipy
```

### 4. Set Environment Variables
```bash
# Create .env file or set directly
export ALPACA_PAPER_API_KEY="PKCX98VZSJBQF79C1SD8"
export ALPACA_PAPER_API_SECRET="KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt"
export OPENROUTER_API_KEY="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"
export TRADING_MODE="paper"  # Use paper trading for safety
```

### 5. Test Installation
```bash
# Test basic functionality
python simplified_real_data_demo.py

# Test historical data (if yfinance installed)
python historical_data_engine.py

# Test complete system
python fixed_realistic_ai_system.py
```

---

## 🔧 Detailed Installation

### Option 1: Full Installation (Recommended)
```bash
# Install all dependencies for complete functionality
pip install yfinance alpaca-trade-api pandas-datareader numpy pandas scipy matplotlib plotly asyncio requests scikit-learn

# Test complete system
python real_data_ai_system.py
```

### Option 2: Minimal Installation (No External APIs)
```bash
# Install only basic scientific computing
pip install numpy pandas scipy matplotlib

# Run systems with simulated data
python fixed_realistic_ai_system.py
python optimized_ultimate_ai_system.py
```

### Option 3: Docker Installation (Advanced)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install yfinance alpaca-trade-api pandas-datareader numpy pandas scipy

ENV ALPACA_PAPER_API_KEY="PKCX98VZSJBQF79C1SD8"
ENV ALPACA_PAPER_API_SECRET="KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt"
ENV TRADING_MODE="paper"

CMD ["python", "real_data_ai_system.py"]
```

---

## 🔑 API Configuration

### 1. Alpaca Markets Setup
```python
# Paper Trading (Safe for testing)
ALPACA_PAPER_API_KEY = "PKCX98VZSJBQF79C1SD8"
ALPACA_PAPER_API_SECRET = "KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"

# Live Trading (REAL MONEY - Use with caution)
ALPACA_LIVE_API_KEY = "AK7LZKPVTPZTOTO9VVPM"
ALPACA_LIVE_API_SECRET = "2TjtRymW9aWXkWWQFwTThQGAKQrTbSWwLdz1LGKI"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
```

### 2. OpenRouter AI Setup
```python
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

### 3. Yahoo Finance (Free - No API Key Required)
```python
# Yahoo Finance works automatically with yfinance package
import yfinance as yf
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")
```

---

## 🎮 Running the Systems

### 1. Basic Realistic System (No External APIs)
```bash
python fixed_realistic_ai_system.py
```
**Output**: Realistic trading opportunities with proper financial calculations

### 2. Advanced Optimization System
```bash
python optimized_ultimate_ai_system.py
```
**Output**: Portfolio optimization with multiple algorithms (GA, PSO, Linear Programming)

### 3. Real Historical Data System
```bash
python historical_data_engine.py
```
**Output**: Analysis using actual market data from Yahoo Finance/Alpaca

### 4. Complete AI System
```bash
python real_data_ai_system.py
```
**Output**: Full AI trading system with real data integration

### 5. Multi-LLM Arbitrage Discovery
```bash
python autonomous_ai_arbitrage_agent.py
```
**Output**: AI-powered arbitrage discovery using multiple language models

### 6. Demo Systems
```bash
# Simplified demo (always works)
python simplified_real_data_demo.py

# AI arbitrage demo
python ai_arbitrage_demo.py
```

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
```
Error: ModuleNotFoundError: No module named 'yfinance'
Solution: pip install yfinance
```

#### 2. API Connection Errors
```
Error: Alpaca API connection failed
Solution: Check API keys and internet connection
```

#### 3. Permission Errors
```
Error: externally-managed-environment
Solution: Use virtual environment or add --break-system-packages
```

#### 4. Data Access Errors
```
Error: No price data found
Solution: Check symbol names and try different time periods
```

### Verification Commands
```bash
# Test API connections
python -c "import yfinance as yf; print('Yahoo Finance:', 'OK' if yf.Ticker('AAPL').history(period='1d').empty == False else 'FAIL')"

# Test Alpaca connection
python test_connection.py

# Test basic system
python -c "import numpy as np; import pandas as pd; print('Core packages: OK')"
```

---

## 📊 System Performance Expectations

### Realistic Performance Targets
- **Discovery Rate**: 5-50 opportunities per session
- **Confidence Levels**: 60-85% (realistic range)
- **Expected Returns**: 8-15% annually
- **Sharpe Ratios**: 0.4-2.0 (realistic range)
- **Processing Time**: <5 seconds per cycle

### Resource Requirements
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ for basic, 8GB+ for full system
- **Network**: Stable internet for real-time data
- **Storage**: 1GB+ for historical data caching

---

## 🛡️ Security Best Practices

### 1. API Key Security
```bash
# Never commit API keys to version control
echo ".env" >> .gitignore

# Use environment variables
export ALPACA_PAPER_API_KEY="your_key_here"

# Rotate keys regularly
```

### 2. Trading Safety
```bash
# Always start with paper trading
export TRADING_MODE="paper"

# Set position limits
export MAX_POSITION_SIZE="10000"  # $10K max

# Monitor all trades
tail -f *.log
```

### 3. Access Control
```bash
# Run with limited permissions
chmod 700 *.py

# Use dedicated trading user account
sudo useradd -m trader
su trader
```

---

## 📈 Next Steps

### 1. Basic Usage (Day 1)
- Install system and dependencies
- Run `simplified_real_data_demo.py`
- Review realistic opportunities

### 2. Advanced Features (Week 1)
- Configure API keys
- Run real data integration
- Analyze actual market correlations

### 3. Production Deployment (Month 1)
- Set up database for historical data
- Implement monitoring and alerting
- Deploy with proper security measures

### 4. Live Trading (When Ready)
- Extensive backtesting
- Paper trading validation
- Gradual position size increases
- Continuous monitoring

---

## 📞 Support & Resources

### Documentation Files
- `PROJECT_CONTEXT.md` - Complete system overview
- `FIXES_SUMMARY.md` - Details on realistic calculations
- `CLAUDE.md` - System context and achievements
- `realistic_fixes_patch.py` - Reusable fix library

### Key System Files
- `fixed_realistic_ai_system.py` - Main realistic system
- `historical_data_engine.py` - Real data integration
- `optimized_ultimate_ai_system.py` - Advanced optimization
- `autonomous_ai_arbitrage_agent.py` - Multi-LLM discovery

### Contact & Support
- Check log files for detailed error messages
- Review system documentation in repository
- Verify API key permissions and limits

---

**🚀 You're ready to start! Begin with `python simplified_real_data_demo.py` for a quick test.**