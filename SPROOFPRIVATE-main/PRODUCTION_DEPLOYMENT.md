# 🚀 Production Deployment Guide

## Overview

This guide covers the complete production deployment of the Alpaca Trading System with all integrated components.

## 📋 System Components

### 1. **Core Trading Systems**
- **Position Monitor & Rebalancer** - Real-time monitoring of 60+ positions
- **HFT Integrated System** - High-frequency trading with <100ms latency
- **ML Trading System** - Machine learning predictions and signals
- **Options Trading System** - Advanced Greeks calculations and strategies
- **Execution Engine** - Smart order routing with TWAP/VWAP algorithms

### 2. **Infrastructure**
- **Free Monitoring System** - SQLite-based monitoring (no Prometheus fees)
- **MinIO Historical Data** - Access to 2002-2025 options data
- **Advanced Vector Database** - Multi-vector embeddings with proximity graph indexing
- **Security & Compliance** - Encryption, audit trails, regulatory checks
- **MCP Server** - LLM integration for AI-assisted trading

### 3. **Data Sources**
- **Alpaca Live Data** - Real-time market data via WebSocket
- **MinIO Historical** - 23+ years of options data (2002-2025, ~250k contracts/day)
- **Yahoo Finance** - Fallback data source

## 🔧 Installation

### Prerequisites
```bash
# System requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended for ML)
- GPU (optional, for HFT acceleration)
- Linux/macOS (Windows WSL2 supported)
```

### Quick Setup
```bash
# 1. Clone and setup
cd /home/harry/alpaca-mcp

# 2. Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/macOS
# or
trading_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements_real.txt

# 4. GPU support (optional)
pip install cupy-cuda11x  # For NVIDIA GPUs
```

## 🔐 Configuration

### 1. Environment Variables
Your `.env` file is already configured with:
```bash
# Paper Trading (Currently Active)
ALPACA_API_KEY_PAPER=PKEP9PIBDKOSUGHHY44Z
ALPACA_SECRET_KEY_PAPER=VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ

# Live Trading (Use with caution)
ALPACA_API_KEY=AK7LZKPVTPZTOTO9VVPM
ALPACA_SECRET_KEY=2TjtRymW9aWXkWWQFwTThQGAKQrTbSWwLdz1LGKI

# MinIO Historical Data
MINIO_ENDPOINT=uschristmas.us
MINIO_ACCESS_KEY=AKSTOCKDB2024
MINIO_SECRET_KEY=StockDB-Secret-Access-Key-2024-Secure!
```

### 2. Master Configuration
Create `config.json` for custom settings:
```json
{
  "mode": "paper",
  "components": {
    "position_monitor": true,
    "free_monitoring": true,
    "hft_system": false,
    "ml_trading": true,
    "mcp_server": true,
    "security": true
  },
  "symbols": {
    "stocks": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    "options": ["SPY", "QQQ"],
    "hft": ["SPY", "QQQ"]
  },
  "risk": {
    "max_position_size": 0.05,
    "max_daily_loss": 0.02,
    "max_positions": 20
  }
}
```

## 🚀 Running the System

### 1. Master Trading System (Recommended)
```bash
# Start all components
python src/production/master_trading_system.py

# This will start:
# ✅ Position monitoring
# ✅ Free monitoring with dashboard
# ✅ ML trading signals
# ✅ Security & compliance
# ✅ MCP server for LLM access
```

### 2. Individual Components

#### Position Monitor
```bash
# Monitor your 60 existing positions
python src/production/position_monitor_rebalancer.py

# Features:
# - Real-time P&L tracking
# - Stop loss alerts
# - Rebalancing recommendations
# - Tax loss harvesting
```

#### Free Monitoring Dashboard
```bash
# Start monitoring system
python src/production/free_monitoring_system.py

# Access dashboard:
# Open monitoring_dashboard.html in browser
# Auto-refreshes every 30 seconds
```

#### HFT System (Advanced Users)
```bash
# WARNING: Only enable after thorough testing
python src/production/hft_integrated_system.py

# Features:
# - <100ms execution target
# - GPU acceleration
# - 10 concurrent symbols
# - Rate limiting protection
```

#### Historical Backtesting
```bash
# Run backtests with MinIO data
python src/production/minio_historical_data.py

# Access 2002-2025 options data
# ~250,000 contracts per day
# 23+ years of historical data
```

#### MCP Server (LLM Integration)
```bash
# Start MCP server
python src/production/alpaca_mcp_server.py

# Use with Claude, GPT-4, etc.
# Provides trading tools and resources
```

## 📊 Monitoring & Dashboards

### 1. Free Monitoring Dashboard
- Location: `monitoring_dashboard.html`
- Updates: Every 5 minutes
- Metrics:
  - System performance (CPU, Memory)
  - Trading metrics (P&L, positions)
  - Active alerts
  - Recent errors

### 2. Position Dashboard
- Location: `position_dashboard.json`
- Updates: Every minute
- Contents:
  - All positions with real-time prices
  - Risk metrics
  - Recommendations

### 3. Logs
- System logs: `trading_system.log`
- Master logs: `master_trading.log`
- Audit trail: `.security/audit.db`

## 🛡️ Security Features

### 1. Encryption
- All sensitive data encrypted at rest
- Encryption key: `.security/encryption.key`
- Password hashing with PBKDF2

### 2. Access Control
- IP whitelist: `.security/ip_whitelist.txt`
- Rate limiting per user/action
- Failed attempt tracking

### 3. Compliance
- Pattern Day Trader (PDT) checks
- Position concentration limits
- Wash sale rule enforcement
- Order size limits

### 4. Audit Trail
- All actions logged to SQLite
- Compliance checks recorded
- Generate reports:
```bash
# View compliance report
cat compliance_report.json
```

## 🚨 Alerts & Notifications

### Current Alerts File
- Location: `current_alerts.json`
- Updated in real-time
- Integration ready for:
  - Email notifications
  - Slack webhooks
  - SMS alerts

### Alert Types
- **STOP_LOSS_WARNING** - Position near stop loss
- **HIGH_CPU** - System resource alerts
- **LARGE_DRAWDOWN** - Portfolio loss alerts
- **HIGH_LATENCY** - Performance degradation

## 🔍 Troubleshooting

### Common Issues

#### 1. "No module named 'alpaca'"
```bash
pip install alpaca-py
```

#### 2. "403 Forbidden" errors
```bash
# Check credentials in .env
# Ensure using correct API (paper vs live)
```

#### 3. High latency warnings
```bash
# Disable debug logging
export PYTHONOPTIMIZE=1

# Enable GPU acceleration
pip install cupy-cuda11x
```

#### 4. MinIO connection issues
```bash
# Test connection
python -c "from src.production.minio_historical_data import MinIOHistoricalData; m = MinIOHistoricalData(); print(m.get_statistics())"
```

## 📈 Performance Optimization

### 1. System Optimizations
```bash
# Run with optimizations
python -O src/production/master_trading_system.py

# Set process priority (Linux)
sudo nice -n -20 python src/production/master_trading_system.py
```

### 2. GPU Acceleration
```bash
# Check GPU availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Monitor GPU usage
nvidia-smi -l 1
```

### 3. Database Optimization
```bash
# Vacuum SQLite databases
sqlite3 monitoring_metrics.db "VACUUM;"
sqlite3 .security/audit.db "VACUUM;"
```

## 🚀 Production Checklist

### Before Going Live
- [ ] Test all components in paper mode
- [ ] Review all 60 existing positions
- [ ] Set appropriate risk limits
- [ ] Configure IP whitelist
- [ ] Enable security features
- [ ] Test stop loss orders
- [ ] Verify compliance rules
- [ ] Set up monitoring alerts
- [ ] Create backup of credentials
- [ ] Document emergency procedures

### Daily Operations
1. Check monitoring dashboard
2. Review position alerts
3. Verify system performance
4. Check compliance report
5. Monitor error logs

### Weekly Tasks
1. Review trading performance
2. Update ML models
3. Clean old log files
4. Backup configuration
5. Update security whitelist

## 🆘 Emergency Procedures

### 1. Emergency Shutdown
```bash
# Graceful shutdown
Ctrl+C in master terminal

# Force shutdown
pkill -f "python.*trading"
```

### 2. Disable Trading
```bash
# Edit config.json
"mode": "paper"  # Switch to paper
"components": {
  "hft_system": false  # Disable HFT
}
```

### 3. Position Liquidation
```python
# Emergency liquidation script
from src.production.position_monitor_rebalancer import PositionMonitorRebalancer
monitor = PositionMonitorRebalancer(api_key, api_secret, paper=False)
# WARNING: This would close all positions
# positions = monitor.emergency_liquidate_all()
```

## 📞 Support

### Resources
- Alpaca API Docs: https://alpaca.markets/docs/
- MinIO Data: Contact admin for access issues
- System Logs: Check `*.log` files
- Audit Trail: Query `.security/audit.db`

### Current Status
- ✅ 60 positions being monitored
- ✅ $1.7M buying power available
- ✅ Paper trading active
- ✅ All systems operational

---

**⚠️ IMPORTANT**: Always test changes in paper mode first. The system has access to both paper and live trading APIs. Double-check the `mode` setting before running in production.