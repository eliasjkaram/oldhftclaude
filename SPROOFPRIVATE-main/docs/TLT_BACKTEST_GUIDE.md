# TLT Covered Call Backtest Guide

This guide explains how to run TLT (iShares 20+ Year Treasury Bond ETF) covered call backtests using the existing infrastructure at `/home/harry/alpaca-mcp/`.

## Overview

The TLT covered call strategy involves:
1. Holding 10,000 shares of TLT (100 contracts)
2. Selling call options when implied volatility (IV) is at peak levels
3. Using various IV prediction algorithms to determine optimal selling times

## Key Requirements

**IMPORTANT**: Use ONLY existing backtesting infrastructure - do not create new/simpler programs. The goal is to fix and use the existing comprehensive backtest system.

## Data Sources

The system uses real historical data from multiple sources in priority order:

1. **MinIO (Primary)**: Historical data stored at `uschristmas.us`
   - Path: `stockdb/stocks-2010-2025/TLT.csv`
   - Date column: `trade_date`
   - Contains data from 2010-2025

2. **Alpaca API (Secondary)**: Real-time and historical market data
   - Uses paper trading credentials from `.env`

3. **yfinance (Fallback)**: If other sources fail

## Key Files

### 1. `/home/harry/alpaca-mcp/comprehensive_backtest_system.py`
Main backtesting infrastructure with TLT-specific functionality:
- Fixed 30+ syntax errors to make it operational
- Contains IV prediction algorithms
- Handles options pricing and execution

### 2. `/home/harry/alpaca-mcp/enhanced_data_loader.py`
Multi-source data loader with MinIO integration:
```python
# MinIO configuration (line 56-63)
endpoint = 'uschristmas.us'  # NOT 'uschristmas.us:9000'
secure = True  # Uses HTTPS on port 443

# Date column handling for MinIO (line 135-139)
date_col = None
for col in ['Date', 'date', 'trade_date', 'timestamp']:
    if col in df.columns:
        date_col = col
        break

# Data quality filtering (line 161-168)
price_cols = ['Open', 'High', 'Low', 'Close']
valid_mask = (df[price_cols] > 0).all(axis=1)
df_clean = df[valid_mask]
```

### 3. `/home/harry/alpaca-mcp/tlt_3year_backtest.py`
Example implementation testing multiple IV algorithms:
```python
algorithms = {
    'IV_Percentile_75': lambda df, i: iv_percentile_threshold(df, i, 75),
    'Local_Peak_IV': local_peak_iv,
    'Regional_Peak_IV': regional_peak_iv,
    'Strike_Anomaly_IV': strike_anomaly_iv,
    'Expiration_Anomaly_IV': expiration_anomaly_iv,
    'Cross_Strike_IV': cross_strike_iv,
    'Term_Structure_IV': term_structure_iv,
    'Mean_Reversion_IV': mean_reversion_iv,
    'Volatility_Smile_IV': volatility_smile_iv,
    'Combined_Signal_IV': combined_signal_iv
}
```

### 4. `/home/harry/alpaca-mcp/.env`
Contains necessary credentials:
```
ALPACA_PAPER_API_KEY=PKEP9PIBDKOSUGHHY44Z
ALPACA_PAPER_API_SECRET=vCFAgqyJPRB5ESFNOnBR63lODruojVvoqtcUSVBP
MINIO_ACCESS_KEY=AKSTOCKDB2024
MINIO_SECRET_KEY=StockDB-Secret-Access-Key-2024-Secure!
MINIO_ENDPOINT=https://uschristmas.us
```

## Running a TLT Backtest

### Basic Usage

```bash
# Run comprehensive backtest with MinIO data
python comprehensive_backtest_system.py

# Run TLT-specific 3-year backtest
python tlt_3year_backtest.py

# Test MinIO data quality
python check_minio_data_quality.py
```

### Example Code

```python
from comprehensive_backtest_system import ComprehensiveBacktester
from datetime import datetime

# Initialize backtester
backtester = ComprehensiveBacktester(
    symbols=['TLT'],
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Run backtest with IV-based covered call strategy
results = backtester.run_covered_call_backtest(
    initial_shares=10000,
    iv_threshold=75  # Sell calls when IV percentile > 75
)
```

## IV Prediction Algorithms

The system includes 10 different IV prediction algorithms:

1. **IV_Percentile_75**: Sells when IV is in the 75th percentile or higher
2. **Local_Peak_IV**: Detects local maxima in IV (5-day window)
3. **Regional_Peak_IV**: Finds regional peaks (20-day window)
4. **Strike_Anomaly_IV**: Detects unusual IV patterns across strikes
5. **Expiration_Anomaly_IV**: Finds IV anomalies in term structure
6. **Cross_Strike_IV**: Analyzes IV relationships between strikes
7. **Term_Structure_IV**: Uses calendar spread opportunities
8. **Mean_Reversion_IV**: Trades IV mean reversion
9. **Volatility_Smile_IV**: Exploits volatility smile distortions
10. **Combined_Signal_IV**: Ensemble of multiple signals

## Common Issues and Solutions

### Issue: MinIO Connection Failed
```
Error: Connection to localhost:9000 failed
```
**Solution**: MinIO is hosted at `uschristmas.us` on HTTPS (port 443), not localhost:
```python
endpoint = 'uschristmas.us'  # Correct
# NOT: endpoint = 'uschristmas.us:9000'  # Wrong
```

### Issue: Date Column Not Found
```
KeyError: 'Date'
```
**Solution**: MinIO data uses `trade_date` column, not `Date`. The enhanced_data_loader.py handles this automatically.

### Issue: Zero Price Data
MinIO data contains some rows with zero prices (5 rows found).
**Solution**: The data loader automatically filters these out:
```python
valid_mask = (df[price_cols] > 0).all(axis=1)
df_clean = df[valid_mask]
```

### Issue: Syntax Errors in Backtest System
**Solution**: Run the comprehensive fix script or manually fix:
- Unmatched parentheses
- Incorrect dictionary/list syntax
- Missing colons in function definitions

## Results Interpretation

### Example Results (9-year MinIO backtest 2015-2023):
```
Strategy Return: -3.96%
Buy & Hold Return: -22.34%
Excess Return: +18.38%
Sharpe Ratio: 2.31
Win Rate: 85.7%
```

The covered call strategy significantly outperformed buy & hold during the bond bear market by:
- Collecting premium income during high IV periods
- Reducing downside through premium collection
- Maintaining consistent income despite price decline

### Key Metrics:
- **Strategy Return**: Total return including premiums and stock price changes
- **Buy & Hold Return**: Return from just holding TLT without options
- **Excess Return**: Strategy outperformance vs buy & hold
- **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
- **Win Rate**: Percentage of options that expired worthless (kept premium)

## Advanced Features

### 1. Multiple Data Sources
The system automatically tries data sources in order:
1. MinIO (complete historical data)
2. Alpaca (recent market data)
3. yfinance (fallback)
4. Synthetic data (last resort)

### 2. Comprehensive Indicators
- Historical Volatility (10, 20, 30-day)
- IV Proxy and IV Percentile
- RSI, MACD, Bollinger Bands
- Volume analysis

### 3. Options Pricing
Uses Black-Scholes model with:
- Strike: 3% out-of-the-money
- Expiration: 30 days
- Risk-free rate: 4.5%
- Premium bounds: $0.20 - 3% of stock price

## File Structure
```
/home/harry/alpaca-mcp/
├── comprehensive_backtest_system.py    # Main backtest infrastructure
├── enhanced_data_loader.py            # Multi-source data loader
├── tlt_3year_backtest.py             # TLT-specific implementation
├── tlt_minio_backtest.py             # MinIO data test
├── check_minio_data_quality.py       # Data quality checker
├── .env                              # Credentials (API keys)
└── advanced/
    └── ultra_high_accuracy_backtester.py  # ML ensemble system (needs fixes)
```

## Next Steps

1. **Fix remaining syntax errors** in ultra_high_accuracy_backtester.py if ML predictions needed
2. **Optimize IV thresholds** based on backtest results
3. **Add more sophisticated option pricing models** (e.g., Bjerksund-Stensland for American options)
4. **Implement real-time paper trading** using the backtested strategies
5. **Add portfolio risk management** for position sizing

## Important Notes

- Always use existing infrastructure - don't create simplified versions
- Test with real data (MinIO/Alpaca) before using synthetic data
- Monitor data quality - MinIO has some zero-price rows that need filtering
- The system date might show 2025 but actual market data goes through 2023
- Covered calls work well in declining/sideways markets like TLT 2021-2023

This guide should help future Claude instances quickly understand and use the TLT backtest system without recreating functionality that already exists.