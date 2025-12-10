# Historical Data Testing System

## Overview

The Historical Data Testing System is a comprehensive framework that automatically detects market status and switches between live and historical data modes. When markets are closed, it seamlessly transitions to historical data testing using random 5-day periods from 2022-2023.

## Key Features

1. **Automatic Market Detection**: Detects when markets are open/closed
2. **Multi-Source Data**: Supports yfinance, Alpaca, and MinIO data sources
3. **Random Period Selection**: Automatically selects random 5-day periods from 2022-2023
4. **Algorithm Integration**: Works with all existing trading algorithms
5. **Realistic Simulation**: Provides market-like conditions for testing
6. **Performance Tracking**: Tracks all metrics as if trading live
7. **Database Storage**: Stores all results for analysis

## Installation

```bash
# Required dependencies
pip install yfinance pandas numpy alpaca-trade-api minio pytz

# Optional for better performance
pip install torch  # For GPU-accelerated algorithms
```

## Quick Start

### Basic Usage

```python
from historical_data_testing_system import HistoricalDataTestingSystem

# Initialize the system
testing_system = HistoricalDataTestingSystem()

# Check market status
market_status = testing_system.market_detector.get_market_status()
print(f"Market is {'open' if market_status.is_open else 'closed'}")

# Run a testing session (automatically switches to historical if market is closed)
import asyncio
session = asyncio.run(testing_system.run_testing_session(duration_minutes=30))

print(f"Session completed: {session.session_id}")
print(f"Mode: {session.mode}")  # 'live' or 'historical'
```

### Integrating Your Algorithm

```python
# Define your trading algorithm
class MyTradingAlgorithm:
    def __init__(self, config):
        self.config = config
        
    def fetch_market_data(self, symbol):
        # This will be intercepted by the testing system
        pass
        
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        # This will be intercepted by the testing system
        pass
        
    async def analyze_and_trade(self):
        # Your trading logic here
        price_data = self.fetch_market_data('AAPL')
        if price_data['last'] > 150:
            self.place_order('AAPL', 'buy', 100)

# Register your algorithm
testing_system.register_algorithm('my_algo', MyTradingAlgorithm, {
    'param1': 'value1',
    'param2': 'value2'
})

# Run testing
session = asyncio.run(testing_system.run_testing_session(duration_minutes=60))
```

### Integration with Existing Systems

```python
from historical_data_testing_system import integrate_with_existing_system

# Automatically integrate with known systems
testing_system = integrate_with_existing_system('aggressive_trading', {
    'max_positions': 10,
    'confidence_threshold': 0.6
})

# Run testing
session = asyncio.run(testing_system.run_testing_session())
```

## How It Works

### Market Status Detection

The system checks market status using:
1. Alpaca API for accurate market hours
2. Fallback time-based detection
3. Handles weekends and holidays

### Historical Data Selection

When markets are closed:
1. Randomly selects a 5-day period from 2022-2023
2. Ensures no weekends in the selected period
3. Randomly chooses data source (yfinance, Alpaca, or MinIO)

### Data Sources

- **YFinance**: Free, reliable historical data
- **Alpaca**: Professional-grade data (requires API keys)
- **MinIO**: Your custom data storage
- **Mixed**: Automatically tries all sources with fallback

### Algorithm Wrapping

The system wraps your algorithms to:
- Intercept `fetch_market_data()` calls → returns historical data
- Intercept `place_order()` calls → simulates order execution
- Track all trades and performance metrics

### Performance Tracking

Tracks:
- Total return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Final equity

## Advanced Features

### Custom Data Periods

```python
from datetime import datetime
from historical_data_testing_system import HistoricalPeriod

# Create custom testing period
custom_period = HistoricalPeriod(
    start_date=datetime(2022, 6, 1),
    end_date=datetime(2022, 6, 5),
    symbols=['AAPL', 'MSFT', 'GOOGL']
)

# Use in session
session.historical_period = custom_period
```

### Multiple Algorithms

```python
# Register multiple algorithms
testing_system.register_algorithm('momentum', MomentumAlgo, {'threshold': 0.02})
testing_system.register_algorithm('mean_reversion', MeanReversionAlgo, {'window': 20})
testing_system.register_algorithm('breakout', BreakoutAlgo, {'lookback': 10})

# All run simultaneously during testing
```

### Performance Analysis

```python
# Get performance summary of recent sessions
summary_df = testing_system.get_performance_summary(last_n_sessions=10)

print(summary_df[['session_id', 'mode', 'final_equity', 'sharpe_ratio']])
```

### Database Queries

```python
import sqlite3
import pandas as pd

# Connect to results database
conn = sqlite3.connect('historical_testing.db')

# Get all historical sessions
query = """
SELECT * FROM testing_sessions 
WHERE mode = 'historical' 
ORDER BY start_time DESC
"""
df = pd.read_sql_query(query, conn)

# Get trade history
trades_query = """
SELECT * FROM trade_history 
WHERE session_id = ?
"""
trades_df = pd.read_sql_query(trades_query, conn, params=(session_id,))
```

## Configuration

### Environment Variables

```bash
# Alpaca API Keys (optional)
export ALPACA_PAPER_API_KEY="your_key"
export ALPACA_PAPER_API_SECRET="your_secret"

# Trading mode
export TRADING_MODE="paper"  # or "live"
```

### System Configuration

```python
# In historical_data_testing_system.py

# Market hours configuration
MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "16:00"
MARKET_TIMEZONE = "US/Eastern"

# Historical data range
HISTORICAL_START_DATE = "2022-01-01"
HISTORICAL_END_DATE = "2023-12-31"
HISTORICAL_PERIOD_DAYS = 5

# Default test symbols
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'SPY']
```

## Best Practices

1. **Algorithm Design**
   - Always implement `fetch_market_data()` and `place_order()` methods
   - Use `async def analyze_and_trade()` for main logic
   - Handle missing data gracefully

2. **Testing Duration**
   - Short tests (1-5 minutes) for development
   - Medium tests (30-60 minutes) for validation
   - Long tests (hours) for final verification

3. **Data Source Selection**
   - Use 'mixed' for best reliability
   - Use specific source for consistency
   - MinIO for custom/proprietary data

4. **Performance Evaluation**
   - Run multiple sessions for statistical significance
   - Compare different time periods
   - Test various market conditions

## Troubleshooting

### Common Issues

1. **No data available**
   - Check internet connection
   - Verify API keys
   - Try different data source

2. **Algorithm not trading**
   - Verify `analyze_and_trade()` method exists
   - Check algorithm initialization
   - Review order placement logic

3. **Performance metrics missing**
   - Ensure trades are being executed
   - Check position tracking
   - Verify price data availability

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
session = asyncio.run(testing_system.run_testing_session())
```

## Examples

See `test_historical_integration.py` for complete examples:
- Basic integration
- Multiple algorithms
- Existing system integration
- Performance analysis

## Support

For issues or questions:
1. Check the logs in `historical_testing_system.log`
2. Review the database in `historical_testing.db`
3. Enable debug mode for detailed output

---

The Historical Data Testing System provides a complete solution for testing trading algorithms with real market data, whether markets are open or closed. It seamlessly integrates with existing systems and provides comprehensive performance tracking.