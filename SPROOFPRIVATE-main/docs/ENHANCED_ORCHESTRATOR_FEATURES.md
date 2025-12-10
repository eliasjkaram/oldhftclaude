# 🚀 Enhanced Master Orchestrator V2 with Historical Data Testing

## 🎯 New Features Added

### 1. **Automatic Market Detection & Mode Switching**
The system now automatically detects market status and switches between:
- **LIVE MODE**: During market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
- **HISTORICAL MODE**: Nights, weekends, and holidays

### 2. **Historical Data Testing (2022-2023)**
When markets are closed:
- Randomly selects 5-day periods from 2022-2023
- Uses data from multiple sources (yfinance, Alpaca, MinIO)
- Runs all algorithms as if trading live
- Tracks performance metrics

### 3. **Continuous Learning**
- Algorithms train 24/7 using historical data when markets closed
- No downtime for model improvement
- Performance tracked separately for live vs historical

## 📊 How It Works

### Market Status Detection
```python
# Checks every 5 minutes
if market_is_closed():
    switch_to_historical_mode()
else:
    switch_to_live_mode()
```

### Historical Period Selection
```python
# Randomly selects from 2022-2023
start_date = random_date_between('2022-01-01', '2023-12-31')
end_date = start_date + 5_trading_days
```

### Data Source Priority
1. **MinIO** (if available)
2. **Alpaca Markets API** (with valid credentials)
3. **yfinance** (fallback)

## 🔧 Configuration

### Environment Variables
```bash
# For historical mode
export HISTORICAL_MODE=true
export HISTORICAL_START_DATE=2022-06-15
export HISTORICAL_END_DATE=2022-06-21
export HISTORICAL_SYMBOLS=SPY,AAPL,MSFT,GOOGL,TSLA
```

### Launch Options
```bash
# Quick 5-minute test
./launch_smart_orchestrator.sh
# Choose option 1

# Full 360-minute run
./launch_smart_orchestrator.sh
# Choose option 2

# 24-hour continuous run
./launch_smart_orchestrator.sh
# Choose option 4
```

## 📈 Performance Tracking

### Database Schema
New table: `historical_sessions`
- `mode`: 'live' or 'historical'
- `period_start`: Start date of historical period
- `period_end`: End date of historical period
- `symbols`: Symbols tested
- `performance_data`: JSON with metrics
  - Total trades
  - Successful trades
  - Total return
  - Sharpe ratio
  - Max drawdown

### Example Performance Data
```json
{
  "total_trades": 127,
  "successful_trades": 78,
  "total_return": 0.0542,
  "sharpe_ratio": 1.85,
  "max_drawdown": -0.0234,
  "win_rate": 0.614,
  "avg_trade_return": 0.0004
}
```

## 🎮 Usage Examples

### Basic Launch
```bash
# System automatically detects market status
python master_orchestrator_v2.py --runtime 360
```

### Force Historical Mode
```bash
# Useful for testing
export FORCE_HISTORICAL_MODE=true
python master_orchestrator_v2.py --test
```

### Monitor Performance
```bash
# Real-time dashboard
python orchestrator_dashboard.py --watch

# Generate report
python orchestrator_dashboard.py --report --output performance.txt
```

## 🔍 Status Messages

### During Market Hours
```
Market Status: OPEN
Mode: LIVE
Data Source: Real-time market data
```

### After Hours/Weekends
```
Market Status: CLOSED
Mode: HISTORICAL
Data Source: Historical (2022-2023)
Current Period: 2022-10-15 to 2022-10-21
Symbols: SPY, AAPL, MSFT, GOOGL, TSLA
```

## 🛡️ Improvements from V1

1. **No More 17-Hour Runs**: Automatic shutdown at specified time
2. **No Idle Time**: Uses historical data when markets closed
3. **Better Error Handling**: Graceful mode transitions
4. **Performance Tracking**: Separate metrics for each mode
5. **Resource Optimization**: Only runs market-dependent processes when needed

## 📊 Benefits

1. **24/7 Algorithm Training**: Continuous improvement
2. **Risk-Free Testing**: Test strategies on historical data
3. **Performance Validation**: Compare live vs historical performance
4. **Market Replay**: Test how algorithms would have performed in past conditions
5. **Data Source Flexibility**: Automatic fallback between sources

## 🚀 Quick Start

```bash
# Make launcher executable
chmod +x launch_smart_orchestrator.sh

# Run with smart mode detection
./launch_smart_orchestrator.sh

# System will automatically:
# - Detect if markets are open
# - Switch to appropriate mode
# - Select random historical periods if needed
# - Track all performance metrics
```

## 📈 Expected Behavior

### Weekday 10 AM ET
- Mode: LIVE
- Uses real-time market data
- All trading algorithms active

### Weekend or 8 PM ET
- Mode: HISTORICAL
- Selects random 5-day period from 2022-2023
- Runs algorithms on historical data
- Tracks hypothetical performance

### Market Open Transition
- Detects market opening
- Gracefully stops historical mode
- Switches all algorithms to live data
- Continues performance tracking

The enhanced orchestrator provides true 24/7 operation with intelligent mode switching!