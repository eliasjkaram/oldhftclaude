# Multi-Leg Options Trading System - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
1. Alpaca account with Options Level 2+ approved
2. Python 3.8+ installed
3. Required packages installed:
   ```bash
   pip install alpaca-py pandas numpy scipy scikit-learn joblib requests
   ```

### Launch the System
```bash
./launch_multileg_trader.sh
```

## 📊 Available Trading Modes

### 1. Test Execution (Market Hours Only)
- Tests basic multi-leg execution with a simple bull call spread
- Good for verifying your setup works
- Executes on SPY by default

### 2. Advanced Trader (Market Hours Only)
- Uses ML models to find best opportunities
- Executes sophisticated strategies: Iron Condors, Jade Lizards, etc.
- Includes risk management and position sizing

### 3. Continuous Trader (Market Hours Only)
- Scans every 5 minutes for opportunities
- Quick technical analysis on multiple symbols
- Executes strategies based on market conditions

### 4. 24/7 ML Trainer (NEW!)
- **During Market Hours**: Finds and executes live trades
- **During Off-Hours**: 
  - Downloads historical data
  - Backtests 14 different strategies
  - Trains ML models for better predictions
  - Optimizes strategy parameters

### 5. Position Monitor
- Shows all option positions with Greeks (Delta, Gamma, Theta, Vega)
- Identifies multi-leg strategies
- Calculates portfolio-wide risk metrics
- Watch mode available for continuous monitoring

## 🎯 Supported Strategies

1. **Directional**
   - Bull Call Spread
   - Bear Put Spread
   - Bull Put Spread (Credit)
   - Bear Call Spread (Credit)

2. **Neutral**
   - Iron Condor
   - Iron Butterfly
   - Calendar Spread
   - Diagonal Spread

3. **Volatility**
   - Long Straddle
   - Long Strangle
   - Ratio Spread

4. **Advanced**
   - Jade Lizard
   - Broken Wing Butterfly
   - Double Diagonal

## 📈 ML Models

The system trains and uses several ML models:

1. **Strategy Selector**: Chooses best strategy for market conditions
2. **Profit Predictor**: Estimates expected profit
3. **Strike Optimizer**: Finds optimal strike prices
4. **Risk Analyzer**: Evaluates trade risk
5. **Regime Detector**: Identifies market regime

## ⚙️ Configuration

### Symbols Traded
- Primary: SPY, QQQ, AAPL, TSLA, NVDA
- Secondary: META, MSFT, AMD, AMZN, GOOGL

### Risk Parameters
- Max positions: 20
- Max trades per hour: 5
- Position size: 2 contracts (adjustable)

### ML Training
- Retrains every 24 hours
- Uses 1 year of historical data
- Minimum 1,000 samples for training

## 🛡️ Risk Management

1. **Position Limits**: Maximum 20 open positions
2. **Rate Limiting**: Maximum 5 trades per hour
3. **Greeks Monitoring**: Real-time portfolio Greeks calculation
4. **P&L Alerts**: 
   - Profit alert: +$200 per position
   - Loss alert: -$100 per position

## 📊 Example Output

### Live Trading
```
🎯 Found 3 opportunities
   1. SPY - Iron Condor (confidence: 75%)
   2. AAPL - Bull Call Spread (confidence: 72%)
   3. TSLA - Jade Lizard (confidence: 68%)

Executing Iron Condor on SPY...
   BUY 1x SPY250718P00420000
   SELL 1x SPY250718P00430000
   SELL 1x SPY250718C00450000
   BUY 1x SPY250718C00460000
✅ Strategy executed successfully!
```

### ML Training (Off-Hours)
```
📊 Backtesting strategies...
   Testing iron_condor...
   • Trades: 245
   • Win Rate: 68.2%
   • Avg P&L: $124.50
   • Sharpe: 1.85

🧠 Training ML models...
   • Strategy selector accuracy: 74.3%
   • Profit predictor RMSE: $32.15
```

## 🚨 Important Notes

1. **Paper Trading**: System defaults to paper trading. Use `--live` flag for real money
2. **Market Hours**: Options trades only execute 9:30 AM - 4:00 PM ET
3. **Monitoring**: Always monitor your positions, especially in volatile markets
4. **Risk**: Options trading involves significant risk. Start small

## 📞 Troubleshooting

### Common Issues

1. **"Options market orders are only allowed during market hours"**
   - Solution: Run traders 1-3 only during market hours
   - The ML trainer (option 4) can run 24/7

2. **"No option contracts found"**
   - Solution: Market might be closed or symbol might not have liquid options

3. **"Insufficient options level"**
   - Solution: Contact Alpaca to upgrade to Options Level 2+

## 🎉 Quick Start Commands

```bash
# Test your setup
./launch_multileg_trader.sh
# Choose option 1

# Start 24/7 ML trainer
./launch_multileg_trader.sh
# Choose option 4

# Monitor positions with Greeks
./launch_multileg_trader.sh
# Choose option 5
```

## 📈 Performance Tracking

The system tracks:
- Total opportunities found
- Trades executed
- Win/loss ratio
- Total P&L
- Per-strategy performance
- ML model accuracy

Check `ml_models/performance_metrics.json` for detailed metrics.

---

Happy Trading! 🚀📊💰