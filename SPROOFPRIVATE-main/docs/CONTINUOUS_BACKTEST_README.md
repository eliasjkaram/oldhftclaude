# 🔄 Continuous Backtesting & Training System

## Overview

The Continuous Backtesting & Training System automatically discovers, tests, and optimizes all trading strategies in the alpaca-mcp project. It runs continuously, learning from simulated market conditions to improve trading performance over time.

## Key Features

### 1. **Automatic Strategy Discovery**
- Scans entire codebase for trading strategies
- Extracts configurable parameters automatically
- Supports classes with patterns: `*Strategy`, `*Bot`, `*Trader`, `*System`, `*Algo`

### 2. **Multiple Backtest Types**
- **Standard Backtest**: Sequential historical simulation
- **Walk-Forward Analysis**: Out-of-sample validation
- **Monte Carlo Simulation**: Statistical robustness testing
- **Bootstrap Resampling**: Confidence interval estimation

### 3. **Market Scenario Testing**
Tests strategies across 8 different market conditions:
- Bull Market (positive trend, normal volatility)
- Bear Market (negative trend, high volatility)
- High Volatility (sideways, extreme volatility)
- Low Volatility (calm markets)
- Sideways (range-bound)
- Crash (severe downturn)
- Recovery (post-crash rebound)
- Mixed (changing regimes)

### 4. **Machine Learning Optimization**
- Bayesian optimization with Optuna (if available)
- Multi-objective optimization (Sharpe, Return, Drawdown)
- Parameter range exploration
- Performance tracking over time

### 5. **Automatic Configuration Updates**
- Updates strategy config files with optimized parameters
- Only updates when improvement > 5%
- Creates backups before modifications
- Tracks all changes in database

## Installation & Setup

### Quick Start
```bash
# Make launch script executable
chmod +x launch_continuous_backtest.sh

# Run the system
./launch_continuous_backtest.sh
```

### Manual Setup
```bash
# Install dependencies
pip install numpy pandas scipy scikit-learn
pip install optuna  # Optional, for advanced optimization

# Run directly
python continuous_backtest_training_system.py
```

## Configuration

Edit `continuous_backtest_config.json` to customize:

```json
{
  "system_config": {
    "scan_interval_hours": 1,        # How often to scan for strategies
    "max_concurrent_backtests": 8,   # Parallel execution limit
    "optimization_threshold": 0.05,   # Min improvement to update configs
    "auto_update_configs": true       # Enable automatic updates
  },
  
  "backtest_settings": {
    "initial_capital": 100000,       # Starting capital for backtests
    "commission_rate": 0.001,        # 0.1% commission
    "slippage_rate": 0.0005,        # 0.05% slippage
    "max_position_size": 0.1         # 10% max per position
  }
}
```

## Monitoring

### Real-time Dashboard
```bash
# Install Streamlit
pip install streamlit plotly

# Launch dashboard
streamlit run continuous_improvement_dashboard.py
```

The dashboard shows:
- Total backtests run
- Average performance metrics
- Strategy leaderboard
- Optimization impact
- Market scenario analysis
- Real-time activity feed

### Reports

HTML and JSON reports are generated in `backtest_reports/` directory:
- Strategy performance summary
- Market scenario analysis
- Optimization details
- Recommendations

## How It Works

### 1. Strategy Discovery Phase
```python
# The system scans for classes like:
class MomentumStrategy:
    def __init__(self, lookback_period=20, threshold=0.02):
        self.lookback_period = lookback_period
        self.threshold = threshold
```

### 2. Backtest Execution
- Generates synthetic market data for each scenario
- Runs multiple variations of parameters
- Tests across different time periods
- Calculates comprehensive metrics

### 3. Optimization Process
```
Current Parameters → Run Backtests → Analyze Results → Optimize → Update Configs
                           ↑                                              ↓
                           ←──────────────────────────────────────────────
```

### 4. Performance Metrics Tracked
- **Sharpe Ratio**: Risk-adjusted returns
- **Total Return**: Absolute performance
- **Max Drawdown**: Worst peak-to-trough
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Database Schema

SQLite database (`continuous_training.db`) stores:

### backtest_results
- strategy_name
- parameters (JSON)
- market_scenario
- backtest_type
- sharpe_ratio
- total_return
- max_drawdown
- win_rate
- total_trades
- timestamp

### optimization_history
- strategy_name
- old_parameters (JSON)
- new_parameters (JSON)
- performance_improvement
- timestamp

## Example Output

### Console Output
```
============================================================
Starting training iteration 1
============================================================

📊 Phase 1: Scanning for strategies...
Found 42 strategies to analyze

📊 Phase 2: Running backtests...
Completed 630 backtests

📊 Phase 3: Optimizing parameters...
Optimized 12 strategies

Updated momentum_trader.lookback_period: 20 -> 23
Updated momentum_trader.threshold: 0.02 -> 0.018
Performance improvement: 12.3%

Report generated: backtest_reports/continuous_training_report_20241207_143022.html
```

### Strategy Performance Report
| Strategy | Best Sharpe | Best Return | Win Rate | Status |
|----------|-------------|-------------|----------|---------|
| momentum_trader | 1.82 | 24.3% | 58% | Optimized ✅ |
| mean_reversion_bot | 1.45 | 18.7% | 62% | Optimized ✅ |
| arbitrage_system | 2.13 | 31.2% | 71% | No Change |

## Best Practices

### 1. **Start Small**
- Begin with 1-2 strategies to test the system
- Gradually add more as you verify results

### 2. **Validate Results**
- Always paper trade optimized strategies before live trading
- Compare backtest results with forward performance

### 3. **Monitor Overfitting**
- Use walk-forward analysis
- Check out-of-sample performance
- Don't over-optimize

### 4. **Resource Management**
- Limit concurrent backtests based on CPU cores
- Use smaller parameter ranges initially
- Schedule intensive runs during off-hours

## Troubleshooting

### System Not Finding Strategies
- Check strategy class naming conventions
- Ensure parameters have default values
- Verify file permissions

### Poor Optimization Results
- Increase parameter variation range
- Add more market scenarios
- Extend backtest period

### High Memory Usage
- Reduce concurrent backtests
- Limit Monte Carlo simulations
- Clear old database records

## Advanced Features

### Custom Strategy Integration
```python
# Your strategy must have:
# 1. Recognizable class name
# 2. Parameters with defaults
# 3. Standard interface

class CustomTradingBot:
    def __init__(self, 
                 fast_period=10,      # Will be optimized
                 slow_period=30,      # Will be optimized
                 risk_percent=0.02):  # Will be optimized
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.risk_percent = risk_percent
```

### Adding New Market Scenarios
Edit `MarketSimulator.generate_market_data()` to add custom scenarios with specific characteristics.

### Custom Optimization Objectives
Modify `ParameterOptimizer.optimize_strategy()` to change the optimization scoring function.

## Safety Features

1. **Config Backups**: Always creates `.backup` files
2. **Improvement Threshold**: Only updates if >5% improvement
3. **Parameter Bounds**: Prevents extreme values
4. **Database Retention**: Keeps 90 days of history

## Future Enhancements

- [ ] Real-time strategy adaptation
- [ ] Multi-asset correlation analysis
- [ ] Deep learning optimization
- [ ] Cloud-based distributed backtesting
- [ ] Integration with paper trading system

---

The Continuous Backtesting & Training System transforms your trading strategies from static rules to dynamic, self-improving algorithms that adapt to changing market conditions.