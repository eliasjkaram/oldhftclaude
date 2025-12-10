# Comprehensive Backtest System

## Overview

This comprehensive backtest system tests all trading algorithms against historical market data from 2005-2009, providing detailed performance metrics and comparisons.

## Features

### 1. Data Integration
- Downloads historical stock data from MinIO and Yahoo Finance
- Retrieves options data for arbitrage analysis
- Adds technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Implements intelligent caching to reduce download times

### 2. AI Models Tested
- **Enhanced Transformer V3**: Advanced transformer architecture for price prediction
- **Mamba Trading Model**: State-space model for sequential market data
- **Financial CLIP Model**: Multi-modal model combining price and text data
- **PPO Trading Agent**: Reinforcement learning agent for dynamic trading
- **Multi-Agent System**: Ensemble of specialized trading agents
- **TimeGAN Simulator**: Generative model for market simulation
- **Options Arbitrage System**: Put-call parity and volatility arbitrage

### 3. Backtest Methodology
- **Rolling Window**: 90-day windows with 30-day steps
- **Walk-Forward Analysis**: Models are fine-tuned on historical data before each window
- **Transaction Costs**: Includes commission (0.1%) and slippage (0.05%)
- **Position Sizing**: Maximum 10% of portfolio per position
- **Risk Management**: Built-in drawdown and exposure limits

### 4. Performance Metrics
- Total Return & Annualized Return
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown & Duration
- Win Rate & Profit Factor
- Average Win/Loss
- Calmar Ratio
- Value at Risk (VaR) & Conditional VaR
- Beta, Alpha, and Information Ratio
- Volatility, Skewness, and Kurtosis

## Usage

### Quick Start
```bash
# Run the comprehensive backtest
./launch_comprehensive_backtest.sh
```

### Python Script
```bash
# Direct execution
python3 run_comprehensive_backtest.py
```

### Custom Configuration
```python
from run_comprehensive_backtest import EnhancedBacktestRunner

# Create custom runner
runner = EnhancedBacktestRunner(
    start_date="2005-01-01",
    end_date="2009-12-31"
)

# Run backtest
report = await runner.run_comprehensive_backtest()
```

## Configuration Options

Edit the `BacktestConfig` in `run_comprehensive_backtest.py`:

```python
config = BacktestConfig(
    # Data settings
    start_date="2005-01-01",
    end_date="2009-12-31",
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    
    # Backtest settings
    rolling_window_days=90,
    step_days=30,
    initial_capital=1000000,
    max_position_size=0.1,
    commission=0.001,
    slippage=0.0005,
    
    # Fine-tuning settings
    fine_tune_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    
    # Hardware settings
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=4
)
```

## Output Files

### 1. Results Directory (`./backtest_results/`)
- `backtest_results_YYYYMMDD_HHMMSS.json`: Detailed results for all models
- `performance_summary_YYYYMMDD_HHMMSS.csv`: Performance metrics table
- `backtest_results.db`: SQLite database with all trades

### 2. Reports Directory (`./backtest_reports/`)
- `backtest_report_YYYYMMDD_HHMMSS.json`: Comprehensive analysis report
- Executive summary with recommendations
- Aggregate performance metrics

### 3. Plots Directory (`./backtest_reports/plots/`)
- `model_comparison.png`: Bar charts comparing model performance
- `performance_heatmap.png`: Normalized performance heatmap
- Individual equity curves and drawdown charts

### 4. Model Weights (`./fine_tuned_models/`)
- `transformer_best.pth`: Fine-tuned transformer weights
- `mamba_best.pth`: Fine-tuned Mamba model
- Other model checkpoints

## Interpreting Results

### Executive Summary
The system automatically identifies:
- **Best Overall Model**: Highest weighted score across all metrics
- **Highest Returns**: Model with maximum total return
- **Best Risk-Adjusted**: Highest Sharpe ratio
- **Most Consistent**: Lowest standard deviation of returns
- **Lowest Drawdown**: Minimal peak-to-trough loss

### Performance Metrics
- **Sharpe Ratio > 1.0**: Good risk-adjusted returns
- **Win Rate > 50%**: More winning trades than losing
- **Profit Factor > 1.5**: Winners outweigh losers significantly
- **Max Drawdown < 20%**: Acceptable risk level
- **Calmar Ratio > 1.0**: Good return relative to drawdown

## Troubleshooting

### Common Issues

1. **MinIO Connection Error**
   ```bash
   # Check MinIO credentials in minio_config.py
   # Verify internet connection
   # Falls back to Yahoo Finance automatically
   ```

2. **Insufficient Memory**
   ```bash
   # Reduce batch size in config
   # Test fewer symbols
   # Use CPU instead of GPU
   ```

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Advanced Usage

### Adding New Models
1. Implement model class with standard interface
2. Add to model initialization in `ComprehensiveBacktestSystem`
3. Ensure forward() method returns price predictions

### Custom Strategies
1. Modify trading logic in `run_backtest()` method
2. Add new metrics to `PerformanceMetrics` class
3. Update report generation accordingly

### Extending Data Sources
1. Add new data providers in `download_historical_data()`
2. Implement data validation and normalization
3. Update technical indicator calculations

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster training
2. **Parallel Processing**: Increase `num_workers` for data loading
3. **Caching**: Reuse downloaded data across runs
4. **Selective Testing**: Test specific models or time periods

## Support

For issues or questions:
1. Check logs in `comprehensive_backtest_run.log`
2. Review error messages in console output
3. Verify data availability for selected symbols
4. Ensure all dependencies are installed

## Future Enhancements

- Real-time data integration
- More sophisticated portfolio optimization
- Additional risk metrics
- Interactive web dashboard
- Cloud deployment support
- Automated hyperparameter tuning