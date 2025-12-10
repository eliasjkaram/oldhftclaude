# Comprehensive Backtesting and Fine-Tuning System

## Overview

This system provides a complete backtesting and fine-tuning framework for multiple AI trading algorithms. It downloads historical data from MinIO and Yahoo Finance, fine-tunes models on this data, and runs comprehensive backtests with rolling windows.

## Features

### 1. Data Integration
- **MinIO Integration**: Downloads option chain data from MinIO storage
- **Yahoo Finance**: Downloads historical stock price data
- **Technical Indicators**: Automatically calculates RSI, MACD, Bollinger Bands, moving averages, ATR, and more
- **Data Caching**: Intelligent caching to reduce redundant downloads

### 2. AI Models Supported
- **Enhanced Transformer V3**: State-of-the-art transformer with attention mechanisms
- **Mamba Trading Model**: Linear-time sequence modeling with selective state spaces
- **Financial CLIP**: Multi-modal analysis combining charts, text, and numerical data
- **PPO Trading Agent**: Reinforcement learning with proximal policy optimization
- **Multi-Agent System**: Ensemble of specialized trading agents
- **TimeGAN**: Generative model for market simulation and synthetic data

### 3. Backtesting Features
- **Rolling Window Analysis**: Tests strategies over multiple time periods
- **Comprehensive Metrics**: 
  - Returns (total, annualized)
  - Risk metrics (Sharpe, Sortino, Calmar ratios)
  - Drawdown analysis
  - Trade statistics (win rate, profit factor)
  - Statistical measures (skewness, kurtosis, VaR, CVaR)
- **Transaction Costs**: Includes commission and slippage modeling
- **Position Sizing**: Risk-based position management

### 4. Fine-Tuning Pipeline
- **Automated Training**: Fine-tunes each model on historical data
- **Validation Split**: Prevents overfitting with train/validation splits
- **Early Stopping**: Optimizes training time and prevents overfitting
- **Model Checkpointing**: Saves best performing models

### 5. Reporting and Visualization
- **Performance Comparison**: Side-by-side model comparisons
- **Visual Reports**: Charts showing equity curves, drawdowns, and metrics
- **Database Storage**: SQLite database for all results and trades
- **JSON Reports**: Detailed reports in machine-readable format

## Quick Start

### Basic Usage

```python
import asyncio
from comprehensive_backtest_system import ComprehensiveBacktestSystem, BacktestConfig

async def run_backtest():
    # Configure the backtest
    config = BacktestConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date="2020-01-01",
        end_date="2024-11-01",
        rolling_window_days=90,
        step_days=30,
        initial_capital=1000000
    )
    
    # Create and run system
    system = ComprehensiveBacktestSystem(config)
    report = await system.run_full_backtest()
    
    # Access results
    print(report['summary']['recommendations'])

# Run the backtest
asyncio.run(run_backtest())
```

### Test Script

Run the provided test script for a quick demo:

```bash
python test_backtest_system.py
```

## Configuration Options

### BacktestConfig Parameters

- `symbols`: List of stock symbols to test
- `start_date`: Backtest start date (YYYY-MM-DD)
- `end_date`: Backtest end date
- `rolling_window_days`: Size of each backtest window
- `step_days`: Days to step forward between windows
- `initial_capital`: Starting capital for each backtest
- `max_position_size`: Maximum position as fraction of portfolio
- `commission`: Transaction cost rate
- `slippage`: Market impact cost
- `fine_tune_epochs`: Training epochs for model fine-tuning
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `device`: 'cuda' or 'cpu' for computation

## Output Structure

### Reports Directory
```
backtest_reports/
├── backtest_results.db       # SQLite database with all results
├── backtest_report_*.json    # Detailed JSON reports
└── plots/
    ├── model_comparison.png  # Model performance comparison
    └── performance_heatmap.png # Normalized metrics heatmap
```

### Models Directory
```
fine_tuned_models/
├── transformer_best.pth
├── mamba_best.pth
├── clip_best.pth
├── ppo_best.pth
├── multi_agent_best.pth
└── timegan_best.pth
```

## Performance Metrics Explained

### Return Metrics
- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Yearly return rate
- **Alpha**: Excess return over market
- **Beta**: Market correlation

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return over maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns

### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses
- **Average Win/Loss**: Mean profit/loss per trade
- **Total Trades**: Number of completed trades
- **Average Holding Period**: Mean days per position

### Statistical Metrics
- **Skewness**: Return distribution asymmetry
- **Kurtosis**: Return distribution tail weight
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk

## Advanced Usage

### Custom Model Integration

```python
# Add your own model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        # Return price predictions
        return predictions

# Register with system
system.models['custom'] = CustomModel().to(device)
```

### Custom Metrics

```python
# Add custom metrics to calculation
def custom_metric(portfolio, initial_capital):
    # Calculate your metric
    return value

# Use in system
metrics = system._calculate_metrics(portfolio, initial_capital)
metrics.custom_metric = custom_metric(portfolio, initial_capital)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch_size or use smaller rolling windows
2. **Slow Training**: Decrease fine_tune_epochs or use GPU
3. **No MinIO Access**: System will use Yahoo Finance data only
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space for caching

## Dependencies

Key packages:
- torch (PyTorch)
- pandas
- numpy
- yfinance
- minio
- matplotlib
- seaborn
- sqlite3

## Future Enhancements

- Real-time backtesting
- Distributed computing support
- Additional AI models
- Live trading integration
- Web-based dashboard
- Cloud deployment options