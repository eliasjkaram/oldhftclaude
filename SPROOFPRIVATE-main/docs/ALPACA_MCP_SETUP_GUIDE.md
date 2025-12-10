# 🚀 Alpaca MCP Integration Setup Guide - V19

## Overview

The V19 Alpaca MCP Integration System combines:
- ✅ 76+ optimized trading algorithms from V18
- ✅ Alpaca trading API for live/paper trading
- ✅ Model Context Protocol (MCP) server for AI integration
- ✅ Real-time market data streaming
- ✅ Advanced portfolio optimization
- ✅ GPU-accelerated machine learning

## 📋 Prerequisites

### 1. Install Required Dependencies

```bash
# Core dependencies
pip install alpaca-trade-api
pip install alpaca-py
pip install mcp
pip install fastmcp
pip install python-dotenv

# Data analysis
pip install pandas numpy
pip install yfinance
pip install ta-lib

# Machine learning (optional, for GPU acceleration)
pip install torch torchvision torchaudio
pip install scikit-learn
pip install xgboost

# Visualization (optional)
pip install matplotlib seaborn
```

### 2. Alpaca Account Setup

1. Create an Alpaca account at https://alpaca.markets
2. Generate API keys from the dashboard
3. Save credentials to `/home/harry/alpaca-mcp/alpaca_config.json`:

```json
{
    "api_key": "YOUR_ALPACA_API_KEY",
    "secret_key": "YOUR_ALPACA_SECRET_KEY",
    "base_url": "https://paper-api.alpaca.markets",
    "use_paper": true
}
```

## 🔧 Installation & Configuration

### 1. Clone or Copy V19 System Files

Ensure you have these files in `/home/harry/alpaca-mcp/`:
- `v19_alpaca_mcp_integration.py` - Main integration system
- `v18_optimized_algorithms.py` - 76+ trading algorithms
- `alpaca_config.json` - API credentials
- `src/server.py` - MCP server implementation

### 2. Configure MCP Server for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alpaca-v19": {
      "command": "python",
      "args": ["/home/harry/alpaca-mcp/v19_alpaca_mcp_integration.py", "--server"],
      "env": {
        "PYTHONPATH": "/home/harry/alpaca-mcp",
        "ALPACA_CONFIG_FILE": "/home/harry/alpaca-mcp/alpaca_config.json"
      }
    }
  }
}
```

### 3. Test the Installation

```bash
# Test basic functionality
cd /home/harry/alpaca-mcp
python v19_integration_test.py

# Run the MCP server
python v19_alpaca_mcp_integration.py --server
```

## 📊 Available MCP Resources & Tools

### Resources (Read-Only Data)

| Resource | Description | Example |
|----------|-------------|---------|
| `account://info` | Get account information | Returns equity, cash, buying power |
| `algorithms://list` | List all 76+ algorithms | Shows categories, risk levels |
| `performance://summary` | Algorithm performance metrics | Win rates, returns, top performers |
| `market://data/{symbol}` | Real-time market data | Quote, bars, indicators |
| `positions://all` | All open positions | Current P&L, quantities |
| `orders://recent/{limit}` | Recent order history | Status, fills, timestamps |

### Tools (Actions)

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_with_algorithms` | Run analysis with multiple algorithms | symbol, algorithms[], timeframe |
| `execute_best_signal` | Execute highest confidence trade | symbol, min_confidence, max_position_size |
| `run_portfolio_optimization` | Optimize entire portfolio | capital, max_positions, risk_level |
| `monitor_positions` | Monitor and manage positions | rebalance, close_losers, profit_target |
| `place_market_order` | Place market order | symbol, qty, side |
| `place_limit_order` | Place limit order | symbol, qty, side, limit_price |

## 🎯 Usage Examples

### In Claude Desktop (with MCP configured)

```
User: "Analyze TSLA with all machine learning algorithms"
Claude: [Uses analyze_with_algorithms tool to run ML algorithms on TSLA]

User: "Execute the best trading signal for NVDA with at least 80% confidence"
Claude: [Uses execute_best_signal tool with min_confidence=0.8]

User: "Optimize my portfolio with $100,000 across 10 positions"
Claude: [Uses run_portfolio_optimization tool with specified parameters]

User: "Monitor my positions and close any with losses over 2%"
Claude: [Uses monitor_positions tool with close_losers=true]
```

### Programmatic Usage

```python
# Initialize the integration
from v19_alpaca_mcp_integration import AlpacaMCPIntegration

integration = AlpacaMCPIntegration()

# Analyze a symbol
result = await integration.analyze_with_algorithms(
    symbol="AAPL",
    algorithms=["Neural_Network", "LSTM_Prediction", "Mean_Reversion"]
)

# Execute best signal
trade = await integration.execute_best_signal(
    symbol="TSLA",
    min_confidence=0.75,
    max_position_size=10000
)

# Start live trading
integration.start_live_trading()
```

## 📈 Algorithm Categories

### Technical Analysis (12 algorithms)
- RSI_Oversold, MACD_Crossover, Bollinger_Squeeze
- Volume_Breakout, Support_Resistance, Fibonacci_Retracement
- Elliott_Wave, Ichimoku_Cloud, Pivot_Points
- Candlestick_Patterns, Chart_Patterns, Trend_Following

### Statistical & Quantitative (15 algorithms)
- Mean_Reversion, Momentum_Alpha, Pairs_Trading
- Statistical_Arbitrage, Cointegration, Kalman_Filter
- GARCH_Volatility, Correlation_Trading, Bayesian_Inference
- Monte_Carlo, Factor_Model, Risk_Parity

### Machine Learning (15 algorithms)
- Neural_Network, LSTM_Prediction, Transformer_Model
- XGBoost, Random_Forest, Deep_Learning
- Reinforcement_Learning, SVM_Classifier, Ensemble_Model
- CNN_Pattern, GAN_Prediction, Autoencoder

### Options Trading (12 algorithms)
- Volatility_Smile, Greeks_Optimization, Gamma_Scalping
- Vega_Trading, Theta_Decay, Delta_Neutral
- Volatility_Arbitrage, Dispersion_Trading, Skew_Trading
- Term_Structure, Options_Flow, Options_Sentiment

### High-Frequency Trading (8 algorithms)
- Order_Flow, Market_Making, Latency_Arbitrage
- HFT_Momentum, Cross_Exchange, Dark_Pool
- Liquidity_Detection, Order_Imbalance

### Advanced Strategies (14+ algorithms)
- Quantum_Algorithm, Fractal_Analysis, Wavelet_Transform
- Hidden_Markov, Genetic_Algorithm, Adaptive_Strategy
- Chaos_Theory, Swarm_Intelligence, Fuzzy_Logic

## 🛡️ Risk Management Features

1. **Position Sizing**: Dynamic Kelly Criterion-based sizing
2. **Stop Loss**: Automatic stop loss orders for all positions
3. **Take Profit**: Configurable profit targets
4. **Portfolio Limits**: Maximum position and sector exposure
5. **Drawdown Protection**: Automatic trading halt on excessive losses
6. **Market Regime Detection**: Adapts to market conditions

## ⚡ Performance Optimization

### GPU Acceleration
- Enable GPU for ML algorithms: `use_gpu=True`
- Supports CUDA-enabled NVIDIA GPUs
- 4-5x speedup for neural networks

### Parallel Processing
- Concurrent algorithm execution
- Multi-threaded data fetching
- Asynchronous order placement

### Caching
- Market data caching (5-second TTL)
- Algorithm result caching
- Historical data persistence

## 🔍 Monitoring & Debugging

### Logs
- Location: `/home/harry/alpaca-mcp/logs/`
- Levels: INFO, WARNING, ERROR
- Rotation: Daily, 7-day retention

### Performance Metrics
- Algorithm execution time
- Signal generation latency
- Order fill statistics
- Portfolio performance tracking

### Health Checks
```bash
# Check system status
python v19_alpaca_mcp_integration.py --status

# Run diagnostic tests
python v19_integration_test.py

# View recent trades
python v19_alpaca_mcp_integration.py --trades
```

## 🚨 Troubleshooting

### Common Issues

1. **"Alpaca client not initialized"**
   - Check API credentials in `alpaca_config.json`
   - Verify internet connection
   - Ensure API keys are active

2. **"Algorithm system not loaded"**
   - Install required dependencies (talib, torch)
   - Check `v18_optimized_algorithms.py` exists
   - Verify Python path includes project directory

3. **"MCP server not starting"**
   - Install mcp and fastmcp: `pip install mcp fastmcp`
   - Check Claude Desktop config syntax
   - Verify Python path in MCP config

4. **"GPU not detected"**
   - Install PyTorch with CUDA support
   - Check NVIDIA drivers are installed
   - Run `nvidia-smi` to verify GPU

## 📞 Support & Resources

- **Alpaca Documentation**: https://alpaca.markets/docs/
- **MCP Documentation**: https://modelcontextprotocol.io/
- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: https://forum.alpaca.markets/

## 🎉 Next Steps

1. **Paper Trading**: Start with paper trading to test strategies
2. **Backtest**: Use V18 backtesting on historical data
3. **Optimize**: Fine-tune algorithm parameters
4. **Monitor**: Set up alerts and monitoring
5. **Scale**: Add more algorithms and symbols
6. **Production**: Switch to live trading when ready

---

**Version**: 19.0.0  
**Last Updated**: June 2025  
**License**: MIT  
**Author**: AI-Enhanced Trading System