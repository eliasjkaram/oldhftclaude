# Premium Harvesting Options Trading Bot

## 🎯 Overview

You now have a sophisticated options trading bot that implements a premium harvesting strategy on Alpaca's paper trading platform. The bot sells out-of-the-money calls when implied volatility is elevated and buys them back when they decay or hit profit targets.

## 📁 Files Created

### Core Bots
1. **`working_options_bot.py`** - Main production bot (RECOMMENDED)
   - Uses real Alpaca options API
   - Simplified and reliable
   - Handles actual option orders

2. **`real_options_bot.py`** - Advanced version with full Greeks
   - More sophisticated analysis
   - Complex risk management
   - Higher computational overhead

3. **`advanced_premium_bot.py`** - Original sophisticated version
   - Black-Scholes calculations
   - Portfolio Greeks tracking
   - Market regime detection

### Testing & Utilities
- **`test_working_bot.py`** - Test the main bot
- **`simple_options_test.py`** - Basic API verification
- **`premium_bot_dashboard.py`** - Web monitoring dashboard

## 🚀 Quick Start

### 1. Run the Main Bot
```bash
cd /home/harry/alpaca-mcp
source .venv/bin/activate
python working_options_bot.py
```

### 2. Test First (Recommended)
```bash
python test_working_bot.py
```

### 3. Monitor with Dashboard
```bash
python premium_bot_dashboard.py
# Then open http://localhost:5000
```

## 🎛️ Bot Configuration

### Target Symbols
The bot scans these symbols for options opportunities:
- **SPY, QQQ, IWM** - Major ETFs
- **AAPL, MSFT, TSLA, NVDA** - Tech stocks
- **AA** - Other liquid options

### Strategy Parameters
```python
MIN_PREMIUM_YIELD = 1.0%     # Minimum premium yield
MAX_DTE = 56 days            # Maximum days to expiration
MIN_DTE = 7 days             # Minimum days to expiration
PROFIT_TARGET = 50%          # Take profit target
STOP_LOSS = 100%             # Maximum loss threshold
MAX_POSITIONS = 3            # Maximum concurrent positions
```

## 📊 Strategy Details

### Entry Criteria
- **Sell OTM calls** when:
  - Strike > current stock price
  - Premium yield > 1%
  - 1-8 weeks to expiration
  - Decent bid-ask spread

### Exit Criteria
- **Buy to close** when:
  - 50% profit achieved
  - 100% loss hit (stop loss)
  - Approaching expiration (< 7 days)

### Risk Management
- Risk 2% of portfolio per trade
- Maximum 3 concurrent positions
- Position sizing based on potential loss

## 💰 Expected Performance

### Typical Results
- **Win Rate**: 60-80% (time decay favors sellers)
- **Average Profit**: 25-50% per winning trade
- **Average Loss**: -50% to -100% per losing trade
- **Monthly Return**: 2-5% in normal markets

### Risk Factors
- **Volatility Spikes**: Can cause quick losses
- **Earnings Events**: Avoid positions through earnings
- **Market Crashes**: Naked calls can have unlimited risk

## 🛡️ Safety Features

### Built-in Protections
1. **Paper Trading Only** - No real money at risk
2. **Position Limits** - Max 3 positions
3. **Stop Losses** - Automatic loss limiting
4. **Expiration Management** - Closes before expiry

### Manual Overrides
- **Ctrl+C** to stop bot anytime
- **Edit parameters** in code for different risk levels
- **Monitor logs** for transparency

## 📈 Monitoring

### Log Files
- `working_options_bot.log` - Main bot activity
- `real_options_bot.log` - Advanced bot logs

### Key Metrics to Watch
- **Total P&L** - Overall profitability
- **Win Rate** - Percentage of profitable trades
- **Active Positions** - Current exposure
- **Days to Expiry** - Time decay progress

## ⚙️ Customization

### Adjust Risk Level
```python
# Conservative (in working_options_bot.py)
MAX_POSITIONS = 2
RISK_PER_TRADE = 0.01  # 1% risk

# Aggressive
MAX_POSITIONS = 5
RISK_PER_TRADE = 0.05  # 5% risk
```

### Target Different Symbols
```python
# Focus on ETFs only
TEST_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'IEF', 'TLT']

# Add more stocks
TEST_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'AMZN']
```

## 🔧 Troubleshooting

### Common Issues

1. **"No options found"**
   - Check if markets are open
   - Verify API credentials
   - Some symbols may not have options

2. **"Order rejected"**
   - Insufficient buying power
   - Invalid option symbol
   - Market hours restriction

3. **Bot stops running**
   - Check logs for errors
   - Restart with `python working_options_bot.py`
   - Verify internet connection

### Support Commands
```bash
# Check account status
python -c "from alpaca.trading.client import TradingClient; import os; client = TradingClient(os.getenv('ALPACA_PAPER_API_KEY'), os.getenv('ALPACA_PAPER_API_SECRET'), paper=True); print(client.get_account())"

# Test options access
python simple_options_test.py

# View recent orders
python -c "from alpaca.trading.client import TradingClient; import os; client = TradingClient(os.getenv('ALPACA_PAPER_API_KEY'), os.getenv('ALPACA_PAPER_API_SECRET'), paper=True); [print(order) for order in client.get_orders()]"
```

## 🎯 Next Steps

1. **Start with Paper Trading** - Get comfortable with the strategy
2. **Monitor Performance** - Track results for 1-2 months
3. **Adjust Parameters** - Optimize based on results
4. **Consider Live Trading** - Only after proven profitability

## ⚠️ Important Disclaimers

- **Paper Trading Only** - This is for educational purposes
- **No Financial Advice** - Not investment advice
- **Risk of Loss** - Options trading carries significant risk
- **Test Thoroughly** - Understand the strategy before using real money

---

## 🚀 Ready to Start!

Your premium harvesting bot is ready to run. Start with:

```bash
python working_options_bot.py
```

The bot will automatically scan for opportunities and execute trades based on the strategy parameters. Monitor the logs and enjoy watching your premium harvesting strategy in action! 🎯