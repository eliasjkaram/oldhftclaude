
# 📚 Alpaca SDK Best Practices Guide

## 1. Imports (✅ CORRECT)
```python
# Trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOptionContractsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetStatus

# Market Data
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.live import StockDataStream, OptionDataStream
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
```

## 2. Client Initialization (✅ CORRECT)
```python
# Paper Trading
trading_client = TradingClient(api_key, secret_key, paper=True)

# Live Trading (be careful!)
trading_client = TradingClient(api_key, secret_key, paper=False)

# Data Clients
stock_data_client = StockHistoricalDataClient(api_key, secret_key)
option_data_client = OptionHistoricalDataClient(api_key, secret_key)
```

## 3. Option Contracts (✅ CORRECT)
```python
# ✅ CORRECT: Use underlying_symbols as LIST
contracts_request = GetOptionContractsRequest(
    underlying_symbols=['AAPL'],  # List, not string!
    strike_price_gte=str(190),    # String, not float!
    strike_price_lte=str(210),    # String, not float!
    expiration_date_gte='2024-01-15',
    status=AssetStatus.ACTIVE
)

# ❌ INCORRECT: Don't use these patterns
contracts_request = GetOptionContractsRequest(
    underlying_symbol='AAPL',     # Wrong parameter name
    strike_price_gte=190.0,       # Wrong type (should be string)
    strike_price_lte=210.0        # Wrong type (should be string)
)
```

## 4. Multi-Leg Orders (✅ CORRECT)
```python
# ✅ CORRECT: Use OptionLegRequest and OrderClass.MLEG
legs = [
    OptionLegRequest(
        symbol='AAPL240119C00200000',
        side=OrderSide.BUY,
        ratio_qty=1
    ),
    OptionLegRequest(
        symbol='AAPL240119C00210000',
        side=OrderSide.SELL,
        ratio_qty=1
    )
]

order_request = LimitOrderRequest(
    qty=1,
    order_class=OrderClass.MLEG,  # Multi-leg class
    time_in_force=TimeInForce.DAY,
    legs=legs,
    limit_price=2.50
)
```

## 5. Error Handling (✅ BEST PRACTICE)
```python
try:
    order = trading_client.submit_order(order_request)
    logger.info(f"Order submitted: {order.id}")
except Exception as e:
    logger.error(f"Order failed: {e}")
    # Handle specific error types
    if "insufficient buying power" in str(e).lower():
        logger.warning("Insufficient buying power")
    elif "market closed" in str(e).lower():
        logger.warning("Market is closed")
```

## 6. Async Patterns (✅ RECOMMENDED)
```python
async def trading_strategy():
    # Concurrent data fetching
    tasks = [
        get_market_data('AAPL'),
        get_option_chain('AAPL'),
        check_positions()
    ]
    results = await asyncio.gather(*tasks)
    
    # Make trading decisions
    decisions = analyze_data(results)
    
    # Execute trades
    for decision in decisions:
        await execute_trade(decision)
```

## 7. Resource Management (✅ BEST PRACTICE)
```python
class TradingBot:
    def __init__(self):
        self.clients = {
            'trading': TradingClient(api_key, secret_key, paper=True),
            'stock_data': StockHistoricalDataClient(api_key, secret_key),
            'option_data': OptionHistoricalDataClient(api_key, secret_key)
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        pass
```

## 8. Common Mistakes to Avoid (❌ WRONG)
```python
# ❌ Don't use deprecated imports
import alpaca_trade_api
from alpaca_trade_api import REST

# ❌ Don't use wrong parameter names
GetOptionContractsRequest(underlying_symbol='AAPL')  # Wrong!

# ❌ Don't use wrong data types
GetOptionContractsRequest(
    underlying_symbols=['AAPL'],
    strike_price_gte=190.0  # Should be string!
)

# ❌ Don't ignore error handling
order = trading_client.submit_order(order_request)  # No try/except!
```

## 9. Performance Optimization (✅ RECOMMENDED)
```python
# Batch requests when possible
symbols = ['AAPL', 'TSLA', 'MSFT']
stock_request = StockBarsRequest(
    symbol_or_symbols=symbols,  # Batch request
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date
)

# Use appropriate timeframes
TimeFrame.Minute  # For intraday strategies
TimeFrame.Hour    # For short-term strategies  
TimeFrame.Day     # For swing trading
TimeFrame.Week    # For position trading
```

## 10. Testing Patterns (✅ BEST PRACTICE)
```python
# Always test with paper trading first
def create_trading_client(paper=True):
    return TradingClient(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        paper=paper  # Start with True!
    )

# Validate data before trading
def validate_market_data(data):
    if data is None:
        raise ValueError("No market data available")
    if len(data) == 0:
        raise ValueError("Empty market data")
    return True

# Test connectivity
def test_connection():
    client = create_trading_client(paper=True)
    try:
        account = client.get_account()
        logger.info(f"Connection OK: {account.status}")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False
```
        