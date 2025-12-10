# Alpaca Trading System - Fixed Version

## Overview
This fixed version addresses the common sub-penny pricing issues and order type errors encountered with the Alpaca API.

## Fixed Issues

### 1. Sub-Penny Pricing Errors
**Problem**: Prices like `$615.4353` or `$142.63920000000002` cause validation errors.

**Solution**: All prices are rounded to 2 decimal places using the `round_price()` function:
```python
def round_price(price: float, decimals: int = 2) -> float:
    return float(Decimal(str(price)).quantize(Decimal(f'0.{"0" * decimals}'), rounding=ROUND_DOWN))
```

### 2. Bracket Order Implementation
**Problem**: Invalid take_profit and stop_loss prices due to floating-point precision.

**Solution**: Proper bracket order creation with rounded prices:
```python
# Calculate and round bracket prices
take_profit_price = round_price(current_price * (1 + take_profit_pct))
stop_loss_price = round_price(current_price * (1 - stop_loss_pct))

# Create bracket order
order_request = MarketOrderRequest(
    symbol=symbol,
    qty=qty,
    side=order_side,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    take_profit={"limit_price": take_profit_price},
    stop_loss={"stop_price": stop_loss_price}
)
```

### 3. Trailing Stop Orders
**Problem**: "market orders must not have trail_percent" error.

**Solution**: Use dedicated `TrailingStopOrderRequest` instead of adding trail_percent to market orders:
```python
order_request = TrailingStopOrderRequest(
    symbol=symbol,
    qty=qty,
    side=order_side,
    time_in_force=TimeInForce.DAY,
    trail_percent=trail_percent
)
```

### 4. Options Trading
The system now supports both real options trading (if available) and synthetic strategies:

- **Real Options**: Uses `get_option_contracts()` to find and trade actual option contracts
- **Synthetic Strategies**: Falls back to stock-based strategies that mimic option behavior

## Files

1. **execute_trades_fixed.py** - Main trading system with all fixes
2. **check_alpaca_options.py** - Utility to check options trading capabilities
3. **demo_fixed_trading.py** - Comprehensive demonstration of all fixes

## Usage

### Basic Usage
```bash
python execute_trades_fixed.py
```

### Check Options Capability
```bash
python check_alpaca_options.py
```

### Run Demo
```bash
python demo_fixed_trading.py
```

## API Credentials
The system uses the provided credentials:
- API Key: `PKCX98VZSJBQF79C1SD8`
- Secret Key: `KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt`

## Key Features

1. **Automatic Price Rounding**: Prevents sub-penny pricing errors
2. **Proper Order Types**: Uses correct order classes for different strategies
3. **Options Support**: Automatically detects and uses options if available
4. **Synthetic Strategies**: Falls back to stock-based strategies when options aren't available
5. **Comprehensive Error Handling**: Graceful fallbacks and informative error messages

## Example Output
```
📊 ACCOUNT INFORMATION
==================================================
Equity: $100,000.00
Cash: $100,000.00
Buying Power: $200,000.00

🚀 EXECUTING TRADES
==================================================
1. Testing Bracket Order on SPY...
   Current Price: $450.25
   Take Profit: $454.75
   Stop Loss: $447.74
   ✅ Bracket order submitted

2. Testing Options Trading on AAPL...
   ✅ Options trading methods available
   📊 Trading option: AAPL240125C00150000
   ✅ Option order submitted
```

## Notes
- All monetary values are rounded to avoid sub-penny issues
- The system automatically falls back to synthetic strategies if options aren't available
- Paper trading is used by default for safety