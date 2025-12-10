# Alpaca Deployment Guide for GPI Deep Learning HFT Bot

## Alpaca Account Requirements

### Account Setup Checklist

- [ ] **Alpaca Account Type**: Paid subscription (10,000 requests/minute)
- [ ] **Options Trading**: Enabled on your account
- [ ] **Account Minimum**: $100,000+ for proper position sizing
- [ ] **API Keys**: Both paper and live keys configured
- [ ] **Market Data**: Alpaca Market Data subscription active

### Alpaca API Configuration

```yaml
# config/alpaca_production.yaml
trading:
  # Paper Trading First
  paper_trading: true
  base_url: "https://paper-api.alpaca.markets"
  
  # Production URLs (when ready)
  # paper_trading: false
  # base_url: "https://api.alpaca.markets"
  
  # API Credentials
  api_key: "YOUR_ALPACA_API_KEY"
  api_secret: "YOUR_ALPACA_API_SECRET"
  
  # Data URLs
  data_url: "https://data.alpaca.markets"
  stream_url: "wss://stream.data.alpaca.markets/v2"
  
  # Rate Limiting
  max_requests_per_minute: 10000
  request_timeout: 30
  retry_attempts: 3

# Alpaca-Specific Settings
alpaca:
  # Order Settings
  extended_hours: false
  order_class: "simple"  # simple, bracket, oco, oto
  time_in_force: "day"   # day, gtc, ioc, fok
  
  # Position Settings
  max_position_value: 50000
  max_positions: 20
  
  # Options Settings
  options_enabled: true
  options_level: 2  # Your approved level
  
  # Data Subscriptions
  subscriptions:
    - "trades"
    - "quotes"
    - "bars"
    - "options"
```

## Pre-Deployment Validation

### 1. Validate Alpaca Connection

```bash
# Test paper trading connection
python -c "
from alpaca.trading.client import TradingClient
client = TradingClient('YOUR_KEY', 'YOUR_SECRET', paper=True)
account = client.get_account()
print(f'Account Status: {account.status}')
print(f'Buying Power: ${float(account.buying_power):,.2f}')
print(f'PDT Check: {account.pattern_day_trader}')
"
```

### 2. Verify Data Access

```bash
# Test market data access
python -c "
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

client = StockHistoricalDataClient('YOUR_KEY', 'YOUR_SECRET')
request = StockBarsRequest(
    symbol_or_symbols='SPY',
    timeframe=TimeFrame.Minute,
    start=datetime.now() - timedelta(days=1)
)
bars = client.get_stock_bars(request)
print(f'Retrieved {len(bars["SPY"])} bars')
"
```

### 3. Check Options Access

```bash
# Test options data access
python -c "
from alpaca.data.historical import OptionsHistoricalDataClient
client = OptionsHistoricalDataClient('YOUR_KEY', 'YOUR_SECRET')
# Test options chain retrieval
print('Options access verified')
"
```

## Alpaca-Specific Risk Management

### PDT (Pattern Day Trading) Compliance

```python
# Automatic PDT compliance in risk manager
class AlpacaRiskManager:
    def check_pdt_compliance(self, account):
        if account.pattern_day_trader:
            # PDT account - can day trade freely
            self.day_trade_limit = float('inf')
        else:
            # Regular account - limit to 3 day trades per 5 days
            self.day_trade_limit = 3
            self.day_trade_count = account.daytrade_count
```

### Alpaca Order Types

```python
# Market Orders (fastest execution)
from alpaca.trading.requests import MarketOrderRequest
order = MarketOrderRequest(
    symbol="AAPL",
    qty=100,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# Limit Orders (price control)
from alpaca.trading.requests import LimitOrderRequest
order = LimitOrderRequest(
    symbol="AAPL",
    qty=100,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    limit_price=150.00
)

# Bracket Orders (built-in risk management)
from alpaca.trading.requests import BracketOrderRequest
order = BracketOrderRequest(
    symbol="AAPL",
    qty=100,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    take_profit={"limit_price": 155.00},
    stop_loss={"stop_price": 145.00}
)
```

## Deployment Steps for Alpaca

### Step 1: Paper Trading Validation (1-2 weeks)

```bash
# Start with paper trading
cd /home/harry/alpaca-mcp/production_roadmap

# Run with Alpaca paper endpoint
python month3_robustness.py \
    --mode live \
    --config config/alpaca_paper.yaml \
    --symbols AAPL MSFT SPY QQQ TSLA
```

**Success Criteria**:
- 95%+ order fill rate
- <100ms average latency
- No API rate limit hits
- Consistent daily operation

### Step 2: Limited Live Testing (1 week)

```yaml
# config/alpaca_limited_live.yaml
trading:
  paper_trading: false  # LIVE TRADING
  base_url: "https://api.alpaca.markets"
  
risk_management:
  max_position_size: 10000  # Start small
  max_daily_loss: 500       # Tight stop
  max_positions: 5          # Limited exposure
```

### Step 3: Full Production Deployment

```yaml
# config/alpaca_production.yaml
trading:
  paper_trading: false
  base_url: "https://api.alpaca.markets"
  
risk_management:
  max_position_size: 50000
  max_daily_loss: 5000
  max_positions: 20
  
alpaca:
  # Production optimizations
  use_adaptive_throttling: true
  batch_orders: true
  smart_routing: true
```

## Alpaca Monitoring & Alerts

### Real-time Position Monitoring

```python
# Monitor positions via Alpaca
async def monitor_alpaca_positions():
    positions = trading_client.get_all_positions()
    
    for position in positions:
        print(f"{position.symbol}: {position.qty} @ {position.avg_entry_price}")
        print(f"  P&L: ${float(position.unrealized_pl):,.2f}")
        print(f"  Market Value: ${float(position.market_value):,.2f}")
```

### Alpaca Webhooks

```python
# Set up Alpaca webhooks for order updates
webhook_config = {
    "url": "https://your-server.com/alpaca-webhook",
    "events": ["trade_updates", "account_updates"]
}
```

## Alpaca Best Practices

### 1. **Rate Limit Management**
```python
# Implement exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def place_alpaca_order(order_request):
    return trading_client.submit_order(order_request)
```

### 2. **Connection Management**
```python
# Use connection pooling
async def create_alpaca_session():
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=100,
            ttl_dns_cache=300
        )
    )
```

### 3. **Error Handling**
```python
# Handle Alpaca-specific errors
from alpaca.common.exceptions import APIError

try:
    order = trading_client.submit_order(order_request)
except APIError as e:
    if e.code == 'insufficient_buying_power':
        # Reduce position size
    elif e.code == 'order_not_fillable':
        # Try different order type
```

## Alpaca Performance Optimization

### 1. **Colocation Benefits**
- Alpaca servers: US-EAST-1 (Virginia)
- Recommended VPS: AWS EC2 in us-east-1
- Expected latency: <10ms to Alpaca

### 2. **WebSocket Streams**
```python
# Use WebSocket for real-time data
async def start_alpaca_stream():
    stream = StockDataStream(api_key, secret_key)
    
    async def handle_trade(trade):
        # Process trade in <1ms
        await process_trade(trade)
    
    stream.subscribe_trades(handle_trade, "AAPL", "MSFT")
    await stream.run()
```

### 3. **Batch Operations**
```python
# Batch order status checks
order_ids = ["order1", "order2", "order3"]
orders = trading_client.get_orders(filter={"symbols": "AAPL,MSFT,GOOGL"})
```

## Production Checklist for Alpaca

### Pre-Launch
- [ ] API keys secured in environment variables
- [ ] Rate limiting tested under load
- [ ] WebSocket reconnection logic implemented
- [ ] Error handling for all Alpaca exceptions
- [ ] PDT compliance logic verified

### Launch Day
- [ ] Start with 10% of intended capital
- [ ] Monitor order fill rates closely
- [ ] Check for any API deprecation warnings
- [ ] Verify all risk limits working
- [ ] Test emergency shutdown

### Post-Launch
- [ ] Daily reconciliation with Alpaca reports
- [ ] Monitor API usage vs limits
- [ ] Review order execution quality
- [ ] Track slippage and fees
- [ ] Optimize based on performance

## Alpaca Support Resources

- **API Documentation**: https://alpaca.markets/docs/
- **Status Page**: https://status.alpaca.markets/
- **Community Forum**: https://forum.alpaca.markets/
- **Support Email**: support@alpaca.markets
- **GitHub**: https://github.com/alpacahq

## Emergency Contacts

### Alpaca Trading Issues
- Check status page first
- Email: support@alpaca.markets
- Response time: Usually within 24 hours

### Critical Issues
- Use emergency shutdown procedures
- Document issue with screenshots
- Contact support with details

---

**Remember**: Start small, validate thoroughly, and scale gradually. Alpaca's modern infrastructure supports sophisticated strategies, but always prioritize risk management over performance.