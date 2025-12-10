# Enhanced Price Provider - Multi-Source Real-Time Market Data

## Overview

The Enhanced Price Provider is a production-ready system that retrieves **LIVE market data** from multiple sources with automatic failover, validation, and caching. It prioritizes real-time accuracy over static references.

## Features

### 🔄 Multiple Data Sources
1. **YFinance** - Free, reliable, with enhanced error handling
2. **Alpha Vantage API** - Professional market data
3. **Polygon.io/Alpaca** - Direct market access (highest priority)
4. **IEX Cloud** - Reliable backup source
5. **Finnhub API** - Additional validation source

### 🎯 Key Capabilities
- **Priority-based source selection** - Uses best available source
- **Automatic failover** - Switches sources on failure
- **Price validation** - Compares prices across sources
- **Confidence scoring** - Based on source agreement
- **Smart caching** - TTL-based with market hours awareness
- **After-hours/pre-market** support
- **Bulk price fetching** - Efficient multi-symbol requests
- **Comprehensive logging** - Tracks all sources and discrepancies

## Installation

```bash
# Install required dependencies
pip install yfinance alpaca-py requests numpy pandas python-dotenv

# Optional: Install additional dependencies for specific sources
pip install alpha-vantage finnhub-python
```

## Configuration

### API Keys Setup

Create a `.env` file or set environment variables:

```bash
# Alpaca (Highest Priority - Direct Market Data)
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# Alpha Vantage (Free tier available)
ALPHA_VANTAGE_API_KEY=your_av_key

# IEX Cloud
IEX_CLOUD_API_KEY=your_iex_key

# Finnhub
FINNHUB_API_KEY=your_finnhub_key
```

### Provider Configuration

```python
from enhanced_price_provider import EnhancedPriceProvider

config = {
    # API Keys
    'alpaca_api_key': 'your_key',
    'alpaca_api_secret': 'your_secret',
    'alpha_vantage_api_key': 'your_av_key',
    'iex_cloud_api_key': 'your_iex_key',
    'finnhub_api_key': 'your_finnhub_key',
    
    # Cache Settings
    'cache_ttl_seconds': 60,  # Cache duration
    
    # Validation Settings
    'max_price_deviation': 0.05,  # 5% max deviation between sources
    'min_sources_for_confidence': 2,  # Min sources for high confidence
    
    # Fallback Reference Prices (last resort only)
    'reference_prices': {
        'AAPL': 150.0,
        'MSFT': 400.0
    }
}

provider = EnhancedPriceProvider(config)
```

## Usage Examples

### Basic Price Fetching

```python
# Get single price
price_data = provider.get_price('AAPL')

if price_data:
    print(f"Price: ${price_data.price:.2f}")
    print(f"Source: {price_data.source}")
    print(f"Confidence: {price_data.confidence:.2%}")
    print(f"Bid/Ask: ${price_data.bid:.2f}/${price_data.ask:.2f}")
```

### Bulk Price Fetching

```python
# Get multiple prices efficiently
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
prices = provider.get_bulk_prices(symbols)

for symbol, price_data in prices.items():
    if price_data:
        print(f"{symbol}: ${price_data.price:.2f}")
```

### Integration with Trading Systems

```python
from integrate_enhanced_price_provider import TradingSystemPriceAdapter

# Create adapter for easy integration
adapter = TradingSystemPriceAdapter(config)

# Drop-in replacement for existing price functions
current_price = adapter.get_current_price('AAPL')
bid, ask = adapter.get_bid_ask('AAPL')
is_market_open = adapter.get_market_hours_status()
```

## Data Source Priority

Sources are tried in priority order (0 = highest):

| Priority | Source | Requirements | Best For |
|----------|--------|--------------|----------|
| 0 | Alpaca/Polygon | API Key + Secret | Real-time trading |
| 1 | YFinance | None | General use |
| 2 | Alpha Vantage | API Key | Professional data |
| 3 | IEX Cloud | API Key | Reliable backup |
| 4 | Finnhub | API Key | Additional validation |

## Price Data Structure

```python
@dataclass
class PriceData:
    symbol: str          # Stock symbol
    price: float         # Current price
    source: str          # Data source used
    timestamp: datetime  # When price was fetched
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    is_market_hours: bool = True
    confidence: float = 1.0  # 0-1 confidence score
```

## Confidence Scoring

Confidence scores indicate price reliability:

- **95-100%**: Direct market data, multiple source agreement
- **80-95%**: Single reliable source or good agreement
- **70-80%**: Single source, market closed, or minor disagreement
- **< 70%**: Major source disagreement or stale data
- **10%**: Reference price fallback (avoid trading)

## Performance Optimization

### Caching Strategy
- Market hours: 60-second default TTL
- After hours: 3x longer TTL
- Low confidence: Reduced TTL (30 seconds max)

### Bulk Fetching
- Use `get_bulk_prices()` for multiple symbols
- Leverages source-specific bulk APIs when available
- Significantly faster than individual requests

### Source Management
- Automatic backoff on failures
- Source availability tracking
- Performance statistics per source

## Error Handling

The provider handles various error scenarios:

- **Network timeouts**: Automatic retry with backoff
- **Invalid symbols**: Returns None, logs warning
- **Rate limiting**: Built-in rate limiters per source
- **Empty responses**: Validates all responses
- **Source failures**: Automatic failover to next source

## Monitoring and Statistics

```python
# Get provider statistics
stats = provider.get_stats()

# Source performance
for source, source_stats in stats['sources'].items():
    print(f"{source}:")
    print(f"  Successes: {source_stats['success_count']}")
    print(f"  Failures: {source_stats['failure_count']}")
    print(f"  Avg Latency: {source_stats['avg_latency_ms']:.1f}ms")

# Cache information
print(f"Cache Size: {stats['cache_size']} symbols")
```

## Best Practices

1. **API Key Priority**: Always provide Alpaca keys for best real-time data
2. **Cache Usage**: Enable caching for better performance
3. **Confidence Monitoring**: Log/alert on low confidence prices
4. **Bulk Requests**: Use bulk fetching for portfolios
5. **Error Handling**: Always check if price_data is not None
6. **Market Hours**: Adjust strategies based on `is_market_hours`

## Migration Guide

### From Static Prices

```python
# Old system
STATIC_PRICES = {'AAPL': 150.0}
price = STATIC_PRICES['AAPL']

# New system
price_data = provider.get_price('AAPL')
price = price_data.price if price_data else STATIC_PRICES['AAPL']
```

### From Single Source

```python
# Old system (yfinance only)
import yfinance as yf
ticker = yf.Ticker('AAPL')
price = ticker.info['regularMarketPrice']

# New system (multi-source)
price_data = provider.get_price('AAPL')
price = price_data.price  # Automatically uses best source
```

## Troubleshooting

### No prices returned
1. Check API keys are correctly set
2. Verify at least one source is enabled
3. Check network connectivity
4. Review logs for specific errors

### Low confidence scores
1. Add more API keys for additional sources
2. Check for large price discrepancies in logs
3. Verify sources are returning current data

### Performance issues
1. Enable caching if disabled
2. Use bulk fetching for multiple symbols
3. Check rate limit settings
4. Monitor source latencies

## Demo Scripts

1. **Basic Demo**: `python demo_enhanced_price_provider.py`
2. **Integration Examples**: `python integrate_enhanced_price_provider.py`

## Future Enhancements

- WebSocket support for real-time streaming
- Additional sources (TD Ameritrade, Interactive Brokers)
- Historical data integration
- Options chain pricing
- Crypto and forex support

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify API keys and permissions
3. Test with demo scripts
4. Review source-specific documentation