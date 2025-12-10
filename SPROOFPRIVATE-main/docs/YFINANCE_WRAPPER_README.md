# YFinance Wrapper - Robust Error Handling

## Overview

The `yfinance_wrapper.py` module provides a production-ready wrapper around the yfinance library that handles common errors:

- **"Expecting value: line 1 column 1 (char 0)"** - Empty JSON responses from Yahoo Finance
- **Rate limiting** - Too many requests to Yahoo Finance
- **Network timeouts** - Slow or interrupted connections
- **Invalid tickers** - Non-existent stock symbols
- **Missing data** - Incomplete or unavailable historical data

## Features

- ✅ **Automatic retry with exponential backoff** - Retries failed requests intelligently
- ✅ **Response validation** - Ensures data integrity before returning
- ✅ **Request caching** - Reduces API calls for repeated requests
- ✅ **Rate limit management** - Prevents hitting Yahoo's rate limits
- ✅ **Graceful degradation** - Returns empty DataFrames instead of crashing
- ✅ **Drop-in replacement** - Minimal code changes required
- ✅ **Comprehensive logging** - Track issues and debug problems

## Installation

1. The wrapper uses the standard yfinance library:
```bash
pip install yfinance pandas numpy requests
```

2. Copy `yfinance_wrapper.py` to your project directory

## Quick Start

### Method 1: Drop-in Replacement (Recommended)

Replace your import statement:

```python
# Old code:
# import yfinance as yf

# New code:
import yfinance_wrapper as yf

# Everything else stays the same!
data = yf.download('AAPL', period='1mo')
ticker = yf.Ticker('MSFT')
```

### Method 2: Direct Usage

```python
from yfinance_wrapper import YFinanceWrapper

# Create wrapper with custom config
wrapper = YFinanceWrapper({
    'max_retries': 5,
    'cache_duration_minutes': 30,
    'calls_per_minute': 30
})

# Download data with error handling
data = wrapper.download('AAPL', period='1mo')
```

## Configuration Options

```python
config = {
    'max_retries': 3,              # Number of retry attempts
    'retry_delay': 1.0,            # Initial retry delay in seconds
    'backoff_factor': 2.0,         # Exponential backoff multiplier
    'calls_per_minute': 60,        # Rate limit per minute
    'calls_per_hour': 1800,        # Rate limit per hour
    'cache_duration_minutes': 15,  # Cache TTL
    'timeout': 10,                 # Request timeout in seconds
    'log_level': 'INFO'           # Logging level
}

wrapper = YFinanceWrapper(config)
```

## Common Use Cases

### 1. Handling the JSON Error

```python
import yfinance_wrapper as yf

# This will retry automatically if Yahoo returns empty response
data = yf.download('AAPL', period='5d', interval='1m')

if data.empty:
    print("No data available")
else:
    print(f"Got {len(data)} data points")
```

### 2. Batch Processing with Rate Limiting

```python
wrapper = YFinanceWrapper({'calls_per_minute': 30})

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
for symbol in symbols:
    # Automatically rate-limited
    data = wrapper.download(symbol, period='1mo')
    print(f"{symbol}: {len(data)} days")
```

### 3. Options Chain with Error Handling

```python
wrapper = YFinanceWrapper()

options = wrapper.get_options('AAPL')
if options['calls'].empty:
    print("No options data available")
else:
    print(f"Found {len(options['calls'])} call options")
```

### 4. Using Cache for Repeated Requests

```python
wrapper = YFinanceWrapper({'cache_duration_minutes': 60})

# First call - hits API
data1 = wrapper.download('SPY', period='1d')  # Slow

# Second call - from cache  
data2 = wrapper.download('SPY', period='1d')  # Fast!
```

## Error Handling Examples

### Empty Response Error
```python
# Old code that crashes:
# JSONDecodeError: Expecting value: line 1 column 1 (char 0)

# New code that handles it:
data = yf.download('AAPL', period='1d')
if data.empty:
    print("Yahoo returned empty response, handled gracefully")
```

### Invalid Ticker
```python
# Returns empty DataFrame instead of crashing
data = yf.download('INVALID123', period='1d')
assert data.empty  # True
```

### Network Issues
```python
# Automatically retries with exponential backoff
wrapper = YFinanceWrapper({
    'max_retries': 5,
    'timeout': 20
})
data = wrapper.download('AAPL', period='1y')
```

## Integration Guide

### For Existing Projects

1. Find all imports of yfinance:
```bash
grep -r "import yfinance" .
grep -r "from yfinance" .
```

2. Replace with wrapper:
```python
# Replace: import yfinance as yf
# With:    import yfinance_wrapper as yf
```

3. No other code changes needed!

### For New Projects

```python
# Always use the wrapper for production code
import yfinance_wrapper as yf

# Configure once at startup
from yfinance_wrapper import YFinanceWrapper
YFinanceWrapper.default_config = {
    'cache_duration_minutes': 30,
    'calls_per_minute': 30
}
```

## Testing

Run the test suite:
```bash
python test_yfinance_wrapper.py
```

Run examples:
```bash
python yfinance_wrapper_example.py
```

## Troubleshooting

### Still Getting JSON Errors?

1. Increase retries:
```python
wrapper = YFinanceWrapper({'max_retries': 5})
```

2. Add longer delays:
```python
wrapper = YFinanceWrapper({
    'retry_delay': 2.0,
    'backoff_factor': 3.0
})
```

3. Use stricter rate limiting:
```python
wrapper = YFinanceWrapper({'calls_per_minute': 20})
```

### Performance Issues?

1. Enable caching:
```python
wrapper = YFinanceWrapper({'cache_duration_minutes': 60})
```

2. Batch requests:
```python
# Good - one API call
data = wrapper.download(['AAPL', 'MSFT', 'GOOGL'], period='1d')

# Bad - three API calls
data1 = wrapper.download('AAPL', period='1d')
data2 = wrapper.download('MSFT', period='1d')
data3 = wrapper.download('GOOGL', period='1d')
```

## Best Practices

1. **Always check for empty DataFrames**:
```python
data = yf.download('AAPL', period='1d')
if not data.empty:
    # Process data
    pass
```

2. **Use appropriate intervals**:
```python
# Yahoo limits minute data to 7 days
data = yf.download('AAPL', period='7d', interval='1m')  # OK
data = yf.download('AAPL', period='30d', interval='1m')  # Will fail
```

3. **Cache for repeated requests**:
```python
wrapper = YFinanceWrapper({'cache_duration_minutes': 30})
# Use same wrapper instance to benefit from cache
```

4. **Handle options carefully**:
```python
options = wrapper.get_options('AAPL')
if options['calls'].empty:
    # No options available
    pass
```

## Support

For issues or questions:
1. Check the test files for examples
2. Enable debug logging: `YFinanceWrapper({'log_level': 'DEBUG'})`
3. Review the error messages - the wrapper provides detailed error information

## License

Same as your project license.