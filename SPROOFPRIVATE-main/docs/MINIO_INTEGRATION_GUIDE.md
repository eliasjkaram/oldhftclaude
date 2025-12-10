# MinIO Data Integration Guide for LEAPS Arbitrage System

## Overview

The Integrated LEAPS Arbitrage System has been updated to use MinIO as the primary data source, replacing simulated data with real historical market data. This integration provides:

- Real-time access to historical stock data
- LEAPS options data retrieval
- Intelligent caching for performance optimization
- Automatic fallback to simulated data when needed
- Configuration management for multiple data sources

## Key Features

### 1. **Real Data Access**
- Historical stock prices (OHLCV data)
- LEAPS options chains with Greeks
- Multiple timeframes (minute, hourly, daily)
- Automatic data validation and preprocessing

### 2. **Intelligent Caching**
- Local cache to reduce API calls
- Configurable cache expiration
- Automatic cache cleanup
- Performance optimization through bulk downloads

### 3. **Error Handling**
- Graceful fallback to simulated data
- Comprehensive error logging
- Data validation and cleaning
- Network resilience with retries

### 4. **Configuration Management**
- Easy switching between data sources
- Support for multiple data providers
- Environment-specific configurations
- Secure credential management

## Usage

### Basic Usage

```bash
# Run with MinIO data (default)
python integrated_leaps_arbitrage_system.py

# Run with simulated data
python integrated_leaps_arbitrage_system.py --use-simulated

# Analyze specific symbols
python integrated_leaps_arbitrage_system.py --symbols AAPL MSFT GOOGL

# Clear cache before running
python integrated_leaps_arbitrage_system.py --clear-cache
```

### Python API Usage

```python
from integrated_leaps_arbitrage_system import IntegratedLEAPSArbitrageSystem

# Initialize with MinIO
system = IntegratedLEAPSArbitrageSystem(data_source='minio')

# Run analysis with real data
analysis = await system.analyze_leaps_opportunity(
    symbol='AAPL',
    use_real_data=True
)

# Access data directly
historical_data = await system._get_real_historical_data('AAPL', lookback_days=365)
options_data = await system._get_real_options_data('AAPL')
```

### Configuration Management

```bash
# List available data sources
python data_source_config.py list

# Set active data source
python data_source_config.py set --source minio

# Enable/disable sources
python data_source_config.py enable --source alpaca
python data_source_config.py disable --source yahoo

# Validate configuration
python data_source_config.py validate --source minio
```

## Data Structure

### Historical Stock Data
```python
{
    'timestamp': datetime,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'returns': float,        # Added by preprocessing
    'log_returns': float,    # Added by preprocessing
    'typical_price': float,  # Added by preprocessing
    'vwap': float           # Added by preprocessing
}
```

### LEAPS Options Data
```python
{
    'symbol': str,              # e.g., 'AAPL_20251219_C150'
    'type': str,                # 'call' or 'put'
    'strike': float,
    'expiration': datetime,
    'bid': float,
    'ask': float,
    'spread': float,
    'volume': int,
    'open_interest': int,
    'implied_volatility': float,
    'delta': float,             # Optional
    'gamma': float,             # Optional
    'theta': float,             # Optional
    'vega': float               # Optional
}
```

## Configuration Files

### data_sources.json
```json
{
  "active": "minio",
  "sources": {
    "minio": {
      "type": "minio",
      "enabled": true,
      "config": {
        "endpoint": "uschristmas.us/minio",
        "access_key": "AKSTOCKDBUSER001",
        "secret_key": "StockDB-User-Secret-Key-Secure-2024!",
        "bucket_name": "stockdb",
        "secure": true
      },
      "cache": {
        "cache_dir": "/home/harry/alpaca-mcp/minio_cache",
        "max_cache_age_hours": 24,
        "max_cache_size_gb": 50
      }
    }
  },
  "fallback_order": ["minio", "alpaca", "yahoo", "simulated"]
}
```

## Error Handling

The system includes comprehensive error handling:

1. **Connection Errors**: Automatically falls back to simulated data
2. **Missing Data**: Uses interpolation or simulated data for gaps
3. **Invalid Data**: Validates and cleans data automatically
4. **Cache Errors**: Rebuilds cache from source if corrupted

## Performance Optimization

1. **Parallel Downloads**: Bulk data retrieval with configurable workers
2. **Smart Caching**: Only downloads changed data
3. **Data Compression**: Efficient storage formats (Parquet support)
4. **Memory Management**: Streaming for large datasets

## Security Considerations

1. **Credentials**: Store sensitive data in environment variables
2. **Encryption**: HTTPS connections by default
3. **Access Control**: Read-only access to MinIO bucket
4. **Audit Logging**: All data access is logged

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check network connectivity
   - Verify MinIO endpoint is accessible
   - Confirm credentials are correct

2. **No Data Found**
   - Verify symbol exists in database
   - Check date range is valid
   - Ensure bucket permissions are correct

3. **Cache Issues**
   - Clear cache with `--clear-cache` flag
   - Check disk space availability
   - Verify cache directory permissions

### Debug Mode

```bash
# Enable debug logging
export TRADING_ENV=development
python integrated_leaps_arbitrage_system.py
```

## Future Enhancements

1. **Additional Data Sources**
   - Alpaca Markets API integration
   - Yahoo Finance backup
   - Bloomberg Terminal support

2. **Advanced Features**
   - Real-time streaming data
   - WebSocket connections
   - Delta-neutral portfolio optimization
   - Multi-timeframe analysis

3. **Performance Improvements**
   - GPU-accelerated data processing
   - Distributed caching
   - Predictive prefetching

## Testing

Run the test suite to verify integration:

```bash
# Test MinIO integration
python test_integrated_minio.py

# Run original tests with real data
python test_minio_integration.py
```

## Support

For issues or questions:
1. Check logs in `~/alpaca-mcp/logs/`
2. Review cache statistics
3. Validate configurations
4. Test with simulated data as baseline