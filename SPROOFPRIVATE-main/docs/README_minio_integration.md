# MinIO Data Integration System

A comprehensive data integration system that connects to MinIO stockdb bucket, providing efficient data access for all trading algorithms with local caching, validation, and preprocessing capabilities.

## Features

### Core Integration (`minio_data_integration.py`)
- **MinIO Connection**: Secure connection to MinIO stockdb bucket
- **Local Caching**: Intelligent caching system with configurable TTL
- **Data Validation**: Comprehensive validation for trading data integrity
- **Preprocessing**: Automatic calculation of technical indicators and features
- **Bulk Operations**: Parallel download capabilities for efficiency
- **Unified Interface**: Single point of access for all data types

### Trading Data Adapter (`trading_data_adapter.py`)
- **Algorithm-Specific Processing**: Tailored data preprocessing for each trading strategy
- **Supported Algorithms**:
  - Momentum Tracker
  - Mean Reversion
  - Pairs Trading
  - ML Predictions
  - LEAPS Arbitrage
  - Options Flow
  - Volume Analysis

### LEAPS Integration (`leaps_data_integration.py`)
- **LEAPS Chain Retrieval**: Specialized handling of long-dated options
- **Arbitrage Detection**: Automated opportunity scanning
- **Portfolio Analysis**: Multi-symbol LEAPS portfolio analytics
- **Strategy Support**:
  - Calendar Spreads
  - Diagonal Spreads
  - Synthetic Positions
  - Volatility Arbitrage

## Installation

```bash
# Install required dependencies
pip install -r requirements_minio.txt

# Verify MinIO connection
python test_minio_integration.py
```

## Configuration

Edit `minio_config.py` to customize:
- MinIO connection settings
- Cache configuration
- Validation rules
- Performance parameters

## Usage Examples

### Basic Data Retrieval
```python
from minio_data_integration import MinIODataIntegration

# Initialize
minio = MinIODataIntegration()

# Get historical data
df = minio.get_historical_data(
    symbol='AAPL',
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='daily'
)
```

### Algorithm Integration
```python
from trading_data_adapter import TradingDataAdapter

# Create adapter
adapter = TradingDataAdapter()

# Get preprocessed data for momentum tracking
momentum_data = adapter.get_data_for_algorithm(
    'momentum_tracker',
    'AAPL',
    lookback_days=60
)
```

### LEAPS Analysis
```python
from leaps_data_integration import LEAPSDataIntegration

# Initialize LEAPS integration
leaps = LEAPSDataIntegration()

# Find arbitrage opportunities
opportunities = leaps.find_arbitrage_opportunities('MSFT')
```

### Unified Data Access
```python
# Create unified interface
unified = minio.create_unified_interface()

# Get feature matrix for ML models
features = unified.get_feature_matrix(['AAPL', 'MSFT', 'GOOGL'])

# Prepare data for specific model type
lstm_data = unified.prepare_for_model(features, model_type='lstm')
```

## Architecture

```
MinIO Data Integration System
├── Core Layer
│   ├── MinIO Client Connection
│   ├── Cache Management
│   └── Data Validation
├── Processing Layer
│   ├── Data Preprocessing
│   ├── Technical Indicators
│   └── Feature Engineering
├── Adapter Layer
│   ├── Algorithm-Specific Adapters
│   ├── LEAPS Specialization
│   └── Unified Interface
└── Application Layer
    ├── Trading Algorithms
    ├── ML Models
    └── Analytics Tools
```

## Cache Management

The system implements intelligent caching:
- **Automatic TTL**: Configurable cache expiration
- **Parallel Downloads**: Efficient bulk data retrieval
- **Cache Statistics**: Monitor cache usage and performance
- **Cleanup**: Automatic removal of stale data

```python
# Get cache statistics
stats = minio.get_cache_stats()

# Clear old cache
minio.clear_cache(older_than_days=7)
```

## Data Validation

Comprehensive validation ensures data quality:
- Required columns verification
- Price range validation
- Volume consistency checks
- Duplicate detection
- Data consistency (high >= low, etc.)

## Performance Optimization

- Parallel downloads with configurable workers
- Efficient caching with TTL management
- Bulk operations for multiple symbols
- Optimized data structures for quick access

## Testing

Run the comprehensive test suite:
```bash
python test_minio_integration.py
```

This will test:
- MinIO connection
- Download and caching
- Data validation
- Algorithm adapters
- LEAPS integration
- Performance benchmarks

## Integration with Existing Systems

The system seamlessly integrates with your existing trading infrastructure:

1. **Direct Algorithm Integration**: Use `TradingDataAdapter` for algorithm-specific data
2. **ML Model Support**: Unified interface provides standardized features
3. **LEAPS Strategies**: Specialized support for long-dated options
4. **Real-time Updates**: Async support for live data integration

## Error Handling

Robust error handling throughout:
- Connection retry logic
- Graceful degradation on cache miss
- Validation error reporting
- Detailed logging for debugging

## Future Enhancements

- Real-time data streaming
- Advanced caching strategies
- Additional technical indicators
- Enhanced arbitrage detection
- WebSocket support for live updates

## Support

For issues or questions:
1. Check the test output for diagnostics
2. Review logs in `~/alpaca-mcp/logs/`
3. Verify MinIO credentials and connectivity
4. Ensure all dependencies are installed

The system is designed for reliability, performance, and ease of integration with your existing trading algorithms and AI models.