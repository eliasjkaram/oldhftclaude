# MinIO Options Data Processor

A comprehensive system for downloading, processing, and analyzing options data from MinIO, with full integration into the backtesting system.

## Overview

The MinIO Options Processor provides:
- **Data Download**: Connects to MinIO and downloads options/LEAPS data
- **Data Processing**: Processes raw options data with Greeks calculation
- **Database Storage**: Unified SQLite database for all options data
- **Arbitrage Detection**: Identifies various arbitrage opportunities
- **Backtesting Integration**: Seamless integration with the backtesting system
- **Strategy Support**: Multiple options strategies including wheel, LEAPS, spreads

## Features

### 1. Options Data Management
- Download options data from MinIO `options/` directory
- Support for regular options and LEAPS (365+ days to expiration)
- Automatic caching with configurable TTL
- Parallel download support for efficiency

### 2. Greeks Calculation
- Black-Scholes pricing model
- Full Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied volatility calculation
- Support for both calls and puts

### 3. Arbitrage Strategies
- **Conversion/Reversal**: Put-call parity violations
- **Calendar Spreads**: IV differences across expirations  
- **Butterfly Spreads**: Mispriced volatility smiles
- **Box Spreads**: Risk-free arbitrage opportunities
- **Custom Strategies**: Extensible framework for new strategies

### 4. Database Schema

```sql
-- Main options table
CREATE TABLE options (
    symbol TEXT,
    underlying TEXT,
    strike REAL,
    expiration DATE,
    option_type TEXT,
    bid REAL,
    ask REAL,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    rho REAL,
    -- ... additional fields
);

-- Arbitrage opportunities table
CREATE TABLE arbitrage_opportunities (
    strategy_type TEXT,
    underlying TEXT,
    theoretical_profit REAL,
    execution_profit REAL,
    probability REAL,
    risk_score REAL,
    -- ... additional fields
);
```

## Installation

```bash
# Install required dependencies
pip install minio pandas numpy scipy sqlite3

# Optional: GPU acceleration
pip install cupy torch
```

## Usage

### Basic Usage

```python
from minio_options_processor import MinIOOptionsProcessor

# Initialize processor
processor = MinIOOptionsProcessor()

# Download options data
symbols = ['AAPL', 'MSFT', 'GOOGL']
raw_data = processor.download_options_data(
    symbols=symbols,
    include_leaps=True
)

# Process and calculate Greeks
contracts = processor.process_options_data(raw_data, calculate_greeks=True)

# Save to database
processor.save_to_database(contracts)

# Find arbitrage opportunities
opportunities = processor.find_arbitrage_opportunities(
    underlying='AAPL',
    min_profit=50,
    max_risk=5000
)
```

### Integration with Backtesting

```python
from options_backtest_integration import OptionsBacktestEngine
from minio_options_processor import MinIOOptionsProcessor

# Initialize components
processor = MinIOOptionsProcessor()
backtest_engine = OptionsBacktestEngine(processor)

# Add strategies
backtest_engine.add_strategy(ArbitrageStrategy())
backtest_engine.add_strategy(WheelStrategy())
backtest_engine.add_strategy(LEAPSStrategy())

# Run backtest
results = backtest_engine.run_backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 1),
    symbols=['AAPL', 'MSFT'],
    initial_capital=100000
)

# Generate report
backtest_engine.generate_report("backtest_results.html")
```

### Custom Strategy Example

```python
class IronCondorStrategy(OptionsStrategy):
    def evaluate(self, contracts, market_data):
        signals = []
        
        # Find iron condor opportunities
        for underlying in set(c.underlying for c in contracts):
            # Get options for this underlying
            options = [c for c in contracts if c.underlying == underlying]
            
            # Look for appropriate strikes
            signal = self._find_iron_condor(options, market_data)
            if signal:
                signals.append(signal)
                
        return {'signals': signals}
```

## Configuration

The system uses configuration from `minio_config.py`:

```python
MINIO_CONFIG = {
    'endpoint': 'uschristmas.us/minio',
    'access_key': 'AKSTOCKDBUSER001',
    'secret_key': 'StockDB-User-Secret-Key-Secure-2024!',
    'bucket_name': 'stockdb'
}

LEAPS_CONFIG = {
    'min_days_to_expiration': 365,
    'symbols': ['AAPL', 'MSFT', 'GOOGL', ...]
}
```

## Scripts

### 1. `minio_options_processor.py`
Main processor with all core functionality:
- MinIO connection and data download
- Options data processing and Greeks calculation
- Database management
- Arbitrage detection algorithms

### 2. `test_minio_options_processor.py`
Comprehensive test suite:
- Unit tests for Greeks calculator
- Integration tests with MinIO
- Arbitrage detection tests
- Database operation tests

### 3. `options_backtest_integration.py`
Backtesting integration:
- Strategy implementations (Arbitrage, LEAPS, Wheel)
- Backtest engine with options support
- Performance metrics calculation
- Report generation

## Performance Optimization

### Parallel Processing
```python
# Download multiple symbols in parallel
processor.download_options_data(
    symbols=symbols,
    max_workers=10  # Parallel downloads
)
```

### Caching
- Local file cache reduces MinIO calls
- Configurable cache TTL
- Automatic cache cleanup

### Database Indexing
- Indexes on underlying, expiration, strike
- Optimized queries for arbitrage detection

## Arbitrage Detection

### Put-Call Parity
```
C - P = S - K * exp(-r*T)
```

### Box Spread Value
```
Box Value = (K2 - K1) * exp(-r*T)
```

### Calendar Spread Criteria
- IV differential > 20%
- Same strike, different expirations
- Sufficient liquidity

## Reports

The system generates HTML reports with:
- Summary statistics by underlying
- Top arbitrage opportunities
- Backtest results by strategy
- Equity curves and performance metrics

## Troubleshooting

### Common Issues

1. **MinIO Connection Failed**
   ```python
   # Check endpoint and credentials
   processor = MinIOOptionsProcessor(
       endpoint="your-endpoint",
       access_key="your-key",
       secret_key="your-secret"
   )
   ```

2. **No Data Found**
   - Verify bucket name and prefix
   - Check date ranges
   - Ensure options/ directory exists in MinIO

3. **Greeks Calculation Error**
   - Ensure positive time to expiration
   - Check for valid volatility values
   - Verify spot price data

## Future Enhancements

1. **Real-time Integration**
   - WebSocket support for live data
   - Real-time arbitrage alerts
   - Automatic trade execution

2. **Advanced Strategies**
   - Volatility arbitrage
   - Statistical arbitrage
   - Machine learning predictions

3. **Risk Management**
   - Portfolio-level risk metrics
   - Stress testing
   - Monte Carlo simulations

## Contributing

To add new strategies:
1. Inherit from `OptionsStrategy` base class
2. Implement `evaluate()` method
3. Add to backtest engine
4. Test with historical data

## License

This software is provided as-is for educational and research purposes.

## Contact

For questions or issues, please refer to the main project documentation.