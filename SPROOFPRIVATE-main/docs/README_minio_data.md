# MinIO Data Downloader and Integration System

This system provides comprehensive tools for downloading, organizing, and integrating stock market data from MinIO storage with trading algorithms.

## Components

### 1. MinIO Data Downloader (`minio_data_downloader.py`)

A robust downloader that handles:
- Historical ZIP files (2002-2009)
- Directory contents (financials/, options/, samples/, us_financials/)
- Data integrity verification
- Incremental updates (skips already downloaded files)
- Parallel downloads for efficiency

### 2. Data Integration Module (`data_integration.py`)

Integrates downloaded data with trading algorithms:
- Formats data for specific algorithm types (pairs trading, momentum, mean reversion)
- Creates unified datasets across multiple symbols
- Handles different data types (historical prices, financials, options)
- Provides caching for improved performance

### 3. Example Usage (`example_minio_usage.py`)

Demonstrates practical usage patterns:
- Downloading and extracting historical data
- Downloading directory contents
- Using data with pairs trading algorithms
- Momentum strategy analysis
- Data verification

## Quick Start

### 1. Download Historical Data

```bash
# Download specific years
python minio_data_downloader.py --download-historical 2008 2009

# Download all available years
python minio_data_downloader.py --download-historical

# Extract downloaded data
python minio_data_downloader.py --extract-historical 2008
```

### 2. Download Directory Data

```bash
# Download specific directories
python minio_data_downloader.py --download-directory financials/
python minio_data_downloader.py --download-directory options/
python minio_data_downloader.py --download-directory samples/
```

### 3. Full Data Sync

```bash
# Sync all available data
python minio_data_downloader.py --sync-all
```

### 4. Verify Data Integrity

```bash
# Check downloaded files
python minio_data_downloader.py --verify

# Show integration configuration
python minio_data_downloader.py --integration-config
```

## Using with Trading Algorithms

### Example: Pairs Trading

```python
from data_integration import DataIntegration
from datetime import datetime

# Initialize integration
integration = DataIntegration()

# Get data for pairs trading
data = integration.get_data_for_algorithm(
    algorithm_type="pairs_trading",
    symbols=["AAPL", "MSFT"],
    start_date=datetime(2008, 1, 1),
    end_date=datetime(2009, 12, 31)
)

# Access formatted data
price_data = data['price_data']
correlation_matrix = data['correlation_data']
```

### Example: Momentum Strategy

```python
# Get momentum-formatted data
data = integration.get_data_for_algorithm(
    algorithm_type="momentum",
    symbols=["AAPL", "GOOGL", "AMZN"],
    start_date=datetime(2008, 1, 1),
    end_date=datetime(2009, 12, 31)
)

# Access returns and volume data
returns = data['returns_data']
volume = data['volume_data']
```

## Data Organization

Downloaded data is organized in the cache directory:

```
data/minio_cache/
├── historical/          # ZIP files and extracted data
│   ├── 2008.zip
│   ├── 2008/           # Extracted data
│   ├── 2009.zip
│   └── 2009/
├── financials/         # Financial statements
├── options/            # Options data
├── samples/            # Sample datasets
├── us_financials/      # US financial data
└── metadata/           # Download metadata and checksums
```

## Features

### Incremental Downloads
- Checks file timestamps and sizes
- Skips files that are already up-to-date
- Maintains checksums for integrity verification

### Parallel Processing
- Uses thread pool for concurrent downloads
- Significantly faster for directory downloads

### Data Integrity
- Calculates and stores SHA256 checksums
- Verifies file integrity on demand
- Tracks download history and metadata

### Algorithm Integration
- Pre-formats data for common algorithm types
- Handles date filtering and symbol selection
- Provides unified datasets for multi-symbol analysis

## Advanced Usage

### Custom Cache Directory

```bash
python minio_data_downloader.py --cache-dir /path/to/cache --sync-all
```

### Programmatic Usage

```python
from minio_data_downloader import MinIODataDownloader

# Initialize with custom cache
downloader = MinIODataDownloader(cache_dir="/path/to/cache")

# Download specific years
downloaded = downloader.download_historical_zips([2008, 2009])

# Extract data
downloader.extract_historical_data(2008, cleanup=True)

# Download directory
files = downloader.download_directory("financials/", recursive=True)

# Verify integrity
results = downloader.verify_data_integrity()
```

## Integration with Existing Systems

The data integration module provides a standardized interface for accessing MinIO data:

1. **Ensure Data Availability**: Automatically downloads missing data
2. **Format for Algorithms**: Transforms data into algorithm-specific formats
3. **Create Unified Datasets**: Combines multiple symbols and data types
4. **Cache Management**: Efficient data access with local caching

## Troubleshooting

### Connection Issues
- Ensure MinIO credentials are configured in `minio_client.py`
- Check network connectivity to MinIO server
- Verify bucket permissions

### Memory Issues
- Use year-by-year downloads for large datasets
- Extract historical data one year at a time
- Consider using data streaming for very large files

### Data Format Issues
- Check column names in downloaded files
- Verify date formats match expected patterns
- Use example scripts to test data loading

## Performance Tips

1. **Initial Sync**: Run full sync during off-hours
2. **Incremental Updates**: Use regular syncs to stay current
3. **Parallel Downloads**: Leverage multi-threading for directories
4. **Local Cache**: Keep frequently used data in cache
5. **Extract on Demand**: Extract ZIP files only when needed

## Next Steps

1. Run `python example_minio_usage.py` to see examples
2. Perform initial data sync with `--sync-all`
3. Integrate with your trading algorithms using `DataIntegration`
4. Set up regular sync schedule for updates