# MinIO Connection Setup Guide

## Overview
Successfully connected to MinIO server at `https://uschristmas.us` with access to the `stockdb` bucket containing ~84 GB of financial data across 7,614+ objects.

## Credentials Used
- **Endpoint**: https://uschristmas.us
- **Access Key**: AKSTOCKDB2024
- **Secret Key**: StockDB-Secret-Access-Key-2024-Secure!
- **Bucket**: stockdb

## Data Available

### Summary
- **Total Objects**: 7,614+ files
- **Total Size**: ~84.5 GB
- **Data Types**: Options, Stocks, Financial Reports, Notebooks
- **Date Range**: Historical data from 2002-2009

### Data Categories
- **Options Data**: 7,492 files (CSV format)
- **Financial Reports**: 87 files (XLSX format)
- **Stock Data**: 4 files (CSV format)
- **Historical ZIPs**: 8 files (compressed archives)
- **Jupyter Notebooks**: 5 files
- **Documentation**: 1 file
- **Other**: 17 files

### Directory Structure
```
stockdb/
├── options/ (3,844 files, 39.4 GB)
│   ├── 2002/
│   ├── 2003/
│   └── 2004/
├── aws_backup/ (3,717 files, 36.5 GB)
├── us_financials/ (27 files, 313 MB)
├── financials/ (16 files, 35 MB)
└── samples/ (2 files, 0.7 MB)
```

## Installation

### Required Packages
```bash
pip install minio pandas numpy python-dateutil urllib3
```

Or install from requirements:
```bash
pip install -r minio_requirements.txt
```

## Usage

### Basic Connection
```python
from final_minio_connection_script import MinIOStockDBClient

# Initialize client
client = MinIOStockDBClient()

# Get bucket overview
overview = client.get_bucket_overview()
print(overview)
```

### Get Options Data
```python
# Get all options files for 2002
options_2002 = client.get_options_data(year=2002)

# Get options data for specific date range
options_may = client.get_options_data(
    year=2002,
    start_date='2002-05-01',
    end_date='2002-05-31'
)

# Load specific options file as DataFrame
df = client.load_options_csv('options/2002/2002-05-01options.csv')
```

### Sample Data Format

#### Options Data Columns
```
contract, underlying, expiration, type, strike, style, bid, bid_size, 
ask, ask_size, volume, open_interest, quote_date, delta, gamma, 
theta, vega, implied_volatility
```

#### Example Options Row
```
A020518C00015000,A,2002-05-18,call,15,A,15.2,,15.6,,0,9,2002-05-01,0.9966,0.0001,1.9669,0.0023,0.9158
```

### Search and Filter
```python
# Search for specific symbol
aapl_files = client.search_objects('AAPL')

# Get financial reports
reports = client.get_financial_reports()

# Get stock data
stocks = client.get_stock_data(year=2002)
```

### Download Files
```python
# Download single file
local_path = client.download_object('options/2002/2002-05-01options.csv')

# Load as pandas DataFrame
df = client.load_options_csv('options/2002/2002-05-01options.csv')
```

## Key Scripts Created

1. **test_minio_connection_custom.py** - Initial connection test
2. **comprehensive_minio_explorer.py** - Full data exploration tool
3. **final_minio_connection_script.py** - Complete client library
4. **minio_stockdb_connection.py** - Auto-generated connection script

## Available Years
Historical options data is available for: **2002, 2003, 2004**

## Performance Notes
- Data is stored in CSV format for easy processing
- Files range from small (80KB stocks) to large (12MB options)
- Use the client's built-in methods for efficient data access
- Temporary files are automatically cleaned up after loading

## Next Steps
1. Explore specific years or date ranges of interest
2. Analyze options data for specific underlying assets
3. Combine with existing trading algorithms
4. Use financial reports for fundamental analysis

## Troubleshooting
- Ensure network connectivity to uschristmas.us
- Verify credentials if connection fails
- Use the connection test script to validate setup
- Check logs for detailed error information