# MinIO Web Browser Interface Scraper

## Overview

The `web_minio_scraper.py` script is designed to scrape MinIO browser interfaces to discover available datasets, extract file listings, and create comprehensive inventories of data. It handles both traditional HTML-based interfaces and modern JavaScript-based MinIO Console applications.

## Features

- **Multi-Method Discovery**: Uses both HTML scraping and API endpoint discovery
- **Modern MinIO Console Support**: Works with JavaScript-heavy MinIO Console interfaces
- **Authentication Detection**: Identifies when authentication is required
- **Comprehensive File Analysis**: Extracts file sizes, dates, types, and ETags
- **Dataset Organization**: Automatically groups files into logical datasets
- **Multiple Output Formats**: Generates JSON inventory and detailed text reports
- **Error Handling**: Robust error handling with detailed logging

## Installation

### Dependencies
```bash
pip install requests beautifulsoup4 lxml
```

Or install from the existing requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from web_minio_scraper import MinIOWebScraper

# Initialize scraper
scraper = MinIOWebScraper("https://uschristmas.us/browser/stockdb")

# Perform scraping
inventory = scraper.scrape_minio()

# Print summary
scraper.print_summary()

# Save results
scraper.save_inventory("inventory.json")
```

### Command Line Usage

```bash
# Run the test script
python3 test_minio_scraper.py

# Run the main scraper directly
python3 web_minio_scraper.py
```

## Discovered Data Structure

The scraper successfully discovered the following data structure at `https://uschristmas.us/browser/stockdb`:

### Summary
- **Total Files**: 13
- **Total Size**: 10.01 GB
- **File Types**: 2 (zip, directories)
- **Datasets**: 13

### Main Data Files
1. **2002.zip** - 513.58 MB
2. **2003.zip** - 798.74 MB
3. **2004.zip** - 940.16 MB
4. **2005.zip** - 1.06 GB
5. **2006.zip** - 1.23 GB
6. **2007.zip** - 1.51 GB
7. **2008.zip** - 1.96 GB
8. **2009.zip** - 2.04 GB

### Directory Structure
- **aws_backup/** - Backup directory
- **financials/** - Financial data directory
- **options/** - Options data directory
- **samples/** - Sample data directory
- **us_financials/** - US financial data directory

## API Discovery Results

The scraper successfully identified the working API endpoint:
- **Primary Endpoint**: `https://uschristmas.us/api/v1/buckets/stockdb/objects`
- **Response Format**: JSON with objects array
- **Authentication**: Some endpoints require authentication

## Output Files

### 1. JSON Inventory (`minio_inventory_stockdb.json`)
Contains complete structured data including:
- File metadata (name, size, dates, ETags)
- Dataset organization
- API discovery results
- Summary statistics

### 2. Detailed Report (`minio_detailed_report.txt`)
Human-readable report with:
- Executive summary
- File type distribution
- Dataset breakdown with sizes and date ranges

## Technical Implementation

### Discovery Methods

1. **HTML Scraping**: Parses traditional MinIO browser interfaces
   - Table-based file listings
   - List-based file listings
   - Div-based file listings

2. **API Endpoint Discovery**: Tests multiple potential API endpoints
   - MinIO admin APIs
   - S3-compatible APIs
   - Modern MinIO Console APIs

### Supported API Response Formats

- **MinIO list-objects**: JSON with `objects` array
- **S3 ListBucketResult**: XML format
- **Bucket listings**: JSON with `buckets` array
- **Generic object lists**: Various JSON formats

### Authentication Handling

The scraper detects and reports:
- HTTP 401 (Authentication Required)
- HTTP 403 (Access Denied)
- Login forms in HTML content

## Customization

### Changing Target URL

```python
scraper = MinIOWebScraper("https://your-minio-server.com/browser/bucket-name")
```

### Adjusting Discovery Depth

```python
# Limit folder traversal depth
files = scraper.navigate_folders(url, max_depth=2)
```

### Custom API Endpoints

```python
# Add custom endpoints to discovery
endpoints = scraper.discover_api_endpoints(base_url)
endpoints.append("https://custom-api-endpoint.com/list")
```

## Error Handling

The scraper includes comprehensive error handling:
- Network timeouts and connection errors
- HTTP status code handling
- JSON/XML parsing errors
- File access permissions
- Missing dependencies

## Logging

Detailed logging is provided at multiple levels:
- INFO: Successful operations and discoveries
- WARNING: Authentication issues and access problems
- ERROR: Critical failures and parsing errors
- DEBUG: Detailed diagnostic information

## Performance Considerations

- **Rate Limiting**: Built-in delays between requests
- **Timeout Handling**: Configurable request timeouts
- **Memory Efficiency**: Streaming JSON parsing for large responses
- **Concurrent Discovery**: Parallel API endpoint testing

## Security Notes

- Does not store or transmit authentication credentials
- Respects robots.txt and rate limiting
- Uses standard HTTP headers and user agents
- No data modification - read-only operations

## Troubleshooting

### Common Issues

1. **No data found**: Check if authentication is required
2. **Connection timeouts**: Verify network connectivity and firewall settings
3. **Permission denied**: Ensure proper access credentials if needed
4. **Parsing errors**: Check if the MinIO interface has changed format

### Debug Mode

Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Support for authenticated sessions
- Recursive directory exploration for nested buckets
- Integration with MinIO client libraries
- Export to additional formats (CSV, Excel)
- Real-time monitoring capabilities
- Dashboard web interface