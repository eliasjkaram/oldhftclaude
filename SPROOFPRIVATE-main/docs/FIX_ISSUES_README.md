# Alpaca-MCP Codebase Fix Guide

This document describes the comprehensive fix process for the Alpaca-MCP trading system codebase.

## Issues Identified

### 1. **Syntax Errors**
- Invalid escape sequence in `gpu_cluster_deployment_system.py` (line 969)
- Issue: `'nvidia\.com/gpu'` should be `'nvidia\\.com/gpu'`

### 2. **Import Errors**
- Missing modules that need to be installed
- Incorrect relative imports in some files
- Module name mismatches (e.g., `alpaca_trade_api` vs `alpaca-trade-api`)

### 3. **Configuration Issues**
- Missing `.env` file for environment variables
- Potential hardcoded credentials that should use environment variables
- Missing or incomplete `alpaca_config.json`

### 4. **Missing Dependencies**
- Core packages: `alpaca-trade-api`, `yfinance`, `pandas`, `numpy`
- MinIO integration: `minio`, `boto3`
- ML/AI packages: `torch`, `scikit-learn`, `tensorflow`
- Web frameworks: `fastapi`, `flask`, `uvicorn`
- Database: `psycopg2-binary`, `redis`

### 5. **Directory Structure Issues**
- Missing required directories: `logs`, `models`, `reports`, `backtest_results`
- Missing subdirectories for deployment and options-wheel modules

### 6. **File Permission Issues**
- Python scripts and shell scripts need executable permissions

### 7. **Database Issues**
- SQLite database files may need initialization
- Database connection strings may need updating

## Fix Scripts Created

### 1. **quick_fixes.py**
Quick fixes for immediate issues:
- Fixes escape sequence errors
- Installs essential packages only
- Creates `.env` template
- Creates required directories
- Validates Alpaca configuration

### 2. **check_runtime_errors.py**
Diagnostic tool that:
- Checks all module imports
- Validates configuration files
- Tests database connections
- Checks syntax of key scripts
- Scans log files for errors
- Generates diagnostic report

### 3. **fix_all_issues.py**
Comprehensive fix tool that:
- Fixes all syntax errors
- Installs all missing dependencies
- Fixes import statements
- Creates missing configuration files
- Sets correct file permissions
- Creates all required directories
- Validates database connections
- Checks for hardcoded credentials
- Generates detailed fix report

### 4. **run_all_fixes.sh**
Master script that runs all fixes in the correct order:
1. Quick fixes
2. Runtime error check
3. Comprehensive fixes
4. Additional manual fixes
5. Directory creation
6. Permission setting

## How to Use

### Quick Start (Recommended)
```bash
# Run the master fix script
./run_all_fixes.sh

# Or if permissions issue:
bash run_all_fixes.sh
```

### Manual Process
```bash
# Step 1: Run quick fixes
python3 quick_fixes.py

# Step 2: Check for runtime errors
python3 check_runtime_errors.py

# Step 3: Run comprehensive fixes
python3 fix_all_issues.py

# Step 4: Set up Alpaca configuration
python3 alpaca_config_setup.py

# Step 5: Update .env file with your credentials
nano .env  # or use your preferred editor
```

## After Fixing

1. **Review Reports**
   - `diagnostic_report.json` - System diagnostic information
   - `fix_report.json` - Detailed list of fixes applied
   - `fix_issues_*.log` - Detailed logs of fix process

2. **Configure API Keys**
   ```bash
   # Set up Alpaca API configuration
   python3 alpaca_config_setup.py
   
   # Edit .env file with your actual credentials
   nano .env
   ```

3. **Test the System**
   ```bash
   # Test basic connectivity
   python3 test_connection.py
   
   # Test MinIO connection (if using)
   python3 test_minio_connection.py
   
   # Run a simple demo
   python3 simple_demo.py
   ```

4. **Start Trading**
   ```bash
   # Paper trading (recommended for testing)
   python3 alpaca_paper_trading_system.py
   
   # Options bot
   python3 simple_options_bot.py
   
   # Live trading (use with caution)
   python3 alpaca_live_trading_system.py
   ```

## Common Issues and Solutions

### Issue: "Module not found"
**Solution**: Run `pip install -r requirements.txt` or use the fix scripts

### Issue: "Invalid API credentials"
**Solution**: 
1. Run `python3 alpaca_config_setup.py`
2. Update `.env` file with correct credentials
3. Ensure you're using paper trading credentials for testing

### Issue: "Permission denied"
**Solution**: Run `chmod +x *.py *.sh` or use the fix scripts

### Issue: "Database error"
**Solution**: Delete the `.db` file and let the system recreate it

### Issue: "MinIO connection failed"
**Solution**: 
1. Check if MinIO is running: `docker ps`
2. Verify credentials in `.env` file
3. Test with `python3 test_minio_connection.py`

## Support

For additional help:
1. Check the log files in the `logs/` directory
2. Review the diagnostic and fix reports
3. Ensure all dependencies are installed
4. Verify your API credentials are correct

## Safety Notes

- Always test with paper trading first
- Keep your API credentials secure
- Don't commit `.env` or `alpaca_config.json` to version control
- Review all trades before enabling live trading
- Set appropriate risk limits in your trading configuration