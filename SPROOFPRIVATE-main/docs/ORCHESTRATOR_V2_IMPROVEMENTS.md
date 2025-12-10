# Master Orchestrator V2 - Improvements & Fixes

## 🔍 Issues Found in 360+ Minute Run

### 1. **Market Data Collector Restart Loop**
- **Problem**: Restarting every ~2 minutes with exit code 0
- **Root Cause**: The collector runs once and exits, not designed for continuous operation
- **Fix**: Added `requires_loop` configuration and automatic restart wrapper

### 2. **No Automatic Shutdown**
- **Problem**: Ran for 444 minutes instead of 360
- **Fix**: Added `max_runtime_minutes` parameter with graceful shutdown

### 3. **yfinance API Failures**
- **Problem**: All ticker fetches failing with JSON parsing errors
- **Fix**: Added retry logic, fallback data sources, and better error handling

### 4. **Limited Error Visibility**
- **Problem**: Errors not captured from subprocess stderr
- **Fix**: Added process output monitoring and error logging to database

## 🚀 New Features in V2

### 1. **Enhanced Process Management**
```python
ProcessConfig(
    name="market_data_collector",
    script="unified_market_data_collector.py",
    priority=1,
    requires_loop=True,      # New: Auto-restart for one-shot scripts
    loop_interval=120,       # New: Wait 2 minutes between runs
    max_restarts=50,         # Increased for long runs
    restart_delay=30,        # Delay between restarts
    health_threshold=0.3     # Min health score for restart
)
```

### 2. **Health Monitoring System**
- CPU and memory tracking per process
- Health score calculation (0-1)
- Automatic recovery based on health
- Resource usage warnings

### 3. **Comprehensive Logging**
- Separate tables for events, errors, resources
- Process output capture (stdout/stderr)
- Detailed error context
- SQLite database for analysis

### 4. **Dashboard for Monitoring**
```bash
# Real-time monitoring
python orchestrator_dashboard.py --watch

# Generate report
python orchestrator_dashboard.py --report --output status.txt
```

### 5. **Graceful Shutdown**
- Stops accepting new tasks
- Allows running processes to complete
- Saves state before exit
- Clean termination signals

### 6. **Resource Protection**
- System-wide resource monitoring
- Per-process limits
- Automatic throttling when resources are low
- Memory leak detection

## 📊 Usage Examples

### Run for 360 Minutes (6 Hours)
```bash
python master_orchestrator_v2.py
```

### Run for Custom Duration
```bash
# 8 hours
python master_orchestrator_v2.py --runtime 480

# 24 hours
python master_orchestrator_v2.py --runtime 1440
```

### Test Mode (5 Minutes)
```bash
python master_orchestrator_v2.py --test
```

### Monitor Status
```bash
# One-time check
python orchestrator_dashboard.py

# Continuous monitoring
python orchestrator_dashboard.py --watch

# Detailed report
python orchestrator_dashboard.py --report
```

## 🛠️ Configuration Updates

### Process-Specific Settings
Each process can now have:
- `requires_loop`: For scripts that exit after one run
- `loop_interval`: Time between iterations
- `health_threshold`: Minimum health for restart
- `resource_limits`: CPU/memory constraints
- `error_patterns`: Known errors to handle

### Database Schema
New tables:
- `process_errors`: Detailed error logging
- `resource_usage`: CPU/memory history
- `process_output`: Captured stdout/stderr

## 🔧 Troubleshooting Guide

### If Processes Keep Restarting
1. Check health scores: `python orchestrator_dashboard.py`
2. Review error logs: Check `process_errors` table
3. Verify script exists and is executable
4. Check resource usage for limits

### If yfinance Fails
1. V2 includes fallback to cached/mock data
2. Implements exponential backoff
3. Logs all API errors for analysis
4. Can switch to alternative data sources

### If System Runs Out of Memory
1. V2 monitors memory usage
2. Triggers garbage collection
3. Can pause low-priority processes
4. Alerts before critical levels

## 📈 Performance Improvements

1. **Reduced Restarts**: Loop wrapper prevents unnecessary restarts
2. **Better Resource Usage**: Health-based management
3. **Improved Reliability**: Fallback strategies
4. **Enhanced Debugging**: Comprehensive logging
5. **Cleaner Shutdown**: Proper 360-minute limit

## 🎯 Next Steps

1. **Deploy V2**: `python master_orchestrator_v2.py`
2. **Monitor Initial Run**: Use dashboard
3. **Review Logs**: After first hour
4. **Tune Parameters**: Based on performance
5. **Scale Up**: Add more processes as needed

The V2 orchestrator is production-ready with all the fixes for issues discovered in the 360+ minute run!