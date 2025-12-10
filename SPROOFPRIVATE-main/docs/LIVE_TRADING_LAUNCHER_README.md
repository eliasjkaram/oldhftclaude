# Live Trading System Launcher

## Overview

The Live Trading System Launcher is a comprehensive, production-ready launcher for the AI-Enhanced High-Frequency Trading System. It provides a robust framework for starting, monitoring, and managing all system components with automatic error recovery and health monitoring.

## Features

### 🚀 Core Capabilities
- **Complete Component Management**: Starts all system components in proper dependency order
- **Multiple Trading Modes**: Paper trading (default), live trading, backtest, and research modes
- **Health Monitoring**: Continuous health checks with automatic component restart
- **Comprehensive Logging**: Detailed logging with rotation and multiple output channels
- **Error Recovery**: Automatic error handling and component restart capabilities
- **Resource Monitoring**: Real-time CPU, memory, and disk usage tracking
- **Graceful Shutdown**: Clean shutdown of all components in reverse dependency order

### 🛡️ Safety Features
- **Paper Trading Default**: System defaults to paper trading for safety
- **Live Trading Confirmation**: Requires explicit confirmation for live trading
- **Pre-flight Checks**: Validates system resources, API credentials, and dependencies
- **Market Hours Check**: Warns if market is closed (non-blocking)
- **Component Health Checks**: Regular health monitoring with configurable thresholds

### 📊 Real-time Monitoring
- **System Status Display**: Periodic status updates showing all component states
- **Performance Metrics**: Tracks opportunities found, trades executed, and P&L
- **Resource Usage**: Monitors CPU, memory, and disk utilization
- **Error Tracking**: Counts and logs all errors with detailed information

## Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version

# Install required packages
pip install -r requirements.txt
```

### Environment Setup
```bash
# Set Alpaca API credentials
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"

# Set OpenRouter API key for AI features (optional)
export OPENROUTER_API_KEY="your_openrouter_key"
```

## Usage

### Basic Launch (Paper Trading)
```bash
# Default paper trading mode
python LIVE_TRADING_SYSTEM_LAUNCHER.py

# Or explicitly specify paper mode
python LIVE_TRADING_SYSTEM_LAUNCHER.py --mode paper
```

### Live Trading (Real Money)
```bash
# Requires confirmation
python LIVE_TRADING_SYSTEM_LAUNCHER.py --mode live
```

### Other Modes
```bash
# Backtest mode
python LIVE_TRADING_SYSTEM_LAUNCHER.py --mode backtest

# Research mode
python LIVE_TRADING_SYSTEM_LAUNCHER.py --mode research

# Enable debug logging
python LIVE_TRADING_SYSTEM_LAUNCHER.py --debug
```

## System Components

### Core Infrastructure
- **Health Monitor**: System-wide health monitoring and alerting
- **Logging System**: Centralized logging with multiple outputs

### Market Data
- **Alpaca Client**: Connection to Alpaca trading API
- **Market Data Collector**: Real-time and historical data collection
- **MinIO Pipeline**: High-performance data storage (optional)

### AI Components
- **AI Arbitrage Agent**: Multi-LLM powered arbitrage discovery
- **Strategy Optimizer**: AI-driven strategy optimization
- **Integrated AI HFT System**: Complete AI trading integration

### Trading Components
- **Order Executor**: High-speed order execution
- **Position Manager**: Position tracking and management
- **Risk Management**: Advanced risk controls and limits

### Advanced Features
- **Options Trader**: Options strategy execution (optional)
- **Arbitrage Scanner**: Traditional arbitrage scanning (optional)
- **GPU Cluster**: GPU acceleration for AI models (optional)

### Monitoring
- **Performance Tracker**: Real-time performance metrics
- **Realtime Monitor**: Web-based monitoring dashboard (optional)

## Pre-flight Checks

The launcher performs comprehensive checks before starting:

1. **Python Version**: Ensures Python 3.8+
2. **System Resources**: Validates CPU, memory, and disk space
3. **Network Connectivity**: Tests Alpaca and OpenRouter APIs
4. **API Credentials**: Verifies authentication credentials
5. **Market Hours**: Checks if market is open (warning only)
6. **Dependencies**: Validates all required packages

## Component Dependencies

Components are started in dependency order:
```
logging_system
    └── alpaca_client
            ├── market_data_collector
            │       ├── minio_pipeline
            │       ├── ai_arbitrage_agent
            │       │       └── strategy_optimizer
            │       │               └── ai_hft_system
            │       │                       └── gpu_cluster
            │       └── arbitrage_scanner
            └── order_executor
                    └── position_manager
                            ├── risk_management
                            └── performance_tracker
health_monitor
    └── realtime_monitor
```

## Monitoring and Health Checks

### Component Health
- Health checks run every 10 seconds
- Components with health endpoints are checked via HTTP
- Components with health_check() methods are called directly
- Failed components are automatically restarted (configurable)

### Metrics Tracked
- Total system uptime
- Component restart counts
- Errors handled
- Opportunities discovered
- Trades executed
- Profit/Loss

### Status Display
The system displays comprehensive status every 60 seconds:
```
================================================================================
SYSTEM STATUS
================================================================================

Components:
  ✅ health_monitor              running         [REQUIRED]
  ✅ logging_system              running         [REQUIRED]
  ✅ alpaca_client               running         [REQUIRED]
  ✅ market_data_collector       running         [REQUIRED]
  ⚠️  minio_pipeline              warning         [OPTIONAL]
  ✅ ai_arbitrage_agent          running         [REQUIRED]
  ...

Metrics:
  Uptime: 3600 seconds
  Opportunities Found: 5432
  Trades Executed: 127
  P&L: $1,234.56
  Errors Handled: 3

Resource Usage:
  CPU: 45%
  Memory: 62%
  Disk: 78%
================================================================================
```

## Error Handling

### Automatic Recovery
- Failed components are automatically restarted
- Maximum restart attempts configurable per component
- Exponential backoff for restart attempts
- Dependencies respected during restart

### Error Logging
- All errors logged with full stack traces
- Component-specific error counts maintained
- Last error message stored per component

## Graceful Shutdown

The system supports clean shutdown via:
- `Ctrl+C` (SIGINT)
- `kill` command (SIGTERM)
- Programmatic shutdown

Shutdown process:
1. Stops components in reverse dependency order
2. Allows components to clean up resources
3. Displays final metrics summary
4. Ensures all data is saved

## Configuration

### Component Configuration
Each component can be configured with:
- `required`: Whether component must start successfully
- `restart_on_failure`: Enable automatic restart
- `max_restarts`: Maximum restart attempts
- `health_check_endpoint`: HTTP endpoint for health checks
- `startup_delay`: Delay after starting component
- `config_overrides`: Component-specific configuration

### System Configuration
- Logging levels and outputs
- Monitoring intervals
- Resource thresholds
- API rate limits

## Troubleshooting

### Common Issues

1. **API Credentials Missing**
   ```bash
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
   ```

2. **High Resource Usage**
   - Check CPU/Memory in status display
   - Reduce number of active components
   - Adjust component configurations

3. **Component Fails to Start**
   - Check logs in `logs/` directory
   - Verify dependencies are met
   - Check component-specific configuration

4. **Network Issues**
   - Verify internet connectivity
   - Check firewall settings
   - Validate API endpoints

### Debug Mode
Enable detailed logging:
```bash
python LIVE_TRADING_SYSTEM_LAUNCHER.py --debug
```

## Best Practices

1. **Always Start in Paper Mode**: Test strategies thoroughly before live trading
2. **Monitor Resource Usage**: Keep CPU and memory below 80%
3. **Regular Backups**: System automatically creates backups in `backups/`
4. **Review Logs**: Check daily logs for warnings and errors
5. **Update Dependencies**: Keep all packages up to date

## Safety Warnings

⚠️ **IMPORTANT**: 
- Live trading involves real money and risk of loss
- Always test strategies in paper mode first
- Set appropriate risk limits and stop losses
- Monitor the system actively during trading hours
- Have a disaster recovery plan

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review component-specific documentation
3. Ensure all dependencies are installed
4. Verify API credentials are correct

## License

This system is provided as-is for educational and research purposes. Use at your own risk.