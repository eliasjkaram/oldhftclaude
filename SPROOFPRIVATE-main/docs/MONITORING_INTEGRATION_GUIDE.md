# Comprehensive Monitoring and Logging Integration Guide

## Overview

This guide explains how to integrate comprehensive structured logging and monitoring throughout the Alpaca trading system. The monitoring system provides:

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Audit Trails**: Compliance-ready audit logs for all trading operations
- **Performance Metrics**: Real-time performance tracking
- **Prometheus Integration**: Export metrics to Prometheus for visualization
- **Health Checks**: Automated system health monitoring
- **Dashboards**: Web-based monitoring dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring System                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Structured  │  │   Metrics    │  │  Health Checks   │  │
│  │  Logger     │  │  Collector   │  │    Manager       │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│         │                 │                    │            │
│         └─────────────────┴────────────────────┘           │
│                           │                                 │
│                    ┌──────────────┐                        │
│                    │ Correlation  │                        │
│                    │   Context    │                        │
│                    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Trading Systems                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Master    │  │   Trading    │  │      Data        │  │
│  │Orchestrator │  │   Engines    │  │   Providers      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. **Install Required Dependencies**:
```bash
pip install prometheus-client flask psutil
```

2. **Import Monitoring Components**:
```python
from comprehensive_monitoring_system import (
    get_monitoring_system,
    get_trading_monitor,
    MonitoredLogger,
    CorrelationContext
)
from monitoring_integration import (
    MonitoringMixin,
    monitor_class,
    monitor_api_call,
    monitor_trading_operation
)
```

## Integration Examples

### 1. Basic Function Monitoring

```python
from comprehensive_monitoring_system import get_monitoring_system

monitoring = get_monitoring_system()

@monitoring.monitor_function
def calculate_portfolio_value(positions, prices):
    """This function will be automatically monitored"""
    return sum(positions[symbol] * prices[symbol] for symbol in positions)
```

### 2. Class Monitoring with Mixin

```python
from monitoring_integration import MonitoringMixin

class TradingStrategy(MonitoringMixin):
    def __init__(self):
        super().__init__()
        # All public methods will be automatically monitored
    
    def analyze_market(self, symbol):
        # This method is automatically monitored
        return {'signal': 'buy', 'confidence': 0.8}
```

### 3. Trading Operation Monitoring

```python
from monitoring_integration import monitor_trading_operation

@monitor_trading_operation("place_order")
def place_market_order(symbol, quantity, side):
    """Place order with automatic audit logging"""
    # Order placement logic
    return order_id
```

### 4. API Call Monitoring

```python
from monitoring_integration import monitor_api_call

@monitor_api_call("alpaca/orders", "POST")
async def submit_order_to_alpaca(order_data):
    """API calls are monitored with latency tracking"""
    # API call logic
    return response
```

### 5. Manual Correlation Tracking

```python
from comprehensive_monitoring_system import get_monitoring_system

monitoring = get_monitoring_system()

with monitoring.track_request('complex_operation', user_id=12345) as correlation_id:
    # All logs within this context will have the same correlation_id
    logger.info("Starting complex operation")
    
    # Perform operations
    result = perform_calculation()
    
    logger.info("Operation completed", result=result)
```

## Monitoring Dashboard

The system includes a web-based monitoring dashboard accessible at:

- **Dashboard**: http://localhost:5000
- **Metrics**: http://localhost:5000/metrics
- **Health Check**: http://localhost:5000/health
- **Recent Logs**: http://localhost:5000/logs/recent

## Prometheus Integration

1. **Metrics Endpoint**: The system exposes Prometheus metrics at `/metrics`
2. **Available Metrics**:
   - `trading_system_orders_total`: Total orders by type, side, and status
   - `trading_system_order_execution_duration_seconds`: Order execution time
   - `trading_system_portfolio_value_usd`: Current portfolio value
   - `trading_system_active_positions_count`: Number of active positions
   - `trading_system_api_requests_total`: API request counts
   - `trading_system_errors_total`: Error counts by type
   - `trading_system_model_predictions_total`: ML model predictions
   - `trading_system_model_accuracy_percent`: Model accuracy metrics

3. **Prometheus Configuration**:
```yaml
scrape_configs:
  - job_name: 'trading_system'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Audit Logging

All trading operations are automatically audit logged:

```python
from comprehensive_monitoring_system import get_trading_monitor

trading_monitor = get_trading_monitor()

# Automatic audit logging for orders
with trading_monitor.monitor_order_execution('market', 'AAPL', 'buy', 100) as order_id:
    # Execute order
    pass

# Manual audit logging
monitoring.logger.audit(
    'RISK_LIMIT_UPDATED',
    {
        'old_limit': 10000,
        'new_limit': 15000,
        'updated_by': 'admin',
        'reason': 'Increased capital'
    },
    user='admin',
    ip_address='192.168.1.1'
)
```

## Health Checks

Register custom health checks:

```python
monitoring = get_monitoring_system()

def check_database_connection():
    """Custom health check for database"""
    try:
        # Check database
        return {'status': 'healthy', 'tables': 10}
    except Exception as e:
        raise Exception(f"Database unhealthy: {e}")

monitoring.register_health_check('database', check_database_connection)
```

## Integration with Existing Systems

### Master Orchestrator Integration

```python
from enhanced_master_orchestrator import EnhancedMasterOrchestrator

# The enhanced orchestrator includes comprehensive monitoring
orchestrator = EnhancedMasterOrchestrator()
await orchestrator.start()
```

### Trading System Integration

```python
from monitoring_integration import integrate_monitoring_with_trading_system

# Add monitoring to existing trading system
trading_system = YourTradingSystem()
monitored_system = integrate_monitoring_with_trading_system(trading_system)
```

## Performance Optimization

1. **Asynchronous Logging**: All logs are written asynchronously to avoid blocking
2. **Metric Batching**: Metrics are batched before export
3. **Correlation Context**: Thread-local storage for efficient correlation tracking
4. **Lazy Initialization**: Components are initialized only when needed

## Best Practices

1. **Use Correlation IDs**: Always use correlation IDs for request tracking
2. **Structured Data**: Include structured data in logs for better querying
3. **Appropriate Log Levels**: Use INFO for normal operations, ERROR for failures
4. **Audit Critical Operations**: Always audit log trading operations
5. **Monitor Performance**: Track execution times for critical paths
6. **Health Checks**: Implement health checks for all external dependencies

## Troubleshooting

### Common Issues

1. **Prometheus Not Available**:
   - The system will fall back to local metrics storage
   - Install prometheus-client: `pip install prometheus-client`

2. **Port Conflicts**:
   - Dashboard runs on port 5000
   - Prometheus metrics on port 9090
   - Change ports in configuration if needed

3. **Performance Impact**:
   - Monitoring adds ~1-2ms overhead per operation
   - Use sampling for high-frequency operations

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('comprehensive_monitoring_system').setLevel(logging.DEBUG)
```

## Security Considerations

1. **Sensitive Data**: Never log passwords, API keys, or sensitive user data
2. **Audit Logs**: Store audit logs securely with proper retention
3. **Access Control**: Restrict access to monitoring endpoints
4. **Data Encryption**: Consider encrypting logs at rest

## Compliance Features

The monitoring system supports compliance requirements:

1. **Audit Trail**: Complete audit trail for all trading operations
2. **Data Retention**: Configurable log retention policies
3. **User Attribution**: Track user and IP for all operations
4. **Immutable Logs**: Audit logs are append-only
5. **Correlation Tracking**: Full request tracing capability

## Example: Complete Integration

```python
#!/usr/bin/env python3
"""Example of complete monitoring integration"""

from comprehensive_monitoring_system import get_monitoring_system, get_trading_monitor
from monitoring_integration import monitor_class, monitor_trading_operation

# Initialize monitoring
monitoring = get_monitoring_system()
trading_monitor = get_trading_monitor()

@monitor_class
class EnhancedTradingBot:
    """Trading bot with comprehensive monitoring"""
    
    def __init__(self):
        self.positions = {}
    
    @monitor_trading_operation("analyze_signal")
    def analyze_market(self, symbol):
        """Analyze market for trading signals"""
        # Analysis logic
        return {'signal': 'buy', 'confidence': 0.85}
    
    @monitor_trading_operation("execute_trade")
    def execute_trade(self, symbol, signal, quantity):
        """Execute trade with full monitoring"""
        with trading_monitor.monitor_order_execution('market', symbol, signal, quantity) as order_id:
            # Execute order
            print(f"Executing {signal} order for {quantity} shares of {symbol}")
            
            # Update metrics
            trading_monitor.monitor_portfolio_update(
                portfolio_value=100000,
                positions={symbol: quantity}
            )
            
            return order_id

# Usage
bot = EnhancedTradingBot()
signal_result = bot.analyze_market('AAPL')
if signal_result['confidence'] > 0.8:
    order_id = bot.execute_trade('AAPL', signal_result['signal'], 100)
```

## Maintenance

1. **Log Rotation**: Configure log rotation to prevent disk space issues
2. **Metric Retention**: Set appropriate retention for Prometheus data
3. **Health Check Frequency**: Adjust health check intervals based on needs
4. **Dashboard Updates**: Regularly update dashboard queries

## Support

For issues or questions:
1. Check the logs in `monitoring.log`
2. View health status at http://localhost:5000/health
3. Check metrics at http://localhost:5000/metrics
4. Review audit logs for compliance tracking