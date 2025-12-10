# Alpaca-MCP Operations Runbook

## Table of Contents
1. [System Startup Procedures](#system-startup-procedures)
2. [Daily Operations](#daily-operations)
3. [Trading Session Management](#trading-session-management)
4. [Monitoring and Alerts](#monitoring-and-alerts)
5. [Incident Response](#incident-response)
6. [Performance Optimization](#performance-optimization)
7. [Backup and Recovery](#backup-and-recovery)
8. [Maintenance Procedures](#maintenance-procedures)
9. [Emergency Procedures](#emergency-procedures)
10. [Troubleshooting Guide](#troubleshooting-guide)

## System Startup Procedures

### Pre-Market Startup Checklist

**Time: 30 minutes before market open**

```bash
#!/bin/bash
# Pre-market startup script

echo "=== Alpaca-MCP Pre-Market Startup ==="
echo "Time: $(date)"

# 1. Check system resources
echo "Checking system resources..."
df -h | grep -E "/$|/data"
free -h
nvidia-smi

# 2. Verify network connectivity
echo "Checking network connectivity..."
ping -c 3 api.alpaca.markets
ping -c 3 data.alpaca.markets

# 3. Test API connections
echo "Testing API connections..."
python3 scripts/test_connections.py

# 4. Verify data feeds
echo "Verifying data feeds..."
python3 scripts/verify_data_feeds.py

# 5. Check MinIO storage
echo "Checking MinIO storage..."
mc admin info minio/

# 6. Start core services
echo "Starting core services..."
systemctl start alpaca-mcp-redis
systemctl start alpaca-mcp-postgres
systemctl start alpaca-mcp-kafka

# 7. Initialize trading engine
echo "Initializing trading engine..."
python3 master_trading_orchestrator.py --mode=init

echo "=== Startup Complete ==="
```

### Service Start Order

1. **Infrastructure Services** (Start first)
   ```bash
   # Redis (cache and session storage)
   redis-server /etc/redis/redis.conf
   
   # PostgreSQL (trade history)
   pg_ctl start -D /var/lib/postgresql/data
   
   # Kafka (event streaming)
   kafka-server-start.sh config/server.properties
   
   # MinIO (historical data)
   minio server /data/minio
   ```

2. **Data Services** (Start second)
   ```bash
   # Market data manager
   python3 market_data_manager.py &
   
   # Historical data loader
   python3 minio_data_loader.py &
   
   # Real-time feed handler
   python3 realtime_feed_handler.py &
   ```

3. **Trading Services** (Start third)
   ```bash
   # Risk management engine
   python3 risk_management_engine.py &
   
   # ML prediction service
   python3 ml_prediction_service.py &
   
   # Options executor
   python3 comprehensive_options_executor.py &
   
   # Master orchestrator
   python3 master_trading_orchestrator.py
   ```

4. **Monitoring Services** (Start last)
   ```bash
   # Prometheus
   prometheus --config.file=prometheus.yml &
   
   # Grafana
   grafana-server &
   
   # Alert manager
   alertmanager --config.file=alertmanager.yml &
   ```

## Daily Operations

### Market Hours Schedule

```python
MARKET_SCHEDULE = {
    'pre_market': {
        'start': '04:00:00 ET',
        'end': '09:30:00 ET',
        'activities': [
            'System warmup',
            'Model updates',
            'Risk limit review',
            'News scanning'
        ]
    },
    'regular_hours': {
        'start': '09:30:00 ET',
        'end': '16:00:00 ET',
        'activities': [
            'Active trading',
            'Position monitoring',
            'Risk management',
            'Performance tracking'
        ]
    },
    'after_hours': {
        'start': '16:00:00 ET',
        'end': '20:00:00 ET',
        'activities': [
            'Limited trading',
            'Position reconciliation',
            'Report generation',
            'Next day preparation'
        ]
    }
}
```

### Daily Tasks

#### Pre-Market (4:00 AM - 9:30 AM ET)

1. **System Health Check**
   ```python
   def pre_market_health_check():
       checks = {
           'api_connectivity': test_api_connections(),
           'data_feeds': verify_data_feeds(),
           'model_status': check_ml_models(),
           'risk_limits': validate_risk_parameters(),
           'capital_available': check_buying_power()
       }
       
       for check, status in checks.items():
           if not status:
               alert_operations_team(f"FAILED: {check}")
               
       return all(checks.values())
   ```

2. **Update Market Intelligence**
   ```python
   def update_market_intelligence():
       # Economic calendar
       events = fetch_economic_calendar()
       high_impact_events = filter_high_impact(events)
       
       # Earnings releases
       earnings = fetch_earnings_calendar()
       portfolio_earnings = filter_portfolio_companies(earnings)
       
       # News sentiment
       news = scan_market_news()
       sentiment = analyze_sentiment(news)
       
       return {
           'economic_events': high_impact_events,
           'earnings': portfolio_earnings,
           'sentiment': sentiment
       }
   ```

3. **Model Refresh**
   ```bash
   # Retrain models with latest data
   python3 production_ml_training_system.py --mode=incremental
   
   # Update feature statistics
   python3 scripts/update_feature_stats.py
   
   # Validate model performance
   python3 scripts/validate_models.py --date=today
   ```

#### Market Hours (9:30 AM - 4:00 PM ET)

1. **Active Monitoring Dashboard**
   ```
   ┌─────────────────────────────────────────────────────────┐
   │ ALPACA-MCP TRADING DASHBOARD          [2024-12-06 10:15]│
   ├─────────────────────────────────────────────────────────┤
   │ POSITIONS (12 active)                                   │
   │ Symbol  Qty    Entry    Current  P&L      Risk         │
   │ AAPL    100   $180.50  $181.75  +$125    Low          │
   │ MSFT    -50   $420.25  $419.50  +$37.50  Medium       │
   │ SPY IC  1     $2.45    $1.80    +$65     Low          │
   ├─────────────────────────────────────────────────────────┤
   │ PERFORMANCE                                             │
   │ Daily P&L: +$1,245.67 (+0.62%)                        │
   │ Win Rate: 73.2%                                        │
   │ Sharpe: 2.45                                           │
   ├─────────────────────────────────────────────────────────┤
   │ RISK METRICS                                           │
   │ VaR (95%): $3,456                                      │
   │ Position Risk: 45% of limit                            │
   │ Margin Used: 38%                                       │
   ├─────────────────────────────────────────────────────────┤
   │ ACTIVE ALGORITHMS                                      │
   │ ✓ ML Ensemble (87% confidence)                        │
   │ ✓ Quantum Analysis (3 signals)                         │
   │ ✓ Swarm Intelligence (78% consensus)                   │
   │ ✓ Options Arbitrage (2 opportunities)                 │
   └─────────────────────────────────────────────────────────┘
   ```

2. **Real-Time Adjustments**
   ```python
   def monitor_and_adjust():
       while market_is_open():
           # Check positions
           positions = get_current_positions()
           
           for position in positions:
               # Stop loss check
               if position.unrealized_loss > position.stop_loss:
                   close_position(position, reason="Stop loss triggered")
               
               # Profit target check
               elif position.unrealized_profit > position.take_profit:
                   close_position(position, reason="Profit target reached")
               
               # Risk adjustment
               elif position_risk(position) > risk_threshold:
                   reduce_position(position, reason="Risk limit exceeded")
           
           # Check for new opportunities
           signals = aggregate_trading_signals()
           validated_signals = risk_validate_signals(signals)
           execute_signals(validated_signals)
           
           time.sleep(1)  # 1-second loop
   ```

#### After Hours (4:00 PM - 8:00 PM ET)

1. **End of Day Reconciliation**
   ```python
   def end_of_day_reconciliation():
       # Download trade confirmations
       trades = fetch_todays_trades()
       
       # Reconcile with internal records
       discrepancies = reconcile_trades(trades)
       
       if discrepancies:
           alert_operations_team("Trade reconciliation discrepancies found")
           
       # Update position records
       update_position_database()
       
       # Calculate final P&L
       daily_pnl = calculate_daily_pnl()
       
       # Generate reports
       generate_daily_report(daily_pnl)
       
       return daily_pnl
   ```

2. **Performance Analysis**
   ```python
   def analyze_daily_performance():
       metrics = {
           'total_trades': count_trades(),
           'winning_trades': count_winning_trades(),
           'losing_trades': count_losing_trades(),
           'win_rate': calculate_win_rate(),
           'average_win': calculate_average_win(),
           'average_loss': calculate_average_loss(),
           'profit_factor': calculate_profit_factor(),
           'sharpe_ratio': calculate_daily_sharpe(),
           'max_drawdown': calculate_max_drawdown()
       }
       
       # Algorithm performance
       for algo in ['ml', 'quantum', 'swarm', 'options', 'hft']:
           metrics[f'{algo}_performance'] = analyze_algorithm_performance(algo)
       
       # Save to database
       save_performance_metrics(metrics)
       
       return metrics
   ```

## Trading Session Management

### Session States

```python
class TradingSessionState(Enum):
    INITIALIZING = "initializing"
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"
    HALTED = "halted"
    ERROR = "error"

class TradingSessionManager:
    def __init__(self):
        self.state = TradingSessionState.INITIALIZING
        self.start_time = None
        self.positions = []
        self.active_orders = []
        
    def transition_state(self, new_state):
        valid_transitions = {
            TradingSessionState.INITIALIZING: [TradingSessionState.PRE_MARKET],
            TradingSessionState.PRE_MARKET: [TradingSessionState.MARKET_OPEN, TradingSessionState.HALTED],
            TradingSessionState.MARKET_OPEN: [TradingSessionState.MARKET_CLOSE, TradingSessionState.HALTED],
            TradingSessionState.MARKET_CLOSE: [TradingSessionState.AFTER_HOURS],
            TradingSessionState.AFTER_HOURS: [TradingSessionState.INITIALIZING],
            TradingSessionState.HALTED: [TradingSessionState.MARKET_OPEN, TradingSessionState.ERROR],
            TradingSessionState.ERROR: [TradingSessionState.INITIALIZING]
        }
        
        if new_state in valid_transitions.get(self.state, []):
            self.state = new_state
            self.handle_state_transition()
        else:
            raise ValueError(f"Invalid state transition: {self.state} -> {new_state}")
```

### Position Management

```python
def manage_positions():
    """Real-time position management"""
    
    # 1. Load current positions
    positions = load_positions()
    
    # 2. Calculate metrics
    for position in positions:
        position.current_price = get_current_price(position.symbol)
        position.unrealized_pnl = calculate_pnl(position)
        position.risk_score = calculate_position_risk(position)
        
    # 3. Apply position limits
    total_exposure = sum(p.market_value for p in positions)
    if total_exposure > MAX_PORTFOLIO_EXPOSURE:
        reduce_largest_positions(positions)
        
    # 4. Rebalance if needed
    if should_rebalance(positions):
        rebalance_portfolio(positions)
        
    # 5. Update hedges
    update_portfolio_hedges(positions)
```

## Monitoring and Alerts

### Key Metrics to Monitor

```yaml
monitoring:
  system_metrics:
    - metric: cpu_usage
      threshold: 80%
      action: alert_warning
    - metric: memory_usage
      threshold: 90%
      action: alert_critical
    - metric: gpu_usage
      threshold: 95%
      action: scale_up
      
  trading_metrics:
    - metric: daily_loss
      threshold: 2%
      action: reduce_trading
    - metric: position_concentration
      threshold: 30%
      action: force_rebalance
    - metric: win_rate
      threshold: 40%
      action: pause_trading
      
  risk_metrics:
    - metric: portfolio_var
      threshold: 5%
      action: reduce_risk
    - metric: margin_usage
      threshold: 80%
      action: close_positions
    - metric: correlation_spike
      threshold: 0.9
      action: hedge_positions
```

### Alert Configuration

```python
ALERT_RULES = {
    'critical': {
        'conditions': [
            'daily_loss > 3%',
            'system_error_rate > 1%',
            'api_disconnection > 30s',
            'position_limit_breach'
        ],
        'actions': [
            'send_pagerduty_alert',
            'pause_trading',
            'close_risky_positions',
            'notify_risk_team'
        ]
    },
    'warning': {
        'conditions': [
            'daily_loss > 1.5%',
            'high_correlation_detected',
            'unusual_volume_spike',
            'model_confidence < 60%'
        ],
        'actions': [
            'send_slack_alert',
            'reduce_position_sizes',
            'increase_monitoring_frequency'
        ]
    },
    'info': {
        'conditions': [
            'new_high_confidence_signal',
            'successful_trade_execution',
            'model_update_complete'
        ],
        'actions': [
            'log_event',
            'update_dashboard'
        ]
    }
}
```

### Monitoring Dashboard

```python
def create_monitoring_dashboard():
    """Create real-time monitoring dashboard"""
    
    dashboard = {
        'system_health': {
            'api_status': check_api_status(),
            'data_feed_status': check_data_feeds(),
            'model_status': check_model_health(),
            'database_status': check_database_connections()
        },
        'trading_performance': {
            'active_positions': count_active_positions(),
            'daily_pnl': calculate_daily_pnl(),
            'win_rate': calculate_win_rate(),
            'sharpe_ratio': calculate_sharpe_ratio()
        },
        'risk_metrics': {
            'var_95': calculate_var(0.95),
            'max_drawdown': calculate_max_drawdown(),
            'position_concentration': calculate_concentration(),
            'margin_usage': calculate_margin_usage()
        },
        'algorithm_performance': {
            'ml_accuracy': get_ml_accuracy(),
            'quantum_signals': count_quantum_signals(),
            'swarm_consensus': get_swarm_consensus(),
            'options_opportunities': count_options_opportunities()
        }
    }
    
    return dashboard
```

## Incident Response

### Incident Classification

```python
class IncidentSeverity(Enum):
    CRITICAL = 1  # System down, major loss
    HIGH = 2      # Degraded performance, risk breach
    MEDIUM = 3    # Minor issues, warnings
    LOW = 4       # Info only

class IncidentType(Enum):
    SYSTEM_FAILURE = "system_failure"
    DATA_ISSUE = "data_issue"
    TRADING_ERROR = "trading_error"
    RISK_BREACH = "risk_breach"
    PERFORMANCE_DEGRADATION = "performance_degradation"
```

### Response Procedures

#### Critical Incident Response

```python
def handle_critical_incident(incident):
    """Handle critical incidents"""
    
    # 1. Immediate actions
    logger.critical(f"CRITICAL INCIDENT: {incident}")
    
    # 2. Pause all trading
    pause_all_trading_activities()
    
    # 3. Close risky positions
    close_risky_positions(risk_threshold=0.7)
    
    # 4. Notify team
    notify_incident_response_team(incident)
    
    # 5. Start incident log
    incident_id = start_incident_log(incident)
    
    # 6. Execute runbook
    if incident.type == IncidentType.SYSTEM_FAILURE:
        execute_system_failure_runbook()
    elif incident.type == IncidentType.RISK_BREACH:
        execute_risk_breach_runbook()
    elif incident.type == IncidentType.TRADING_ERROR:
        execute_trading_error_runbook()
        
    return incident_id
```

#### System Failure Runbook

```bash
#!/bin/bash
# System failure runbook

echo "=== SYSTEM FAILURE RESPONSE ==="
echo "Time: $(date)"
echo "Severity: CRITICAL"

# 1. Stop trading engine
echo "Stopping trading engine..."
systemctl stop alpaca-mcp-trading

# 2. Preserve state
echo "Preserving system state..."
mkdir -p /emergency/$(date +%Y%m%d_%H%M%S)
cp -r /var/log/alpaca-mcp/* /emergency/$(date +%Y%m%d_%H%M%S)/

# 3. Check core services
echo "Checking core services..."
for service in redis postgresql kafka minio; do
    systemctl status $service
    if [ $? -ne 0 ]; then
        echo "Service $service is down. Attempting restart..."
        systemctl restart $service
    fi
done

# 4. Validate data integrity
echo "Validating data integrity..."
python3 scripts/validate_data_integrity.py

# 5. Test API connectivity
echo "Testing API connectivity..."
python3 scripts/test_api_connectivity.py

# 6. Restart in safe mode
echo "Restarting in safe mode..."
python3 master_trading_orchestrator.py --mode=safe

echo "=== RUNBOOK COMPLETE ==="
```

## Performance Optimization

### Query Optimization

```python
# Optimized data queries
class OptimizedDataAccess:
    def __init__(self):
        self.cache = Redis()
        self.connection_pool = create_connection_pool()
        
    def get_market_data(self, symbol, timeframe='1d'):
        # Check cache first
        cache_key = f"market_data:{symbol}:{timeframe}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return pickle.loads(cached_data)
            
        # Use connection pool for database
        with self.connection_pool.get_connection() as conn:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = %s AND timeframe = %s
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            data = pd.read_sql(query, conn, params=(symbol, timeframe))
            
        # Cache for 60 seconds
        self.cache.setex(cache_key, 60, pickle.dumps(data))
        
        return data
```

### Model Inference Optimization

```python
class OptimizedModelInference:
    def __init__(self):
        self.model_cache = {}
        self.batch_queue = Queue()
        self.gpu_device = torch.device('cuda:0')
        
    def batch_predict(self, features_list, model_name='ensemble'):
        """Batch multiple predictions for efficiency"""
        
        # Load model to GPU if not cached
        if model_name not in self.model_cache:
            model = load_model(model_name)
            model.to(self.gpu_device)
            model.eval()
            self.model_cache[model_name] = model
            
        model = self.model_cache[model_name]
        
        # Batch features
        batch_tensor = torch.tensor(features_list).to(self.gpu_device)
        
        # Inference
        with torch.no_grad():
            predictions = model(batch_tensor)
            
        return predictions.cpu().numpy()
```

### Memory Management

```python
def optimize_memory_usage():
    """Optimize memory usage for long-running processes"""
    
    # 1. Clear unused objects
    gc.collect()
    
    # 2. Limit DataFrame memory
    for df_name in ['market_data', 'positions', 'orders']:
        if df_name in globals():
            df = globals()[df_name]
            # Convert to efficient dtypes
            df = reduce_memory_usage(df)
            
    # 3. Clear model cache periodically
    if len(model_cache) > 10:
        # Keep only most recently used models
        sorted_models = sorted(model_cache.items(), key=lambda x: x[1]['last_used'])
        for model_name, _ in sorted_models[:-5]:
            del model_cache[model_name]
            
    # 4. Compress historical data
    compress_old_logs()
```

## Backup and Recovery

### Backup Strategy

```yaml
backup_strategy:
  databases:
    postgresql:
      frequency: hourly
      retention: 7_days
      method: pg_dump
      location: s3://alpaca-mcp-backups/postgres/
      
    redis:
      frequency: every_6_hours
      retention: 3_days
      method: rdb_snapshot
      location: s3://alpaca-mcp-backups/redis/
      
  models:
    ml_models:
      frequency: daily
      retention: 30_days
      versioning: true
      location: s3://alpaca-mcp-backups/models/
      
  configurations:
    frequency: on_change
    retention: unlimited
    git_backup: true
    location: git@github.com:org/alpaca-mcp-config.git
```

### Recovery Procedures

```python
class DisasterRecovery:
    def __init__(self):
        self.backup_location = 's3://alpaca-mcp-backups/'
        self.recovery_point = None
        
    def restore_system(self, target_timestamp):
        """Restore system to specific point in time"""
        
        steps = [
            self.stop_all_services,
            self.restore_databases,
            self.restore_models,
            self.restore_configurations,
            self.validate_restoration,
            self.start_services_safe_mode,
            self.run_integrity_checks,
            self.resume_normal_operations
        ]
        
        for step in steps:
            success = step(target_timestamp)
            if not success:
                logger.error(f"Recovery failed at step: {step.__name__}")
                return False
                
        return True
```

## Maintenance Procedures

### Weekly Maintenance

```bash
#!/bin/bash
# Weekly maintenance script

echo "=== WEEKLY MAINTENANCE ==="
echo "Start time: $(date)"

# 1. Database maintenance
echo "Running database maintenance..."
psql -U postgres -d alpaca_mcp -c "VACUUM ANALYZE;"
psql -U postgres -d alpaca_mcp -c "REINDEX DATABASE alpaca_mcp;"

# 2. Log rotation
echo "Rotating logs..."
logrotate -f /etc/logrotate.d/alpaca-mcp

# 3. Clear old cache entries
echo "Clearing old cache..."
redis-cli --scan --pattern "expired:*" | xargs redis-cli del

# 4. Model performance review
echo "Reviewing model performance..."
python3 scripts/model_performance_review.py --period=weekly

# 5. System updates
echo "Checking for updates..."
pip list --outdated
apt list --upgradable

# 6. Backup verification
echo "Verifying backups..."
python3 scripts/verify_backups.py --period=7d

echo "=== MAINTENANCE COMPLETE ==="
echo "End time: $(date)"
```

### Model Retraining Schedule

```python
RETRAINING_SCHEDULE = {
    'ml_ensemble': {
        'frequency': 'weekly',
        'trigger_conditions': [
            'accuracy_drop > 5%',
            'market_regime_change',
            'new_data_threshold > 10000'
        ]
    },
    'quantum_parameters': {
        'frequency': 'daily',
        'optimization_method': 'grid_search'
    },
    'swarm_agents': {
        'frequency': 'continuous',
        'evolution_rate': 0.1
    },
    'neural_architecture': {
        'frequency': 'monthly',
        'search_budget': 100  # GPU hours
    }
}
```

## Emergency Procedures

### Market Crash Response

```python
def market_crash_response():
    """Emergency response to market crash"""
    
    # 1. Detect crash conditions
    market_drop = calculate_market_drop()
    vix_spike = get_vix_spike()
    
    if market_drop > 7 or vix_spike > 50:
        logger.critical("MARKET CRASH DETECTED")
        
        # 2. Immediate actions
        # Close all speculative positions
        close_positions(position_type='speculative')
        
        # Hedge remaining positions
        hedge_portfolio(hedge_ratio=1.0)
        
        # Switch to defensive mode
        set_trading_mode('defensive')
        
        # Reduce position limits
        update_risk_limits(reduction_factor=0.3)
        
        # 3. Notify stakeholders
        send_emergency_notification("Market crash protocol activated")
```

### API Outage Response

```python
def handle_api_outage():
    """Handle Alpaca API outage"""
    
    # 1. Switch to backup data feed
    switch_to_backup_feed()
    
    # 2. Use cached data for positions
    positions = load_cached_positions()
    
    # 3. Estimate current prices
    estimated_prices = estimate_prices_from_alternates()
    
    # 4. Run in degraded mode
    set_trading_mode('degraded')
    
    # 5. Monitor for API recovery
    start_api_recovery_monitor()
```

## Troubleshooting Guide

### Common Issues and Solutions

```yaml
troubleshooting:
  - issue: "ML model predictions returning NaN"
    symptoms:
      - "ValueError: Input contains NaN"
      - "Model confidence: 0%"
    causes:
      - "Missing market data"
      - "Feature engineering error"
    solutions:
      - "Check data feed connectivity"
      - "Validate feature pipeline"
      - "Use fallback model"
      
  - issue: "High latency in order execution"
    symptoms:
      - "Execution time > 100ms"
      - "Slippage > 0.1%"
    causes:
      - "Network congestion"
      - "API rate limiting"
    solutions:
      - "Check network metrics"
      - "Implement order queuing"
      - "Use smart order routing"
      
  - issue: "Risk limits being breached"
    symptoms:
      - "Position limit exceeded"
      - "VaR > threshold"
    causes:
      - "Correlation spike"
      - "Volatility increase"
    solutions:
      - "Force portfolio rebalance"
      - "Increase hedge ratio"
      - "Reduce position sizes"
```

### Debug Commands

```bash
# Check system status
alpaca-mcp status --verbose

# View recent errors
alpaca-mcp logs --level=error --last=1h

# Test specific component
alpaca-mcp test --component=risk_engine

# Force model reload
alpaca-mcp models --reload --model=ml_ensemble

# Check position reconciliation
alpaca-mcp positions --reconcile

# View performance metrics
alpaca-mcp metrics --period=today

# Run diagnostics
alpaca-mcp diagnose --full
```

---

This operations runbook provides comprehensive procedures for running the Alpaca-MCP trading system in production, handling incidents, optimizing performance, and maintaining system reliability.