# 📊 Real-Time Monitoring Guide - Alpaca MCP Trading System

## Overview

This guide shows you all the ways to monitor what your trading system is doing in real-time. You can see live trades, AI discoveries, system performance, and more.

## 🚀 Quick Start - Web Dashboard

The easiest way to monitor everything:

```bash
# Start the web monitoring dashboard
python web_monitor.py

# Open in browser
http://localhost:8888
```

This shows:
- Live system metrics (CPU, memory, network)
- Trading performance (P&L, success rate)
- AI discovery status
- Recent activity logs
- Recent trades

## 📺 Terminal-Based Monitoring

### 1. **Interactive Monitor**
```bash
python realtime_monitor.py
```
Shows a comprehensive terminal dashboard with:
- System resources
- Active processes
- Recent logs
- Database activity
- Quick access to other monitoring tools

### 2. **System Dashboard**
```bash
python system_dashboard.py
```
Beautiful ASCII dashboard showing:
- System status
- Trading capabilities
- Recent AI discoveries
- Quick commands

### 3. **TMux Multi-View** (Recommended)
```bash
./start_monitoring.sh
```
Creates multiple windows:
- Window 1: System Dashboard
- Window 2: 4-panel log view (trading, AI, orders, monitoring)
- Window 3: Process monitor (htop)
- Window 4: Real-time monitor

Navigation:
- `Ctrl+B, N` = Next window
- `Ctrl+B, P` = Previous window
- `Ctrl+B, 0-3` = Jump to specific window
- `Ctrl+B, D` = Detach (keeps running)

## 📝 Log File Monitoring

### Real-Time Log Tailing
```bash
# Watch all logs
tail -f *.log

# Watch specific logs
tail -f production_trading.log    # Trading activity
tail -f ai_arbitrage.log         # AI discoveries
tail -f order_execution.log      # Order details
tail -f monitoring.log           # System monitoring

# Watch with highlighting
tail -f production_trading.log | grep --color=always -E "(EXECUTED|PROFIT|ERROR)"
```

### Log Analysis
```bash
# Count trades today
grep "EXECUTED" production_trading.log | grep "$(date +%Y-%m-%d)" | wc -l

# See all profits
grep "P&L:" production_trading.log | tail -20

# Find errors
grep -i "error" *.log
```

## 🔍 Process Monitoring

### See What's Running
```bash
# List all Python trading processes
ps aux | grep python | grep -E "(trading|arbitrage|alpaca)"

# Watch process changes
watch -n 1 'ps aux | grep python | grep trading'

# Detailed process info
pgrep -f "alpaca" | xargs -I {} ps -p {} -o pid,cmd,%cpu,%mem,etime
```

### Resource Usage
```bash
# Interactive process monitor
htop
# Press F4 and type "python" to filter

# Simple top view
top -c
# Press 'o' and type 'COMMAND=python' to filter
```

## 💾 Database Monitoring

### View Recent Trades
```bash
# Open database
sqlite3 trading_system.db

# Recent trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;

# Today's P&L
SELECT SUM(pnl) FROM trades WHERE date(timestamp) = date('now');

# Success rate
SELECT 
  COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as success_rate 
FROM trades;

# Exit
.quit
```

### Watch Database Changes
```bash
# Monitor database file
watch -n 1 'ls -la *.db'

# See database writes in real-time
inotifywait -m trading_system.db
```

## 🌐 Network Monitoring

### API Connections
```bash
# See active connections
netstat -tuln | grep -E "(443|8080|9090|3000)"

# Watch network traffic
iftop -i any -f "port 443"

# Monitor specific API calls
tcpdump -i any -A 'host api.alpaca.markets'
```

### Service Ports
- `8080` - Monitoring API
- `8888` - Web Dashboard
- `9090` - Prometheus metrics
- `9091` - Prometheus server
- `3000` - Grafana

## 📈 Performance Monitoring

### Prometheus Metrics
```bash
# View raw metrics
curl http://localhost:9090/metrics

# Specific metrics
curl -s http://localhost:9090/metrics | grep trading_

# Grafana Dashboard
Open http://localhost:3000
Login: admin/admin
```

### Custom Metrics
```bash
# Orders per minute
curl -s http://localhost:9090/metrics | grep orders_total

# AI discovery rate
curl -s http://localhost:9090/metrics | grep discoveries_per_second

# Current P&L
curl -s http://localhost:9090/metrics | grep portfolio_value
```

## 🔔 Real-Time Alerts

### Set Up Notifications
```bash
# Watch for losses
tail -f production_trading.log | grep --line-buffered "LOSS" | while read line; do
    echo "ALERT: $line"
    # Add notification command here (e.g., notify-send, mail, etc.)
done

# Watch for errors
tail -f *.log | grep --line-buffered -i "error" | while read line; do
    echo "ERROR DETECTED: $line"
done
```

## 🎯 Quick Commands

### Status Checks
```bash
# Full system status
python system_status_check.py

# Quick health check
python -c "from PRODUCTION_FIXES import HealthMonitor; import asyncio; hm = HealthMonitor(); print(asyncio.run(hm.run_health_checks()))"

# Test API connection
python -c "from production_data_manager import ProductionDataManager; dm = ProductionDataManager(); print('Connected' if dm else 'Failed')"
```

### Live Monitoring Commands
```bash
# Start everything
python run_all_concurrent.py

# Run trading demo
python live_trading_demo.py

# Run AI discovery
python ai_arbitrage_demo.py
```

## 🖥️ Advanced Monitoring

### Custom Watch Commands
```bash
# Watch multiple metrics
watch -n 1 'echo "=== SYSTEM ===" && \
  ps aux | grep python | grep trading | wc -l && \
  echo "=== LOGS ===" && \
  tail -5 production_trading.log && \
  echo "=== DATABASE ===" && \
  sqlite3 trading_system.db "SELECT COUNT(*) as trades FROM trades;"'
```

### Performance Profiling
```bash
# CPU profiling
py-spy top -- python alpaca_live_trading_system.py

# Memory profiling
mprof run python ai_arbitrage_demo.py
mprof plot
```

## 📱 Remote Monitoring

### SSH Monitoring
```bash
# SSH to server and attach to tmux
ssh user@server
tmux attach -t alpaca-monitor

# Port forward web dashboard
ssh -L 8888:localhost:8888 user@server
# Then open http://localhost:8888
```

### Mobile Access
1. Start web monitor on server
2. Access via phone browser: `http://server-ip:8888`
3. See real-time updates on mobile

## 🆘 Troubleshooting

### If Nothing Shows Up
1. Check if processes are running: `ps aux | grep python`
2. Check if log files exist: `ls -la *.log`
3. Create log files: `touch production_trading.log ai_arbitrage.log`
4. Check permissions: `ls -la`

### Common Issues
- **No logs**: System might not be running
- **No database**: Run a trade first to create DB
- **Port in use**: Change port in web_monitor.py
- **Permission denied**: Use `chmod +x` on scripts

## 🎉 Summary

You now have multiple ways to monitor your trading system:
1. **Web Dashboard** - Best for visual monitoring
2. **Terminal Monitor** - Best for detailed system info
3. **Log Files** - Best for debugging
4. **Database** - Best for trade history
5. **TMux** - Best for multi-view monitoring

Choose the method that works best for your needs!