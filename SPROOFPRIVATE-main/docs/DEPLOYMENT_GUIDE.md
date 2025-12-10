# AI Trading System - Complete Deployment Guide
## Production-Ready Docker Deployment

### 📋 Overview

This guide provides complete instructions for deploying the AI Trading System in production using Docker containers with full monitoring, database persistence, and security features.

---

## 🚀 Quick Deployment (5 Minutes)

### Prerequisites
- Docker 20.0+ and Docker Compose 2.0+
- 8GB+ RAM, 4+ CPU cores
- 50GB+ disk space
- Internet connection

### 1. Clone and Prepare
```bash
# Navigate to project directory
cd /home/harry/alpaca-mcp/

# Create necessary directories
mkdir -p logs data config monitoring nginx

# Set permissions
chmod 755 logs data config
```

### 2. Quick Start
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ai-trading-system
```

### 3. Access Services
- **Main Trading System**: http://localhost:8000
- **Monitoring Dashboard**: http://localhost:3000 (admin/trading_admin_2024)
- **Metrics**: http://localhost:9090
- **Database**: localhost:5432

---

## 🏗️ Architecture Overview

### Container Services

#### Core Trading Services
1. **ai-trading-system** - Main AI trading engine
2. **data-engine** - Historical data collection and processing
3. **arbitrage-agent** - Multi-LLM arbitrage discovery
4. **optimizer** - Portfolio optimization service

#### Infrastructure Services
5. **postgres** - Primary database for historical data
6. **redis** - High-speed cache and message broker
7. **prometheus** - Metrics collection
8. **grafana** - Monitoring dashboards
9. **nginx** - Reverse proxy and load balancer

### Network Architecture
```
Internet
    ↓
[Nginx Proxy]
    ↓
[Trading Services] ←→ [Redis Cache]
    ↓
[PostgreSQL DB]
    ↓
[Monitoring Stack]
```

---

## 🔧 Detailed Configuration

### 1. Environment Configuration

Create `.env` file:
```bash
# Trading Configuration
TRADING_MODE=paper
LOG_LEVEL=INFO
MAX_POSITION_SIZE=100000
RISK_LIMIT=0.02

# API Keys
ALPACA_PAPER_API_KEY=PKCX98VZSJBQF79C1SD8
ALPACA_PAPER_API_SECRET=KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt
OPENROUTER_API_KEY=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

# Database
POSTGRES_PASSWORD=secure_trading_password_2024
GRAFANA_PASSWORD=trading_admin_2024

# Security
JWT_SECRET=your_jwt_secret_here_change_me
ENCRYPTION_KEY=your_encryption_key_here_change_me
```

### 2. Database Initialization

Create `init.sql`:
```sql
-- Trading Database Schema
CREATE DATABASE IF NOT EXISTS trading_db;
USE trading_db;

-- Historical Data Tables
CREATE TABLE IF NOT EXISTS historical_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Opportunities Table
CREATE TABLE IF NOT EXISTS opportunities (
    id SERIAL PRIMARY KEY,
    opportunity_id VARCHAR(100) UNIQUE NOT NULL,
    arbitrage_type VARCHAR(50),
    underlying_assets TEXT[],
    expected_profit DECIMAL(12,2),
    confidence_score DECIMAL(4,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'discovered'
);

-- Portfolio Positions
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(12,4),
    avg_price DECIMAL(10,4),
    current_value DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading Performance
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    opportunity_id VARCHAR(100),
    symbol VARCHAR(10),
    side VARCHAR(10),
    quantity DECIMAL(12,4),
    price DECIMAL(10,4),
    executed_at TIMESTAMP,
    profit_loss DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50),
    metric_value DECIMAL(12,4),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_historical_prices_symbol_date ON historical_prices(symbol, date);
CREATE INDEX idx_opportunities_created_at ON opportunities(created_at);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_positions_symbol ON positions(symbol);
```

### 3. Monitoring Configuration

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'ai-trading-system'
    static_configs:
      - targets: ['ai-trading-system:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'data-engine'
    static_configs:
      - targets: ['data-engine:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'arbitrage-agent'
    static_configs:
      - targets: ['arbitrage-agent:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 4. Nginx Configuration

Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream trading_backend {
        server ai-trading-system:8000;
    }

    upstream monitoring_backend {
        server grafana:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # Main trading system
        location / {
            proxy_pass http://trading_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws {
            proxy_pass http://trading_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # Monitoring dashboard
        location /monitoring/ {
            proxy_pass http://monitoring_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Health checks
        location /health {
            access_log off;
            return 200 "healthy\n";
        }
    }
}
```

---

## 🔄 Deployment Commands

### Build and Deploy
```bash
# Build all images
docker-compose build

# Start services (detached)
docker-compose up -d

# Start specific service
docker-compose up -d ai-trading-system

# Scale services
docker-compose up -d --scale arbitrage-agent=3
```

### Management Commands
```bash
# View service status
docker-compose ps

# View logs
docker-compose logs -f ai-trading-system
docker-compose logs --tail=100 data-engine

# Restart services
docker-compose restart ai-trading-system
docker-compose restart

# Stop services
docker-compose stop
docker-compose down

# Remove everything (CAUTION!)
docker-compose down -v --remove-orphans
```

### Update and Maintenance
```bash
# Update code and rebuild
git pull
docker-compose build --no-cache
docker-compose up -d

# Database backup
docker exec trading-postgres pg_dump -U trader trading_db > backup_$(date +%Y%m%d).sql

# View resource usage
docker stats

# Clean up unused resources
docker system prune -a
```

---

## 📊 Monitoring & Observability

### 1. Grafana Dashboards
Access: http://localhost:3000
- Username: admin
- Password: trading_admin_2024

**Key Dashboards:**
- Trading System Performance
- Opportunity Discovery Metrics
- Portfolio Performance
- System Resource Usage
- API Response Times

### 2. Prometheus Metrics
Access: http://localhost:9090

**Key Metrics:**
- `trading_opportunities_discovered_total`
- `trading_profit_generated_total`
- `api_request_duration_seconds`
- `system_memory_usage_bytes`
- `database_connections_active`

### 3. Application Logs
```bash
# Real-time logs
docker-compose logs -f ai-trading-system

# Error logs only
docker-compose logs ai-trading-system | grep ERROR

# Save logs to file
docker-compose logs ai-trading-system > trading_logs.txt
```

### 4. Health Checks
```bash
# Check service health
curl http://localhost:8000/health

# Database health
docker exec trading-postgres pg_isready -U trader

# Redis health
docker exec trading-redis redis-cli ping
```

---

## 🛡️ Security Configuration

### 1. Network Security
```bash
# Create isolated network
docker network create trading-secure-network

# Update docker-compose.yml to use secure network
networks:
  trading-secure-network:
    external: true
```

### 2. SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/trading.key \
  -out nginx/ssl/trading.crt

# Update nginx for HTTPS
# (Add SSL configuration to nginx.conf)
```

### 3. API Key Security
```bash
# Use Docker secrets instead of environment variables
echo "ALPACA_PAPER_API_KEY" | docker secret create alpaca_key -
echo "OPENROUTER_API_KEY" | docker secret create openrouter_key -

# Update compose file to use secrets
secrets:
  alpaca_key:
    external: true
  openrouter_key:
    external: true
```

### 4. Database Security
```bash
# Strong passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Database encryption
# (Enable in PostgreSQL config)

# Connection limits
# (Configure in PostgreSQL)
```

---

## ⚡ Performance Optimization

### 1. Resource Allocation
```yaml
# In docker-compose.yml
services:
  ai-trading-system:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 2. Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
SELECT pg_reload_conf();
```

### 3. Redis Optimization
```bash
# Redis configuration
echo "maxmemory 2gb" >> redis.conf
echo "maxmemory-policy allkeys-lru" >> redis.conf
```

### 4. Application Tuning
```python
# Environment variables for performance
WORKERS=4
MAX_CONNECTIONS=100
POOL_SIZE=20
CACHE_TTL=300
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker-compose logs ai-trading-system

# Check resource usage
docker stats

# Restart service
docker-compose restart ai-trading-system
```

#### 2. Database Connection Issues
```bash
# Test database connection
docker exec -it trading-postgres psql -U trader -d trading_db

# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down
docker volume rm alpaca-mcp_postgres_data
docker-compose up -d postgres
```

#### 3. API Connection Errors
```bash
# Test API connectivity
docker exec -it ai-trading-system python -c "
import yfinance as yf
print(yf.Ticker('AAPL').history(period='1d'))
"

# Check API key configuration
docker exec -it ai-trading-system env | grep API
```

#### 4. Memory Issues
```bash
# Check memory usage
docker stats --no-stream

# Increase memory limits
# (Update docker-compose.yml)

# Clear caches
docker exec trading-redis redis-cli FLUSHALL
```

### Performance Monitoring
```bash
# System resources
htop
iostat -x 1
free -h

# Container resources
docker stats

# Network traffic
nethogs

# Disk usage
df -h
du -sh /var/lib/docker/
```

---

## 📈 Scaling Instructions

### 1. Horizontal Scaling
```bash
# Scale specific services
docker-compose up -d --scale arbitrage-agent=3
docker-compose up -d --scale data-engine=2

# Load balancer configuration
# (Update nginx.conf for multiple backends)
```

### 2. Vertical Scaling
```yaml
# Increase resource limits
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

### 3. Database Scaling
```bash
# Read replicas
# (Configure PostgreSQL streaming replication)

# Sharding
# (Implement application-level sharding)
```

### 4. Cluster Deployment
```bash
# Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml trading-stack

# Kubernetes
kubectl apply -f kubernetes/
```

---

## 🚀 Production Checklist

### Pre-Deployment
- [ ] Review and update all API keys
- [ ] Configure SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Test all services individually
- [ ] Backup strategy implemented
- [ ] Security review completed
- [ ] Load testing performed

### Post-Deployment
- [ ] Verify all services are running
- [ ] Check monitoring dashboards
- [ ] Test API endpoints
- [ ] Verify database connectivity
- [ ] Monitor logs for errors
- [ ] Set up automated backups
- [ ] Configure alerts

### Ongoing Maintenance
- [ ] Regular security updates
- [ ] Database maintenance
- [ ] Log rotation
- [ ] Performance monitoring
- [ ] Backup verification
- [ ] Capacity planning

---

## 📞 Support & Maintenance

### Log Locations
- Application logs: `/app/logs/`
- Database logs: `/var/lib/postgresql/data/log/`
- System logs: `docker-compose logs`

### Key Configuration Files
- `docker-compose.yml` - Service orchestration
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `init.sql` - Database schema
- `prometheus.yml` - Monitoring configuration

### Emergency Procedures
```bash
# Emergency stop
docker-compose down

# Emergency database backup
docker exec trading-postgres pg_dumpall -U trader > emergency_backup.sql

# Service recovery
docker-compose up -d --force-recreate
```

---

**🚀 Your AI Trading System is now ready for production deployment with Docker!**

Use `docker-compose up -d` to start all services and access the system at http://localhost:8000