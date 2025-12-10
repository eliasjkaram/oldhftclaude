# Alpaca MCP Trading System - Deployment Guide

This directory contains all deployment configurations, scripts, and documentation for the Alpaca MCP Trading System.

## Directory Structure

```
deployment/
├── ci-cd/                      # CI/CD pipeline configurations
│   └── .github/
│       └── workflows/
│           └── deploy.yml      # GitHub Actions workflow
├── monitoring/                 # Monitoring configurations
│   ├── prometheus.yml         # Prometheus configuration
│   ├── alertmanager.yml       # Alert manager configuration
│   └── alerts/                # Alert rules
│       └── trading_alerts.yml
├── scripts/                   # Deployment scripts
│   ├── deploy.sh             # Main deployment script
│   ├── rollback.sh           # Rollback script
│   ├── setup_environment.sh  # Environment setup
│   ├── health_check.sh       # Health check script
│   └── backup.sh             # Backup script
├── docker-compose.prod.yml   # Production overrides
├── docker-compose.staging.yml # Staging overrides
├── .env.production.template  # Production env template
├── .env.staging.template     # Staging env template
├── Makefile                  # Deployment commands
└── PRODUCTION_DEPLOYMENT_CHECKLIST.md
```

## Quick Start

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/alpaca-mcp.git
cd alpaca-mcp

# Run environment setup
sudo ./deployment/scripts/setup_environment.sh

# Copy environment templates
cp deployment/.env.staging.template .env.staging
cp deployment/.env.production.template .env.production

# Edit environment files with your credentials
nano .env.staging  # or .env.production
```

### 2. Deploy to Staging

```bash
# Using Make
make deploy ENVIRONMENT=staging

# Or directly
./deployment/scripts/deploy.sh staging
```

### 3. Deploy to Production

```bash
# Review checklist first
cat deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md

# Deploy
make prod  # Will ask for confirmation
```

## Deployment Commands

Use the Makefile for common deployment tasks:

```bash
# View all available commands
make help

# Build Docker images
make build

# Start services
make up ENVIRONMENT=staging

# View logs
make logs

# Check system health
make health

# Create backup
make backup

# Rollback deployment
make rollback

# View service status
make status

# Access shell
make shell

# Emergency stop
make emergency-stop
```

## Environment Configuration

### Required Environment Variables

Each environment requires a `.env.{environment}` file with the following variables:

- **Alpaca API**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- **Database**: `REDIS_PASSWORD`, `DATABASE_URL`
- **Monitoring**: `GRAFANA_ADMIN_PASSWORD`, `SLACK_WEBHOOK_URL`
- **Trading**: `MAX_POSITION_SIZE`, `DAILY_LOSS_LIMIT`

See `.env.*.template` files for complete list.

### Environments

1. **Development** (local)
   - Uses local services
   - Mock data enabled
   - Debug logging

2. **Staging**
   - Paper trading account
   - Limited resources
   - Testing features enabled

3. **Production**
   - Live trading account
   - Full resources
   - High availability setup
   - Monitoring and alerting

## CI/CD Pipeline

The GitHub Actions workflow (`ci-cd/.github/workflows/deploy.yml`) provides:

1. **Automated Testing**
   - Unit tests
   - Integration tests
   - Security scanning

2. **Build Process**
   - Docker image building
   - Dependency scanning
   - Code quality checks

3. **Deployment**
   - Staging deployment on merge to main
   - Production deployment on merge to production branch
   - Automatic rollback on failure

## Monitoring

### Prometheus Metrics

Access at: `http://your-server:9090`

Key metrics:
- Trading performance
- System resources
- API latency
- Error rates

### Grafana Dashboards

Access at: `http://your-server:3000`

Available dashboards:
- Trading Overview
- System Performance
- Risk Metrics
- ML Model Performance

### Alerts

Configured alerts include:
- Service downtime
- High error rates
- Trading anomalies
- Resource exhaustion

## Backup and Recovery

### Automated Backups

Backups run daily at 2 AM via cron:
```bash
0 2 * * * /opt/alpaca-mcp/deployment/scripts/backup.sh
```

### Manual Backup

```bash
make backup
```

### Recovery

```bash
# List available backups
./deployment/scripts/rollback.sh list

# Rollback to specific backup
./deployment/scripts/rollback.sh backup-20240101-020000
```

## Security

### SSL/TLS

- Production uses Let's Encrypt certificates
- Staging uses self-signed certificates
- All internal communication encrypted

### Secrets Management

- Never commit `.env` files
- Use environment-specific secrets
- Rotate credentials regularly
- Minimal permission principle

### Network Security

- Firewall rules configured
- Service isolation via Docker networks
- API rate limiting enabled

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   make logs
   
   # Check disk space
   df -h
   
   # Check Docker
   docker ps -a
   ```

2. **Database connection failed**
   ```bash
   # Test Redis connection
   make redis-cli
   
   # Check network
   docker network ls
   ```

3. **High memory usage**
   ```bash
   # Check resource usage
   docker stats
   
   # Restart services
   make restart
   ```

### Debug Mode

Enable debug logging:
```bash
# Set in environment file
LOG_LEVEL=DEBUG
DEBUG=true

# Restart services
make restart
```

## Maintenance

### Regular Tasks

- **Daily**: Check health status, review logs
- **Weekly**: Update dependencies, review metrics
- **Monthly**: Rotate logs, update SSL certificates
- **Quarterly**: Security audit, performance review

### Updates

```bash
# Update code
git pull origin main

# Update dependencies
make update-deps

# Rebuild and deploy
make deploy
```

## Support

For issues or questions:

1. Check logs: `make logs`
2. Review documentation
3. Contact DevOps team
4. Emergency: See contacts in checklist

---

**Last Updated**: December 2024
**Version**: 1.0
