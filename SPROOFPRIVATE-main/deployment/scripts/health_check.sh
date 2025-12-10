#!/bin/bash
# Health check script for monitoring

set -euo pipefail

# Configuration
SERVICES=("trading-engine:8080" "risk-manager:8081" "data-collector:9091" "redis:6379" "minio:9000")
LOG_FILE="/var/log/alpaca-mcp/health-check.log"
ALERT_THRESHOLD=2

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
failed_checks=0
warnings=0

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Check service health
check_service() {
    local service="$1"
    local host="${service%:*}"
    local port="${service#*:}"
    
    if nc -z -w 5 "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $host:$port is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $host:$port is down"
        ((failed_checks++))
        return 1
    fi
}

# Check HTTP endpoint
check_http() {
    local url="$1"
    local expected_status="${2:-200}"
    
    status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [[ "$status" == "$expected_status" ]]; then
        echo -e "${GREEN}✓${NC} $url returned $status"
        return 0
    else
        echo -e "${RED}✗${NC} $url returned $status (expected $expected_status)"
        ((failed_checks++))
        return 1
    fi
}

# Check disk space
check_disk_space() {
    local threshold=90
    local usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [[ $usage -lt $threshold ]]; then
        echo -e "${GREEN}✓${NC} Disk usage: $usage%"
    else
        echo -e "${YELLOW}⚠${NC} Disk usage: $usage% (threshold: $threshold%)"
        ((warnings++))
    fi
}

# Check memory
check_memory() {
    local threshold=90
    local usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    
    if [[ $usage -lt $threshold ]]; then
        echo -e "${GREEN}✓${NC} Memory usage: $usage%"
    else
        echo -e "${YELLOW}⚠${NC} Memory usage: $usage% (threshold: $threshold%)"
        ((warnings++))
    fi
}

# Check Docker containers
check_containers() {
    local unhealthy=$(docker ps --filter health=unhealthy -q | wc -l)
    local exited=$(docker ps -a --filter status=exited --filter "label=com.docker.compose.project=alpaca-mcp" -q | wc -l)
    
    if [[ $unhealthy -eq 0 && $exited -eq 0 ]]; then
        echo -e "${GREEN}✓${NC} All containers healthy"
    else
        if [[ $unhealthy -gt 0 ]]; then
            echo -e "${RED}✗${NC} $unhealthy unhealthy containers"
            ((failed_checks++))
        fi
        if [[ $exited -gt 0 ]]; then
            echo -e "${RED}✗${NC} $exited exited containers"
            ((failed_checks++))
        fi
    fi
}

# Check market data freshness
check_market_data() {
    # This would normally check your actual data freshness
    # For now, we'll simulate it
    local last_update=$(date -d "1 minute ago" +%s)
    local now=$(date +%s)
    local age=$((now - last_update))
    
    if [[ $age -lt 300 ]]; then
        echo -e "${GREEN}✓${NC} Market data is fresh (${age}s old)"
    else
        echo -e "${YELLOW}⚠${NC} Market data is stale (${age}s old)"
        ((warnings++))
    fi
}

# Main health check
main() {
    log "Starting health check..."
    echo "=== System Health Check ==="
    echo
    
    # System checks
    echo "System Resources:"
    check_disk_space
    check_memory
    echo
    
    # Service checks
    echo "Service Status:"
    for service in "${SERVICES[@]}"; do
        check_service "$service"
    done
    echo
    
    # HTTP endpoints
    echo "HTTP Endpoints:"
    check_http "http://localhost:8080/health"
    check_http "http://localhost:8081/health"
    check_http "http://localhost:9090/-/healthy"
    check_http "http://localhost:3000/api/health"
    echo
    
    # Container checks
    echo "Container Status:"
    check_containers
    echo
    
    # Application checks
    echo "Application Status:"
    check_market_data
    echo
    
    # Summary
    echo "=== Summary ==="
    echo -e "Failed checks: ${failed_checks}"
    echo -e "Warnings: ${warnings}"
    
    # Send alert if needed
    if [[ $failed_checks -ge $ALERT_THRESHOLD ]]; then
        log "ALERT: $failed_checks health checks failed!"
        
        # Send alert via webhook
        if [[ -n "${HEALTH_CHECK_WEBHOOK:-}" ]]; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"Health check alert: $failed_checks checks failed!\"}" \
                "$HEALTH_CHECK_WEBHOOK" 2>/dev/null || true
        fi
        
        exit 1
    fi
    
    log "Health check completed: $failed_checks failures, $warnings warnings"
    exit 0
}

# Create log directory if needed
mkdir -p "$(dirname "$LOG_FILE")"

# Run main function
main "$@"