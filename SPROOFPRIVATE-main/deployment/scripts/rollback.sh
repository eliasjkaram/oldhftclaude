#!/bin/bash
# Emergency rollback script

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="/var/backups/alpaca-mcp"
LOG_FILE="/var/log/alpaca-mcp/rollback-$(date +%Y%m%d-%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Find backup
find_backup() {
    local backup_id="${1:-}"
    
    if [[ -n "$backup_id" ]]; then
        # Specific backup requested
        backup_file="$BACKUP_DIR/backup-$backup_id-data.tar.gz"
        if [[ ! -f "$backup_file" ]]; then
            error "Backup not found: $backup_file"
        fi
    else
        # Find latest backup
        backup_file=$(ls -t "$BACKUP_DIR"/backup-*-data.tar.gz 2>/dev/null | head -1)
        if [[ -z "$backup_file" ]]; then
            error "No backups found in $BACKUP_DIR"
        fi
    fi
    
    echo "$backup_file"
}

# Perform rollback
perform_rollback() {
    local backup_file="$1"
    local backup_name=$(basename "$backup_file" -data.tar.gz)
    
    log "Starting rollback to: $backup_name"
    
    # Stop current services
    log "Stopping current services..."
    cd "$PROJECT_ROOT"
    docker-compose down || true
    
    # Backup current broken state
    log "Backing up current state..."
    mkdir -p "$BACKUP_DIR/failed"
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        tar -czf "$BACKUP_DIR/failed/failed-$(date +%Y%m%d-%H%M%S).tar.gz" -C "$PROJECT_ROOT" data/
    fi
    
    # Restore data
    log "Restoring data from backup..."
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        rm -rf "$PROJECT_ROOT/data.old"
        mv "$PROJECT_ROOT/data" "$PROJECT_ROOT/data.old"
    fi
    tar -xzf "$backup_file" -C "$PROJECT_ROOT"
    
    # Restore Redis data if available
    redis_backup="$BACKUP_DIR/$backup_name-redis.rdb"
    if [[ -f "$redis_backup" ]]; then
        log "Restoring Redis data..."
        mkdir -p "$PROJECT_ROOT/redis-data"
        cp "$redis_backup" "$PROJECT_ROOT/redis-data/dump.rdb"
    fi
    
    # Restore MinIO data if available
    minio_backup="$BACKUP_DIR/$backup_name-minio.tar.gz"
    if [[ -f "$minio_backup" ]]; then
        log "Restoring MinIO data..."
        docker volume create alpaca-mcp_minio-data || true
        docker run --rm -v alpaca-mcp_minio-data:/data -v "$BACKUP_DIR":/backup \
            alpine tar -xzf "/backup/$backup_name-minio.tar.gz" -C /data
    fi
    
    # Determine environment from backup
    environment="production"
    if [[ "$backup_name" =~ staging ]]; then
        environment="staging"
    fi
    
    # Start services
    log "Starting services..."
    if [[ "$environment" == "production" ]]; then
        docker-compose -f docker-compose.yml -f deployment/docker-compose.prod.yml up -d
    else
        docker-compose -f docker-compose.yml -f deployment/docker-compose.staging.yml up -d
    fi
    
    # Wait for services
    log "Waiting for services to start..."
    sleep 30
    
    # Verify rollback
    log "Verifying rollback..."
    if docker-compose ps | grep -q "Up"; then
        log "Services are running"
    else
        error "Services failed to start after rollback"
    fi
    
    # Check health
    if curl -f http://localhost:8080/health 2>/dev/null; then
        log "Health check passed"
    else
        warning "Health check failed - manual intervention may be required"
    fi
    
    log "Rollback completed successfully!"
}

# List available backups
list_backups() {
    log "Available backups:"
    ls -la "$BACKUP_DIR"/backup-*.tar.gz 2>/dev/null | awk '{print $9, $5, $6, $7, $8}' || echo "No backups found"
}

# Main function
main() {
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case "${1:-}" in
        list)
            list_backups
            ;;
        "")
            # Rollback to latest
            backup_file=$(find_backup)
            perform_rollback "$backup_file"
            ;;
        *)
            # Rollback to specific backup
            backup_file=$(find_backup "$1")
            perform_rollback "$backup_file"
            ;;
    esac
    
    # Send notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Emergency rollback completed!\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
}

# Run main
main "$@"