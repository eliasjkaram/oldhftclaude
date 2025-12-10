#!/bin/bash
# Production deployment script with safety checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ENVIRONMENT="${1:-staging}"
COMPOSE_PROJECT_NAME="alpaca-mcp"
BACKUP_DIR="/var/backups/alpaca-mcp"
LOG_FILE="/var/log/alpaca-mcp/deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
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

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed!"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed!"
    fi
    
    # Check environment files
    if [[ ! -f "$PROJECT_ROOT/.env.$ENVIRONMENT" ]]; then
        error "Environment file .env.$ENVIRONMENT not found!"
    fi
    
    # Check disk space
    available_space=$(df -BG /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//') || available_space=100
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space! At least 10GB required."
    fi
    
    log "Pre-deployment checks passed!"
}

# Backup current state
backup_current_state() {
    log "Backing up current state..."
    
    mkdir -p "$BACKUP_DIR"
    backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    
    # Backup databases
    if docker-compose ps 2>/dev/null | grep -q "redis"; then
        docker-compose exec -T redis redis-cli BGSAVE || true
        sleep 5
        docker cp "$(docker-compose ps -q redis)":/data/dump.rdb "$BACKUP_DIR/$backup_name-redis.rdb" || true
    fi
    
    # Backup application data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        tar -czf "$BACKUP_DIR/$backup_name-data.tar.gz" -C "$PROJECT_ROOT" data/ || true
    fi
    
    log "Backup completed: $backup_name"
}

# Deploy application
deploy() {
    log "Starting deployment to $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    if [[ -f ".env.$ENVIRONMENT" ]]; then
        set -a
        source ".env.$ENVIRONMENT"
        set +a
    fi
    
    # Build images
    log "Building Docker images..."
    docker-compose build
    
    # Pull external images
    docker-compose pull
    
    # Deploy services
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Performing production deployment..."
        docker-compose -f docker-compose.yml -f deployment/docker-compose.prod.yml up -d
    else
        docker-compose -f docker-compose.yml -f "deployment/docker-compose.$ENVIRONMENT.yml" up -d
    fi
    
    log "Deployment completed!"
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."
    
    # Wait for services to start
    sleep 30
    
    # Check service health
    unhealthy_services=$(docker-compose ps 2>/dev/null | grep -c "unhealthy" || true)
    if [[ $unhealthy_services -gt 0 ]]; then
        warning "$unhealthy_services services are unhealthy!"
    fi
    
    # Check API endpoints
    if curl -f http://localhost:8080/health 2>/dev/null; then
        log "API health check passed!"
    else
        warning "API health check failed!"
    fi
    
    log "Post-deployment validation completed!"
}

# Rollback function
rollback() {
    error "Deployment failed! Starting rollback..."
    
    # Find latest backup
    if [[ -d "$BACKUP_DIR" ]]; then
        latest_backup=$(ls -t "$BACKUP_DIR"/backup-*.tar.gz 2>/dev/null | head -1)
        if [[ -n "$latest_backup" ]]; then
            log "Rolling back to: $latest_backup"
            tar -xzf "$latest_backup" -C "$PROJECT_ROOT"
        fi
    fi
    
    # Restart services
    docker-compose down
    docker-compose up -d
    
    log "Rollback completed!"
}

# Main deployment flow
main() {
    log "Starting deployment process for environment: $ENVIRONMENT"
    
    # Create directories
    mkdir -p "$(dirname "$LOG_FILE")" "$BACKUP_DIR"
    
    # Set trap for rollback on error
    trap rollback ERR
    
    # Run deployment steps
    pre_deployment_checks
    backup_current_state
    deploy
    post_deployment_validation
    
    # Remove trap if successful
    trap - ERR
    
    log "Deployment successful!"
    
    # Send notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Deployment to $ENVIRONMENT completed successfully!\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
}

# Run main function
main "$@"
