#!/bin/bash
# Automated backup script

set -euo pipefail

# Configuration
BACKUP_ROOT="/var/backups/alpaca-mcp"
BACKUP_RETENTION_DAYS=30
PROJECT_ROOT="/opt/alpaca-mcp"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="backup-$TIMESTAMP"
LOG_FILE="/var/log/alpaca-mcp/backup-$TIMESTAMP.log"

# S3 Configuration (optional)
USE_S3_BACKUP="${USE_S3_BACKUP:-false}"
S3_BUCKET="${S3_BACKUP_BUCKET:-}"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[ERROR] $*" | tee -a "$LOG_FILE"
    exit 1
}

# Create backup directory
create_backup_dir() {
    local backup_dir="$BACKUP_ROOT/$BACKUP_NAME"
    mkdir -p "$backup_dir"
    echo "$backup_dir"
}

# Backup databases
backup_databases() {
    local backup_dir="$1"
    log "Backing up databases..."
    
    # Redis backup
    if docker-compose ps 2>/dev/null | grep -q redis; then
        log "Backing up Redis..."
        docker-compose exec -T redis redis-cli BGSAVE || true
        sleep 5
        docker cp "$(docker-compose ps -q redis)":/data/dump.rdb "$backup_dir/redis.rdb" || true
    fi
    
    # SQLite databases
    find "$PROJECT_ROOT" -name "*.db" -type f 2>/dev/null | while read -r db; do
        db_name=$(basename "$db")
        log "Backing up SQLite: $db_name"
        cp "$db" "$backup_dir/$db_name"
    done
}

# Backup application data
backup_application_data() {
    local backup_dir="$1"
    log "Backing up application data..."
    
    # Backup data directory
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        tar -czf "$backup_dir/data.tar.gz" -C "$PROJECT_ROOT" data/
    fi
    
    # Backup models
    if [[ -d "$PROJECT_ROOT/models" ]]; then
        tar -czf "$backup_dir/models.tar.gz" -C "$PROJECT_ROOT" models/
    fi
    
    # Backup logs
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        tar -czf "$backup_dir/logs.tar.gz" -C "$PROJECT_ROOT" logs/
    fi
    
    # Backup configuration (without secrets)
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        tar -czf "$backup_dir/config.tar.gz" -C "$PROJECT_ROOT" \
            --exclude="*.env" --exclude="*secret*" config/
    fi
}

# Backup Docker volumes
backup_docker_volumes() {
    local backup_dir="$1"
    log "Backing up Docker volumes..."
    
    # MinIO data
    if docker volume ls | grep -q minio-data; then
        log "Backing up MinIO volume..."
        docker run --rm -v alpaca-mcp_minio-data:/data -v "$backup_dir":/backup \
            alpine tar -czf "/backup/minio-data.tar.gz" -C /data .
    fi
}

# Create backup metadata
create_metadata() {
    local backup_dir="$1"
    
    cat > "$backup_dir/metadata.json" <<EOF
{
    "timestamp": "$TIMESTAMP",
    "hostname": "$(hostname)",
    "environment": "${ENVIRONMENT:-production}",
    "docker_compose_version": "$(docker-compose version --short 2>/dev/null || echo 'unknown')",
    "git_commit": "$(cd $PROJECT_ROOT && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "backup_size": "$(du -sh $backup_dir | cut -f1)"
}
EOF
}

# Upload to S3
upload_to_s3() {
    local backup_dir="$1"
    
    if [[ "$USE_S3_BACKUP" == "true" ]] && [[ -n "$S3_BUCKET" ]]; then
        log "Uploading backup to S3..."
        
        # Create tarball
        local tarball="$BACKUP_ROOT/$BACKUP_NAME.tar.gz"
        tar -czf "$tarball" -C "$BACKUP_ROOT" "$BACKUP_NAME"
        
        # Upload to S3
        aws s3 cp "$tarball" "s3://$S3_BUCKET/backups/" \
            --storage-class STANDARD_IA || true
        
        # Cleanup local tarball
        rm -f "$tarball"
        
        log "Backup uploaded to S3: s3://$S3_BUCKET/backups/$BACKUP_NAME.tar.gz"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Local cleanup
    find "$BACKUP_ROOT" -name "backup-*" -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
}

# Verify backup
verify_backup() {
    local backup_dir="$1"
    log "Verifying backup..."
    
    # Check if essential files exist
    if [[ ! -f "$backup_dir/metadata.json" ]]; then
        error "Backup verification failed: metadata.json missing"
    fi
    
    log "Backup verified successfully"
}

# Main backup function
main() {
    mkdir -p "$(dirname "$LOG_FILE")" "$BACKUP_ROOT"
    log "Starting backup process..."
    
    # Create backup directory
    backup_dir=$(create_backup_dir)
    log "Backup directory: $backup_dir"
    
    # Perform backups
    backup_databases "$backup_dir"
    backup_application_data "$backup_dir"
    backup_docker_volumes "$backup_dir"
    create_metadata "$backup_dir"
    
    # Verify backup
    verify_backup "$backup_dir"
    
    # Upload to S3 if configured
    upload_to_s3 "$backup_dir"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Send notification
    if [[ -n "${BACKUP_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Backup completed successfully: $BACKUP_NAME\"}" \
            "$BACKUP_WEBHOOK_URL" 2>/dev/null || true
    fi
    
    log "Backup completed successfully!"
}

# Run main function
main "$@"
