#!/bin/bash

# Maintenance and lifecycle automation script for Self-Healing Pipeline Guard
# Handles routine maintenance tasks, health checks, and system optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/healing-guard"
BACKUP_DIR="/var/backups/healing-guard"
RETENTION_DAYS=30
MAX_LOG_SIZE="100M"

# Functions
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up maintenance directories..."
    
    sudo mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    sudo mkdir -p "$LOG_DIR/maintenance" "$LOG_DIR/health-checks"
    sudo mkdir -p "$BACKUP_DIR/database" "$BACKUP_DIR/config"
    
    # Set proper permissions
    sudo chown -R app:app "$LOG_DIR" "$BACKUP_DIR" 2>/dev/null || true
    
    log_success "Directories created successfully"
}

# Health check function
health_check() {
    log_info "Performing comprehensive health check..."
    
    local health_status=0
    local health_report="$LOG_DIR/health-checks/health-$(date +%Y%m%d_%H%M%S).json"
    
    # Initialize health report
    cat > "$health_report" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "overall_status": "checking",
    "checks": {}
}
EOF
    
    # Check API health
    log_info "Checking API health..."
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
        jq '.checks.api = {"status": "healthy", "response_time": 0}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_error "API health check failed"
        health_status=1
        jq '.checks.api = {"status": "unhealthy", "error": "API not responding"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if docker exec healing-guard-postgres-1 pg_isready -U healing_user > /dev/null 2>&1; then
        log_success "Database connectivity check passed"
        jq '.checks.database = {"status": "healthy"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_error "Database connectivity check failed"
        health_status=1
        jq '.checks.database = {"status": "unhealthy", "error": "Database not responding"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Check Redis connectivity
    log_info "Checking Redis connectivity..."
    if docker exec healing-guard-redis-1 redis-cli ping > /dev/null 2>&1; then
        log_success "Redis connectivity check passed"
        jq '.checks.redis = {"status": "healthy"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_error "Redis connectivity check failed"
        health_status=1
        jq '.checks.redis = {"status": "unhealthy", "error": "Redis not responding"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Check disk space
    log_info "Checking disk space..."
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 85 ]; then
        log_success "Disk space check passed ($disk_usage% used)"
        jq --arg usage "$disk_usage" '.checks.disk_space = {"status": "healthy", "usage_percent": ($usage | tonumber)}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_warning "Disk space usage is high ($disk_usage% used)"
        jq --arg usage "$disk_usage" '.checks.disk_space = {"status": "warning", "usage_percent": ($usage | tonumber), "message": "High disk usage"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Check memory usage
    log_info "Checking memory usage..."
    local memory_usage=$(free | awk 'NR==2{printf "%.2f", $3/$2*100}')
    local memory_int=${memory_usage%.*}
    if [ "$memory_int" -lt 85 ]; then
        log_success "Memory usage check passed ($memory_usage% used)"
        jq --arg usage "$memory_usage" '.checks.memory = {"status": "healthy", "usage_percent": ($usage | tonumber)}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_warning "Memory usage is high ($memory_usage% used)"
        jq --arg usage "$memory_usage" '.checks.memory = {"status": "warning", "usage_percent": ($usage | tonumber), "message": "High memory usage"}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Check container status
    log_info "Checking container status..."
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
    if [ "$unhealthy_containers" -eq 0 ]; then
        log_success "All containers are healthy"
        jq '.checks.containers = {"status": "healthy", "unhealthy_count": 0}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    else
        log_error "$unhealthy_containers unhealthy containers found"
        health_status=1
        jq --arg count "$unhealthy_containers" '.checks.containers = {"status": "unhealthy", "unhealthy_count": ($count | tonumber)}' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
    fi
    
    # Update overall status
    if [ $health_status -eq 0 ]; then
        jq '.overall_status = "healthy"' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
        log_success "Overall health check passed"
    else
        jq '.overall_status = "unhealthy"' "$health_report" > tmp.$$ && mv tmp.$$ "$health_report"
        log_error "Health check failed - issues detected"
    fi
    
    return $health_status
}

# Database maintenance
database_maintenance() {
    log_info "Performing database maintenance..."
    
    # Vacuum and analyze database
    log_info "Running database vacuum and analyze..."
    docker exec healing-guard-postgres-1 psql -U healing_user -d healing_guard -c "VACUUM ANALYZE;" || log_warning "Database vacuum failed"
    
    # Check database size
    local db_size=$(docker exec healing-guard-postgres-1 psql -U healing_user -d healing_guard -t -c "SELECT pg_size_pretty(pg_database_size('healing_guard'));" | xargs)
    log_info "Database size: $db_size"
    
    # Check for long-running queries
    log_info "Checking for long-running queries..."
    docker exec healing-guard-postgres-1 psql -U healing_user -d healing_guard -c "
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
    FROM pg_stat_activity 
    WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes' 
    AND state = 'active';
    " || log_warning "Could not check for long-running queries"
    
    # Update table statistics
    log_info "Updating table statistics..."
    docker exec healing-guard-postgres-1 psql -U healing_user -d healing_guard -c "ANALYZE;" || log_warning "Table statistics update failed"
    
    log_success "Database maintenance completed"
}

# Log cleanup and rotation
log_cleanup() {
    log_info "Performing log cleanup..."
    
    # Find and remove old log files
    local removed_files=0
    
    # Clean application logs older than retention period
    if [ -d "$LOG_DIR" ]; then
        removed_files=$(find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS | wc -l)
        find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS -delete
        log_info "Removed $removed_files old log files"
    fi
    
    # Clean Docker logs
    log_info "Cleaning Docker logs..."
    docker system prune -f --filter "until=24h" > /dev/null 2>&1 || log_warning "Docker cleanup failed"
    
    # Rotate large log files
    log_info "Checking for large log files..."
    find "$LOG_DIR" -name "*.log" -type f -size +$MAX_LOG_SIZE -exec logrotate -f {} \; 2>/dev/null || true
    
    # Clean temporary files
    log_info "Cleaning temporary files..."
    find /tmp -name "healing-guard-*" -type f -mtime +1 -delete 2>/dev/null || true
    
    log_success "Log cleanup completed"
}

# Performance optimization
performance_optimization() {
    log_info "Performing performance optimization..."
    
    # Clean Redis memory
    log_info "Optimizing Redis memory usage..."
    docker exec healing-guard-redis-1 redis-cli MEMORY PURGE > /dev/null 2>&1 || log_warning "Redis memory purge failed"
    
    # Check Redis memory usage
    local redis_memory=$(docker exec healing-guard-redis-1 redis-cli INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
    log_info "Redis memory usage: $redis_memory"
    
    # Optimize database connections
    log_info "Checking database connection pool..."
    local active_connections=$(docker exec healing-guard-postgres-1 psql -U healing_user -d healing_guard -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" | xargs)
    log_info "Active database connections: $active_connections"
    
    # Clear application caches
    log_info "Clearing application caches..."
    curl -X POST http://localhost:8000/api/v1/admin/clear-cache > /dev/null 2>&1 || log_warning "Cache clear failed"
    
    log_success "Performance optimization completed"
}

# Security audit
security_audit() {
    log_info "Performing security audit..."
    
    # Check for failed login attempts
    log_info "Checking for security events..."
    local failed_logins=$(grep -c "401\|403" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum}')
    if [ "$failed_logins" -gt 100 ]; then
        log_warning "High number of failed authentication attempts: $failed_logins"
    else
        log_info "Authentication attempts within normal range: $failed_logins"
    fi
    
    # Check SSL certificate expiration
    log_info "Checking SSL certificate expiration..."
    local cert_file="/etc/nginx/ssl/healing-guard.crt"
    if [ -f "$cert_file" ]; then
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        if [ $days_until_expiry -lt 30 ]; then
            log_warning "SSL certificate expires in $days_until_expiry days"
        else
            log_info "SSL certificate expires in $days_until_expiry days"
        fi
    fi
    
    # Check for suspicious network activity
    log_info "Checking network activity..."
    local unusual_ips=$(netstat -tn | awk '{print $5}' | grep -v '127.0.0.1\|0.0.0.0' | sort | uniq -c | sort -nr | head -5)
    echo "Top network connections:"
    echo "$unusual_ips"
    
    log_success "Security audit completed"
}

# Backup operations
backup_operations() {
    log_info "Performing backup operations..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Database backup
    log_info "Creating database backup..."
    docker exec healing-guard-postgres-1 pg_dump -U healing_user healing_guard | gzip > "$BACKUP_DIR/database/healing_guard_$backup_timestamp.sql.gz"
    
    # Configuration backup
    log_info "Creating configuration backup..."
    tar -czf "$BACKUP_DIR/config/config_$backup_timestamp.tar.gz" \
        "$PROJECT_ROOT/config" \
        "$PROJECT_ROOT/.env" \
        "$PROJECT_ROOT/docker-compose.yml" \
        2>/dev/null || log_warning "Some configuration files missing"
    
    # Remove old backups
    log_info "Cleaning old backups..."
    find "$BACKUP_DIR" -name "*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
    
    log_success "Backup operations completed"
}

# System updates check
system_updates_check() {
    log_info "Checking for system updates..."
    
    # Check for package updates
    if command -v apt &> /dev/null; then
        local updates=$(apt list --upgradable 2>/dev/null | wc -l)
        if [ "$updates" -gt 1 ]; then
            log_warning "$((updates-1)) package updates available"
        else
            log_info "System packages are up to date"
        fi
    fi
    
    # Check for Docker image updates
    log_info "Checking for Docker image updates..."
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | grep healing-guard
    
    # Check for Python dependency updates
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        log_info "Checking for Python dependency updates..."
        cd "$PROJECT_ROOT"
        poetry show --outdated > /tmp/outdated_deps.txt 2>/dev/null || true
        local outdated_count=$(wc -l < /tmp/outdated_deps.txt)
        if [ "$outdated_count" -gt 0 ]; then
            log_warning "$outdated_count Python dependencies are outdated"
            head -5 /tmp/outdated_deps.txt
        else
            log_info "Python dependencies are up to date"
        fi
        rm -f /tmp/outdated_deps.txt
    fi
    
    log_success "System updates check completed"
}

# Generate maintenance report
generate_report() {
    log_info "Generating maintenance report..."
    
    local report_file="$LOG_DIR/maintenance/maintenance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Maintenance Report - $(date)

## Summary
- Maintenance started: $(date)
- Duration: $SECONDS seconds
- Status: $1

## Health Check Results
$(cat "$LOG_DIR/health-checks/health-"*.json | tail -1 | jq -r '
"- Overall Status: " + .overall_status + "
- API: " + .checks.api.status + "
- Database: " + .checks.database.status + "
- Redis: " + .checks.redis.status + "
- Disk Usage: " + (.checks.disk_space.usage_percent | tostring) + "%
- Memory Usage: " + (.checks.memory.usage_percent | tostring) + "%"')

## System Information
- Server Time: $(date)
- Uptime: $(uptime)
- Load Average: $(uptime | awk -F'load average:' '{print $2}')
- Disk Usage: $(df -h / | awk 'NR==2 {print $5}') used
- Memory Usage: $(free -h | awk 'NR==2{printf "%.2f%%", $3/$2*100}')

## Docker Containers
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

## Recent Logs (Last 10 lines)
\`\`\`
$(tail -10 "$LOG_DIR"/*.log 2>/dev/null | tail -10)
\`\`\`

## Action Items
$([ -f /tmp/maintenance_actions.txt ] && cat /tmp/maintenance_actions.txt || echo "- No action items")

---
Report generated by automated maintenance script
EOF
    
    log_success "Maintenance report generated: $report_file"
}

# Send notifications
send_notifications() {
    local status=$1
    
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color="good"
        local message="🔧 Routine maintenance completed successfully"
        
        if [ "$status" != "success" ]; then
            color="warning"
            message="⚠️ Maintenance completed with warnings - please review logs"
        fi
        
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Healing Guard Maintenance Report\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Server\", \"value\": \"$(hostname)\", \"short\": true},
                        {\"title\": \"Duration\", \"value\": \"${SECONDS}s\", \"short\": true}
                    ]
                }]
            }" > /dev/null 2>&1 || log_warning "Failed to send Slack notification"
    fi
}

# Main maintenance function
run_maintenance() {
    local maintenance_type=${1:-"full"}
    local start_time=$(date +%s)
    local status="success"
    
    log_info "Starting $maintenance_type maintenance..."
    
    case $maintenance_type in
        "health")
            health_check || status="warning"
            ;;
        "database")
            database_maintenance
            ;;
        "cleanup")
            log_cleanup
            performance_optimization
            ;;
        "security")
            security_audit
            ;;
        "backup")
            backup_operations
            ;;
        "update-check")
            system_updates_check
            ;;
        "full"|*)
            setup_directories
            health_check || status="warning"
            database_maintenance
            log_cleanup
            performance_optimization
            security_audit
            backup_operations
            system_updates_check
            ;;
    esac
    
    local end_time=$(date +%s)
    SECONDS=$((end_time - start_time))
    
    generate_report "$status"
    send_notifications "$status"
    
    if [ "$status" = "success" ]; then
        log_success "Maintenance completed successfully in ${SECONDS} seconds"
    else
        log_warning "Maintenance completed with warnings in ${SECONDS} seconds"
    fi
    
    return 0
}

# Main execution
main() {
    case ${1:-"full"} in
        "health"|"database"|"cleanup"|"security"|"backup"|"update-check"|"full")
            run_maintenance "$1"
            ;;
        "--help"|"-h")
            cat << EOF
Usage: $0 [MAINTENANCE_TYPE]

Maintenance Types:
    full         Run all maintenance tasks (default)
    health       Health check only
    database     Database maintenance only
    cleanup      Log cleanup and performance optimization
    security     Security audit only
    backup       Backup operations only
    update-check Check for system updates

Environment Variables:
    SLACK_WEBHOOK_URL    Slack webhook for notifications
    RETENTION_DAYS       Log retention period (default: 30)
    MAX_LOG_SIZE         Maximum log file size (default: 100M)

Examples:
    $0                # Run full maintenance
    $0 health         # Health check only
    $0 cleanup        # Cleanup and optimization only
EOF
            ;;
        *)
            log_error "Unknown maintenance type: $1"
            log_info "Use --help to see available options"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"