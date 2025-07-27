#!/bin/bash

# Security hardening script for Self-Healing Pipeline Guard
# This script applies security best practices to the deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Generate secure secrets
generate_secrets() {
    log "Generating secure secrets..."
    
    # Create secrets directory
    mkdir -p secrets
    chmod 700 secrets
    
    # Generate random secrets if they don't exist
    if [ ! -f secrets/secret_key ]; then
        openssl rand -hex 32 > secrets/secret_key
        log_success "Generated SECRET_KEY"
    fi
    
    if [ ! -f secrets/jwt_secret ]; then
        openssl rand -hex 32 > secrets/jwt_secret
        log_success "Generated JWT_SECRET_KEY"
    fi
    
    if [ ! -f secrets/database_password ]; then
        openssl rand -base64 32 > secrets/database_password
        log_success "Generated database password"
    fi
    
    # Set proper permissions
    chmod 600 secrets/*
    log_success "Set secure permissions on secret files"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    mkdir -p ssl
    chmod 700 ssl
    
    if [ ! -f ssl/healing-guard.crt ]; then
        # Generate private key
        openssl genrsa -out ssl/healing-guard.key 2048
        
        # Generate certificate signing request
        openssl req -new -key ssl/healing-guard.key -out ssl/healing-guard.csr \
            -subj "/C=US/ST=CA/L=San Francisco/O=Terragon Labs/OU=Engineering/CN=healing-guard.local"
        
        # Generate self-signed certificate
        openssl x509 -req -days 365 -in ssl/healing-guard.csr \
            -signkey ssl/healing-guard.key -out ssl/healing-guard.crt \
            -extensions v3_req -extfile <(cat <<EOF
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = healing-guard.local
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
)
        
        # Clean up CSR
        rm ssl/healing-guard.csr
        
        # Set proper permissions
        chmod 600 ssl/healing-guard.key
        chmod 644 ssl/healing-guard.crt
        
        log_success "Generated SSL certificates"
    else
        log "SSL certificates already exist"
    fi
}

# Harden Docker configuration
harden_docker() {
    log "Applying Docker security hardening..."
    
    # Create docker daemon configuration
    sudo mkdir -p /etc/docker
    
    # Docker daemon security configuration
    cat << EOF | sudo tee /etc/docker/daemon.json > /dev/null
{
    "icc": false,
    "userland-proxy": false,
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp.json",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "storage-driver": "overlay2"
}
EOF
    
    log_success "Applied Docker daemon security configuration"
}

# Configure firewall rules
configure_firewall() {
    log "Configuring firewall rules..."
    
    # Enable UFW firewall
    sudo ufw --force enable
    
    # Default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH (adjust port as needed)
    sudo ufw allow 22/tcp
    
    # Allow HTTP and HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow application ports (restrict to specific IPs in production)
    sudo ufw allow from 10.0.0.0/8 to any port 8000
    sudo ufw allow from 172.16.0.0/12 to any port 8000
    sudo ufw allow from 192.168.0.0/16 to any port 8000
    
    # Database ports (only from app network)
    sudo ufw allow from 172.20.0.0/16 to any port 5432
    sudo ufw allow from 172.20.0.0/16 to any port 6379
    
    # Monitoring ports
    sudo ufw allow from 10.0.0.0/8 to any port 9090
    sudo ufw allow from 10.0.0.0/8 to any port 3000
    
    log_success "Configured firewall rules"
}

# Set up log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat << EOF | sudo tee /etc/logrotate.d/healing-guard > /dev/null
/var/log/healing-guard/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 app app
    postrotate
        docker kill -s USR1 \$(docker ps -q --filter "label=app=healing-guard") 2>/dev/null || true
    endscript
}
EOF
    
    log_success "Configured log rotation"
}

# Harden system settings
harden_system() {
    log "Applying system hardening..."
    
    # Kernel parameters for security
    cat << EOF | sudo tee /etc/sysctl.d/99-healing-guard-security.conf > /dev/null
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# TCP SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Memory protection
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1

# File system protections
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
EOF
    
    # Apply sysctl settings
    sudo sysctl -p /etc/sysctl.d/99-healing-guard-security.conf
    
    log_success "Applied system hardening"
}

# Configure fail2ban
setup_fail2ban() {
    log "Setting up fail2ban..."
    
    # Install fail2ban if not present
    if ! command -v fail2ban-client &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y fail2ban
    fi
    
    # Create custom jail for healing guard
    cat << EOF | sudo tee /etc/fail2ban/jail.d/healing-guard.conf > /dev/null
[healing-guard-auth]
enabled = true
port = http,https
filter = healing-guard-auth
logpath = /var/log/nginx/access.log
maxretry = 5
bantime = 3600
findtime = 600

[nginx-limit-req]
enabled = true
port = http,https
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600
findtime = 600
EOF
    
    # Create filter for authentication failures
    cat << EOF | sudo tee /etc/fail2ban/filter.d/healing-guard-auth.conf > /dev/null
[Definition]
failregex = ^<HOST> -.*"(GET|POST|PUT|DELETE|PATCH) .* HTTP/.*" 401 .*$
            ^<HOST> -.*"(GET|POST|PUT|DELETE|PATCH) .* HTTP/.*" 403 .*$
ignoreregex =
EOF
    
    # Restart fail2ban
    sudo systemctl restart fail2ban
    sudo systemctl enable fail2ban
    
    log_success "Configured fail2ban"
}

# Set up monitoring for security events
setup_security_monitoring() {
    log "Setting up security monitoring..."
    
    # Create security log directory
    sudo mkdir -p /var/log/healing-guard/security
    sudo chown app:app /var/log/healing-guard/security
    
    # Create script to monitor security events
    cat << 'EOF' > scripts/security-monitor.sh
#!/bin/bash

# Security monitoring script
LOG_FILE="/var/log/healing-guard/security/security-events.log"

# Function to log security events
log_security_event() {
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - $1" >> "$LOG_FILE"
}

# Monitor for suspicious activities
while true; do
    # Check for high number of failed authentication attempts
    FAILED_AUTH=$(tail -n 100 /var/log/nginx/access.log | grep -c " 401 ")
    if [ "$FAILED_AUTH" -gt 10 ]; then
        log_security_event "HIGH_FAILED_AUTH: $FAILED_AUTH failed authentication attempts in last 100 requests"
    fi
    
    # Check for rate limiting triggers
    RATE_LIMITED=$(tail -n 100 /var/log/nginx/access.log | grep -c " 429 ")
    if [ "$RATE_LIMITED" -gt 20 ]; then
        log_security_event "HIGH_RATE_LIMITING: $RATE_LIMITED rate limited requests in last 100 requests"
    fi
    
    # Check for suspicious user agents
    SUSPICIOUS_AGENTS=$(tail -n 100 /var/log/nginx/access.log | grep -i -E "(sqlmap|nikto|scanner|bot)" | wc -l)
    if [ "$SUSPICIOUS_AGENTS" -gt 0 ]; then
        log_security_event "SUSPICIOUS_USER_AGENTS: $SUSPICIOUS_AGENTS suspicious user agents detected"
    fi
    
    sleep 60
done
EOF
    
    chmod +x scripts/security-monitor.sh
    log_success "Created security monitoring script"
}

# Validate Docker Compose security settings
validate_docker_security() {
    log "Validating Docker Compose security settings..."
    
    # Check for security best practices in docker-compose files
    local issues=0
    
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ]; then
            log "Checking $compose_file..."
            
            # Check for privileged containers
            if grep -q "privileged.*true" "$compose_file"; then
                log_warning "Found privileged containers in $compose_file"
                ((issues++))
            fi
            
            # Check for host network mode
            if grep -q "network_mode.*host" "$compose_file"; then
                log_warning "Found host network mode in $compose_file"
                ((issues++))
            fi
            
            # Check for volume mounts to sensitive paths
            if grep -q "/etc:" "$compose_file" || grep -q "/usr:" "$compose_file"; then
                log_warning "Found potentially sensitive volume mounts in $compose_file"
                ((issues++))
            fi
        fi
    done
    
    if [ $issues -eq 0 ]; then
        log_success "Docker Compose security validation passed"
    else
        log_warning "Found $issues potential security issues in Docker Compose files"
    fi
}

# Create security checklist
create_security_checklist() {
    log "Creating security checklist..."
    
    cat << EOF > SECURITY_CHECKLIST.md
# Security Checklist for Self-Healing Pipeline Guard

## Pre-deployment Security Checklist

### Secrets Management
- [ ] All default passwords changed
- [ ] Strong, unique passwords generated for all services
- [ ] Secrets stored securely (not in environment variables)
- [ ] Database credentials rotated
- [ ] API keys generated and secured

### SSL/TLS Configuration
- [ ] Valid SSL certificates installed
- [ ] TLS 1.2+ enforced
- [ ] Weak ciphers disabled
- [ ] HSTS headers configured

### Network Security
- [ ] Firewall rules configured
- [ ] Network segmentation implemented
- [ ] VPN access configured for management
- [ ] Database access restricted to application network
- [ ] Monitoring ports secured

### Application Security
- [ ] Security headers configured in Nginx
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Error messages sanitized
- [ ] Logging configured properly

### Infrastructure Security
- [ ] Operating system hardened
- [ ] fail2ban configured
- [ ] Log rotation set up
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### Docker Security
- [ ] Non-root users in containers
- [ ] Minimal base images used
- [ ] Container scanning completed
- [ ] Resource limits set
- [ ] Security contexts configured

### Monitoring and Alerting
- [ ] Security monitoring enabled
- [ ] Alert thresholds configured
- [ ] Incident response plan documented
- [ ] Log aggregation configured
- [ ] Metrics collection enabled

## Post-deployment Security Checklist

### Verification
- [ ] Security scan completed
- [ ] Penetration testing performed
- [ ] Vulnerability assessment completed
- [ ] Compliance requirements met
- [ ] Documentation updated

### Ongoing Maintenance
- [ ] Security update schedule established
- [ ] Monitoring dashboard configured
- [ ] Incident response tested
- [ ] Backup and recovery tested
- [ ] Security training completed

## Emergency Procedures

### Security Incident Response
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Begin incident analysis
5. Implement containment measures
6. Document lessons learned

### Contact Information
- Security Team: security@terragonlabs.com
- Emergency Contact: +1-555-SECURITY
- Incident Response: incidents@terragonlabs.com

---
Generated by security hardening script on $(date)
EOF
    
    log_success "Created security checklist"
}

# Main execution
main() {
    log "Starting security hardening for Self-Healing Pipeline Guard..."
    
    check_root
    generate_secrets
    generate_ssl_certificates
    harden_docker
    configure_firewall
    setup_log_rotation
    harden_system
    setup_fail2ban
    setup_security_monitoring
    validate_docker_security
    create_security_checklist
    
    log_success "Security hardening completed successfully!"
    log ""
    log "Next steps:"
    log "1. Review the generated security checklist: SECURITY_CHECKLIST.md"
    log "2. Test the application with the new security settings"
    log "3. Configure monitoring alerts"
    log "4. Schedule regular security audits"
    log ""
    log_warning "Remember to:"
    log_warning "- Update default credentials before going to production"
    log_warning "- Configure proper SSL certificates for production"
    log_warning "- Review and test all security settings"
    log_warning "- Keep security configurations up to date"
}

# Run main function
main "$@"