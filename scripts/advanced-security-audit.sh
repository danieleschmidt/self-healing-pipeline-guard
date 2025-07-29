#!/bin/bash
# Advanced Security Audit Script for Self-Healing Pipeline Guard
# Performs comprehensive security analysis beyond basic scanning

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUDIT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
AUDIT_DIR="$PROJECT_ROOT/security_audit_$AUDIT_TIMESTAMP"
FINDINGS_FILE="$AUDIT_DIR/security_findings.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Create audit directory
create_audit_dir() {
    log_info "Creating audit directory: $AUDIT_DIR"
    mkdir -p "$AUDIT_DIR"/{reports,logs,configs,evidence}
}

# Check tool availability
check_dependencies() {
    log_info "Checking security tool dependencies..."
    
    local tools=("bandit" "safety" "semgrep" "gitleaks" "trivy" "grype")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_warn "$tool not found. Install with: pip install $tool (or appropriate package manager)"
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing tools: ${missing_tools[*]}"
        log_info "Installing missing Python security tools..."
        pip install bandit safety semgrep || log_warn "Failed to install some tools"
    fi
}

# Static Application Security Testing (SAST)
run_sast_analysis() {
    log_info "Running Static Application Security Testing (SAST)..."
    
    # Bandit for Python security issues
    log_info "Running Bandit security scan..."
    bandit -r "$PROJECT_ROOT" -f json -o "$AUDIT_DIR/reports/bandit_results.json" || true
    
    # Semgrep for advanced pattern matching
    if command -v semgrep &> /dev/null; then
        log_info "Running Semgrep analysis..."
        semgrep --config=auto --json --output="$AUDIT_DIR/reports/semgrep_results.json" "$PROJECT_ROOT" || true
    fi
    
    # Custom security patterns
    log_info "Checking for custom security patterns..."
    grep -r -n -E "(password|secret|key|token).*=" "$PROJECT_ROOT" --include="*.py" > "$AUDIT_DIR/reports/potential_secrets.txt" || true
}

# Dependency vulnerability analysis
analyze_dependencies() {
    log_info "Analyzing dependency vulnerabilities..."
    
    # Safety for Python dependencies
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]] || [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_info "Running Safety check for Python dependencies..."
        safety check --json --output "$AUDIT_DIR/reports/safety_results.json" || true
    fi
    
    # Trivy for comprehensive vulnerability scanning
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy vulnerability scan..."
        trivy fs --format json --output "$AUDIT_DIR/reports/trivy_results.json" "$PROJECT_ROOT" || true
    fi
    
    # Grype for additional vulnerability scanning
    if command -v grype &> /dev/null; then
        log_info "Running Grype vulnerability scan..."
        grype dir:"$PROJECT_ROOT" -o json > "$AUDIT_DIR/reports/grype_results.json" || true
    fi
}

# Secret scanning
scan_for_secrets() {
    log_info "Scanning for secrets and sensitive data..."
    
    # Gitleaks for secret detection
    if command -v gitleaks &> /dev/null; then
        log_info "Running Gitleaks secret scan..."
        cd "$PROJECT_ROOT"
        gitleaks detect --report-format json --report-path "$AUDIT_DIR/reports/gitleaks_results.json" || true
        cd - > /dev/null
    fi
    
    # Custom secret patterns
    log_info "Checking for custom secret patterns..."
    local secret_patterns=(
        "AKIA[0-9A-Z]{16}"  # AWS Access Key
        "sk-[a-zA-Z0-9]{48}"  # OpenAI API Key
        "xoxb-[0-9]{11}-[0-9]{11}-[0-9A-Za-z]{24}"  # Slack Bot Token
        "ghp_[0-9a-zA-Z]{36}"  # GitHub Personal Access Token
        "glpat-[0-9a-zA-Z_-]{20}"  # GitLab Personal Access Token
    )
    
    for pattern in "${secret_patterns[@]}"; do
        grep -r -E "$pattern" "$PROJECT_ROOT" --exclude-dir=.git > "$AUDIT_DIR/reports/custom_secrets_$pattern.txt" 2>/dev/null || true
    done
}

# Container security analysis
analyze_containers() {
    log_info "Analyzing container security..."
    
    local dockerfiles=("$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT/Dockerfile.dev" "$PROJECT_ROOT/Dockerfile.test")
    
    for dockerfile in "${dockerfiles[@]}"; do
        if [[ -f "$dockerfile" ]]; then
            log_info "Analyzing $(basename "$dockerfile")..."
            
            # Check for security best practices
            {
                echo "=== Dockerfile Security Analysis: $(basename "$dockerfile") ==="
                echo
                
                # Check for root user usage
                if grep -q "USER root" "$dockerfile" || ! grep -q "USER " "$dockerfile"; then
                    echo "WARNING: Container may be running as root"
                fi
                
                # Check for package manager cache cleanup
                if grep -q "apt-get install" "$dockerfile" && ! grep -q "rm -rf /var/lib/apt/lists/*" "$dockerfile"; then
                    echo "WARNING: Package manager cache not cleaned up"
                fi
                
                # Check for hardcoded secrets
                if grep -qE "(password|secret|key|token)" "$dockerfile"; then
                    echo "WARNING: Potential hardcoded secrets found"
                fi
                
                # Check for specific security practices
                if ! grep -q "COPY --chown=" "$dockerfile"; then
                    echo "INFO: Consider using COPY --chown for better security"
                fi
                
                echo
            } >> "$AUDIT_DIR/reports/dockerfile_security_$(basename "$dockerfile").txt"
        fi
    done
    
    # Trivy container image scanning (if images exist)
    if command -v trivy &> /dev/null && command -v docker &> /dev/null; then
        local images=($(docker images --filter "reference=*healing*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || true))
        for image in "${images[@]}"; do
            if [[ "$image" != "<none>:<none>" ]]; then
                log_info "Scanning container image: $image"
                trivy image --format json --output "$AUDIT_DIR/reports/trivy_image_$(echo "$image" | tr '/:' '_').json" "$image" || true
            fi
        done
    fi
}

# Infrastructure security checks
check_infrastructure_security() {
    log_info "Checking infrastructure security configurations..."
    
    # Docker Compose security analysis
    local compose_files=("$PROJECT_ROOT/docker-compose.yml" "$PROJECT_ROOT/docker-compose.prod.yml")
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log_info "Analyzing Docker Compose security: $(basename "$compose_file")"
            
            {
                echo "=== Docker Compose Security Analysis: $(basename "$compose_file") ==="
                echo
                
                # Check for privileged containers
                if grep -q "privileged: true" "$compose_file"; then
                    echo "CRITICAL: Privileged containers detected"
                fi
                
                # Check for host network mode
                if grep -q "network_mode: host" "$compose_file"; then
                    echo "WARNING: Host network mode detected"
                fi
                
                # Check for volume mounts to sensitive paths
                if grep -qE "- /.*:/.*" "$compose_file"; then
                    echo "WARNING: Host volume mounts detected - review for security implications"
                    grep -E "- /.*:/.*" "$compose_file" | head -5
                fi
                
                # Check for default passwords
                if grep -qE "(POSTGRES_PASSWORD|MYSQL_PASSWORD|REDIS_PASSWORD).*=" "$compose_file"; then
                    echo "WARNING: Database passwords may be hardcoded"
                fi
                
                echo
            } >> "$AUDIT_DIR/reports/compose_security_$(basename "$compose_file").txt"
        fi
    done
}

# API security analysis
analyze_api_security() {
    log_info "Analyzing API security configurations..."
    
    # Look for FastAPI security configurations
    if find "$PROJECT_ROOT" -name "*.py" -exec grep -l "FastAPI\|@app\." {} \; | head -1 > /dev/null; then
        {
            echo "=== API Security Analysis ==="
            echo
            
            # Check for CORS configuration
            if grep -r "CORSMiddleware" "$PROJECT_ROOT" --include="*.py"; then
                echo "INFO: CORS middleware found - verify configuration"
            else
                echo "WARNING: No CORS middleware configuration found"
            fi
            
            # Check for rate limiting
            if grep -r "rate.*limit" "$PROJECT_ROOT" --include="*.py"; then
                echo "INFO: Rate limiting detected"
            else
                echo "WARNING: No rate limiting detected"
            fi
            
            # Check for authentication/authorization
            if grep -r -E "(jwt|token|auth)" "$PROJECT_ROOT" --include="*.py" | head -5; then
                echo "INFO: Authentication mechanisms found"
            else
                echo "WARNING: No authentication mechanisms detected"
            fi
            
            echo
        } >> "$AUDIT_DIR/reports/api_security_analysis.txt"
    fi
}

# Generate comprehensive security report
generate_security_report() {
    log_info "Generating comprehensive security report..."
    
    local report_file="$AUDIT_DIR/SECURITY_AUDIT_REPORT.md"
    
    cat > "$report_file" << EOF
# Security Audit Report

**Generated**: $(date)
**Project**: Self-Healing Pipeline Guard
**Audit ID**: $AUDIT_TIMESTAMP

## Executive Summary

This comprehensive security audit analyzed the Self-Healing Pipeline Guard project for:
- Static Application Security Testing (SAST)
- Dependency vulnerabilities
- Secret exposure
- Container security
- Infrastructure security
- API security configurations

## Findings Summary

### Critical Issues
$(find "$AUDIT_DIR/reports" -name "*.txt" -exec grep -l "CRITICAL" {} \; | wc -l) critical issues found

### High-Risk Issues
$(find "$AUDIT_DIR/reports" -name "*.txt" -exec grep -l "WARNING" {} \; | wc -l) high-risk issues found

### Informational Items
$(find "$AUDIT_DIR/reports" -name "*.txt" -exec grep -l "INFO" {} \; | wc -l) informational items noted

## Detailed Analysis

### Static Code Analysis
- **Bandit Results**: See \`reports/bandit_results.json\`
- **Semgrep Results**: See \`reports/semgrep_results.json\`
- **Custom Pattern Matches**: See \`reports/potential_secrets.txt\`

### Dependency Vulnerabilities
- **Safety Results**: See \`reports/safety_results.json\`
- **Trivy Results**: See \`reports/trivy_results.json\`
- **Grype Results**: See \`reports/grype_results.json\`

### Secret Scanning
- **Gitleaks Results**: See \`reports/gitleaks_results.json\`
- **Custom Secret Pattern**: See \`reports/custom_secrets_*.txt\`

### Container Security
- **Dockerfile Analysis**: See \`reports/dockerfile_security_*.txt\`
- **Image Scanning**: See \`reports/trivy_image_*.json\`

### Infrastructure Security
- **Docker Compose Analysis**: See \`reports/compose_security_*.txt\`

### API Security
- **API Configuration Analysis**: See \`reports/api_security_analysis.txt\`

## Recommendations

1. **Address Critical Issues**: Prioritize resolution of all critical security findings
2. **Implement Security Headers**: Ensure proper security headers in API responses
3. **Regular Vulnerability Scanning**: Automate dependency vulnerability scanning
4. **Secret Management**: Implement proper secret management solutions
5. **Container Hardening**: Apply container security best practices
6. **Access Controls**: Review and strengthen access control mechanisms

## Next Steps

- Review all findings in the \`reports/\` directory
- Create tracking issues for each security finding
- Implement remediation plans with timelines
- Schedule regular security audits
- Update security policies and procedures

---

**Audit Tools Used**: Bandit, Safety, Semgrep, Gitleaks, Trivy, Grype
**Report Location**: \`$AUDIT_DIR\`
**Contact**: Security Team <security@terragonlabs.com>
EOF

    log_success "Security audit report generated: $report_file"
}

# Archive audit results
archive_results() {
    log_info "Archiving audit results..."
    
    local archive_name="security_audit_${AUDIT_TIMESTAMP}.tar.gz"
    tar -czf "$PROJECT_ROOT/$archive_name" -C "$(dirname "$AUDIT_DIR")" "$(basename "$AUDIT_DIR")"
    
    log_success "Audit results archived: $archive_name"
    log_info "To extract: tar -xzf $archive_name"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup operations here if needed
}

# Main execution
main() {
    log_info "Starting Advanced Security Audit for Self-Healing Pipeline Guard"
    log_info "Audit timestamp: $AUDIT_TIMESTAMP"
    
    create_audit_dir
    check_dependencies
    
    # Run all security analyses
    run_sast_analysis
    analyze_dependencies
    scan_for_secrets
    analyze_containers
    check_infrastructure_security
    analyze_api_security
    
    # Generate reports
    generate_security_report
    archive_results
    
    log_success "Security audit completed successfully!"
    log_info "Review the audit report at: $AUDIT_DIR/SECURITY_AUDIT_REPORT.md"
    log_info "All findings are available in: $AUDIT_DIR/reports/"
    
    cleanup
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi