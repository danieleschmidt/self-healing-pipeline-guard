#!/bin/bash
#
# Production Deployment Script for Healing Guard
# Deploys the self-healing pipeline guard to production environment
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
REGISTRY="${REGISTRY:-ghcr.io/terragon-labs}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-healing-guard}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "kubectl" "helm")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command '$cmd' not found. Please install it first."
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running or not accessible."
    fi
    
    # Check Kubernetes connectivity
    if ! kubectl version --client &> /dev/null; then
        error_exit "kubectl is not properly configured."
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster."
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main image
    log_info "Building main application image..."
    docker build \
        -f Dockerfile.production \
        -t "${REGISTRY}/healing-guard:${IMAGE_TAG}" \
        --target production \
        .
    
    # Build dashboard image
    log_info "Building dashboard image..."
    docker build \
        -f Dockerfile.production \
        -t "${REGISTRY}/healing-guard-dashboard:${IMAGE_TAG}" \
        --target dashboard \
        .
    
    # Push images
    log_info "Pushing images to registry..."
    docker push "${REGISTRY}/healing-guard:${IMAGE_TAG}"
    docker push "${REGISTRY}/healing-guard-dashboard:${IMAGE_TAG}"
    
    log_success "Images built and pushed successfully"
}

# Generate secrets if they don't exist
generate_secrets() {
    log_info "Generating secrets..."
    
    local secrets_file="$PROJECT_ROOT/.env.production"
    
    if [[ ! -f "$secrets_file" ]]; then
        log_info "Generating new secrets file..."
        
        cat > "$secrets_file" << EOF
# Generated on $(date)
DB_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
SMTP_HOST=smtp.example.com
SMTP_USER=alerts@example.com
SMTP_PASSWORD=$(openssl rand -base64 16)
SMTP_FROM=healing-guard@example.com
EOF
        
        chmod 600 "$secrets_file"
        log_success "Secrets generated in $secrets_file"
        log_warning "Please review and update the SMTP settings in $secrets_file"
    else
        log_info "Using existing secrets from $secrets_file"
    fi
    
    # Source secrets
    source "$secrets_file"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    cd "$PROJECT_ROOT"
    
    # Create namespace
    kubectl apply -f k8s/production/namespace.yaml
    
    # Create secrets
    kubectl create secret generic healing-guard-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=database-url="postgresql://healing_user:${DB_PASSWORD}@healing-guard-postgres:5432/healing_guard" \
        --from-literal=jwt-secret="$JWT_SECRET" \
        --from-literal=encryption-key="$ENCRYPTION_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL database..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install healing-guard-postgres bitnami/postgresql \
        --namespace="$NAMESPACE" \
        --set auth.postgresPassword="$DB_PASSWORD" \
        --set auth.username=healing_user \
        --set auth.password="$DB_PASSWORD" \
        --set auth.database=healing_guard \
        --set primary.persistence.enabled=true \
        --set primary.persistence.size=20Gi \
        --set metrics.enabled=true \
        --wait
    
    # Deploy Redis
    log_info "Deploying Redis cache..."
    helm upgrade --install healing-guard-redis bitnami/redis \
        --namespace="$NAMESPACE" \
        --set auth.enabled=false \
        --set master.persistence.enabled=true \
        --set master.persistence.size=5Gi \
        --set metrics.enabled=true \
        --wait
    
    log_success "Infrastructure components deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    log_info "Deploying Prometheus..."
    helm upgrade --install healing-guard-prometheus prometheus-community/kube-prometheus-stack \
        --namespace="$NAMESPACE" \
        --set grafana.adminPassword="$GRAFANA_PASSWORD" \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=5Gi \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi \
        --wait
    
    # Deploy Loki for log aggregation
    log_info "Deploying Loki..."
    helm upgrade --install healing-guard-loki grafana/loki-stack \
        --namespace="$NAMESPACE" \
        --set loki.persistence.enabled=true \
        --set loki.persistence.size=10Gi \
        --wait
    
    log_success "Monitoring stack deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying Healing Guard application..."
    
    cd "$PROJECT_ROOT"
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/production/configmap.yaml
    kubectl apply -f k8s/production/serviceaccount.yaml
    kubectl apply -f k8s/production/rbac.yaml
    kubectl apply -f k8s/production/pvc.yaml
    
    # Update image references in deployments
    sed -i.bak "s|healing-guard:latest|${REGISTRY}/healing-guard:${IMAGE_TAG}|g" k8s/production/deployment.yaml
    kubectl apply -f k8s/production/deployment.yaml
    
    # Apply services and ingress
    kubectl apply -f k8s/production/service.yaml
    kubectl apply -f k8s/production/ingress.yaml
    
    # Apply HPA
    kubectl apply -f k8s/production/hpa.yaml
    
    log_success "Application deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployments to be ready..."
    
    local deployments=("healing-guard-api" "healing-guard-dashboard" "healing-guard-nginx")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/"$deployment" -n "$NAMESPACE"
    done
    
    log_success "All deployments are ready"
}

# Run post-deployment checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoints
    local api_endpoint
    api_endpoint=$(kubectl get service healing-guard-nginx -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$api_endpoint" ]]; then
        log_warning "LoadBalancer IP not yet assigned. Using port-forward for health check..."
        kubectl port-forward service/healing-guard-nginx 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        api_endpoint="localhost:8080"
    fi
    
    # Test API health
    log_info "Testing API health..."
    if curl -f "http://$api_endpoint/api/health" &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
    fi
    
    # Test dashboard
    log_info "Testing dashboard..."
    if curl -f "http://$api_endpoint/dashboard/api/dashboard/health" &> /dev/null; then
        log_success "Dashboard health check passed"
    else
        log_error "Dashboard health check failed"
    fi
    
    # Kill port-forward if we started it
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid &> /dev/null || true
    fi
    
    log_success "Health checks completed"
}

# Display deployment information
display_deployment_info() {
    log_info "Deployment Information:"
    echo
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Registry: $REGISTRY"
    echo
    
    log_info "Application Endpoints:"
    kubectl get ingress -n "$NAMESPACE" -o wide
    echo
    
    log_info "Service Status:"
    kubectl get services -n "$NAMESPACE"
    echo
    
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    echo
    
    log_info "Resource Usage:"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || log_warning "Metrics server not available"
}

# Backup current deployment
backup_deployment() {
    log_info "Creating backup of current deployment..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/kubernetes_resources.yaml"
    
    # Backup database if possible
    if kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=postgresql &> /dev/null; then
        log_info "Creating database backup..."
        kubectl exec -n "$NAMESPACE" -c postgresql \
            $(kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=postgresql -o jsonpath='{.items[0].metadata.name}') -- \
            pg_dump -U healing_user healing_guard > "$backup_dir/database_backup.sql"
    fi
    
    log_success "Backup created in $backup_dir"
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    log_info "Starting production deployment of Healing Guard..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    echo
    
    # Run deployment steps
    check_prerequisites
    generate_secrets
    
    # Ask for confirmation in production
    if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        echo
        read -p "This will deploy to PRODUCTION environment. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
        backup_deployment
    fi
    
    build_and_push_images
    deploy_infrastructure
    deploy_monitoring
    deploy_application
    wait_for_deployment
    run_health_checks
    display_deployment_info
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo
    log_success "🎉 Production deployment completed successfully!"
    log_info "Total deployment time: ${duration} seconds"
    log_info "Access the dashboard at: http://<your-domain>/dashboard/"
    log_info "API documentation at: http://<your-domain>/api/docs"
    
    # Save deployment info
    cat > "$PROJECT_ROOT/deployment-info.txt" << EOF
Healing Guard Production Deployment
===================================
Deployed: $(date)
Environment: $DEPLOYMENT_ENV
Namespace: $NAMESPACE
Image Tag: $IMAGE_TAG
Registry: $REGISTRY
Duration: ${duration} seconds

Next Steps:
1. Configure DNS to point to the LoadBalancer IP
2. Set up SSL certificates
3. Configure monitoring alerts
4. Update backup retention policies
5. Review security settings

For support, visit: https://docs.terragon.ai/healing-guard
EOF
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        log_info "Rolling back deployment..."
        kubectl rollout undo deployment/healing-guard-api -n "$NAMESPACE"
        kubectl rollout undo deployment/healing-guard-dashboard -n "$NAMESPACE"
        log_success "Rollback completed"
        ;;
    "status")
        display_deployment_info
        ;;
    "health")
        run_health_checks
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|health}"
        exit 1
        ;;
esac