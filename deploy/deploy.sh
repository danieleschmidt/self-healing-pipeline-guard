#!/bin/bash

# Self-Healing Pipeline Guard - Production Deployment Script
# Version: 1.0.0
# Author: Terragon Labs

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-healing-guard}"
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-docker.io/terragonlabs}"
IMAGE_NAME="${IMAGE_NAME:-healing-guard}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.kube/config}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubeconfig
    if [[ ! -f "$KUBECONFIG_PATH" ]]; then
        log_error "Kubeconfig not found at $KUBECONFIG_PATH"
        exit 1
    fi
    
    # Test Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build multi-stage image
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
        -t "$REGISTRY/$IMAGE_NAME:$VERSION" \
        -t "$REGISTRY/$IMAGE_NAME:latest" \
        .
    
    log_success "Docker image built successfully"
}

# Push Docker image
push_image() {
    log_info "Pushing Docker image to registry..."
    
    # Login to registry if credentials are provided
    if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin "$REGISTRY"
    fi
    
    docker push "$REGISTRY/$IMAGE_NAME:$VERSION"
    docker push "$REGISTRY/$IMAGE_NAME:latest"
    
    log_success "Docker image pushed successfully"
}

# Create namespace
create_namespace() {
    log_info "Creating Kubernetes namespace..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
    
    # Label namespace
    kubectl label namespace "$NAMESPACE" \
        app.kubernetes.io/name=healing-guard \
        app.kubernetes.io/managed-by=healing-guard-deploy \
        --overwrite
}

# Deploy using Helm
deploy_helm() {
    log_info "Deploying using Helm..."
    
    cd "$PROJECT_ROOT"
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Deploy using Helm
    helm upgrade --install healing-guard \
        ./helm/healing-guard \
        --namespace "$NAMESPACE" \
        --set image.tag="$VERSION" \
        --set image.registry="$REGISTRY" \
        --set config.environment="$ENVIRONMENT" \
        --wait \
        --timeout=600s
    
    log_success "Helm deployment completed"
}

# Deploy using kubectl (alternative)
deploy_kubectl() {
    log_info "Deploying using kubectl..."
    
    cd "$PROJECT_ROOT"
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/serviceaccount.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=600s deployment/healing-guard -n "$NAMESPACE"
    
    log_success "kubectl deployment completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service endpoint
    if kubectl get ingress healing-guard-ingress -n "$NAMESPACE" &> /dev/null; then
        # Use ingress endpoint
        ENDPOINT=$(kubectl get ingress healing-guard-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
        ENDPOINT="https://$ENDPOINT"
    else
        # Use port-forward for testing
        kubectl port-forward -n "$NAMESPACE" svc/healing-guard-service 8080:80 &
        PORT_FORWARD_PID=$!
        sleep 5
        ENDPOINT="http://localhost:8080"
    fi
    
    # Test health endpoint
    if curl -f "$ENDPOINT/health" &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
    
    # Test API endpoint
    if curl -f "$ENDPOINT/api/metrics/summary" &> /dev/null; then
        log_success "API endpoint test passed"
    else
        log_warning "API endpoint test failed (might be expected if authentication is required)"
    fi
    
    # Clean up port-forward if used
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        kill $PORT_FORWARD_PID &> /dev/null || true
    fi
    
    log_success "Smoke tests completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Image: $REGISTRY/$IMAGE_NAME:$VERSION"
    echo ""
    
    # Show pods
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=healing-guard
    echo ""
    
    # Show services
    echo "Services:"
    kubectl get services -n "$NAMESPACE"
    echo ""
    
    # Show ingress
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE"
        echo ""
    fi
    
    # Show HPA
    if kubectl get hpa -n "$NAMESPACE" &> /dev/null; then
        echo "Horizontal Pod Autoscaler:"
        kubectl get hpa -n "$NAMESPACE"
        echo ""
    fi
}

# Rollback deployment
rollback() {
    log_info "Rolling back deployment..."
    
    if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q healing-guard; then
        helm rollback healing-guard -n "$NAMESPACE"
        log_success "Helm rollback completed"
    else
        log_warning "Helm release not found, using kubectl rollout undo"
        kubectl rollout undo deployment/healing-guard -n "$NAMESPACE"
        log_success "kubectl rollback completed"
    fi
}

# Cleanup deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q healing-guard; then
        helm uninstall healing-guard -n "$NAMESPACE"
    else
        kubectl delete -f k8s/ --ignore-not-found=true
    fi
    
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of Self-Healing Pipeline Guard"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Namespace: $NAMESPACE"
    
    check_prerequisites
    
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_image
    fi
    
    if [[ "${SKIP_PUSH:-false}" != "true" ]]; then
        push_image
    fi
    
    create_namespace
    
    # Choose deployment method
    if [[ -d "$PROJECT_ROOT/helm/healing-guard" ]] && [[ "${USE_HELM:-true}" == "true" ]]; then
        deploy_helm
    else
        deploy_kubectl
    fi
    
    run_smoke_tests
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at the ingress endpoint or use port-forward:"
    log_info "kubectl port-forward -n $NAMESPACE svc/healing-guard-service 8080:80"
}

# Usage information
usage() {
    cat << EOF
Self-Healing Pipeline Guard - Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Deploy the application (default)
    build       Build Docker image only
    push        Push Docker image only
    rollback    Rollback to previous version
    cleanup     Remove all resources
    status      Show deployment status
    help        Show this help message

Environment Variables:
    NAMESPACE         Kubernetes namespace (default: healing-guard)
    ENVIRONMENT       Deployment environment (default: production)
    VERSION           Image version tag (default: latest)
    REGISTRY          Docker registry (default: docker.io/terragonlabs)
    IMAGE_NAME        Docker image name (default: healing-guard)
    KUBECONFIG_PATH   Path to kubeconfig (default: ~/.kube/config)
    USE_HELM          Use Helm for deployment (default: true)
    SKIP_BUILD        Skip Docker build (default: false)
    SKIP_PUSH         Skip Docker image push (default: false)
    DOCKER_USERNAME   Docker registry username
    DOCKER_PASSWORD   Docker registry password

Examples:
    # Full deployment
    $0 deploy

    # Deploy specific version
    VERSION=v1.2.3 $0 deploy

    # Deploy to staging
    ENVIRONMENT=staging NAMESPACE=healing-guard-staging $0 deploy

    # Build and push only
    $0 build
    $0 push

    # Rollback deployment
    $0 rollback

    # Cleanup everything
    $0 cleanup

EOF
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        check_prerequisites
        build_image
        ;;
    push)
        check_prerequisites
        push_image
        ;;
    rollback)
        check_prerequisites
        rollback
        ;;
    cleanup)
        check_prerequisites
        cleanup
        ;;
    status)
        check_prerequisites
        show_status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac