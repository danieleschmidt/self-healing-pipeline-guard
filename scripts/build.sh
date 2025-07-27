#!/bin/bash

# Build script for Self-Healing Pipeline Guard
# Handles building Docker images, running tests, and preparing for deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="self-healing-pipeline-guard"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
VERSION="${VERSION:-$(poetry version -s)}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD)
BUILD_NUMBER="${BUILD_NUMBER:-local}"

# Functions
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

print_header() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "  Self-Healing Pipeline Guard Builder"
    echo "========================================"
    echo -e "${NC}"
    echo "Version: $VERSION"
    echo "Build: $BUILD_NUMBER"
    echo "Commit: $GIT_COMMIT"
    echo "Date: $BUILD_DATE"
    echo ""
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Poetry
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry is not installed or not in PATH"
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

run_tests() {
    log_info "Running test suite..."
    
    # Install dependencies
    poetry install --with dev,test
    
    # Run linting
    log_info "Running code quality checks..."
    poetry run black --check .
    poetry run isort --check-only .
    poetry run flake8 .
    poetry run mypy .
    poetry run ruff check .
    
    # Run security checks
    log_info "Running security checks..."
    poetry run bandit -r healing_guard/ -f json -o reports/bandit-report.json || true
    poetry run safety check --json --output reports/safety-report.json || true
    
    # Run tests
    log_info "Running unit tests..."
    poetry run pytest tests/unit -v --cov=healing_guard --cov-report=xml --cov-report=html
    
    log_info "Running integration tests..."
    poetry run pytest tests/integration -v
    
    log_success "All tests passed"
}

build_docker_images() {
    log_info "Building Docker images..."
    
    # Prepare build arguments
    BUILD_ARGS="--build-arg VERSION=$VERSION"
    BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_DATE=$BUILD_DATE"
    BUILD_ARGS="$BUILD_ARGS --build-arg GIT_COMMIT=$GIT_COMMIT"
    BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_NUMBER=$BUILD_NUMBER"
    
    # Base image name
    IMAGE_NAME="$PROJECT_NAME"
    if [ -n "$DOCKER_REGISTRY" ]; then
        IMAGE_NAME="$DOCKER_REGISTRY/$IMAGE_NAME"
    fi
    
    # Build development image
    log_info "Building development image..."
    docker build $BUILD_ARGS --target development -t "$IMAGE_NAME:dev" .
    
    # Build testing image
    log_info "Building testing image..."
    docker build $BUILD_ARGS --target testing -t "$IMAGE_NAME:test" .
    
    # Build production image
    log_info "Building production image..."
    docker build $BUILD_ARGS --target final -t "$IMAGE_NAME:$VERSION" .
    docker tag "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"
    
    # Build security-scanned image
    log_info "Building security-scanned image..."
    docker build $BUILD_ARGS --target production-secure -t "$IMAGE_NAME:$VERSION-secure" .
    
    log_success "Docker images built successfully"
}

run_docker_tests() {
    log_info "Running tests in Docker container..."
    
    IMAGE_NAME="$PROJECT_NAME"
    if [ -n "$DOCKER_REGISTRY" ]; then
        IMAGE_NAME="$DOCKER_REGISTRY/$IMAGE_NAME"
    fi
    
    # Run tests in container
    docker run --rm \
        -v "$(pwd)/reports:/app/reports" \
        "$IMAGE_NAME:test"
    
    log_success "Docker tests completed"
}

security_scan() {
    log_info "Running security scans..."
    
    IMAGE_NAME="$PROJECT_NAME"
    if [ -n "$DOCKER_REGISTRY" ]; then
        IMAGE_NAME="$DOCKER_REGISTRY/$IMAGE_NAME"
    fi
    
    # Create reports directory
    mkdir -p reports
    
    # Scan with Trivy (if available)
    if command -v trivy &> /dev/null; then
        log_info "Scanning with Trivy..."
        trivy image --format json --output reports/trivy-report.json "$IMAGE_NAME:$VERSION" || true
    else
        log_warning "Trivy not available, skipping container security scan"
    fi
    
    # Scan with Docker Scout (if available)
    if docker scout version &> /dev/null; then
        log_info "Scanning with Docker Scout..."
        docker scout cves --format json --output reports/scout-report.json "$IMAGE_NAME:$VERSION" || true
    else
        log_warning "Docker Scout not available, skipping vulnerability scan"
    fi
    
    log_success "Security scans completed"
}

generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    IMAGE_NAME="$PROJECT_NAME"
    if [ -n "$DOCKER_REGISTRY" ]; then
        IMAGE_NAME="$DOCKER_REGISTRY/$IMAGE_NAME"
    fi
    
    # Create reports directory
    mkdir -p reports
    
    # Generate SBOM with Syft (if available)
    if command -v syft &> /dev/null; then
        log_info "Generating SBOM with Syft..."
        syft "$IMAGE_NAME:$VERSION" -o spdx-json=reports/sbom.spdx.json
        syft "$IMAGE_NAME:$VERSION" -o cyclonedx-json=reports/sbom.cyclonedx.json
    else
        log_warning "Syft not available, generating Python-only SBOM"
        poetry export -f requirements.txt --output reports/requirements.txt
    fi
    
    log_success "SBOM generated"
}

package_artifacts() {
    log_info "Packaging build artifacts..."
    
    # Create artifacts directory
    mkdir -p artifacts
    
    # Package Python wheel
    poetry build
    cp dist/*.whl artifacts/
    cp dist/*.tar.gz artifacts/
    
    # Package configuration files
    tar -czf artifacts/config-$VERSION.tar.gz config/
    
    # Package documentation
    if [ -d "docs/_build" ]; then
        tar -czf artifacts/docs-$VERSION.tar.gz docs/_build/
    fi
    
    # Create deployment package
    tar -czf artifacts/deployment-$VERSION.tar.gz \
        docker-compose.yml \
        docker-compose.prod.yml \
        config/ \
        scripts/deploy.sh \
        scripts/health-check.sh
    
    log_success "Artifacts packaged"
}

push_images() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_warning "No Docker registry specified, skipping image push"
        return
    fi
    
    log_info "Pushing Docker images to registry..."
    
    IMAGE_NAME="$DOCKER_REGISTRY/$PROJECT_NAME"
    
    # Push images
    docker push "$IMAGE_NAME:$VERSION"
    docker push "$IMAGE_NAME:latest"
    docker push "$IMAGE_NAME:dev"
    docker push "$IMAGE_NAME:$VERSION-secure"
    
    log_success "Images pushed to registry"
}

generate_release_notes() {
    log_info "Generating release notes..."
    
    # Create release notes
    cat > artifacts/RELEASE_NOTES.md << EOF
# Release Notes - Version $VERSION

**Build Date:** $BUILD_DATE  
**Build Number:** $BUILD_NUMBER  
**Git Commit:** $GIT_COMMIT  

## Docker Images

- \`$PROJECT_NAME:$VERSION\` - Production image
- \`$PROJECT_NAME:latest\` - Latest production image
- \`$PROJECT_NAME:dev\` - Development image
- \`$PROJECT_NAME:$VERSION-secure\` - Security-scanned image

## Artifacts

- \`$PROJECT_NAME-$VERSION-py3-none-any.whl\` - Python wheel
- \`$PROJECT_NAME-$VERSION.tar.gz\` - Source distribution
- \`config-$VERSION.tar.gz\` - Configuration files
- \`deployment-$VERSION.tar.gz\` - Deployment package

## Security Reports

- \`bandit-report.json\` - Python security analysis
- \`safety-report.json\` - Dependency vulnerability scan
- \`trivy-report.json\` - Container security scan
- \`sbom.spdx.json\` - Software Bill of Materials

## Deployment

1. Extract deployment package: \`tar -xzf deployment-$VERSION.tar.gz\`
2. Configure environment variables in \`.env\`
3. Deploy: \`docker-compose -f docker-compose.prod.yml up -d\`

## Health Check

\`\`\`bash
curl -f http://localhost:8000/health
\`\`\`

EOF
    
    log_success "Release notes generated"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Clean up Docker build cache
    docker builder prune -f
    
    # Clean up Poetry cache
    poetry cache clear --all pypi
    
    log_success "Cleanup completed"
}

main() {
    print_header
    
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_SECURITY=false
    PUSH_IMAGES=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            --push)
                PUSH_IMAGES=true
                shift
                ;;
            --registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-tests      Skip running tests"
                echo "  --skip-security   Skip security scans"
                echo "  --push            Push images to registry"
                echo "  --registry        Docker registry URL"
                echo "  --version         Override version"
                echo "  -h, --help        Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Create reports directory
    mkdir -p reports
    
    # Run build steps
    check_dependencies
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    else
        log_warning "Skipping tests"
    fi
    
    build_docker_images
    
    if [ "$SKIP_TESTS" = false ]; then
        run_docker_tests
    fi
    
    if [ "$SKIP_SECURITY" = false ]; then
        security_scan
        generate_sbom
    else
        log_warning "Skipping security scans"
    fi
    
    package_artifacts
    generate_release_notes
    
    if [ "$PUSH_IMAGES" = true ]; then
        push_images
    fi
    
    cleanup
    
    log_success "Build completed successfully!"
    echo ""
    echo "Build artifacts are available in the 'artifacts' directory"
    echo "Security reports are available in the 'reports' directory"
    echo ""
}

# Run main function
main "$@"