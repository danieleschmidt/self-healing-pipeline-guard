#!/bin/bash

# Release script for Self-Healing Pipeline Guard
# Handles version bumping, changelog generation, and release publishing

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
RELEASE_BRANCH="main"
DEVELOP_BRANCH="develop"

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

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "================================================="
    echo "  Self-Healing Pipeline Guard Release Script"
    echo "================================================="
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if poetry is installed
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry is not installed"
        exit 1
    fi
    
    # Check if semantic-release is available
    if ! command -v semantic-release &> /dev/null; then
        log_warning "semantic-release not found, installing..."
        npm install -g semantic-release @semantic-release/changelog @semantic-release/git @semantic-release/github @semantic-release/exec @semantic-release/slack-webhook
    fi
    
    # Check if required environment variables are set
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN environment variable is required"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Check git status
check_git_status() {
    log_info "Checking git status..."
    
    # Check if working directory is clean
    if ! git diff-index --quiet HEAD --; then
        log_error "Working directory is not clean. Please commit or stash changes."
        git status --porcelain
        exit 1
    fi
    
    # Check if we're on the correct branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "$RELEASE_BRANCH" ]]; then
        log_warning "Not on release branch ($RELEASE_BRANCH). Current branch: $CURRENT_BRANCH"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Switch to $RELEASE_BRANCH branch and try again"
            exit 1
        fi
    fi
    
    # Fetch latest changes
    git fetch origin
    
    # Check if branch is up to date
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse "@{u}" 2>/dev/null || echo "")
    if [[ -n "$REMOTE" && "$LOCAL" != "$REMOTE" ]]; then
        log_error "Branch is not up to date with remote. Please pull latest changes."
        exit 1
    fi
    
    log_success "Git status check passed"
}

# Run tests
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
    
    # Run tests
    log_info "Running tests..."
    poetry run pytest tests/ --cov=healing_guard --cov-fail-under=80
    
    # Run security checks
    log_info "Running security checks..."
    poetry run bandit -r healing_guard/ || log_warning "Security issues found"
    poetry run safety check || log_warning "Dependency vulnerabilities found"
    
    log_success "All tests passed"
}

# Build artifacts
build_artifacts() {
    log_info "Building release artifacts..."
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info/
    
    # Build Python packages
    poetry build
    
    # Build Docker images
    if command -v docker &> /dev/null; then
        log_info "Building Docker images..."
        ./scripts/build.sh --skip-tests --version="$(poetry version -s)"
    fi
    
    # Generate documentation
    if command -v mkdocs &> /dev/null; then
        log_info "Building documentation..."
        mkdocs build
    fi
    
    log_success "Artifacts built successfully"
}

# Generate changelog preview
generate_changelog_preview() {
    log_info "Generating changelog preview..."
    
    # Get current version
    CURRENT_VERSION=$(poetry version -s)
    
    # Get commits since last tag
    LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [[ -n "$LAST_TAG" ]]; then
        log_info "Changes since $LAST_TAG:"
        git log --oneline --pretty=format:"  - %s (%an)" "$LAST_TAG"..HEAD
    else
        log_info "No previous tags found. This will be the first release."
        git log --oneline --pretty=format:"  - %s (%an)" HEAD
    fi
    
    echo
}

# Perform pre-release checks
pre_release_checks() {
    log_info "Performing pre-release checks..."
    
    # Check if version would change
    if command -v semantic-release &> /dev/null; then
        log_info "Checking if release is needed..."
        # This is a dry run to see if a release would be created
        if ! semantic-release --dry-run --no-ci 2>/dev/null; then
            log_warning "No release needed based on commits"
            exit 0
        fi
    fi
    
    # Check Docker registry access
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Testing Docker registry access..."
        docker login "$DOCKER_REGISTRY" || log_warning "Docker registry login failed"
    fi
    
    # Check PyPI access
    if [[ -n "$PYPI_TOKEN" ]]; then
        log_info "PyPI token configured for publishing"
    fi
    
    log_success "Pre-release checks passed"
}

# Create release
create_release() {
    log_info "Creating release..."
    
    # Use semantic-release for automated releases
    if semantic-release --no-ci; then
        log_success "Release created successfully"
        
        # Get the new version
        NEW_VERSION=$(poetry version -s)
        log_info "New version: $NEW_VERSION"
        
        # Push Docker images if registry is configured
        if [[ -n "$DOCKER_REGISTRY" ]]; then
            log_info "Pushing Docker images..."
            docker push "$DOCKER_REGISTRY/self-healing-pipeline-guard:$NEW_VERSION"
            docker push "$DOCKER_REGISTRY/self-healing-pipeline-guard:latest"
        fi
        
        # Publish to PyPI if token is configured
        if [[ -n "$PYPI_TOKEN" ]]; then
            log_info "Publishing to PyPI..."
            poetry config pypi-token.pypi "$PYPI_TOKEN"
            poetry publish
        fi
        
    else
        log_error "Release creation failed"
        exit 1
    fi
}

# Post-release actions
post_release_actions() {
    log_info "Performing post-release actions..."
    
    # Get the new version and tag
    NEW_VERSION=$(poetry version -s)
    NEW_TAG="v$NEW_VERSION"
    
    # Update develop branch if it exists
    if git show-ref --verify --quiet refs/heads/$DEVELOP_BRANCH; then
        log_info "Updating $DEVELOP_BRANCH branch..."
        git checkout "$DEVELOP_BRANCH"
        git merge "$RELEASE_BRANCH" --no-edit
        git push origin "$DEVELOP_BRANCH"
        git checkout "$RELEASE_BRANCH"
    fi
    
    # Create deployment artifacts
    log_info "Creating deployment artifacts..."
    mkdir -p artifacts
    
    # Package configuration files
    tar -czf "artifacts/healing-guard-config-$NEW_VERSION.tar.gz" config/
    
    # Package deployment files
    tar -czf "artifacts/healing-guard-deployment-$NEW_VERSION.tar.gz" \
        docker-compose.yml docker-compose.prod.yml scripts/deploy.sh
    
    # Generate deployment notes
    cat > "artifacts/DEPLOYMENT_NOTES_$NEW_VERSION.md" << EOF
# Deployment Notes - Version $NEW_VERSION

## Pre-deployment Checklist
- [ ] Review changelog for breaking changes
- [ ] Backup current configuration
- [ ] Backup database
- [ ] Notify stakeholders of maintenance window

## Deployment Steps
1. Download deployment package: \`healing-guard-deployment-$NEW_VERSION.tar.gz\`
2. Extract: \`tar -xzf healing-guard-deployment-$NEW_VERSION.tar.gz\`
3. Review configuration changes
4. Deploy: \`docker-compose -f docker-compose.prod.yml up -d\`
5. Verify health: \`curl https://your-domain/health\`

## Rollback Procedure
If issues occur, rollback using:
\`\`\`bash
docker-compose -f docker-compose.prod.yml down
# Restore previous version
docker-compose -f docker-compose.prod.yml up -d
\`\`\`

## Support
- Documentation: https://docs.terragonlabs.com/healing-guard
- Issues: https://github.com/terragon-labs/self-healing-pipeline-guard/issues
- Support: support@terragonlabs.com

Generated on: $(date)
EOF
    
    log_success "Post-release actions completed"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -f .semantic-release-output
    
    # Clean Docker build cache
    if command -v docker &> /dev/null; then
        docker builder prune -f > /dev/null 2>&1 || true
    fi
    
    log_success "Cleanup completed"
}

# Main function
main() {
    local release_type="auto"
    local skip_tests=false
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                release_type="$2"
                shift 2
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                cat << EOF
Usage: $0 [OPTIONS]

Options:
    --type TYPE         Release type (auto, patch, minor, major)
    --skip-tests        Skip running tests
    --dry-run           Show what would be done without making changes
    -h, --help          Show this help message

Environment Variables:
    GITHUB_TOKEN        GitHub token for release creation
    DOCKER_REGISTRY     Docker registry for image publishing
    PYPI_TOKEN          PyPI token for package publishing
    SLACK_WEBHOOK_URL   Slack webhook for notifications

Examples:
    $0                              # Automatic release based on commits
    $0 --type minor                 # Force minor version bump
    $0 --skip-tests --dry-run      # Preview release without tests
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_header
    
    if [[ "$dry_run" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
        export DRY_RUN=true
    fi
    
    # Run release process
    check_prerequisites
    check_git_status
    
    if [[ "$skip_tests" != true ]]; then
        run_tests
    else
        log_warning "Skipping tests"
    fi
    
    build_artifacts
    generate_changelog_preview
    pre_release_checks
    
    if [[ "$dry_run" != true ]]; then
        create_release
        post_release_actions
    else
        log_info "DRY RUN: Would create release here"
    fi
    
    cleanup
    
    log_success "Release process completed!"
    
    if [[ "$dry_run" != true ]]; then
        NEW_VERSION=$(poetry version -s)
        echo
        log_info "🎉 Version $NEW_VERSION has been released!"
        log_info "📦 GitHub Release: https://github.com/terragon-labs/self-healing-pipeline-guard/releases/tag/v$NEW_VERSION"
        log_info "🐳 Docker Image: terragonlabs/healing-guard:$NEW_VERSION"
        log_info "📚 Documentation: https://docs.terragonlabs.com/healing-guard"
    fi
}

# Run main function
main "$@"