#!/bin/bash
# Development environment setup script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "Setting up development environment for Self-Healing Pipeline Guard..."

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    log_info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    log_info "Poetry already installed"
fi

# Install Python dependencies
log_info "Installing Python dependencies..."
poetry install --with dev,test,docs

# Install pre-commit hooks
log_info "Installing pre-commit hooks..."
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg

# Setup database
log_info "Setting up database..."
if [ -f "alembic.ini" ]; then
    poetry run alembic upgrade head
else
    log_warn "No alembic.ini found, skipping database setup"
fi

# Install Node.js dependencies for documentation
if [ -f "package.json" ]; then
    log_info "Installing Node.js dependencies..."
    npm install
else
    log_info "No package.json found, skipping Node.js setup"
fi

# Create necessary directories
log_info "Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p tmp
mkdir -p .coverage

# Set permissions
chmod +x scripts/*.sh

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    log_warn "Please update .env with your configuration"
fi

log_info "Development environment setup complete!"
log_info "You can now run 'make dev' to start the development server"