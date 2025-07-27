#!/bin/bash

# Post-create script for development container setup
set -e

echo "🚀 Setting up Self-Healing Pipeline Guard development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip poetry
poetry config virtualenvs.create false
poetry install --with dev,test,docs

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup database
echo "🗄️ Setting up development database..."
python scripts/setup_db.py

# Generate API documentation
echo "📚 Generating API documentation..."
python scripts/generate_docs.py

# Install additional development tools
echo "🔧 Installing additional tools..."
npm install -g @commitlint/cli @commitlint/config-conventional

# Setup git configuration for development
echo "⚙️ Configuring git for development..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {logs,tmp,data,exports,uploads}
mkdir -p {tests/unit,tests/integration,tests/e2e}
mkdir -p {docs/api,docs/guides,docs/runbooks}

# Set permissions
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Validate environment
echo "✅ Validating environment setup..."
python --version
poetry --version
pre-commit --version
docker --version

echo "🎉 Development environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  make dev          # Start development server"
echo "  make test         # Run test suite"
echo "  make lint         # Run code quality checks"
echo "  make docs         # Generate documentation"
echo ""