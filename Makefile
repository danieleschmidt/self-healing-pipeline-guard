# Self-Healing Pipeline Guard Makefile
# Comprehensive development and deployment automation

.PHONY: help install dev test lint format clean build deploy docs security
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.11
PIP := pip
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := self-healing-pipeline-guard
VERSION := $(shell poetry version -s)

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Self-Healing Pipeline Guard - Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# Environment Setup
install: ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(POETRY) install --with dev,test,docs
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(POETRY) install --only main
	@echo "$(GREEN)Production dependencies installed!$(RESET)"

setup: install ## Complete project setup (alias for install)
	@echo "$(GREEN)Project setup complete!$(RESET)"

# Development
dev: ## Start development server with hot reload
	@echo "$(BLUE)Starting development server...$(RESET)"
	$(POETRY) run uvicorn healing_guard.main:app --reload --host 0.0.0.0 --port 8000

dev-docker: ## Start development environment with Docker Compose
	@echo "$(BLUE)Starting development environment...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up --build

dev-stop: ## Stop development Docker environment
	@echo "$(BLUE)Stopping development environment...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml down

dev-logs: ## Show development logs
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f

shell: ## Open interactive shell in development environment
	$(POETRY) shell

# Database Operations
db-upgrade: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(POETRY) run alembic upgrade head

db-downgrade: ## Rollback last database migration
	@echo "$(BLUE)Rolling back database migration...$(RESET)"
	$(POETRY) run alembic downgrade -1

db-migration: ## Create new database migration
	@echo "$(BLUE)Creating new migration...$(RESET)"
	@read -p "Enter migration name: " name; \
	$(POETRY) run alembic revision --autogenerate -m "$$name"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all database data!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(POETRY) run alembic downgrade base; \
		$(POETRY) run alembic upgrade head; \
		echo "$(GREEN)Database reset complete!$(RESET)"; \
	else \
		echo "$(YELLOW)Database reset cancelled.$(RESET)"; \
	fi

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(RESET)"
	$(POETRY) run pytest

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(POETRY) run pytest tests/unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(POETRY) run pytest tests/integration -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running e2e tests...$(RESET)"
	$(POETRY) run pytest tests/e2e -v

test-ml: ## Run ML/AI tests
	@echo "$(BLUE)Running ML tests...$(RESET)"
	$(POETRY) run pytest tests/ -m ml -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(POETRY) run pytest --cov=healing_guard --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	$(POETRY) run pytest-watch

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(POETRY) run pytest --benchmark-only

# Code Quality
lint: ## Run all linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	$(POETRY) run black --check .
	$(POETRY) run isort --check-only .
	$(POETRY) run flake8 .
	$(POETRY) run mypy .
	$(POETRY) run ruff check .

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(RESET)"
	$(POETRY) run black .
	$(POETRY) run isort .
	$(POETRY) run ruff check --fix .

format: lint-fix ## Format code (alias for lint-fix)

typecheck: ## Run type checking
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(POETRY) run mypy .

# Security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(POETRY) run bandit -r healing_guard/
	$(POETRY) run safety check
	$(POETRY) run semgrep --config=auto healing_guard/

security-deps: ## Check dependencies for vulnerabilities
	@echo "$(BLUE)Checking dependency vulnerabilities...$(RESET)"
	$(POETRY) run safety check

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	$(POETRY) run mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	$(POETRY) run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(RESET)"
	$(POETRY) run mike deploy --push --update-aliases $(VERSION) latest

# Building and Packaging
build: ## Build the project
	@echo "$(BLUE)Building project...$(RESET)"
	$(POETRY) build

build-docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) build -t $(PROJECT_NAME):latest .

build-docker-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(RESET)"
	$(DOCKER) build -f Dockerfile.prod -t $(PROJECT_NAME):$(VERSION)-prod .

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

# Cleanup
clean: ## Clean up build artifacts and cache
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)Cleanup complete!$(RESET)"

clean-docker: ## Clean up Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f

# Utilities
requirements: ## Export requirements.txt from Poetry
	@echo "$(BLUE)Exporting requirements...$(RESET)"
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --with dev,test,docs --without-hashes

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(POETRY) update

check: lint typecheck security test ## Run all quality checks
	@echo "$(GREEN)All checks passed!$(RESET)"

ci: check test-coverage ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed!$(RESET)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(POETRY) run pre-commit run --all-files

version: ## Show current version
	@echo "Current version: $(VERSION)"

bump-patch: ## Bump patch version
	$(POETRY) version patch
	@echo "$(GREEN)Version bumped to: $(shell poetry version -s)$(RESET)"

bump-minor: ## Bump minor version
	$(POETRY) version minor
	@echo "$(GREEN)Version bumped to: $(shell poetry version -s)$(RESET)"

bump-major: ## Bump major version
	$(POETRY) version major
	@echo "$(GREEN)Version bumped to: $(shell poetry version -s)$(RESET)"

# Monitoring and Health
health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	$(POETRY) run python scripts/health_check.py

logs: ## Show application logs
	tail -f logs/app.log

metrics: ## Show application metrics
	@echo "$(BLUE)Application metrics:$(RESET)"
	$(POETRY) run python scripts/show_metrics.py

# Development Utilities
seed-data: ## Seed database with sample data
	@echo "$(BLUE)Seeding database with sample data...$(RESET)"
	$(POETRY) run python scripts/seed_data.py

benchmark: ## Run application benchmarks
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	$(POETRY) run python scripts/benchmark.py

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(RESET)"
	$(POETRY) run python scripts/profile.py

# Git hooks and automation
git-hooks: ## Install git hooks
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit install --hook-type commit-msg

release: ## Create a new release
	@echo "$(BLUE)Creating new release...$(RESET)"
	@read -p "Enter release type (patch|minor|major): " type; \
	$(POETRY) version $$type; \
	git add pyproject.toml; \
	git commit -m "chore: bump version to $(shell poetry version -s)"; \
	git tag -a v$(shell poetry version -s) -m "Release v$(shell poetry version -s)"; \
	echo "$(GREEN)Release v$(shell poetry version -s) created!$(RESET)"; \
	echo "$(YELLOW)Run 'git push origin main --tags' to publish$(RESET)"