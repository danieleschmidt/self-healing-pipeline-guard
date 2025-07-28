# Development Guide

## Quick Setup

```bash
# Clone and setup
git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
cd self-healing-pipeline-guard
poetry install --with dev,test,docs

# Start development environment
docker-compose -f docker-compose.dev.yml up -d
```

## Development Commands

```bash
# Run tests
make test

# Code quality checks
make lint

# Build and serve docs
make docs-serve
```

## Architecture Overview

- **API Layer**: FastAPI endpoints in `healing_guard/api/`
- **Core Logic**: Business logic in `healing_guard/core/`
- **ML Models**: Machine learning components in `healing_guard/ml/`
- **Integrations**: CI/CD platform connectors in `healing_guard/integrations/`

## Resources

- [Contributing Guide](../CONTRIBUTING.md)
- [Architecture Decisions](adr/)
- [Getting Started](getting-started/overview.md)
- [Python Development Best Practices](https://docs.python.org/3/tutorial/)