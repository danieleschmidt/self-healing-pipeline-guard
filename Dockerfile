# Multi-stage production Dockerfile for Self-Healing Pipeline Guard
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Build stage - install all dependencies including dev
FROM base as builder

RUN poetry install --with dev && rm -rf $POETRY_CACHE_DIR

# Production stage - only runtime dependencies
FROM base as production

# Copy only production dependencies
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Security: remove unnecessary packages and files
RUN apt-get update && apt-get remove -y \
    build-essential \
    curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["poetry", "run", "uvicorn", "healing_guard.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM builder as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER appuser

# Development command
CMD ["poetry", "run", "uvicorn", "healing_guard.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage
FROM builder as testing

# Copy test files
COPY tests/ tests/

# Run tests
RUN poetry run pytest tests/ --cov=healing_guard --cov-report=xml

# Security scanning stage
FROM builder as security

# Install security tools
RUN poetry run bandit -r healing_guard/ -f json -o bandit-report.json || true
RUN poetry run safety check --json --output safety-report.json || true

# Production with security scanning
FROM production as production-secure

# Copy security reports from security stage
COPY --from=security /app/bandit-report.json /app/security/
COPY --from=security /app/safety-report.json /app/security/

# Final production image
FROM production as final

LABEL maintainer="Terragon Labs <info@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="Self-Healing Pipeline Guard - AI-powered CI/CD failure detection and remediation"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/self-healing-pipeline-guard"
LABEL org.opencontainers.image.documentation="https://docs.terragonlabs.com/healing-guard"
LABEL org.opencontainers.image.licenses="MIT"

# Set final working directory
WORKDIR /app

# Final health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Production command with proper signal handling
CMD ["poetry", "run", "gunicorn", "healing_guard.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100"]