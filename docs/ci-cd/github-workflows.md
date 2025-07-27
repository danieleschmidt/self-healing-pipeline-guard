# GitHub Actions Workflows

This document contains the GitHub Actions workflow templates that should be added to `.github/workflows/` directory to complete the SDLC automation setup.

> **Note**: These files need to be created manually due to GitHub's security restrictions on workflow files.

## Required Files

### 1. Continuous Integration (`ci.yml`)

Create `.github/workflows/ci.yml`:

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

jobs:
  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  code-quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Install dependencies
        run: poetry install --with dev,test

      - name: Run linting and type checking
        run: |
          poetry run black --check .
          poetry run isort --check-only .
          poetry run ruff check .
          poetry run mypy .

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: healing_guard_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Install dependencies
        run: poetry install --with dev,test

      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/healing_guard_test
          REDIS_URL: redis://localhost:6379/0
        run: |
          poetry run pytest tests/ -v --cov=healing_guard --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    name: Build and Package
    runs-on: ubuntu-latest
    needs: [security-scan, code-quality, test]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Build package
        run: poetry build

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: healing-guard:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 2. Continuous Deployment (`cd.yml`)

Create `.github/workflows/cd.yml`:

```yaml
name: Continuous Deployment

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging'
    environment:
      name: staging
      url: https://staging.healing-guard.terragonlabs.com
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to staging
        run: echo "Deploy to staging logic here"

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    if: github.event_name == 'release' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    environment:
      name: production
      url: https://healing-guard.terragonlabs.com
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to production
        run: echo "Deploy to production logic here"
```

### 3. Security Scanning (`security-scan.yml`)

Create `.github/workflows/security-scan.yml`:

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Install dependencies
        run: poetry install --with dev

      - name: Run security checks
        run: |
          poetry run safety check
          poetry run bandit -r . -f json
```

### 4. Dependency Updates (`dependency-update.yml`)

Create `.github/workflows/dependency-update.yml`:

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    name: Update Dependencies
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Update dependencies
        run: poetry update

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'chore: update dependencies'
          title: 'chore: Weekly dependency updates'
          body: |
            ## Automated Dependency Updates
            
            This PR contains automated updates to dependencies.
          branch: dependency-updates/weekly
          delete-branch: true
          labels: dependencies,automated
```

## Setup Instructions

1. **Create the workflow files**: Copy each of the above YAML configurations into separate files in the `.github/workflows/` directory.

2. **Configure secrets**: Add the following secrets to your GitHub repository:
   - `CODECOV_TOKEN` (for code coverage reporting)
   - Any deployment-specific secrets for staging/production

3. **Enable security features**:
   - Enable Dependabot in repository settings
   - Enable CodeQL code scanning
   - Configure branch protection rules

4. **Set up environments**:
   - Create `staging` and `production` environments in repository settings
   - Configure environment-specific secrets and protection rules

## Additional Considerations

- The workflows are designed to work with the existing project structure
- Security scanning results will appear in the Security tab
- Failed builds will block pull requests if branch protection is enabled
- The workflows support both Python 3.11 and 3.12 for testing compatibility

## Permissions Required

The GitHub App or token used must have the following permissions:
- `actions: write` - To run workflows
- `contents: write` - To create commits and releases
- `security-events: write` - To upload security scan results
- `pull-requests: write` - To create dependency update PRs