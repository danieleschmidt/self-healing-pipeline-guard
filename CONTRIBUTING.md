# Contributing to Self-Healing Pipeline Guard

Thank you for your interest in contributing to Self-Healing Pipeline Guard! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@terragonlabs.com](mailto:conduct@terragonlabs.com).

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Docker and Docker Compose
- Git
- Node.js 18+ (for documentation and tooling)

### First-Time Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/self-healing-pipeline-guard.git
   cd self-healing-pipeline-guard
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/terragon-labs/self-healing-pipeline-guard.git
   ```

4. **Set up the development environment**:
   ```bash
   # Install dependencies
   poetry install --with dev,test,docs
   
   # Install pre-commit hooks
   poetry run pre-commit install
   
   # Start development services
   docker-compose -f docker-compose.dev.yml up -d
   ```

5. **Verify setup**:
   ```bash
   # Run tests
   poetry run pytest
   
   # Check code quality
   poetry run black --check .
   poetry run flake8 .
   poetry run mypy .
   
   # Verify API is running
   curl http://localhost:8000/health
   ```

## Development Setup

### Development Environment

We provide multiple ways to set up your development environment:

#### Option 1: DevContainer (Recommended)

If you use VS Code, open the project in a DevContainer for a fully configured environment:

1. Install the "Remote - Containers" extension
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted

#### Option 2: Local Development

Follow the first-time setup instructions above for local development.

#### Option 3: GitHub Codespaces

Click the "Code" button on GitHub and select "Open with Codespaces" for a cloud development environment.

### Project Structure

```
self-healing-pipeline-guard/
├── healing_guard/          # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core business logic
│   ├── integrations/      # CI/CD platform integrations
│   ├── ml/                # Machine learning models
│   └── services/          # Service layer
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── e2e/               # End-to-end tests
│   └── performance/       # Performance tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── config/                # Configuration files
└── deployment/            # Deployment configurations
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for newcomers
- Check the project roadmap for planned features
- Discuss significant changes in an issue before starting work

### 2. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 3. Make Changes

- Follow the coding standards outlined below
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Check code quality
make lint

# Run security checks
make security
```

## Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **String quotes**: Double quotes preferred
- **Imports**: Organized with isort
- **Type hints**: Required for all public functions
- **Docstrings**: Google style format

Example:

```python
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


async def analyze_failure(
    failure_id: str,
    logs: List[str],
    context: Optional[dict] = None
) -> AnalysisResult:
    """Analyze a pipeline failure and determine remediation strategy.
    
    Args:
        failure_id: Unique identifier for the failure
        logs: List of log lines from the failed pipeline
        context: Optional additional context information
        
    Returns:
        Analysis result containing failure classification and recommendations
        
    Raises:
        AnalysisError: If the failure cannot be analyzed
    """
    logger.info(f"Analyzing failure {failure_id}")
    
    if not logs:
        raise AnalysisError("No logs provided for analysis")
    
    # Implementation here
    return AnalysisResult(failure_id=failure_id, classification="test_failure")
```

### Code Quality Tools

The following tools are automatically run on every commit:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security analysis

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add webhook endpoint for GitLab integration

fix(ml): resolve memory leak in model inference

docs: update installation instructions for Docker setup

test(integration): add tests for GitHub Actions integration
```

## Testing Guidelines

### Test Types

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes
   - Mock external dependencies
   - Fast execution (< 1 second per test)

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Use real dependencies where possible
   - Test API endpoints with database

3. **End-to-End Tests** (`tests/e2e/`)
   - Test complete user workflows
   - Use Docker Compose environment
   - Test actual CI/CD platform integrations

4. **Performance Tests** (`tests/performance/`)
   - Test system performance under load
   - Measure response times and throughput
   - Identify performance regressions

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

from healing_guard.core.failure_detector import FailureDetector
from healing_guard.models.pipeline import PipelineFailure


class TestFailureDetector:
    """Test suite for FailureDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a FailureDetector instance for testing."""
        return FailureDetector()
    
    @pytest.fixture
    def sample_failure(self):
        """Create sample failure data."""
        return PipelineFailure(
            id="test-123",
            platform="github",
            repository="test/repo",
            logs=["Error: Test failed", "Exit code: 1"],
            exit_code=1
        )
    
    @pytest.mark.asyncio
    async def test_detect_failure_success(self, detector, sample_failure):
        """Test successful failure detection."""
        result = await detector.detect(sample_failure)
        
        assert result is not None
        assert result.failure_type == "test_failure"
        assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_detect_failure_with_ml_model(self, detector, sample_failure):
        """Test failure detection with ML model enhancement."""
        with patch.object(detector, 'ml_model') as mock_model:
            mock_model.predict_proba.return_value = [[0.1, 0.9]]
            
            result = await detector.detect(sample_failure)
            
            assert result.confidence > 0.8
            mock_model.predict_proba.assert_called_once()
```

### Test Coverage

- Maintain minimum 80% test coverage
- New features must include comprehensive tests
- Critical paths require 95%+ coverage

## Documentation

### Documentation Types

1. **Code Documentation**
   - Docstrings for all public functions and classes
   - Inline comments for complex logic
   - Type hints for better IDE support

2. **API Documentation**
   - OpenAPI/Swagger documentation
   - Automatically generated from code
   - Include examples and use cases

3. **User Documentation**
   - User guides and tutorials
   - Installation and configuration
   - Troubleshooting guides

4. **Developer Documentation**
   - Architecture decisions
   - Contributing guidelines
   - Development setup

### Writing Documentation

- Use Markdown for all documentation
- Follow the [Microsoft Writing Style Guide](https://docs.microsoft.com/en-us/style-guide/welcome/)
- Include code examples where helpful
- Keep documentation up to date with code changes

### Building Documentation

```bash
# Build documentation locally
make docs

# Serve documentation for preview
make docs-serve

# Check for broken links
make test-links
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all tests and checks**:
   ```bash
   make check  # Runs all quality checks
   ```

3. **Push your branch**:
   ```bash
   git push origin your-branch
   ```

4. **Create a pull request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots for UI changes
   - Testing instructions

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Related Issues
Fixes #123
Related to #456

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented where necessary
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All checks pass

## Screenshots (if applicable)
Include screenshots for UI changes.

## Additional Notes
Any additional information or context.
```

## Review Process

### Review Criteria

Pull requests are reviewed for:

1. **Functionality**: Does it work as intended?
2. **Code Quality**: Is it readable and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is documentation updated?
5. **Performance**: Any performance impact?
6. **Security**: Are there security implications?

### Review Timeline

- **Initial review**: Within 2 business days
- **Follow-up reviews**: Within 1 business day
- **Merge**: After all checks pass and approval

### Addressing Review Comments

1. Make requested changes in new commits
2. Push changes to the same branch
3. Respond to comments explaining changes
4. Request re-review when ready

## Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Slack**: Join our community Slack workspace
- **Office Hours**: Weekly community office hours
- **Documentation**: Comprehensive guides and API reference

### Recognition

We recognize contributors through:

- Contributor spotlight in releases
- Contributor badges on GitHub
- Special recognition for significant contributions
- Speaking opportunities at conferences

### Mentorship

New contributors can request mentorship:

- Pair programming sessions
- Code review guidance
- Architecture discussions
- Career development advice

## Release Process

### Release Schedule

- **Major releases**: Quarterly
- **Minor releases**: Monthly
- **Patch releases**: As needed
- **Security releases**: Immediately

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Questions?

If you have questions about contributing, please:

1. Check the [FAQ](docs/FAQ.md)
2. Search existing [GitHub issues](https://github.com/terragon-labs/self-healing-pipeline-guard/issues)
3. Ask in [GitHub Discussions](https://github.com/terragon-labs/self-healing-pipeline-guard/discussions)
4. Contact the maintainers at [maintainers@terragonlabs.com](mailto:maintainers@terragonlabs.com)

---

Thank you for contributing to Self-Healing Pipeline Guard! 🚀