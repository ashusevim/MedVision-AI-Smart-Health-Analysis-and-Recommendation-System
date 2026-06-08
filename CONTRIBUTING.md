# Contributing to MedVision-AI

First off, thank you for considering contributing to MedVision-AI! It is people like you who make MedVision-AI a great tool for the medical AI community.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)

---

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socioeconomic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

### Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [conduct@medvision-ai.dev](mailto:conduct@medvision-ai.dev). All complaints will be reviewed and investigated promptly and fairly.

---

## How Can I Contribute?

### Reporting Bugs

Bug reports help us improve MedVision-AI. When filing a bug report, please:

1. **Check existing issues** to avoid duplicates
2. Use the **Bug Report template** when creating a new issue
3. Include **clear reproduction steps**, expected vs. actual behavior
4. Provide **environment details** (OS, Python version, CUDA version, etc.)
5. Attach **relevant logs or screenshots**

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. **Check existing issues** for similar suggestions
2. Use the **Feature Request template**
3. Provide a **clear use case** and explain why it would benefit users
4. Include **proposed solution** or API design if possible

### Contributing Code

1. **Fork** the repository to your GitHub account
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/medvision-ai.git
   cd medvision-ai
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/medvision-ai/medvision-ai.git
   ```
4. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Use these branch prefixes:
   - `feature/` — New features or enhancements
   - `fix/` — Bug fixes
   - `docs/` — Documentation changes
   - `refactor/` — Code refactoring
   - `test/` — Adding or updating tests
   - `chore/` — Maintenance tasks

5. **Make your changes** following our [coding standards](#coding-standards)
6. **Write or update tests** as described in [testing requirements](#testing-requirements)
7. **Commit** with [conventional commit messages](#commit-message-conventions)
8. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request** against the `main` branch

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- CUDA 12.1+ (for GPU support)
- Docker & Docker Compose (for containerized development)

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/medvision-ai.git
cd medvision-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=medvision_ai --cov-report=html

# Run specific test file
pytest tests/unit/test_imaging_service.py

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Running Linters

```bash
# Run all linters
make lint

# Or individually:
ruff check .
mypy medvision_ai/
black --check .
```

### Starting Development Services

```bash
# Start Redis and other infra services
docker-compose up -d redis postgres mlflow

# Start the API in development mode
uvicorn medvision_ai.main:app --reload --port 8000
```

---

## Coding Standards

### Python Style

We follow **PEP 8** with the following specifics:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **String quotes**: Double quotes preferred (Black default)
- **Import order**: Standard library → Third-party → Local (sorted with `isort`)

### Type Hints

All public functions and methods **must** include type hints:

```python
from typing import Optional

def analyze_image(
    image_path: str,
    modality: str = "xray",
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    """Analyze a medical image and return classification probabilities.

    Args:
        image_path: Path to the medical image file.
        modality: Imaging modality (xray, ct, mri, etc.).
        confidence_threshold: Minimum confidence for positive predictions.

    Returns:
        Dictionary mapping condition names to confidence scores.

    Raises:
        ValueError: If the image format is unsupported.
        FileNotFoundError: If the image file does not exist.
    """
    ...
```

### Docstrings

We use **Google-style docstrings** for all public modules, classes, functions, and methods:

```python
class RiskScorer:
    """Compute patient risk scores using ensemble models.

    This class orchestrates multiple risk models and combines their
    predictions using a weighted ensemble strategy.

    Attributes:
        models: Dictionary of loaded risk models.
        weights: Ensemble weights for each model.
        cache: Redis cache client for storing computed scores.
    """

    def compute_risk(self, patient_data: dict) -> RiskScore:
        """Compute a composite risk score for a patient.

        Args:
            patient_data: Dictionary containing patient demographics,
                comorbidities, vitals, and lab values.

        Returns:
            RiskScore object containing overall score, category,
            and per-model breakdown.

        Raises:
            ValidationError: If required patient fields are missing.
            ModelNotReadyError: If no risk models are loaded.
        """
        ...
```

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Modules | `snake_case` | `imaging_service.py` |
| Classes | `PascalCase` | `RiskScorer`, `MedViTClassifier` |
| Functions | `snake_case` | `analyze_symptoms()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_IMAGE_SIZE`, `DEFAULT_THRESHOLD` |
| Private methods | `_leading_underscore` | `_preprocess_image()` |
| Type aliases | `PascalCase` | `ImageTensor = torch.Tensor` |

### Code Organization

- One class per file for major components
- Keep functions focused — single responsibility principle
- Maximum function length: ~50 lines (excluding docstrings)
- Use dataclasses or Pydantic models for structured data
- Prefer composition over inheritance

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for all new code
- **Critical paths** (model inference, risk scoring, drug interactions): 95%+
- All bug fixes **must** include a regression test

### Test Structure

```
tests/
├── unit/                       # Unit tests (no external dependencies)
│   ├── test_imaging_service.py
│   ├── test_nlp_service.py
│   ├── test_risk_service.py
│   ├── test_treatment_service.py
│   └── test_models/
│       ├── test_medvit.py
│       └── test_risk_scorer.py
├── integration/                # Integration tests (may require services)
│   ├── test_api_endpoints.py
│   └── test_pipeline.py
└── conftest.py                 # Shared fixtures
```

### Writing Tests

Use **pytest** with the following conventions:

```python
import pytest
from medvision_ai.services.imaging_service import ImagingService


class TestImagingService:
    """Tests for the imaging analysis service."""

    @pytest.fixture
    def service(self) -> ImagingService:
        """Create an imaging service instance for testing."""
        return ImagingService(model_name="test-model")

    def test_analyze_xray_returns_probabilities(self, service: ImagingService):
        """Test that X-ray analysis returns valid probability distribution."""
        result = service.analyze("tests/fixtures/chest_xray.png", modality="xray")
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_analyze_invalid_modality_raises_error(self, service: ImagingService):
        """Test that invalid modality raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            service.analyze("tests/fixtures/image.png", modality="invalid")

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
    def test_confidence_threshold_filters_results(
        self, service: ImagingService, threshold: float
    ):
        """Test that confidence threshold correctly filters low-confidence predictions."""
        result = service.analyze(
            "tests/fixtures/chest_xray.png",
            confidence_threshold=threshold,
        )
        assert all(v >= threshold for v in result.values())
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or descriptive name
- Test methods: `test_<what>_<expected_outcome>` or `test_<behavior>`

### Async Tests

Use `pytest-asyncio` for async code:

```python
@pytest.mark.asyncio
async def test_async_image_analysis():
    """Test asynchronous image analysis endpoint."""
    response = await client.post("/api/v1/imaging/analyze", files={"file": image_file})
    assert response.status_code == 200
```

---

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Changes that do not affect code meaning (formatting, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding missing tests or correcting existing tests |
| `build` | Changes to build system or dependencies |
| `ci` | Changes to CI configuration |
| `chore` | Other changes that don't modify src or test files |
| `revert` | Reverts a previous commit |

### Scopes

Common scopes: `imaging`, `nlp`, `risk`, `treatment`, `api`, `models`, `db`, `config`, `deps`

### Examples

```
feat(imaging): add support for mammography analysis
fix(risk): correct cardiovascular risk score calculation for age > 80
docs(api): update Swagger documentation for v1 endpoints
refactor(nlp): extract shared NER pipeline into base class
perf(models): optimize ViT inference with torch.compile()
test(treatment): add regression test for drug interaction checker
ci: update GitHub Actions to use Python 3.12
chore(deps): bump torch from 2.1.0 to 2.2.0
```

### Breaking Changes

Indicate breaking changes with `!` after the type/scope or with a `BREAKING CHANGE:` footer:

```
feat(api)!: redesign imaging analysis request schema

BREAKING CHANGE: The `/api/v1/imaging/analyze` endpoint now requires
`modality` as a required field instead of auto-detecting. Update your
clients to include this field in all requests.
```

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the full check suite**:
   ```bash
   make lint
   make test
   make type-check
   ```

3. **Ensure all tests pass** and coverage meets thresholds

4. **Update documentation** if your change affects public APIs

### PR Template

When creating a PR, please include:

1. **Description**: What does this PR do and why?
2. **Related Issue**: Link to any related issues (e.g., "Closes #123")
3. **Type of Change**: Bug fix / New feature / Breaking change / Documentation
4. **Testing**: How was this tested? What tests were added?
5. **Checklist**:
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex logic
   - [ ] Documentation updated
   - [ ] No new warnings generated
   - [ ] Tests added and passing
   - [ ] All CI checks pass

### Review Process

1. **Automated checks** must pass (CI, linting, type checking)
2. At least **one approving review** from a maintainer is required
3. Changes requested by reviewers should be addressed before merge
4. PRs are squash-merged to maintain a clean history

### Review Criteria

Reviewers will evaluate:

- **Correctness**: Does the code do what it claims?
- **Design**: Is the approach sound? Does it fit the architecture?
- **Readability**: Is the code clear and well-documented?
- **Testing**: Are there sufficient tests with good coverage?
- **Performance**: Are there any performance regressions?
- **Security**: Are there any security concerns (especially for medical data)?
- **Breaking Changes**: Are breaking changes properly documented?

---

## Issue Reporting Guidelines

### Bug Reports

Please use the Bug Report template and include:

```markdown
**Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Start the server with `uvicorn medvision_ai.main:app`
2. Send a POST request to `/api/v1/imaging/analyze` with...
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: Ubuntu 22.04
- Python: 3.11.5
- PyTorch: 2.1.0
- CUDA: 12.1
- MedVision-AI: v0.1.0

**Additional Context**
Logs, screenshots, or other relevant information.
```

### Feature Requests

Please use the Feature Request template and include:

```markdown
**Problem Statement**
A clear description of the problem this feature would solve.

**Proposed Solution**
Describe your proposed solution or API design.

**Alternatives Considered**
Any alternative solutions you've considered.

**Additional Context**
Any other context, mockups, or references.

**Would you be willing to submit a PR?**
Yes / No
```

### Issue Labels

| Label | Description |
|---|---|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Improvements or additions to documentation |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |
| `security` | Security-related issue |
| `performance` | Performance-related issue |
| `breaking-change` | Breaking change in API or behavior |

---

Thank you for contributing to MedVision-AI! Your efforts help make medical AI more accessible and reliable for everyone.
