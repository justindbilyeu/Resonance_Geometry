# Contributing to Resonance Geometry

Thank you for your interest in contributing! This document provides guidelines
for different types of contributions.

## Types of Contributions

### 1. Bug Reports
- Use GitHub Issues with tag `bug`
- Include: minimal example, expected vs actual behavior, environment details
- Check existing issues first

### 2. Feature Requests
- Use GitHub Issues with tag `enhancement`
- Describe: use case, proposed solution, alternatives considered

### 3. Code Contributions
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Make changes with clear commit messages
- Add tests for new functionality
- Ensure all tests pass (`pytest`)
- Submit Pull Request

### 4. Documentation
- Fix typos, clarify explanations, add examples
- Use clear, concise language
- Follow existing style and structure

### 5. Research Collaboration
- Open an issue with tag `collaboration`
- Describe: your background, what you'd like to work on, timeline

---

## Development Setup

```bash
# Clone repository
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
black src/ tests/
flake8 src/ tests/
```

-----

## Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **Docstrings**: Google style
- **Type hints**: Use for function signatures
- **Names**: Descriptive variable names (not single letters except in math formulas)

Example:

```python
def compute_curvature(
    activations: np.ndarray,
    k_neighbors: int = 15,
    metric: str = "cosine"
) -> float:
    """
    Compute curvature proxy from neural network activations.
    
    Args:
        activations: Hidden states array of shape (n_tokens, n_dims)
        k_neighbors: Number of neighbors for k-NN graph
        metric: Distance metric ('cosine', 'euclidean', etc.)
        
    Returns:
        Curvature estimate (scalar)
        
    Raises:
        ValueError: If k_neighbors >= n_tokens
    """
    # Implementation
    pass
```

-----

## Testing

- Write tests for all new functions
- Use `pytest` framework
- Aim for >80% code coverage
- Include both unit tests and integration tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/resonance_geometry

# Run specific test file
pytest tests/test_hallucination.py
```

-----

## Commit Messages

Use clear, descriptive commit messages:

**Good**:

- `Add Laplacian curvature extraction for LLM activations`
- `Fix off-by-one error in phase boundary calculation`
- `Update README with installation instructions`

**Bad**:

- `fix bug`
- `updates`
- `wip`

-----

## Pull Request Process

1. **Before submitting**:
- Ensure tests pass
- Update documentation
- Add entry to CHANGELOG.md (if exists)
1. **PR description should include**:
- What: Summary of changes
- Why: Motivation and context
- How: Technical approach (if non-obvious)
- Testing: How you verified it works
1. **Review process**:
- Maintainer will review within 1 week
- Address feedback in new commits
- Once approved, maintainer will merge

-----

## Questions?

Open an issue with tag `question` or reach out via [contact method].

-----

*Thank you for contributing to Resonance Geometry!*

