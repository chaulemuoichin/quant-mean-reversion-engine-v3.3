# Contributing to Mean Reversion Backtester

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/yourusername/mean-reversion-backtester.git
cd mean-reversion-backtester
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run Tests**
```bash
python -m unittest test_backtester -v
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters (not strict)
- Docstrings for all public functions/classes

### Testing
- Add tests for new features
- Maintain >80% code coverage
- Test edge cases and error handling
- Use descriptive test names: `test_<feature>_<scenario>_<expected_outcome>`

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Include usage examples

## Pull Request Process

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Write clean, documented code
- Add tests
- Update documentation

3. **Test Locally**
```bash
python -m unittest test_backtester -v
```

4. **Commit**
```bash
git add .
git commit -m "Add feature: brief description"
```

5. **Push and Create PR**
```bash
git push origin feature/your-feature-name
```
Then create a pull request on GitHub.

## Priority Areas

### High Priority
- [ ] Multi-asset portfolio mode with correlation matrix
- [ ] Walk-forward optimization framework
- [ ] Machine learning regime classifiers (HMM, LSTM)
- [ ] Kalman filter for dynamic parameter estimation

### Medium Priority
- [ ] Options overlay strategies (covered calls, protective puts)
- [ ] Monte Carlo simulation for strategy robustness
- [ ] Parameter sensitivity analysis tools
- [ ] Real-time paper trading connector

### Low Priority
- [ ] Web dashboard for results visualization
- [ ] Database backend for historical results
- [ ] Additional technical indicators (Ichimoku, Fibonacci)

## Bug Reports

Include:
- Python version
- Operating system
- Complete error traceback
- Minimal reproducible example
- Expected vs actual behavior

## Feature Requests

Describe:
- Use case and motivation
- Proposed API/interface
- Alternative solutions considered
- Potential implementation approach

## Code Review

Expect feedback on:
- Correctness and edge cases
- Performance implications
- Code clarity and maintainability
- Test coverage
- Documentation completeness

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or contact: chau.le@marquette.edu
