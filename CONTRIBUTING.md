# Contributing to AIDTM

Thank you for your interest in contributing to the AI-Driven Train Maintenance System (AIDTM)! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- UV package manager
- Git
- CUDA-capable GPU (optional, CPU fallback available)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aidtm.git
   cd aidtm
   ```

2. **Set up development environment**
   ```bash
   cd ironsight_dashboard
   uv sync
   ```

3. **Run tests to verify setup**
   ```bash
   uv run python test_imports.py
   uv run python test_nafnet_fixed.py
   uv run python -m pytest tests/ -v
   ```

## 🛠️ Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run import tests
uv run python test_imports.py

# Run NAFNet tests
uv run python test_nafnet_fixed.py

# Run property-based tests
uv run python -m pytest tests/ -v

# Run integration tests
uv run python tests/test_integration.py
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description

- Detailed description of changes
- Any breaking changes
- References to issues"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Code Style Guidelines

### Python Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Example Code Style

```python
def process_image(
    image: np.ndarray,
    model_config: Dict[str, Any]
) -> Tuple[np.ndarray, float]:
    """
    Process an image through the AI pipeline.
    
    Args:
        image: Input image as numpy array (H, W, 3) in BGR format
        model_config: Configuration dictionary for model parameters
        
    Returns:
        Tuple of (processed_image, processing_time_ms)
        
    Raises:
        ValueError: If image format is invalid
        ModelError: If model processing fails
    """
    start_time = time.time()
    
    # Validate input
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H, W, 3) format")
    
    # Process image
    processed = apply_model(image, model_config)
    
    processing_time = (time.time() - start_time) * 1000
    return processed, processing_time
```

### Testing Guidelines

- Write property-based tests for core functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Aim for high test coverage

### Example Test

```python
def test_image_processing_preserves_dimensions():
    """Test that image processing preserves input dimensions."""
    # Arrange
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    config = {"device": "cpu", "batch_size": 1}
    
    # Act
    result, _ = process_image(test_image, config)
    
    # Assert
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
```

## 🧪 Testing Requirements

### Required Tests

All contributions must include appropriate tests:

1. **Unit Tests** - Test individual functions and classes
2. **Property-Based Tests** - Test universal properties using Hypothesis
3. **Integration Tests** - Test component interactions
4. **Performance Tests** - Verify performance requirements

### Test Categories

- **Core Functionality**: NAFNet, YOLO, image processing
- **Error Handling**: Graceful failure and recovery
- **Performance**: Latency and throughput requirements
- **Compatibility**: Different Python versions and platforms

## 📚 Documentation

### Required Documentation

- Update README.md if adding new features
- Add docstrings to all public functions
- Update configuration examples
- Add troubleshooting information if needed

### Documentation Style

- Use clear, concise language
- Provide code examples
- Include performance implications
- Document error conditions

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Verify the bug with latest version
3. Test with minimal reproduction case

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.10.0]
- AIDTM version: [e.g., v1.0.0]
- GPU: [e.g., NVIDIA RTX 3060 or CPU only]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🏗️ Architecture Guidelines

### Adding New Models

1. Create model wrapper in `src/`
2. Implement standard interface
3. Add configuration options
4. Include error handling and fallbacks
5. Write comprehensive tests
6. Update documentation

### Model Interface Example

```python
class ModelInterface:
    """Standard interface for all AI models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """Load the model from checkpoint."""
        raise NotImplementedError
    
    def process(self, input_data: Any) -> Any:
        """Process input through the model."""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        raise NotImplementedError
```

## 🔄 Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Tag created
- [ ] Release notes written

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Other unprofessional conduct

## 📞 Getting Help

- **Documentation**: Check the README and guides first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to AIDTM! 🚂✨