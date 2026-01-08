---
inclusion: always
---

# Technology Stack and Tooling Standards

## Python Environment and Dependency Management

**CRITICAL: Use `uv` exclusively for ALL Python environment and dependency management.**

### Mandatory Rules

1. **NEVER use pip, pip-tools, poetry, conda, or any other package manager**
2. **ALWAYS use `uv` for all Python operations:**
   - Installing packages: `uv add <package>`
   - Removing packages: `uv remove <package>`
   - Running scripts: `uv run <script>`
   - Running commands: `uv run <command>`
   - Creating virtual environments: `uv venv`
   - Syncing dependencies: `uv sync`
   - Locking dependencies: `uv lock`

### HTTP Client Library Standards

**CRITICAL: Use `niquests` exclusively for ALL HTTP operations.**

#### Mandatory HTTP Library Rules

1. **NEVER use requests, httpx, urllib, or any other HTTP library**
2. **ALWAYS use `niquests` for all HTTP operations:**
   - API requests and responses
   - Metadata fetching from IP webcams
   - Frame fetching from HTTP streams
   - Authentication with IP webcam servers
   - Status checks and health monitoring

#### Why niquests?

- **Performance**: Built on top of urllib3 with HTTP/2 and HTTP/3 support
- **Modern**: Async/await support and better connection pooling
- **Compatibility**: Drop-in replacement for requests with better performance
- **Reliability**: Better error handling and timeout management
- **Future-proof**: Supports latest HTTP standards

#### Usage Examples

```python
import niquests

# Basic GET request
response = niquests.get('http://192.168.1.100:8080/status')

# POST with JSON data
response = niquests.post('http://192.168.1.100:8080/api/settings', 
                        json={'quality': 'high'})

# Session for persistent connections (recommended for IP webcams)
session = niquests.Session()
response = session.get('http://192.168.1.100:8080/video')
```

### Naming Convention Override

**CRITICAL: Ignore the 's' prefix naming rule for this project.**

Standard Python naming conventions apply:
- Use descriptive, clear names regardless of prefix
- Prioritize readability and functionality over arbitrary naming rules
- Focus on meaningful variable and function names

### Common Operations

#### Installing Dependencies
```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Add with version constraint
uv add "package-name>=1.0.0"

# Add HTTP and video processing dependencies
uv add niquests opencv-python
```

#### Running Code
```bash
# Run a Python script
uv run python script.py

# Run a module
uv run python -m module_name

# Run pytest
uv run pytest

# Run with specific arguments
uv run python -m pytest tests/ -v
```

#### Managing Environment
```bash
# Sync environment with pyproject.toml
uv sync

# Lock dependencies (creates/updates uv.lock)
uv lock

# Install from lock file
uv sync --frozen
```

#### Project Initialization
```bash
# Initialize new project
uv init

# Add Python version requirement
uv python pin 3.11
```

### Platform-Specific Dependency Resolution

**CRITICAL: Configure uv to resolve dependencies only for the current platform to avoid cross-platform conflicts.**

#### Windows AMD64 Platform Requirements

**MANDATORY: All dependencies MUST strictly target Windows AMD64 (x86_64) architecture.**

When working on Windows AMD64, add this to your `pyproject.toml`:

```toml
[tool.uv]
# Only resolve dependencies for Windows AMD64
environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]
```

**Environment Markers for Windows AMD64:**
- `sys_platform == 'win32'` - Ensures Windows platform
- `platform_machine == 'AMD64'` - Ensures x86_64 architecture (NOT ARM64)

This prevents uv from:
- Resolving dependencies for macOS (darwin) or Linux
- Installing ARM64 packages on AMD64 Windows machines
- Cross-platform conflicts with packages like PyTorch and torchvision

**When adding platform-specific packages (torch, torchvision, etc.):**
```bash
# CORRECT: Explicitly specify platform and architecture
uv add torchvision --extra-index-url https://download.pytorch.org/whl/cpu --platform windows --arch x86_64

# WRONG: Letting uv auto-detect (may install ARM64 on AMD64)
uv add torchvision
```

### Mandatory Development Dependencies

**CRITICAL: The following packages MUST always be present in development dependencies:**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "hypothesis>=6.82.0",
]
```

**Before running any tests:**
1. Ensure pytest is installed: `uv add --dev pytest`
2. Sync the environment: `uv sync`
3. Run tests with: `uv run pytest`

**If pytest fails to spawn**, run:
```bash
uv add --dev pytest hypothesis pytest-asyncio
uv sync
```

### Why uv?

- **Speed**: 10-100x faster than pip
- **Reliability**: Deterministic dependency resolution
- **Modern**: Built in Rust, designed for modern Python workflows
- **Compatibility**: Works with standard pyproject.toml
- **Lock files**: Automatic uv.lock generation for reproducible builds
- **Platform-specific resolution**: Can target specific platforms to avoid cross-platform conflicts

### Integration with This Project

This project uses `uv` for all dependency management. The `pyproject.toml` file defines all dependencies, and `uv.lock` ensures reproducible installations across environments.

**Automatic Locking**: A post-save hook automatically runs `uv lock` whenever `pyproject.toml` is modified, ensuring the lock file stays synchronized.

### Troubleshooting

If you encounter dependency issues:
1. Run `uv lock` to regenerate the lock file
2. Run `uv sync` to synchronize the environment
3. Check `uv.lock` for resolved versions
4. Never manually edit `uv.lock`

#### Windows Torch DLL Loading Issues

**CRITICAL: Windows Access Violations with PyTorch must be resolved via forced re-indexing.**

If you encounter "Windows fatal exception: access violation" when importing torch:

1. **Delete the virtual environment completely:**
   ```bash
   rmdir /s /q .venv
   ```

2. **Clear uv cache (optional but recommended):**
   ```bash
   uv cache clean
   ```

3. **Specify the correct PyTorch index for Windows:**
   
   For CPU-only (recommended for testing):
   ```bash
   uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
   
   For CUDA 11.8:
   ```bash
   uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   
   For CUDA 12.1:
   ```bash
   uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Sync the environment:**
   ```bash
   uv sync
   ```

**Root Cause**: Windows requires platform-specific PyTorch binaries. The default PyPI index may provide incompatible builds. Always use the official PyTorch index URLs for Windows installations.

**Prevention**: When adding torch to a new Windows project, always specify the index URL from the start.

### Forbidden Commands

❌ `pip install`
❌ `pip freeze`
❌ `poetry add`
❌ `conda install`
❌ `pipenv install`

✅ `uv add`
✅ `uv run`
✅ `uv sync`
✅ `uv lock`
