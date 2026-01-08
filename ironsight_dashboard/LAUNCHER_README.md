# IronSight Command Center - Launcher

This directory contains unified launcher scripts to run the complete IronSight Command Center system.

## Quick Start

### Windows
```bash
# Double-click or run from command prompt:
run_ironsight.bat

# Or with options:
run_ironsight.bat --dev --port 8080
```

### Linux/macOS
```bash
# Make executable (first time only):
chmod +x run_ironsight.sh

# Run:
./run_ironsight.sh

# Or with options:
./run_ironsight.sh --dev --port 8080
```

### With UV (Recommended)
```bash
# Install uv first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run - dependencies managed automatically
python run_ironsight.py
```

### Without UV (Fallback)
```bash
# Install dependencies manually
pip install streamlit torch torchvision opencv-python numpy pillow transformers accelerate

# Run
python run_ironsight.py
```

## Options

- `--dev` - Development mode with auto-reload
- `--cpu` - Force CPU-only mode (disable GPU)
- `--port PORT` - Set Streamlit port (default: 8501)
- `--host HOST` - Set host address (default: 0.0.0.0)
- `--no-browser` - Don't open browser automatically
- `--debug` - Enable debug logging
- `--help` - Show help message

## Examples

```bash
# Standard production mode
python run_ironsight.py

# Development mode with auto-reload
python run_ironsight.py --dev

# CPU-only mode on custom port
python run_ironsight.py --cpu --port 8080

# Debug mode without opening browser
python run_ironsight.py --debug --no-browser
```

## What It Does

The launcher performs the following:

1. **System Checks**
   - Python version (3.10-3.12 required)
   - UV package manager availability (optional but recommended)
   - Required packages (streamlit, torch, etc.)
   - CUDA/GPU availability
   - Model file locations
   - Port availability

2. **Environment Setup**
   - Sets CUDA visibility based on --cpu flag
   - Configures Streamlit settings
   - Loads .env file if present
   - Sets project paths

3. **Service Startup**
   - Uses `uv run` if UV is available (faster, better dependency management)
   - Falls back to standard `python -m streamlit` if UV not available
   - Monitors process health
   - Opens browser (unless --no-browser)

4. **Monitoring**
   - Watches for process failures
   - Handles graceful shutdown on Ctrl+C
   - Logs all activities

## Dashboard Access

Once started, the dashboard will be available at:
- **Local**: http://localhost:8501 (or your custom port)
- **Network**: http://YOUR_IP:8501 (accessible from other devices)

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   python run_ironsight.py --port 8080
   ```

2. **CUDA out of memory**
   ```bash
   python run_ironsight.py --cpu
   ```

3. **UV not available (optional)**
   ```bash
   # Install UV for better performance
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or continue with regular Python
   python run_ironsight.py
   ```

4. **Missing packages (without UV)**
   ```bash
   pip install streamlit torch torchvision opencv-python numpy pillow transformers accelerate
   ```

4. **Permission denied (Linux/macOS)**
   ```bash
   chmod +x run_ironsight.sh
   ```

### Log Files

Logs are written to:
- `logs/ironsight_launcher.log` - Launcher logs
- `logs/performance_incidents.jsonl` - Performance logs

### Environment Variables

You can create a `.env` file in this directory with:
```
STREAMLIT_PORT=8501
CUDA_VISIBLE_DEVICES=0
NAFNET_MODEL_PATH=../NAFNet-GoPro-width64.pth
```

## Architecture

The IronSight Command Center consists of:

- **Frontend**: Streamlit web dashboard with 3 tabs
  - Mission Control: Real-time processing
  - Restoration Lab: Interactive image restoration
  - Semantic Search: Natural language search

- **Backend**: Integrated Python services
  - IronSight Engine: Core AI pipeline
  - Model Manager: Neural network loading/optimization
  - Performance Monitor: Real-time metrics
  - Error Handler: Graceful degradation

- **Models**: 8 neural networks
  - Gatekeeper: Pre-filtering
  - SCI Enhancer: Low-light enhancement
  - YOLO (3x): Object detection
  - NAFNet: Deblurring
  - SmolVLM: Vision-language model
  - SigLIP: Semantic search

## Development

For development work:

```bash
# Start in development mode
python run_ironsight.py --dev

# Enable debug logging
python run_ironsight.py --dev --debug

# Use CPU only for testing
python run_ironsight.py --dev --cpu
```

Development mode enables:
- Auto-reload on file changes
- Detailed logging
- Non-headless Streamlit mode
- Browser auto-opening

## Production Deployment

For production deployment:

```bash
# Standard production mode
python run_ironsight.py --host 0.0.0.0 --port 8501

# With systemd service (Linux)
sudo systemctl start ironsight

# With Docker
docker-compose up -d
```

Production mode features:
- Headless operation
- Optimized performance
- Error recovery
- Health monitoring