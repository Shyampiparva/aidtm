# 🚂 IronSight Command Center - Quick Start Guide

## 🚀 One-Click Launch

### Windows Users
```bash
# Double-click this file:
run_ironsight.bat

# Or from Command Prompt/PowerShell:
run_ironsight.bat
```

### Linux/macOS Users
```bash
# Make executable (first time only):
chmod +x run_ironsight.sh

# Run:
./run_ironsight.sh
```

### Python Users (All Platforms)
```bash
python run_ironsight.py
```

## 🎯 What You Get

After running the launcher, you'll have:

1. **🌐 Web Dashboard** at http://localhost:8501
   - Mission Control: Real-time rail inspection
   - Restoration Lab: Image deblurring testing
   - Semantic Search: Natural language queries

2. **🤖 AI Models** (8 neural networks)
   - Automatic loading and optimization
   - GPU acceleration (if available)
   - Graceful fallback to CPU mode

3. **📊 Performance Monitoring**
   - Real-time FPS and latency metrics
   - GPU memory usage tracking
   - Model status indicators

## ⚙️ Common Options

```bash
# Development mode (auto-reload on changes)
python run_ironsight.py --dev

# CPU-only mode (if GPU issues)
python run_ironsight.py --cpu

# Custom port
python run_ironsight.py --port 8080

# Debug mode
python run_ironsight.py --debug
```

## 🔧 System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **UV**: Recommended for faster dependency management (optional)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional, will use CPU fallback)
- **Storage**: 5GB for models and dependencies

## 📦 Dependencies

The launcher will automatically detect and use `uv` if available for better performance and dependency management.

### With UV (Recommended)
```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Dependencies are managed via pyproject.toml
# No manual installation needed!
```

### Without UV (Fallback)
```bash
pip install streamlit torch torchvision opencv-python numpy pillow transformers accelerate
```

## 🎮 Using the Dashboard

### Mission Control Tab
1. Select video input (webcam, file, or RTSP stream)
2. Click "▶️ Start Processing"
3. Watch real-time AI detection overlays
4. Monitor performance metrics

### Restoration Lab Tab
1. Upload a blurry image
2. Click "Restore Image"
3. Compare before/after results
4. Download restored image

### Semantic Search Tab
1. Type natural language queries
2. Search inspection history
3. Find specific damage types
4. Browse results with thumbnails

## 🚨 Troubleshooting

### Port Already in Use
```bash
python run_ironsight.py --port 8080
```

### GPU Out of Memory
```bash
python run_ironsight.py --cpu
```

### UV Not Found
```bash
# Install UV for better performance (optional)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or continue with regular Python
python run_ironsight.py
```

### Missing Models
The system will show "Model Offline" badges and use fallback implementations.

### Permission Errors (Linux/macOS)
```bash
chmod +x run_ironsight.sh
sudo ./run_ironsight.sh  # if needed
```

## 🛑 Stopping the System

- Press **Ctrl+C** in the terminal
- Or close the terminal window
- The system will shutdown gracefully

## 📁 File Structure

```
ironsight_dashboard/
├── run_ironsight.py      # Main launcher (Python)
├── run_ironsight.bat     # Windows launcher
├── run_ironsight.sh      # Linux/macOS launcher
├── run_ironsight.ps1     # PowerShell launcher
├── src/                  # Source code
├── config/               # Configuration files
├── logs/                 # Log files
└── scripts/              # Legacy scripts (deprecated)
```

## 🔗 URLs After Startup

- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health
- **Metrics**: Available in the dashboard sidebar

## 💡 Tips

1. **First Run**: May take 2-3 minutes to download and load models
2. **GPU Memory**: Monitor usage in the performance dashboard
3. **Development**: Use `--dev` flag for auto-reload during development
4. **Production**: Use default settings for best performance
5. **Logs**: Check `logs/` directory for detailed information

## 🆘 Getting Help

1. Run with `--help` flag for all options
2. Check log files in `logs/` directory
3. Use `--debug` flag for verbose output
4. Ensure all system requirements are met

---

**Ready to inspect some rail wagons? 🚂✨**