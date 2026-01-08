# AIDTM - AI-Driven Train Maintenance System

## 🚂 Overview

AIDTM (AI-Driven Train Maintenance System) is a comprehensive computer vision and AI system designed for automated railway inspection and maintenance. The system combines multiple AI models for vehicle detection, damage assessment, image enhancement, and intelligent analysis.

## 🏗️ Architecture

### Core Components

1. **IronSight Dashboard** - Main web interface built with Streamlit
2. **Multi-Model AI Pipeline** - Integrated computer vision models
3. **Image Enhancement** - NAFNet-based deblurring and restoration
4. **Vehicle Detection** - YOLO-based object detection
5. **Damage Assessment** - Specialized damage detection models
6. **Spectral Analysis** - Advanced image processing capabilities

### Key Features

- 🔍 **Real-time Vehicle Detection** - Automated detection of trains and components
- 🖼️ **Image Enhancement** - Advanced deblurring using NAFNet models
- 📊 **Damage Assessment** - AI-powered damage detection and classification
- 🎯 **Crop-First Processing** - Efficient processing with 85% computation reduction
- 📈 **Performance Monitoring** - Real-time performance metrics and optimization
- 🔧 **Modular Architecture** - Easily extensible and maintainable codebase

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager
- CUDA-capable GPU (optional, CPU fallback available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aidtm
   ```

2. **Install dependencies**
   ```bash
   cd ironsight_dashboard
   uv sync
   ```

3. **Download models** (optional - system will use mock models if not available)
   - Place NAFNet models in the root directory
   - Download YOLO models as needed

4. **Run the application**
   ```bash
   uv run python run_ironsight.py
   ```

5. **Access the dashboard**
   - Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
aidtm/
├── ironsight_dashboard/          # Main application
│   ├── src/                      # Source code
│   │   ├── app.py               # Main Streamlit app
│   │   ├── nafnet_fixed.py      # Fixed NAFNet implementation
│   │   ├── crop_first_nafnet.py # Crop-first processing
│   │   ├── multi_yolo_detector.py # YOLO detection
│   │   ├── mission_control.py   # Main control interface
│   │   └── ...                  # Other modules
│   ├── tests/                   # Property-based tests
│   ├── config/                  # Configuration files
│   ├── demo_data/              # Demo data generators
│   └── scripts/                # Utility scripts
├── iron-sight/                  # Legacy component
├── scripts/                     # Project-level scripts
├── .kiro/                      # Kiro IDE specifications
└── models/                     # AI model files (gitignored)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `ironsight_dashboard` directory:

```env
# Model paths
NAFNET_MODEL_PATH=../NAFNet-REDS-width64.pth
YOLO_MODEL_PATH=../yolo_sideview_damage_obb.pt

# Performance settings
DEVICE=cuda  # or cpu
BATCH_SIZE=1
MAX_IMAGE_SIZE=1024

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

The system supports multiple model backends with automatic fallback:

1. **NAFNet Models**
   - Fixed implementation (primary)
   - BasicSR implementation (fallback)
   - Standalone implementation (backup)
   - Mock implementation (testing)

2. **YOLO Models**
   - Custom trained models
   - Pre-trained models
   - Mock detection (testing)

## 🧪 Testing

### Run All Tests
```bash
cd ironsight_dashboard
uv run python -m pytest tests/ -v
```

### Test Specific Components
```bash
# Test NAFNet implementation
uv run python test_nafnet_fixed.py

# Test YOLO detection
uv run python -m pytest tests/test_yolo_properties.py

# Test integration
uv run python test_imports.py
```

### Property-Based Testing

The project uses property-based testing for robust validation:

```bash
# Run property tests
uv run python -m pytest tests/test_*_properties.py -v
```

## 📊 Performance

### Current Benchmarks (CPU)

| Image Size | Processing Time | Use Case |
|------------|----------------|----------|
| 64x64      | ~82ms         | Real-time crops |
| 256x256    | ~544ms        | Standard processing |
| 512x512    | ~2.3s         | High-quality analysis |

### GPU Acceleration

With proper GPU setup, expect 10-15x performance improvement:

- 64x64: ~5-10ms
- 256x256: ~20-40ms  
- 512x512: ~80-150ms

See [GPU Setup Guide](ironsight_dashboard/GPU_SETUP_GUIDE.md) for configuration.

## 🔍 Key Features

### Crop-First Processing

Innovative approach that processes only detected regions of interest:

- **85% computation reduction** vs full-frame processing
- **20ms target latency** per crop
- **Automatic padding** for context preservation
- **Parallel processing** of multiple crops

### Multi-Model Integration

Seamless integration of multiple AI models:

- **NAFNet** - Image deblurring and enhancement
- **YOLO** - Object detection and localization  
- **Custom Models** - Damage assessment and classification
- **Spectral Analysis** - Advanced image processing

### Robust Error Handling

Comprehensive error handling and fallbacks:

- **Graceful degradation** when models unavailable
- **Automatic device detection** (GPU/CPU)
- **Mock implementations** for testing
- **Detailed logging** and monitoring

## 📚 Documentation

- [Quick Start Guide](ironsight_dashboard/QUICK_START.md)
- [GPU Setup Guide](ironsight_dashboard/GPU_SETUP_GUIDE.md)
- [NAFNet Fixes Summary](ironsight_dashboard/NAFNET_FIXES_SUMMARY.md)
- [Video Troubleshooting](ironsight_dashboard/VIDEO_TROUBLESHOOTING.md)
- [Live Processor README](ironsight_dashboard/LIVE_PROCESSOR_README.md)

## 🛠️ Development

### Adding New Models

1. Create model wrapper in `src/`
2. Add configuration to `config/ironsight_config.yaml`
3. Implement property-based tests in `tests/`
4. Update integration in main pipeline

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **NAFNet loading fails**
   - Check model file exists
   - Verify PyTorch installation
   - See [NAFNet Fixes Summary](ironsight_dashboard/NAFNET_FIXES_SUMMARY.md)

2. **GPU not detected**
   - Install NVIDIA drivers
   - Install CUDA toolkit
   - Install PyTorch with CUDA support
   - See [GPU Setup Guide](ironsight_dashboard/GPU_SETUP_GUIDE.md)

3. **Import errors**
   - Run `uv sync` to install dependencies
   - Check Python version compatibility
   - Run `uv run python test_imports.py`

### Getting Help

- Check the documentation in each module
- Run the test suite to identify issues
- Review the troubleshooting guides
- Check the issue tracker

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NAFNet team for the deblurring models
- YOLO team for object detection
- Streamlit team for the web framework
- The open-source computer vision community

## 📈 Roadmap

- [ ] Real-time video processing
- [ ] Advanced damage classification
- [ ] Cloud deployment support
- [ ] Mobile app integration
- [ ] Enhanced reporting features
- [ ] Multi-language support

---

**Built with ❤️ for railway maintenance automation**