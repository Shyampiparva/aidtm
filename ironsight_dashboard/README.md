# IronSight Command Center

Production-grade real-time rail inspection dashboard integrating 5 neural networks into a unified Streamlit interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
  - [Mission Control](#mission-control)
  - [Restoration Lab](#restoration-lab)
  - [Semantic Search](#semantic-search)
- [Model Requirements](#model-requirements)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Targets](#performance-targets)
- [Hardware Requirements](#hardware-requirements)

## Overview

IronSight Command Center transitions from experimental models to a cohesive inspection workflow, leveraging existing trained models including NAFNet-Width64, SmolVLM2 forensic agent, and SigLIP semantic search.

### Architecture

Three-tier AI system:
1. **Tier 1**: Real-time ONNX models (Gatekeeper, SCI, YOLO×3, NAFNet) for 60 FPS processing
2. **Tier 2**: SmolVLM forensic agent for intelligent OCR fallback and damage assessment  
3. **Tier 3**: SigLIP semantic search for natural language querying of inspection history

### Key Innovations

- **Crop-First Strategy**: 85% computation reduction by processing only detected regions
- **SCI Enhancement**: 6x faster than Zero-DCE (~0.5ms vs ~3ms)
- **Graceful Degradation**: System works with missing models using "Model Offline" badges
- **Spectral Processing**: Red channel for OCR, saturation channel for damage detection

## Features

| Feature | Description |
|---------|-------------|
| Mission Control | Live video processing with real-time AI detection overlays |
| Restoration Lab | Interactive image enhancement testing with before/after comparisons |
| Semantic Search | Natural language search interface for historical inspection data |
| Multi-Model YOLO | Three specialized YOLO models for comprehensive detection |
| Smart Gatekeeper | Pre-filtering to skip non-wagon frames and optimize processing |
| Low-Light Enhancement | SCI-based enhancement for dark/night conditions |

## Installation

### Prerequisites

- Python 3.10-3.12
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Standard Installation

```bash
# Navigate to the ironsight_dashboard directory
cd aidtm/ironsight_dashboard

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check that all dependencies are installed
python -c "import streamlit; import torch; import cv2; print('All dependencies OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

```bash
# Start the dashboard
streamlit run src/app.py

# Or use the installed command
ironsight-dashboard
```

The dashboard will open in your default browser at `http://localhost:8501`.

## User Guide

### Mission Control

The Mission Control tab provides real-time wagon inspection with AI detection overlays.

#### Video Input Options

1. **Webcam**: Select camera index (0 for default camera)
2. **RTSP Stream**: Enter RTSP URL (e.g., `rtsp://192.168.1.100:554/stream`)
3. **Video File**: Upload MP4, AVI, MOV, or MKV files

#### Controls

- **▶️ Start Processing**: Begin real-time AI processing
- **⏹️ Stop Processing**: Halt processing and release video source
- **🔄 Refresh**: Update display and metrics

#### Performance Metrics

| Metric | Description |
|--------|-------------|
| FPS | Frames processed per second |
| Latency | Processing time per frame (ms) |
| Queue Depth | Frames waiting to be processed |
| Gatekeeper Skips | Frames skipped (no wagon detected) |
| GPU Memory | Current GPU memory usage |
| GPU Temp | GPU temperature (if available) |

#### Detection Overlay Colors

| Color | Detection Type |
|-------|---------------|
| 🔴 Red | Sideview Damage (dents, holes, rust, scratches) |
| 🟢 Green | Structure (doors, wheels, couplers, brakes) |
| 🔵 Cyan | Wagon Numbers (ID plates, serial numbers) |

### Restoration Lab

The Restoration Lab allows interactive testing of image restoration capabilities.

#### How to Use

1. **Upload Image**: Click the upload area or drag-and-drop a blurry image (JPG/PNG)
2. **View Comparison**: Use the slider to compare original vs. restored image
3. **Check Metrics**: Review processing time and quality improvements

#### Supported Formats

- JPEG/JPG
- PNG

#### Tips for Best Results

- Upload images with motion blur for best deblurring results
- Larger images may take longer to process
- The NAFNet model is optimized for GoPro-style motion blur

### Semantic Search

Search historical inspection data using natural language queries.

#### Example Queries

- "Show me rusted doors"
- "Wagons with damage"
- "Identification plates from last week"
- "Severe dents on wagon sides"

#### Filters

- **Date Range**: Filter results by inspection date
- **Confidence Threshold**: Minimum similarity score (0.0-1.0)
- **Result Limit**: Maximum number of results to display

#### Search Results

Each result displays:
- Thumbnail image
- Similarity score
- Wagon ID
- Timestamp
- Damage description (if applicable)

## Model Requirements

### Required Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| NAFNet-Width64 | `NAFNet-GoPro-width64.pth` | ~270MB | Motion deblurring |

### Optional Models

| Model | File | Purpose | Fallback Behavior |
|-------|------|---------|-------------------|
| Gatekeeper | `gatekeeper.onnx` | Pre-filtering | All frames processed |
| YOLO Sideview | `yolo_sideview_damage_obb.pt` | Damage detection | Mock detections |
| YOLO Structure | `yolo_structure_obb.pt` | Component detection | Mock detections |
| YOLO Wagon Number | `wagon_number_obb.pt` | ID plate detection | Mock detections |
| SmolVLM | HuggingFace download | OCR/Assessment | Skipped |
| SigLIP | HuggingFace download | Semantic search | Disabled |

### Model Paths

Models are searched in the following locations:
1. `ironsight_dashboard/models/`
2. `aidtm/` (parent directory)
3. Paths specified in `config/ironsight_config.yaml`

## Configuration

### Configuration File

Edit `config/ironsight_config.yaml` to customize settings:

```yaml
# Model paths
models:
  nafnet_path: "../NAFNet-GoPro-width64.pth"
  gatekeeper_path: "models/gatekeeper.onnx"
  yolo_sideview_path: "../yolo_sideview_damage_obb.pt"
  yolo_structure_path: "../yolo_structure_obb.pt"

# Performance settings
performance:
  target_fps: 60
  gatekeeper_timeout_ms: 0.5
  sci_timeout_ms: 0.5
  yolo_combined_timeout_ms: 20.0
  nafnet_timeout_ms: 20.0

# Memory optimization
memory:
  use_fp16: true
  smolvlm_quantization_bits: 8
  gpu_memory_fraction: 0.8

# UI settings
ui:
  theme: "dark_industrial"
  enable_performance_monitoring: true
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IRONSIGHT_CONFIG` | Path to config file | `config/ironsight_config.yaml` |
| `IRONSIGHT_MODELS_DIR` | Models directory | `models/` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |
| `STREAMLIT_SERVER_PORT` | Dashboard port | `8501` |

## Troubleshooting

### Common Issues

#### "CUDA out of memory"

**Symptoms**: Error message about GPU memory allocation failure

**Solutions**:
1. Reduce `gpu_memory_fraction` in config (e.g., 0.6)
2. Enable FP16 mode: `use_fp16: true`
3. Close other GPU-intensive applications
4. Use a GPU with more VRAM

#### "Model Offline" badges showing

**Symptoms**: Dashboard shows models as offline

**Solutions**:
1. Check model file paths in configuration
2. Verify model files exist and are not corrupted
3. Check console for specific loading errors
4. Ensure sufficient disk space for model loading

#### Video input not working

**Symptoms**: No video feed in Mission Control

**Solutions**:
1. **Webcam**: Check camera permissions and index
2. **RTSP**: Verify URL format and network connectivity
3. **File**: Ensure file format is supported (MP4, AVI, MOV, MKV)
4. Check OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

#### Slow processing / Low FPS

**Symptoms**: FPS below target, high latency

**Solutions**:
1. Enable GPU acceleration (CUDA)
2. Enable FP16 mode for faster inference
3. Reduce input resolution
4. Check for thermal throttling (GPU temperature)
5. Close background applications

#### Import errors on startup

**Symptoms**: Module not found errors

**Solutions**:
1. Reinstall dependencies: `uv sync` or `pip install -e .`
2. Check Python version (3.10-3.12 required)
3. Verify virtual environment is activated
4. Install missing optional dependencies

### Diagnostic Commands

```bash
# Check system info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check Streamlit
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"

# Run tests
pytest tests/ -v
```

### Getting Help

1. Check the console output for detailed error messages
2. Review the `logs/` directory for incident logs
3. Enable debug mode in configuration for verbose logging

## Performance Targets

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| Gatekeeper | <0.5ms | Per 64x64 thumbnail |
| SCI Enhancement | <0.5ms | Per frame |
| YOLO Combined | <20ms | All 3 models |
| NAFNet Deblurring | <20ms | Per crop |
| Total Pipeline | <16.7ms | For 60 FPS |

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4-core modern processor
- **RAM**: 8GB
- **GPU**: NVIDIA GPU with 4GB VRAM (CUDA 11.0+)
- **Storage**: 2GB free space

### Recommended Requirements

- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB
- **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
- **Storage**: SSD with 10GB free space

### Jetson Deployment

For NVIDIA Jetson deployment:
- Jetson AGX Orin recommended
- Enable 8-bit quantization for SmolVLM
- Use FP16 for all ONNX models
- Limit GPU memory fraction to 0.7

## Project Structure

```
ironsight_dashboard/
├── src/                          # Source code
│   ├── app.py                    # Main Streamlit application
│   ├── ironsight_engine.py       # Main orchestrator
│   ├── gatekeeper_model.py       # Pre-filtering classifier
│   ├── sci_enhancer.py           # Low-light enhancement
│   ├── multi_yolo_detector.py    # Three-model YOLO integration
│   ├── crop_first_nafnet.py      # Optimized deblurring
│   ├── smolvlm_integration.py    # Forensic agent wrapper
│   ├── siglip_integration.py     # Semantic search wrapper
│   ├── mission_control.py        # Live processing interface
│   ├── restoration_lab.py        # Interactive restoration UI
│   ├── semantic_search_ui.py     # Search interface
│   ├── spectral_processor.py     # Channel extraction
│   ├── performance_monitor.py    # Latency tracking
│   ├── error_handler.py          # Error management
│   ├── config_manager.py         # Configuration loading
│   └── asset_discovery.py        # Asset scanning
├── tests/                        # Test suite
├── config/                       # Configuration files
│   └── ironsight_config.yaml     # Main configuration
├── models/                       # Model storage
├── logs/                         # Performance logs
├── demo_data/                    # Demo data generators
└── README.md                     # This file
```

## License

This project is part of the AIDTM rail inspection system.

## Version

IronSight Command Center v1.0
