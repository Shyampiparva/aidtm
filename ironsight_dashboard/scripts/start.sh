#!/bin/bash
# IronSight Command Center - Legacy Startup Script (Linux/macOS)
# DEPRECATED: Use run_ironsight.sh in the parent directory instead
# Usage: ./scripts/start.sh [--dev] [--cpu]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}========================================"
echo -e "  DEPRECATED SCRIPT"
echo -e "========================================${NC}"
echo "This script is deprecated. Please use:"
echo -e "${GREEN}  ./run_ironsight.sh${NC}"
echo ""
echo "Redirecting to new launcher..."
echo ""

# Change to parent directory and run new launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

exec ./run_ironsight.sh "$@"

# Legacy code below (kept for reference)
# =======================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default settings
DEV_MODE=false
CPU_ONLY=false
PORT=${STREAMLIT_PORT:-8501}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "IronSight Command Center Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev     Run in development mode with auto-reload"
            echo "  --cpu     Run without GPU (CPU-only mode)"
            echo "  --port N  Use port N (default: 8501)"
            echo "  -h        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  IronSight Command Center${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.10" ]] || [[ "$PYTHON_VERSION" > "3.12" ]]; then
    echo -e "${RED}Error: Python 3.10-3.12 required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Python version: $PYTHON_VERSION ✓${NC}"

# Check CUDA availability
if [ "$CPU_ONLY" = false ]; then
    echo -e "${YELLOW}Checking CUDA availability...${NC}"
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "${GREEN}CUDA available: $GPU_NAME ✓${NC}"
    else
        echo -e "${YELLOW}Warning: CUDA not available, running in CPU mode${NC}"
        CPU_ONLY=true
    fi
fi

# Set environment variables
if [ "$CPU_ONLY" = true ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo -e "${YELLOW}Running in CPU-only mode${NC}"
fi

# Load .env file if exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}Loading environment from .env...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

# Check for required model files
echo -e "${YELLOW}Checking model files...${NC}"
NAFNET_PATH="${NAFNET_MODEL_PATH:-../NAFNet-GoPro-width64.pth}"
if [ -f "$NAFNET_PATH" ]; then
    echo -e "${GREEN}NAFNet model found ✓${NC}"
else
    echo -e "${YELLOW}Warning: NAFNet model not found at $NAFNET_PATH${NC}"
fi

# Create logs directory
mkdir -p logs

# Build Streamlit command
STREAMLIT_CMD="streamlit run src/app.py"
STREAMLIT_CMD="$STREAMLIT_CMD --server.port=$PORT"
STREAMLIT_CMD="$STREAMLIT_CMD --server.address=0.0.0.0"

if [ "$DEV_MODE" = true ]; then
    echo -e "${YELLOW}Running in development mode with auto-reload${NC}"
    STREAMLIT_CMD="$STREAMLIT_CMD --server.runOnSave=true"
else
    STREAMLIT_CMD="$STREAMLIT_CMD --server.headless=true"
fi

echo ""
echo -e "${GREEN}Starting IronSight Command Center on port $PORT...${NC}"
echo -e "${GREEN}Dashboard URL: http://localhost:$PORT${NC}"
echo ""

# Run Streamlit
exec $STREAMLIT_CMD
