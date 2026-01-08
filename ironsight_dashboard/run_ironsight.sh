#!/bin/bash
# IronSight Command Center - Linux/macOS Launcher
# Simple wrapper for the Python launcher script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is available first
if command -v uv &> /dev/null; then
    echo -e "${GREEN}Found uv package manager - using for better performance${NC}"
    exec uv run python run_ironsight.py "$@"
fi

# Fallback to regular Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Neither uv nor Python 3 is installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install uv (recommended) or Python 3.10-3.12 and try again${NC}"
    echo ""
    echo -e "${CYAN}To install uv: https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.10" ]] || [[ "$PYTHON_VERSION" > "3.12" ]]; then
    echo -e "${RED}Error: Python 3.10-3.12 required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Make sure the script is executable
chmod +x run_ironsight.py

# Run with regular Python
echo -e "${YELLOW}Using regular Python (consider installing uv for better performance)${NC}"
exec python3 run_ironsight.py "$@"