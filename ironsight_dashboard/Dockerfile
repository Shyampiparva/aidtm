# IronSight Command Center - Docker Configuration
# Production-grade real-time rail inspection dashboard

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # IronSight configuration
    IRONSIGHT_CONFIG=/app/config/ironsight_config.yaml \
    IRONSIGHT_MODELS_DIR=/app/models \
    # Streamlit configuration
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY demo_data/ ./demo_data/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["uv", "run", "streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
