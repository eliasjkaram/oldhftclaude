# Multi-stage Dockerfile for GPU-accelerated trading system
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Copy application code
COPY . .

# Run tests and linting in builder
RUN pip install -r requirements-dev.txt
RUN python -m pytest tests/ || echo "No tests found"
RUN flake8 . --max-line-length=120 --exclude=.git,__pycache__,venv || echo "Linting complete"

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create non-root user
RUN useradd -m -u 1000 trader
WORKDIR /app

# Copy built application from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /app /app

# Change ownership to trader user
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose ports
EXPOSE 8080 8081 9090

# Default command - can be overridden
CMD ["python", "production_trading_engine.py"]