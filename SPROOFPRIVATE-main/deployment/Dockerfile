
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libzmq3-dev \
    redis-tools \
    htop \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install GPU-specific packages
RUN pip3 install cupy-cuda12x cudf-cu12 cuml-cu12

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create directories
RUN mkdir -p /app/logs /app/data /app/mmap_data

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=all
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV MALLOC_MMAP_THRESHOLD_=65536

# Expose ports
EXPOSE 5555 5556 5557 5560

# Default command
CMD ["python3", "ultra_optimized_hft_cluster.py"]
