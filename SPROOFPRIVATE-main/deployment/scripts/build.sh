#!/bin/bash
set -e

echo "🚀 Building HFT Cluster..."

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required but not installed. Aborting." >&2; exit 1; }

# Check for NVIDIA Docker runtime
docker info | grep -q nvidia || { echo "NVIDIA Docker runtime required. Install nvidia-docker2." >&2; exit 1; }

# Build the image
echo "📦 Building Docker image..."
docker build -t hft-cluster:latest .

# Verify GPU access
echo "🔍 Verifying GPU access..."
docker run --rm --gpus all hft-cluster:latest nvidia-smi || { echo "GPU access failed!" >&2; exit 1; }

echo "✅ Build complete!"
