#!/bin/bash
set -e

echo "🚀 Optimizing GPU Performance..."

# Set GPU performance mode
sudo nvidia-smi -pm 1

# Set maximum GPU clocks
sudo nvidia-smi -ac 3505,1590

# Set GPU power limit to maximum
sudo nvidia-smi -pl 400

# Check GPU status
nvidia-smi

echo "✅ GPU optimization complete!"
