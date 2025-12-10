#!/bin/bash
set -e

echo "🚀 Optimizing System Performance..."

# CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU power saving
sudo systemctl disable ondemand

# Network optimizations
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Memory optimizations
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.dirty_ratio=15
sudo sysctl -w vm.dirty_background_ratio=5

# Scheduler optimizations
echo mq-deadline | sudo tee /sys/block/*/queue/scheduler

echo "✅ System optimization complete!"
