#!/bin/bash
set -e

echo "🛑 Stopping HFT Cluster..."

# Stop all services
docker-compose down

# Remove volumes (optional)
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning up volumes..."
    docker-compose down -v
    docker system prune -f
fi

echo "✅ Cluster stopped!"
