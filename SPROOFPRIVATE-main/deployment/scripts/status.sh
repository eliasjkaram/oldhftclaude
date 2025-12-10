#!/bin/bash

echo "📊 HFT Cluster Status"
echo "===================="

# Service status
echo "🔍 Service Status:"
docker-compose ps

echo ""
echo "💻 Resource Usage:"
docker stats --no-stream

echo ""
echo "📋 Recent Logs:"
docker-compose logs --tail=10

echo ""
echo "🌐 Access Points:"
echo "  Grafana: http://localhost:3000"
echo "  Prometheus: http://localhost:9090"
