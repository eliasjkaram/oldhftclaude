#!/bin/bash
set -e

echo "🚀 Deploying to Kubernetes..."

# Check kubectl
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed. Aborting." >&2; exit 1; }

# Check cluster connection
kubectl cluster-info || { echo "Cannot connect to Kubernetes cluster!" >&2; exit 1; }

# Apply deployment
echo "📦 Applying Kubernetes deployment..."
kubectl apply -f config/k8s-deployment.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/hft-cluster

# Get status
echo "📊 Deployment status:"
kubectl get pods -l app=hft-cluster

echo "✅ Kubernetes deployment complete!"
