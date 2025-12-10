#!/bin/bash
#
# Production Deployment Script
# ===========================
# Automated deployment of the options trading system to production
#

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
DRY_RUN=${3:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for required tools
    for tool in kubectl docker terraform aws; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check Kubernetes context
    CURRENT_CONTEXT=$(kubectl config current-context)
    if [[ "$CURRENT_CONTEXT" != *"$ENVIRONMENT"* ]]; then
        log_warn "Current k8s context: $CURRENT_CONTEXT"
        read -p "Continue with this context? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Unit tests
    python -m pytest tests/unit -v --cov=src --cov-report=term-missing
    
    # Integration tests
    python -m pytest tests/integration -v
    
    # Backtest validation
    python production_backtest.py
    
    log_info "All tests passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Main trading system
    docker build -t trading-system:$VERSION \
        --build-arg VERSION=$VERSION \
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        -f Dockerfile .
    
    # Model serving
    docker build -t model-server:$VERSION \
        -f docker/model-server/Dockerfile .
    
    # Data pipeline
    docker build -t data-pipeline:$VERSION \
        -f docker/data-pipeline/Dockerfile .
    
    # Tag for registry
    docker tag trading-system:$VERSION $ECR_REGISTRY/trading-system:$VERSION
    docker tag model-server:$VERSION $ECR_REGISTRY/model-server:$VERSION
    docker tag data-pipeline:$VERSION $ECR_REGISTRY/data-pipeline:$VERSION
    
    log_info "Docker images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    # Login to ECR
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Push images
    docker push $ECR_REGISTRY/trading-system:$VERSION
    docker push $ECR_REGISTRY/model-server:$VERSION
    docker push $ECR_REGISTRY/data-pipeline:$VERSION
    
    log_info "Images pushed successfully"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure..."
    
    cd terraform/$ENVIRONMENT
    
    # Initialize Terraform
    terraform init
    
    # Plan changes
    terraform plan -out=tfplan
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "Dry run mode - skipping terraform apply"
        return
    fi
    
    # Apply changes
    terraform apply tfplan
    
    cd ../..
    
    log_info "Infrastructure deployed successfully"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    log_info "Deploying Kubernetes resources..."
    
    # Update image versions in manifests
    sed -i "s|image: trading-system:.*|image: $ECR_REGISTRY/trading-system:$VERSION|g" k8s/$ENVIRONMENT/*.yaml
    sed -i "s|image: model-server:.*|image: $ECR_REGISTRY/model-server:$VERSION|g" k8s/$ENVIRONMENT/*.yaml
    sed -i "s|image: data-pipeline:.*|image: $ECR_REGISTRY/data-pipeline:$VERSION|g" k8s/$ENVIRONMENT/*.yaml
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "Dry run mode - showing what would be deployed"
        kubectl diff -f k8s/$ENVIRONMENT/
        return
    fi
    
    # Deploy namespace and configs first
    kubectl apply -f k8s/$ENVIRONMENT/00-namespace.yaml
    kubectl apply -f k8s/$ENVIRONMENT/01-configmap.yaml
    kubectl apply -f k8s/$ENVIRONMENT/02-secrets.yaml
    
    # Deploy infrastructure components
    kubectl apply -f k8s/$ENVIRONMENT/10-redis.yaml
    kubectl apply -f k8s/$ENVIRONMENT/11-kafka.yaml
    kubectl apply -f k8s/$ENVIRONMENT/12-postgres.yaml
    
    # Wait for infrastructure to be ready
    kubectl wait --for=condition=ready pod -l app=redis -n $ENVIRONMENT --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kafka -n $ENVIRONMENT --timeout=300s
    kubectl wait --for=condition=ready pod -l app=postgres -n $ENVIRONMENT --timeout=300s
    
    # Deploy application components
    kubectl apply -f k8s/$ENVIRONMENT/20-model-server.yaml
    kubectl apply -f k8s/$ENVIRONMENT/21-data-pipeline.yaml
    kubectl apply -f k8s/$ENVIRONMENT/22-trading-system.yaml
    
    # Deploy monitoring
    kubectl apply -f k8s/$ENVIRONMENT/30-prometheus.yaml
    kubectl apply -f k8s/$ENVIRONMENT/31-grafana.yaml
    
    log_info "Kubernetes resources deployed successfully"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment --all -n $ENVIRONMENT --timeout=600s
    
    # Check pod status
    kubectl get pods -n $ENVIRONMENT
    
    # Run smoke tests
    kubectl run smoke-test --image=curlimages/curl --rm -it --restart=Never -- \
        curl -f http://trading-system.$ENVIRONMENT.svc.cluster.local:8080/health
    
    log_info "Health checks passed"
}

# Setup monitoring alerts
setup_monitoring() {
    log_info "Setting up monitoring and alerts..."
    
    # Configure Prometheus alerts
    kubectl apply -f monitoring/alerts.yaml -n $ENVIRONMENT
    
    # Configure Grafana dashboards
    kubectl create configmap grafana-dashboards \
        --from-file=monitoring/dashboards/ \
        -n $ENVIRONMENT \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Configure PagerDuty
    if [[ -n "${PAGERDUTY_TOKEN:-}" ]]; then
        kubectl create secret generic pagerduty-token \
            --from-literal=token=$PAGERDUTY_TOKEN \
            -n $ENVIRONMENT \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    log_info "Monitoring configured"
}

# Main deployment flow
main() {
    log_info "Starting deployment to $ENVIRONMENT (version: $VERSION)"
    
    # Confirmation for production
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$DRY_RUN" != "true" ]]; then
        log_warn "You are about to deploy to PRODUCTION!"
        read -p "Type 'DEPLOY' to confirm: " confirm
        if [[ "$confirm" != "DEPLOY" ]]; then
            log_error "Deployment cancelled"
            exit 1
        fi
    fi
    
    # Run deployment steps
    check_prerequisites
    run_tests
    build_images
    push_images
    deploy_infrastructure
    deploy_kubernetes
    run_health_checks
    setup_monitoring
    
    log_info "Deployment completed successfully!"
    log_info "Access the system at: https://trading.$ENVIRONMENT.company.com"
    log_info "Grafana dashboard: https://grafana.$ENVIRONMENT.company.com"
}

# Run main function
main