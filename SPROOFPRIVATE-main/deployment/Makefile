# Makefile for Alpaca MCP Trading System Deployment

# Variables
ENVIRONMENT ?= staging
COMPOSE_FILE = docker-compose.yml
COMPOSE_PROD_FILE = deployment/docker-compose.prod.yml
COMPOSE_STAGING_FILE = deployment/docker-compose.staging.yml
PROJECT_NAME = alpaca-mcp

# Colors
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

## Help
help:
	@echo "Alpaca MCP Trading System - Deployment Commands"
	@echo ""
	@echo "Usage: make [target] [ENVIRONMENT=staging|production]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

## Build all Docker images
build: ## Build Docker images
	@echo "Building Docker images for $(ENVIRONMENT)..."
	docker-compose -f $(COMPOSE_FILE) build --no-cache

## Deploy the application
deploy: check-env ## Deploy to specified environment
	@echo "Deploying to $(ENVIRONMENT)..."
	@bash deployment/scripts/deploy.sh $(ENVIRONMENT)

## Start services
up: ## Start all services
	@echo "Starting services for $(ENVIRONMENT)..."
ifeq ($(ENVIRONMENT),production)
	docker-compose -f $(COMPOSE_FILE) -f $(COMPOSE_PROD_FILE) up -d
else
	docker-compose -f $(COMPOSE_FILE) -f $(COMPOSE_STAGING_FILE) up -d
endif

## Stop services
down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

## Restart services
restart: down up ## Restart all services

## View logs
logs: ## View service logs
	docker-compose logs -f --tail=100

## Run tests
test: ## Run test suite
	@echo "Running tests..."
	docker-compose run --rm trading-engine pytest tests/

## Run health check
health: ## Check system health
	@bash deployment/scripts/health_check.sh

## Create backup
backup: ## Create system backup
	@echo "Creating backup..."
	@bash deployment/scripts/backup.sh

## Rollback to previous version
rollback: ## Rollback deployment
	@echo "Rolling back deployment..."
	@bash deployment/scripts/rollback.sh

## View service status
status: ## Show service status
	@docker-compose ps
	@echo ""
	@echo "System Health:"
	@curl -s http://localhost:8080/health | jq . || echo "API not responding"

## Run database migrations
migrate: ## Run database migrations
	@echo "Running migrations..."
	docker-compose run --rm trading-engine python scripts/migrate.py

## Clean up system
clean: ## Clean up containers, volumes, and images
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

## Setup environment
setup: ## Initial environment setup
	@echo "Setting up environment..."
	@bash deployment/scripts/setup_environment.sh

## Update dependencies
update-deps: ## Update Python dependencies
	@echo "Updating dependencies..."
	docker-compose run --rm trading-engine pip install --upgrade -r requirements.txt

## Security scan
security-scan: ## Run security scan
	@echo "Running security scan..."
	docker run --rm -v "$$(pwd)":/src \
		aquasec/trivy:latest fs --severity HIGH,CRITICAL /src

## Performance test
perf-test: ## Run performance tests
	@echo "Running performance tests..."
	docker-compose run --rm trading-engine python tests/performance/

## Generate reports
reports: ## Generate system reports
	@echo "Generating reports..."
	@mkdir -p reports
	docker-compose exec trading-engine python scripts/generate_reports.py

## Monitor resources
monitor: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@open http://localhost:3000 || xdg-open http://localhost:3000

## Shell access
shell: ## Get shell access to trading engine
	docker-compose exec trading-engine /bin/bash

## Redis CLI
redis-cli: ## Access Redis CLI
	docker-compose exec redis redis-cli

## View environment
env: ## Display environment configuration
	@echo "Environment: $(ENVIRONMENT)"
	@echo "Compose files:"
	@echo "  - $(COMPOSE_FILE)"
ifeq ($(ENVIRONMENT),production)
	@echo "  - $(COMPOSE_PROD_FILE)"
else
	@echo "  - $(COMPOSE_STAGING_FILE)"
endif

## Validate configuration
validate: ## Validate deployment configuration
	@echo "Validating configuration..."
	@docker-compose config > /dev/null && echo "$(GREEN)✓ Docker Compose configuration valid$(NC)"
	@test -f .env.$(ENVIRONMENT) && echo "$(GREEN)✓ Environment file exists$(NC)" || echo "$(RED)✗ Environment file missing$(NC)"

## Production deployment
prod: ## Deploy to production (requires confirmation)
	@echo "$(RED)WARNING: This will deploy to PRODUCTION!$(NC)"
	@echo "Type 'DEPLOY TO PRODUCTION' to confirm:"
	@read confirm && [ "$$confirm" = "DEPLOY TO PRODUCTION" ] || (echo "Cancelled" && exit 1)
	@$(MAKE) deploy ENVIRONMENT=production

## Emergency stop
emergency-stop: ## Emergency stop all trading
	@echo "$(RED)EMERGENCY STOP - Halting all trading!$(NC)"
	docker-compose exec trading-engine python scripts/emergency_stop.py
	docker-compose stop trading-engine

# Check environment file exists
check-env:
	@test -f .env.$(ENVIRONMENT) || (echo "$(RED)Error: .env.$(ENVIRONMENT) not found!$(NC)" && exit 1)

.PHONY: help build deploy up down restart logs test health backup rollback status \
        migrate clean setup update-deps security-scan perf-test reports monitor \
        shell redis-cli env validate prod emergency-stop check-env