#!/bin/bash
# Environment setup script for new servers

set -euo pipefail

# Configuration
PYTHON_VERSION="3.10"
NODE_VERSION="18"
DOCKER_COMPOSE_VERSION="2.23.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check OS
check_os() {
    if [[ ! -f /etc/os-release ]]; then
        error "Cannot determine OS"
    fi
    
    . /etc/os-release
    log "Detected OS: $NAME $VERSION"
    
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        error "This script only supports Ubuntu/Debian"
    fi
}

# Update system
update_system() {
    log "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        htop \
        iotop \
        net-tools \
        vim \
        tmux \
        jq
}

# Install Python
install_python() {
    log "Installing Python $PYTHON_VERSION..."
    
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip
    
    # Set as default
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    sudo update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
    
    # Install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3
    
    # Verify installation
    python3 --version
    pip3 --version
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    # Start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # Verify installation
    docker --version
    docker-compose --version
}

# Install NVIDIA drivers and container toolkit
install_nvidia() {
    log "Checking for NVIDIA GPU..."
    
    if ! lspci | grep -i nvidia > /dev/null; then
        warning "No NVIDIA GPU detected, skipping NVIDIA setup"
        return
    fi
    
    log "Installing NVIDIA drivers and container toolkit..."
    
    # Install NVIDIA driver
    sudo apt-get install -y nvidia-driver-535
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Verify installation
    nvidia-smi
}

# Install monitoring tools
install_monitoring() {
    log "Installing monitoring tools..."
    
    # Install Node Exporter
    wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
    tar xvfz node_exporter-1.7.0.linux-amd64.tar.gz
    sudo cp node_exporter-1.7.0.linux-amd64/node_exporter /usr/local/bin/
    rm -rf node_exporter-1.7.0.linux-amd64*
    
    # Create systemd service
    sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF
    
    # Create user
    sudo useradd -rs /bin/false node_exporter || true
    
    # Start service
    sudo systemctl daemon-reload
    sudo systemctl enable node_exporter
    sudo systemctl start node_exporter
}

# Setup firewall
setup_firewall() {
    log "Setting up firewall..."
    
    sudo apt-get install -y ufw
    
    # Default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow 22/tcp
    
    # Allow HTTP/HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow monitoring ports (restrict source in production)
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 9090/tcp  # Prometheus
    sudo ufw allow 9100/tcp  # Node Exporter
    
    # Enable firewall
    sudo ufw --force enable
    sudo ufw status
}

# Setup system limits
setup_limits() {
    log "Setting up system limits..."
    
    # Increase file descriptor limits
    sudo tee -a /etc/security/limits.conf > /dev/null <<EOF

# Alpaca MCP Trading System
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
    
    # Sysctl optimizations
    sudo tee -a /etc/sysctl.conf > /dev/null <<EOF

# Alpaca MCP Trading System Optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
vm.swappiness = 10
EOF
    
    sudo sysctl -p
}

# Create directories
create_directories() {
    log "Creating application directories..."
    
    sudo mkdir -p /opt/alpaca-mcp
    sudo mkdir -p /var/log/alpaca-mcp
    sudo mkdir -p /var/backups/alpaca-mcp
    
    # Set permissions
    sudo chown -R $USER:$USER /opt/alpaca-mcp
    sudo chown -R $USER:$USER /var/log/alpaca-mcp
    sudo chown -R $USER:$USER /var/backups/alpaca-mcp
}

# Install application
install_application() {
    log "Installing application..."
    
    cd /opt/alpaca-mcp
    
    # Clone repository (or copy from deployment artifact)
    if [[ -n "${GIT_REPO:-}" ]]; then
        git clone "$GIT_REPO" .
    else
        warning "No GIT_REPO specified, skipping repository clone"
    fi
    
    # Create environment file template
    cat > .env.example <<EOF
# Alpaca API Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_key_here

# Database Configuration
REDIS_PASSWORD=your_redis_password_here

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=your_minio_password_here

# Monitoring Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_grafana_password_here

# Alert Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_WEBHOOK_URL=https://your-alert-webhook.com

# AWS Backup Configuration (optional)
AWS_ACCESS_KEY_ID=your_aws_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_here
S3_BACKUP_BUCKET=your-backup-bucket
EOF
    
    log "Environment template created at .env.example"
    log "Please copy to .env.production and fill in your credentials"
}

# Setup cron jobs
setup_cron() {
    log "Setting up cron jobs..."
    
    # Backup cron
    (crontab -l 2>/dev/null; echo "0 2 * * * /opt/alpaca-mcp/deployment/scripts/backup.sh") | crontab -
    
    # Cleanup cron
    (crontab -l 2>/dev/null; echo "0 3 * * 0 /opt/alpaca-mcp/deployment/scripts/cleanup.sh") | crontab -
    
    # Health check cron
    (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/alpaca-mcp/deployment/scripts/health_check.sh") | crontab -
}

# Main installation
main() {
    log "Starting environment setup..."
    
    check_os
    update_system
    install_python
    install_docker
    install_nvidia
    install_monitoring
    setup_firewall
    setup_limits
    create_directories
    install_application
    setup_cron
    
    log "Environment setup completed!"
    log ""
    log "Next steps:"
    log "1. Log out and back in for docker group changes to take effect"
    log "2. Copy .env.example to .env.production and configure"
    log "3. Run deployment script: ./deployment/scripts/deploy.sh production"
    log ""
    warning "Remember to configure your environment variables before deploying!"
}

# Run main
main "$@"