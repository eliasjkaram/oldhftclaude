# GPU Environment-Specific Deployment Guide

## Overview
This guide provides specific deployment instructions for each GPU script across different environments (AWS, GCP, Azure, On-Premise, Edge).

---

# AWS Deployment

## Instance Selection by Script

### High-Performance Scripts (HFT, Real-time Trading)
```yaml
# GPU Cluster HFT Engine, Ultra Optimized HFT
Instance: p4d.24xlarge
- 8x NVIDIA A100 (40GB each)
- 96 vCPUs, 1.1TB RAM
- 8TB NVMe SSD
- 400 Gbps network
- Cost: $32.77/hour
```

### Training Scripts (ML Models, Backtesting)
```yaml
# GPU Options Pricing Trainer, Production GPU Trainer
Instance: p3.8xlarge
- 4x NVIDIA V100 (16GB each)
- 32 vCPUs, 244GB RAM
- 10 Gbps network
- Cost: $12.24/hour
```

### Inference Scripts (Production Trading)
```yaml
# GPU Options Trader, GPU Trading AI
Instance: g4dn.xlarge
- 1x NVIDIA T4 (16GB)
- 4 vCPUs, 16GB RAM
- Up to 25 Gbps network
- Cost: $0.526/hour
```

## AWS-Specific Setup

### 1. GPU Resource Manager - AWS Configuration
```python
# aws_config.py
class AWSGPUConfig:
    def __init__(self):
        self.instance_type = os.environ.get('EC2_INSTANCE_TYPE')
        self.region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
    def setup_aws_gpu(self):
        # TODO: Configure based on instance type
        if 'p4d' in self.instance_type:
            # A100 optimization
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
            os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Optimize for NVSwitch
        elif 'p3' in self.instance_type:
            # V100 optimization
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
            os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P
        elif 'g4dn' in self.instance_type:
            # T4 optimization
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

### 2. EBS Optimization for Data
```bash
# Setup high-performance storage
sudo mkdir /data
sudo mkfs -t xfs /dev/nvme1n1
sudo mount /dev/nvme1n1 /data
echo '/dev/nvme1n1 /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab

# Enable EBS optimization
aws ec2 modify-instance-attribute --instance-id $INSTANCE_ID --ebs-optimized
```

### 3. S3 Integration for MinIO Data
```python
# s3_data_pipeline.py
class S3DataPipeline:
    def __init__(self):
        self.s3_client = boto3.client('s3',
            endpoint_url='https://uschristmas.us',
            aws_access_key_id='AKSTOCKDB2024',
            aws_secret_access_key='StockDB-Secret-Access-Key-2024-Secure!'
        )
    
    def stream_to_gpu(self, bucket, key):
        # TODO: Direct streaming to GPU memory
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data_stream = response['Body']
        
        # Convert to GPU tensor while streaming
        gpu_tensor = self.process_stream_on_gpu(data_stream)
        return gpu_tensor
```

---

# GCP Deployment

## Instance Selection

### A100 Instances (Best Performance)
```yaml
# For HFT and intensive training
Machine Type: a2-highgpu-8g
- 8x NVIDIA A100 (40GB)
- 96 vCPUs, 680GB RAM
- Cost: ~$29/hour
```

### T4 Instances (Cost-Effective)
```yaml
# For inference and light training
Machine Type: n1-standard-4 + 1x T4
- 1x NVIDIA T4 (16GB)
- 4 vCPUs, 15GB RAM
- Cost: ~$0.35/hour
```

## GCP-Specific Setup

### 1. GPU Allocation with GKE
```yaml
# gke-gpu-nodepool.yaml
apiVersion: container.gke.io/v1
kind: NodePool
metadata:
  name: gpu-pool
spec:
  config:
    machineType: n1-highmem-4
    accelerators:
    - type: nvidia-tesla-t4
      count: 1
    oauthScopes:
    - https://www.googleapis.com/auth/cloud-platform
  autoscaling:
    enabled: true
    minNodeCount: 0
    maxNodeCount: 10
```

### 2. GPU Driver Installation
```bash
# Install NVIDIA drivers on GCE
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
```

### 3. Persistent Disk for Fast I/O
```python
# gcp_storage_config.py
class GCPStorageOptimizer:
    def setup_local_ssd(self):
        # Mount local SSDs for temporary data
        subprocess.run(['sudo', 'mdadm', '--create', '/dev/md0', 
                       '--level=0', '--raid-devices=8', 
                       '/dev/nvme0n1', '/dev/nvme0n2'])
        subprocess.run(['sudo', 'mkfs.ext4', '-F', '/dev/md0'])
        subprocess.run(['sudo', 'mount', '/dev/md0', '/data'])
```

---

# Azure Deployment

## Instance Selection

### NCv3 Series (V100)
```yaml
# For training and HFT
VM Size: Standard_NC24s_v3
- 4x NVIDIA V100 (16GB)
- 24 vCPUs, 448GB RAM
- Cost: ~$12.24/hour
```

### NDv2 Series (V100 with NVLink)
```yaml
# For multi-GPU training
VM Size: Standard_ND40rs_v2
- 8x NVIDIA V100 (32GB) with NVLink
- 40 vCPUs, 672GB RAM
- Cost: ~$22.03/hour
```

## Azure-Specific Setup

### 1. Azure ML Integration
```python
# azure_ml_config.py
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute

class AzureGPUSetup:
    def create_gpu_compute(self):
        ws = Workspace.from_config()
        
        # Create GPU compute cluster
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='Standard_NC24s_v3',
            max_nodes=4,
            idle_seconds_before_scaledown=300
        )
        
        gpu_cluster = ComputeTarget.create(ws, 'gpu-cluster', compute_config)
        gpu_cluster.wait_for_completion(show_output=True)
```

### 2. Azure Blob Storage Integration
```python
# azure_blob_data.py
from azure.storage.blob import BlobServiceClient

class AzureBlobGPUPipeline:
    def __init__(self):
        self.blob_service = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )
    
    def stream_to_gpu(self, container, blob_name):
        blob_client = self.blob_service.get_blob_client(
            container=container, 
            blob=blob_name
        )
        
        # Stream directly to GPU memory
        with blob_client.download_blob() as stream:
            gpu_data = self.process_on_gpu(stream)
        return gpu_data
```

---

# On-Premise Deployment

## Hardware Requirements

### DGX Systems
```yaml
# NVIDIA DGX A100
- 8x A100 (80GB each)
- Optimal for: HFT, large-scale training
- Network: InfiniBand HDR (200Gb/s)

# NVIDIA DGX Station
- 4x A100 (40GB each)
- Optimal for: Development, smaller deployments
- Network: Dual 10GbE
```

### Custom Build
```yaml
# High-Performance Build
- GPUs: 8x RTX 4090 (24GB each)
- CPU: AMD EPYC 7763
- RAM: 512GB DDR4 ECC
- Storage: 8x NVMe in RAID 0
- Network: Mellanox ConnectX-6
```

## On-Premise Specific Setup

### 1. SLURM Integration
```bash
#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load openmpi/4.1.4

# Run distributed training
mpirun -np 16 python gpu_options_pricing_trainer.py \
    --distributed \
    --backend nccl \
    --world-size 16
```

### 2. InfiniBand Configuration
```python
# infiniband_setup.py
class InfiniBandOptimizer:
    def setup_ib_for_nccl(self):
        # Set IB environment variables
        os.environ['NCCL_IB_DISABLE'] = '0'
        os.environ['NCCL_IB_HCA'] = 'mlx5_0:1,mlx5_1:1'
        os.environ['NCCL_IB_GID_INDEX'] = '3'
        os.environ['NCCL_NET_GDR_LEVEL'] = '5'  # GPUDirect RDMA
```

### 3. Local Storage Optimization
```python
# raid_storage_setup.py
class RAIDStorageSetup:
    def create_raid0_array(self):
        # Create RAID 0 for maximum performance
        devices = [f'/dev/nvme{i}n1' for i in range(8)]
        subprocess.run(['mdadm', '--create', '/dev/md0', 
                       '--level=0', f'--raid-devices={len(devices)}'] + devices)
        
        # Format with XFS for better performance
        subprocess.run(['mkfs.xfs', '-f', '/dev/md0'])
        subprocess.run(['mount', '-o', 'noatime,nodiratime', '/dev/md0', '/data'])
```

---

# Edge Deployment

## Edge Devices

### NVIDIA Jetson AGX Orin
```yaml
# For edge inference
- GPU: 2048-core NVIDIA Ampere
- Memory: 32GB LPDDR5
- Power: 15-60W
- Use case: Real-time inference at trading desk
```

### Intel NUC with External GPU
```yaml
# Compact deployment
- GPU: RTX 4060 Ti via Thunderbolt
- CPU: Intel i7-12700H
- RAM: 32GB
- Use case: Branch office deployment
```

## Edge-Specific Optimizations

### 1. Model Quantization
```python
# edge_model_optimizer.py
class EdgeModelOptimizer:
    def quantize_for_edge(self, model):
        # INT8 quantization for edge deployment
        import tensorrt as trt
        
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        
        # Calibrate on representative data
        calibrator = Int8Calibrator(calibration_data)
        config.int8_calibrator = calibrator
        
        engine = builder.build_engine(network, config)
        return engine
```

### 2. Power Management
```python
# edge_power_management.py
class EdgePowerManager:
    def optimize_for_power(self):
        # Set GPU to efficient mode
        subprocess.run(['nvidia-smi', '-pm', '1'])  # Persistence mode
        subprocess.run(['nvidia-smi', '-pl', '150'])  # Power limit 150W
        
        # Dynamic frequency scaling
        subprocess.run(['nvidia-smi', '-ac', '5001,1380'])  # Memory,Core clocks
```

---

# Kubernetes Deployment (All Clouds)

## GPU Operator Setup
```yaml
# gpu-operator-values.yaml
operator:
  defaultRuntime: containerd
driver:
  enabled: true
  version: "525.60.13"
toolkit:
  enabled: true
  version: v1.13.0-cuda12.0
devicePlugin:
  enabled: true
  config:
    name: time-slicing-config
    data:
      any:
        - name: nvidia.com/gpu
          replicas: 4  # GPU time-slicing
```

## Script-Specific Deployments

### 1. GPU Options Pricing Trainer
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: options-pricing-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: your-registry/gpu-options-trainer:latest
        resources:
          limits:
            nvidia.com/gpu: 4
        env:
        - name: WORLD_SIZE
          value: "4"
        - name: MASTER_ADDR
          value: "options-trainer-master"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: models
          mountPath: /models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: model-storage-pvc
```

### 2. HFT Cluster Deployment
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hft-gpu-cluster
spec:
  serviceName: hft-cluster
  replicas: 8
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: p4d.24xlarge
      containers:
      - name: hft-engine
        image: your-registry/hft-gpu-engine:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            hugepages-2Mi: 2Gi
          requests:
            nvidia.com/gpu: 1
            hugepages-2Mi: 2Gi
        securityContext:
          privileged: true  # For kernel bypass networking
        env:
        - name: CUDA_MPS_PIPE_DIRECTORY
          value: /dev/shm/nvidia-mps
        - name: CUDA_MPS_LOG_DIRECTORY
          value: /dev/shm/nvidia-log
```

---

# Docker Configurations

## Base GPU Image
```dockerfile
# Dockerfile.gpu-base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install GPU libraries
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir \
    cupy-cuda11x \
    numba \
    tensorrt \
    pycuda

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

## Script-Specific Dockerfiles

### GPU Options Pricing Trainer
```dockerfile
# Dockerfile.pricing-trainer
FROM your-registry/gpu-base:latest

WORKDIR /app

# Copy application code
COPY src/misc/gpu_options_pricing_trainer.py .
COPY requirements-trainer.txt .

# Install specific dependencies
RUN pip3 install --no-cache-dir -r requirements-trainer.txt

# Create directories
RUN mkdir -p /data /models /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=0
ENV NCCL_DEBUG=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python3 -c "import torch; assert torch.cuda.is_available()"

ENTRYPOINT ["python3", "gpu_options_pricing_trainer.py"]
```

---

# Monitoring Setup

## Prometheus GPU Metrics
```yaml
# prometheus-gpu-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9400']  # nvidia-gpu-exporter
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'nvidia_gpu_.*'
        action: keep
```

## Grafana Dashboard
```json
{
  "dashboard": {
    "title": "GPU Trading System Monitoring",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100"
          }
        ]
      },
      {
        "title": "Model Inference Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, gpu_inference_latency_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

---

# Cost Optimization

## Multi-Cloud Cost Comparison

| Script Type | AWS | GCP | Azure | On-Premise |
|-------------|-----|-----|-------|------------|
| HFT (8x A100) | $32.77/hr | $29/hr | $30/hr | $50k upfront |
| Training (4x V100) | $12.24/hr | $11/hr | $12/hr | $20k upfront |
| Inference (1x T4) | $0.53/hr | $0.35/hr | $0.40/hr | $3k upfront |

## Cost Optimization Strategies

### 1. Spot/Preemptible Instances
```python
# spot_instance_manager.py
class SpotInstanceManager:
    def request_spot_gpu(self, instance_type='p3.2xlarge'):
        ec2 = boto3.client('ec2')
        
        response = ec2.request_spot_instances(
            SpotPrice='3.0',  # Max price
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification={
                'ImageId': 'ami-gpu-ubuntu-2204',
                'InstanceType': instance_type,
                'KeyName': 'gpu-cluster-key',
                'SecurityGroups': ['gpu-trading-sg']
            }
        )
```

### 2. Auto-scaling Based on Load
```python
# gpu_autoscaler.py
class GPUAutoscaler:
    def scale_based_on_queue(self):
        queue_length = self.get_job_queue_length()
        current_nodes = self.get_current_gpu_nodes()
        
        if queue_length > 100 and current_nodes < 10:
            self.scale_up()
        elif queue_length < 10 and current_nodes > 2:
            self.scale_down()
```

---

# Security Considerations

## GPU Memory Isolation
```python
# gpu_security.py
class GPUSecurityManager:
    def setup_mig(self):  # Multi-Instance GPU
        # Partition A100 into isolated instances
        subprocess.run(['nvidia-smi', 'mig', '-cgi', '14,14,14,14'])
        subprocess.run(['nvidia-smi', 'mig', '-cci'])
```

## Encrypted Model Storage
```python
# model_encryption.py
class SecureModelStorage:
    def save_encrypted_model(self, model, path):
        from cryptography.fernet import Fernet
        
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Serialize model
        model_bytes = pickle.dumps(model.state_dict())
        encrypted = cipher.encrypt(model_bytes)
        
        # Save encrypted model
        with open(path, 'wb') as f:
            f.write(encrypted)
```

---

# Deployment Checklist

## Pre-Deployment
- [ ] GPU drivers installed and tested
- [ ] CUDA/cuDNN versions compatible
- [ ] Container runtime configured
- [ ] Network optimized for GPU communication
- [ ] Storage configured for high IOPS
- [ ] Monitoring agents deployed
- [ ] Security policies implemented

## Deployment
- [ ] Health checks passing
- [ ] GPU allocation working
- [ ] Model loading successfully
- [ ] Data pipeline connected
- [ ] API endpoints responding
- [ ] Metrics being collected

## Post-Deployment
- [ ] Performance benchmarks met
- [ ] Cost tracking enabled
- [ ] Alerts configured
- [ ] Backup procedures tested
- [ ] Scaling policies active
- [ ] Documentation updated

---

*This guide ensures each GPU script can be deployed successfully in any environment.*