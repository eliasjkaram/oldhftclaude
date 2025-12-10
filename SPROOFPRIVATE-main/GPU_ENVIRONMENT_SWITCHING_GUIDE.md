# GPU Environment Switching Guide

## Overview
This guide provides a comprehensive framework for seamlessly switching between different GPU environments in production servers. It covers automatic detection, configuration, and optimization for various GPU types.

## Table of Contents
1. [GPU Auto-Detection System](#gpu-auto-detection-system)
2. [Environment Configuration Files](#environment-configuration-files)
3. [GPU-Specific Optimizations](#gpu-specific-optimizations)
4. [Docker Configurations](#docker-configurations)
5. [Kubernetes Deployments](#kubernetes-deployments)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

## GPU Auto-Detection System

### 1. Create GPU Detection Script

Create `detect_gpu_environment.py`:

```python
#!/usr/bin/env python3
"""
GPU Environment Auto-Detection and Configuration
"""

import torch
import os
import json
import subprocess
from pathlib import Path

class GPUEnvironmentDetector:
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.config = self._generate_config()
        
    def _detect_gpu(self):
        """Detect GPU capabilities and specifications"""
        info = {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': []
        }
        
        if info['available']:
            info['count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            
            for i in range(info['count']):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': device_props.name,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'memory_gb': device_props.total_memory / (1024**3),
                    'multi_processor_count': device_props.multi_processor_count,
                    'max_threads_per_block': device_props.max_threads_per_block,
                    'max_threads_per_multiprocessor': device_props.max_threads_per_multiprocessor,
                    'warp_size': device_props.warp_size
                }
                info['devices'].append(device_info)
                
        return info
    
    def _generate_config(self):
        """Generate optimal configuration based on GPU"""
        config = {
            'device': 'cpu',
            'mixed_precision': False,
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': False,
            'optimization_level': 'O0'
        }
        
        if not self.gpu_info['available']:
            return config
            
        # Get primary GPU info
        gpu = self.gpu_info['devices'][0]
        gpu_name = gpu['name'].lower()
        memory_gb = gpu['memory_gb']
        
        # Set device
        config['device'] = 'cuda'
        config['pin_memory'] = True
        
        # GPU-specific configurations
        if 'a100' in gpu_name:
            config.update(self._config_a100(memory_gb))
        elif 'v100' in gpu_name:
            config.update(self._config_v100(memory_gb))
        elif 'rtx 4090' in gpu_name:
            config.update(self._config_rtx4090(memory_gb))
        elif 'rtx 3090' in gpu_name:
            config.update(self._config_rtx3090(memory_gb))
        elif 'rtx 3080' in gpu_name:
            config.update(self._config_rtx3080(memory_gb))
        elif 'rtx 3070' in gpu_name:
            config.update(self._config_rtx3070(memory_gb))
        elif 'rtx 3060' in gpu_name:
            config.update(self._config_rtx3060(memory_gb))
        elif 'rtx 3050' in gpu_name:
            config.update(self._config_rtx3050(memory_gb))
        elif 't4' in gpu_name:
            config.update(self._config_t4(memory_gb))
        else:
            config.update(self._config_generic(memory_gb))
            
        return config
    
    def _config_a100(self, memory_gb):
        """NVIDIA A100 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 256 if memory_gb >= 40 else 128,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True
        }
    
    def _config_v100(self, memory_gb):
        """NVIDIA V100 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 128 if memory_gb >= 32 else 64,
            'num_workers': 8,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': True,
            'gradient_checkpointing': memory_gb < 32,
            'compile_model': False
        }
    
    def _config_rtx4090(self, memory_gb):
        """NVIDIA RTX 4090 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 128,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True
        }
    
    def _config_rtx3090(self, memory_gb):
        """NVIDIA RTX 3090 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 64,
            'num_workers': 6,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True
        }
    
    def _config_rtx3080(self, memory_gb):
        """NVIDIA RTX 3080 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 48,
            'num_workers': 6,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': memory_gb < 12,
            'compile_model': True
        }
    
    def _config_rtx3070(self, memory_gb):
        """NVIDIA RTX 3070 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 32,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': True,
            'compile_model': True
        }
    
    def _config_rtx3060(self, memory_gb):
        """NVIDIA RTX 3060 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 32,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': False,
            'gradient_checkpointing': True,
            'compile_model': False
        }
    
    def _config_rtx3050(self, memory_gb):
        """NVIDIA RTX 3050 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 16,
            'num_workers': 2,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': False,
            'gradient_checkpointing': True,
            'compile_model': False
        }
    
    def _config_t4(self, memory_gb):
        """NVIDIA T4 configuration"""
        return {
            'mixed_precision': True,
            'batch_size': 32,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': True,
            'gradient_checkpointing': True,
            'compile_model': False
        }
    
    def _config_generic(self, memory_gb):
        """Generic GPU configuration"""
        if memory_gb >= 24:
            batch_size = 64
            gradient_checkpointing = False
        elif memory_gb >= 16:
            batch_size = 48
            gradient_checkpointing = False
        elif memory_gb >= 8:
            batch_size = 32
            gradient_checkpointing = True
        else:
            batch_size = 16
            gradient_checkpointing = True
            
        return {
            'mixed_precision': memory_gb >= 8,
            'batch_size': batch_size,
            'num_workers': min(4, os.cpu_count() or 4),
            'optimization_level': 'O1' if memory_gb >= 8 else 'O0',
            'use_tf32': False,
            'use_channels_last': memory_gb >= 16,
            'gradient_checkpointing': gradient_checkpointing,
            'compile_model': False
        }
    
    def save_config(self, path='gpu_config.json'):
        """Save configuration to file"""
        config_data = {
            'gpu_info': self.gpu_info,
            'config': self.config
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"GPU configuration saved to {path}")
        
    def print_info(self):
        """Print GPU information"""
        print("="*60)
        print("GPU Environment Detection")
        print("="*60)
        
        if not self.gpu_info['available']:
            print("No GPU detected. Using CPU configuration.")
            return
            
        print(f"CUDA Available: {self.gpu_info['available']}")
        print(f"CUDA Version: {self.gpu_info['cuda_version']}")
        print(f"cuDNN Version: {self.gpu_info['cudnn_version']}")
        print(f"Number of GPUs: {self.gpu_info['count']}")
        
        for device in self.gpu_info['devices']:
            print(f"\nGPU {device['index']}: {device['name']}")
            print(f"  Compute Capability: {device['compute_capability']}")
            print(f"  Memory: {device['memory_gb']:.1f} GB")
            print(f"  Multiprocessors: {device['multi_processor_count']}")
        
        print("\nOptimal Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    detector = GPUEnvironmentDetector()
    detector.print_info()
    detector.save_config()
```

### 2. Create Adaptive GPU Manager

Create `adaptive_gpu_manager.py`:

```python
#!/usr/bin/env python3
"""
Adaptive GPU Manager for Different Environments
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveGPUManager:
    def __init__(self, config_path: str = 'gpu_config.json'):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self._apply_optimizations()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load GPU configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                return data['config']
        else:
            # Run detection if config doesn't exist
            from detect_gpu_environment import GPUEnvironmentDetector
            detector = GPUEnvironmentDetector()
            detector.save_config(config_path)
            return detector.config
    
    def _setup_device(self) -> torch.device:
        """Setup PyTorch device"""
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Set memory fraction for multi-tenant environments
            if os.environ.get('CUDA_MEMORY_FRACTION'):
                fraction = float(os.environ['CUDA_MEMORY_FRACTION'])
                torch.cuda.set_per_process_memory_fraction(fraction)
                
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
            
        return device
    
    def _apply_optimizations(self):
        """Apply GPU-specific optimizations"""
        if self.config['device'] == 'cuda':
            # Enable TF32 on Ampere GPUs
            if self.config.get('use_tf32', False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for Ampere GPU")
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Set cudnn deterministic if needed
            if os.environ.get('CUDNN_DETERMINISTIC', '0') == '1':
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def get_data_loader_kwargs(self) -> Dict[str, Any]:
        """Get optimal DataLoader arguments"""
        return {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'],
            'persistent_workers': self.config['num_workers'] > 0
        }
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with GPU optimizations"""
        model = model.to(self.device)
        
        # Compile model if supported (PyTorch 2.0+)
        if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")
        
        # Use channels last memory format
        if self.config.get('use_channels_last', False):
            model = model.to(memory_format=torch.channels_last)
            logger.info("Using channels_last memory format")
        
        return model
    
    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get optimizer arguments based on GPU"""
        kwargs = {}
        
        # Use fused optimizers on newer GPUs
        if self.config['device'] == 'cuda':
            gpu_name = torch.cuda.get_device_name(0).lower()
            if any(gpu in gpu_name for gpu in ['a100', 'rtx 40', 'rtx 30']):
                kwargs['fused'] = True
                
        return kwargs
    
    def get_autocast_context(self):
        """Get autocast context manager"""
        if self.config['mixed_precision']:
            device_type = 'cuda' if self.config['device'] == 'cuda' else 'cpu'
            return torch.amp.autocast(device_type=device_type)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def get_grad_scaler(self):
        """Get gradient scaler for mixed precision"""
        if self.config['mixed_precision'] and self.config['device'] == 'cuda':
            return torch.amp.GradScaler('cuda')
        else:
            # Return a dummy scaler for CPU
            class DummyScaler:
                def scale(self, loss):
                    return loss
                def step(self, optimizer):
                    optimizer.step()
                def update(self):
                    pass
                def unscale_(self, optimizer):
                    pass
            return DummyScaler()
```

## Environment Configuration Files

### 1. YAML Configuration

Create `gpu_configs/environments.yaml`:

```yaml
# GPU Environment Configurations
environments:
  development:
    gpu_types:
      - "RTX 3050"
      - "RTX 3060"
      - "GTX 1660"
    settings:
      debug_mode: true
      profiling: true
      memory_fraction: 0.8
      
  staging:
    gpu_types:
      - "RTX 3080"
      - "RTX 3090"
      - "RTX 4070"
    settings:
      debug_mode: false
      profiling: true
      memory_fraction: 0.9
      
  production:
    gpu_types:
      - "A100"
      - "V100"
      - "RTX 4090"
    settings:
      debug_mode: false
      profiling: false
      memory_fraction: 0.95
      
  cloud:
    aws:
      instance_types:
        - "p3.2xlarge"  # V100
        - "p4d.24xlarge" # A100
        - "g4dn.xlarge"  # T4
    gcp:
      instance_types:
        - "n1-standard-8-v100"
        - "a2-highgpu-1g"  # A100
    azure:
      instance_types:
        - "NC6s_v3"  # V100
        - "ND96asr_v4"  # A100

# Optimization Profiles
optimization_profiles:
  memory_constrained:
    gradient_checkpointing: true
    batch_size_multiplier: 0.5
    mixed_precision: true
    compile_model: false
    
  balanced:
    gradient_checkpointing: false
    batch_size_multiplier: 1.0
    mixed_precision: true
    compile_model: true
    
  performance:
    gradient_checkpointing: false
    batch_size_multiplier: 2.0
    mixed_precision: true
    compile_model: true
    use_tf32: true
```

### 2. Docker Multi-Stage Build

Create `Dockerfile.multi-gpu`:

```dockerfile
# Base image with CUDA support
ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install PyTorch with different CUDA versions
ARG TORCH_VERSION=2.5.1
ARG CUDA_VERSION_SHORT=121

# Copy requirements
COPY requirements-gpu.txt /tmp/

# Install PyTorch based on CUDA version
RUN if [ "$CUDA_VERSION_SHORT" = "121" ]; then \
        pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$CUDA_VERSION_SHORT" = "118" ]; then \
        pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install torch==${TORCH_VERSION}; \
    fi

# Install other requirements
RUN pip install -r /tmp/requirements-gpu.txt

# Copy application code
WORKDIR /app
COPY . .

# Run GPU detection on startup
ENTRYPOINT ["python", "detect_gpu_environment.py", "&&"]
CMD ["python", "run_trading_system.py"]
```

### 3. Kubernetes GPU Configuration

Create `k8s/gpu-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alpaca-gpu-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alpaca-gpu-trading
  template:
    metadata:
      labels:
        app: alpaca-gpu-trading
    spec:
      containers:
      - name: trading-system
        image: alpaca-trading:latest
        env:
        - name: GPU_AUTO_DETECT
          value: "1"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: CUDA_MEMORY_FRACTION
          value: "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
      nodeSelector:
        gpu-type: "nvidia-tesla-v100"  # Change based on available GPUs
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

## GPU-Specific Script Adapters

### 1. Create Universal GPU Adapter

Create `gpu_adapter.py`:

```python
#!/usr/bin/env python3
"""
Universal GPU Adapter for Trading Scripts
"""

import torch
import numpy as np
from adaptive_gpu_manager import AdaptiveGPUManager
import logging

logger = logging.getLogger(__name__)

class UniversalGPUAdapter:
    def __init__(self):
        self.gpu_manager = AdaptiveGPUManager()
        self.device = self.gpu_manager.device
        self.config = self.gpu_manager.config
        
    def adapt_options_pricing(self):
        """Adapt options pricing for current GPU"""
        if self.config['batch_size'] >= 64:
            # Use batch processing for high-memory GPUs
            from gpu_options_pricing_trainer_working import GPUOptionsPricingTrainer
            return GPUOptionsPricingTrainer()
        else:
            # Use streaming for low-memory GPUs
            from gpu_options_pricing_streaming import StreamingOptionsPricer
            return StreamingOptionsPricer(batch_size=self.config['batch_size'])
    
    def adapt_hft_engine(self):
        """Adapt HFT engine for current GPU"""
        if 'a100' in torch.cuda.get_device_name(0).lower():
            # Use advanced features on A100
            from gpu_cluster_hft_a100 import A100HFTEngine
            return A100HFTEngine()
        elif self.config.get('memory_gb', 0) >= 16:
            # Use standard HFT for mid-range GPUs
            from gpu_cluster_hft_working import GPUArbitrageScanner
            return GPUArbitrageScanner()
        else:
            # Use memory-efficient version
            from gpu_cluster_hft_lite import LiteHFTEngine
            return LiteHFTEngine()
    
    def adapt_training_system(self):
        """Adapt training system for current GPU"""
        from production_gpu_trainer_working import ProductionGPUTrainer
        
        # Adjust config based on GPU
        training_config = {
            'batch_size': self.config['batch_size'],
            'gradient_checkpointing': self.config.get('gradient_checkpointing', False),
            'mixed_precision': self.config['mixed_precision'],
            'num_workers': self.config['num_workers']
        }
        
        return ProductionGPUTrainer(training_config)
```

### 2. Environment Variables Configuration

Create `set_gpu_env.sh`:

```bash
#!/bin/bash
# GPU Environment Setup Script

# Detect GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

echo "Detected GPU: $GPU_NAME"

# Set environment variables based on GPU
case "$GPU_NAME" in
    *"A100"*)
        export CUDA_MEMORY_FRACTION=0.95
        export TORCH_CUDNN_V8_API_ENABLED=1
        export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
        export CUDA_LAUNCH_BLOCKING=0
        echo "Configured for A100"
        ;;
    *"V100"*)
        export CUDA_MEMORY_FRACTION=0.90
        export TORCH_CUDNN_V8_API_ENABLED=0
        export CUDA_LAUNCH_BLOCKING=0
        echo "Configured for V100"
        ;;
    *"RTX 4090"*)
        export CUDA_MEMORY_FRACTION=0.95
        export TORCH_CUDNN_V8_API_ENABLED=1
        export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
        echo "Configured for RTX 4090"
        ;;
    *"RTX 3090"*)
        export CUDA_MEMORY_FRACTION=0.90
        export TORCH_CUDNN_V8_API_ENABLED=1
        export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
        echo "Configured for RTX 3090"
        ;;
    *"RTX 3080"*)
        export CUDA_MEMORY_FRACTION=0.85
        export TORCH_CUDNN_V8_API_ENABLED=1
        echo "Configured for RTX 3080"
        ;;
    *"RTX 3070"*)
        export CUDA_MEMORY_FRACTION=0.80
        export TORCH_CUDNN_V8_API_ENABLED=1
        echo "Configured for RTX 3070"
        ;;
    *"RTX 3060"*)
        export CUDA_MEMORY_FRACTION=0.80
        export TORCH_CUDNN_V8_API_ENABLED=1
        echo "Configured for RTX 3060"
        ;;
    *"RTX 3050"*)
        export CUDA_MEMORY_FRACTION=0.75
        export TORCH_CUDNN_V8_API_ENABLED=0
        echo "Configured for RTX 3050"
        ;;
    *"T4"*)
        export CUDA_MEMORY_FRACTION=0.85
        export TORCH_CUDNN_V8_API_ENABLED=0
        echo "Configured for T4"
        ;;
    *)
        export CUDA_MEMORY_FRACTION=0.80
        echo "Using default GPU configuration"
        ;;
esac

# Common settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run GPU detection
python detect_gpu_environment.py

echo "GPU environment configured successfully"
```

## Performance Tuning Guide

### 1. Memory Optimization

Create `gpu_memory_optimizer.py`:

```python
#!/usr/bin/env python3
"""
GPU Memory Optimization Utilities
"""

import torch
import gc
import logging

logger = logging.getLogger(__name__)

class GPUMemoryOptimizer:
    @staticmethod
    def optimize_batch_size(model, input_shape, starting_batch=32, max_batch=512):
        """Find optimal batch size for current GPU"""
        device = next(model.parameters()).device
        
        # Binary search for optimal batch size
        low, high = 1, max_batch
        optimal_batch = starting_batch
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Clear cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Try batch size
                dummy_input = torch.randn(mid, *input_shape, device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # If successful, try larger
                optimal_batch = mid
                low = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                else:
                    raise e
            finally:
                # Clean up
                if 'dummy_input' in locals():
                    del dummy_input
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    @staticmethod
    def enable_memory_efficient_mode(model):
        """Enable memory efficient attention and other optimizations"""
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Use memory efficient attention (if available)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Memory efficient attention enabled")
        
        return model
```

### 2. Multi-GPU Support

Create `multi_gpu_manager.py`:

```python
#!/usr/bin/env python3
"""
Multi-GPU Management for Distributed Training
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class MultiGPUManager:
    def __init__(self):
        self.world_size = torch.cuda.device_count()
        self.is_distributed = self.world_size > 1
        
    def setup_distributed(self, rank):
        """Setup distributed training"""
        if not self.is_distributed:
            return
            
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(rank)
        
    def wrap_model(self, model, device_ids=None):
        """Wrap model for distributed training"""
        if self.is_distributed:
            model = DDP(model, device_ids=device_ids)
        return model
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()
```

## Usage Examples

### 1. Basic Usage

```python
# Auto-detect and configure GPU
from detect_gpu_environment import GPUEnvironmentDetector
from adaptive_gpu_manager import AdaptiveGPUManager

# Detect GPU
detector = GPUEnvironmentDetector()
detector.print_info()
detector.save_config()

# Use adaptive manager
gpu_manager = AdaptiveGPUManager()
device = gpu_manager.device

# Load model with optimizations
model = YourModel()
model = gpu_manager.wrap_model(model)

# Get optimal DataLoader settings
loader_kwargs = gpu_manager.get_data_loader_kwargs()
train_loader = DataLoader(dataset, **loader_kwargs)

# Training with mixed precision
scaler = gpu_manager.get_grad_scaler()

for batch in train_loader:
    with gpu_manager.get_autocast_context():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Docker Usage

```bash
# Build with specific CUDA version
docker build \
  --build-arg CUDA_VERSION=12.1.0 \
  --build-arg CUDA_VERSION_SHORT=121 \
  -t alpaca-trading:cuda121 \
  -f Dockerfile.multi-gpu .

# Run with GPU
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e GPU_AUTO_DETECT=1 \
  alpaca-trading:cuda121
```

### 3. Kubernetes Deployment

```bash
# Deploy with GPU node selector
kubectl apply -f k8s/gpu-deployment.yaml

# Check GPU allocation
kubectl describe pod alpaca-gpu-trading-xxxxx | grep nvidia.com/gpu
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   export CUDA_MEMORY_FRACTION=0.7
   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Multi-GPU Communication**
   ```bash
   # Set NCCL debug
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

## Best Practices

1. **Always run GPU detection on startup**
2. **Use environment-specific configurations**
3. **Monitor GPU memory usage**
4. **Enable mixed precision when possible**
5. **Use gradient checkpointing for large models**
6. **Profile before optimizing**

## Conclusion

This guide provides a comprehensive framework for seamlessly switching between different GPU environments. The auto-detection system, adaptive configuration, and optimization strategies ensure optimal performance across various GPU types and deployment scenarios.