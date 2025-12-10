#!/usr/bin/env python3
"""
GPU Environment Auto-Detection and Configuration
"""

import torch
import os
import json
import subprocess
from pathlib import Path
import platform
import psutil

class GPUEnvironmentDetector:
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.system_info = self._detect_system()
        self.config = self._generate_config()
        
    def _detect_system(self):
        """Detect system specifications"""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version()
        }
        return info
        
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
            info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            info['cudnn_enabled'] = torch.backends.cudnn.enabled
            
            for i in range(info['count']):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': device_props.name,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'memory_mb': device_props.total_memory // (1024**2),
                    'multi_processor_count': device_props.multi_processor_count
                }
                
                # Add optional properties that may not exist in all PyTorch versions
                optional_props = [
                    'max_threads_per_block',
                    'max_threads_per_multiprocessor', 
                    'warp_size',
                    'max_shared_memory_per_block',
                    'max_registers_per_block',
                    'clock_rate',
                    'memory_clock_rate',
                    'memory_bus_width',
                    'l2_cache_size'
                ]
                
                for prop in optional_props:
                    if hasattr(device_props, prop):
                        device_info[prop] = getattr(device_props, prop)
                
                # Get current GPU utilization if nvidia-ml-py is available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get current stats
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    
                    device_info['current_utilization'] = {
                        'gpu_util_percent': util.gpu,
                        'memory_util_percent': util.memory,
                        'memory_used_mb': mem_info.used // (1024**2),
                        'memory_free_mb': mem_info.free // (1024**2),
                        'temperature_c': temp,
                        'power_watts': power
                    }
                    
                    pynvml.nvmlShutdown()
                except:
                    pass
                
                info['devices'].append(device_info)
                
        return info
    
    def _generate_config(self):
        """Generate optimal configuration based on GPU"""
        config = {
            'device': 'cpu',
            'mixed_precision': False,
            'batch_size': 32,
            'num_workers': min(4, self.system_info['cpu_count'] // 2),
            'pin_memory': False,
            'optimization_level': 'O0',
            'prefetch_factor': 2,
            'persistent_workers': False
        }
        
        if not self.gpu_info['available']:
            return config
            
        # Get primary GPU info
        gpu = self.gpu_info['devices'][0]
        gpu_name = gpu['name'].lower()
        memory_gb = gpu['memory_gb']
        compute_capability = gpu['compute_capability']
        
        # Set device
        config['device'] = 'cuda'
        config['pin_memory'] = True
        config['persistent_workers'] = True
        
        # GPU-specific configurations
        if 'a100' in gpu_name:
            config.update(self._config_a100(memory_gb))
        elif 'a6000' in gpu_name:
            config.update(self._config_a6000(memory_gb))
        elif 'v100' in gpu_name:
            config.update(self._config_v100(memory_gb))
        elif 'rtx 4090' in gpu_name:
            config.update(self._config_rtx4090(memory_gb))
        elif 'rtx 4080' in gpu_name:
            config.update(self._config_rtx4080(memory_gb))
        elif 'rtx 4070' in gpu_name:
            config.update(self._config_rtx4070(memory_gb))
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
        elif 'a10' in gpu_name:
            config.update(self._config_a10(memory_gb))
        elif 'a40' in gpu_name:
            config.update(self._config_a40(memory_gb))
        else:
            config.update(self._config_generic(memory_gb, compute_capability))
            
        # Add compute capability
        config['compute_capability'] = compute_capability
        
        # Adjust based on available system RAM
        if self.system_info['ram_gb'] < 16:
            config['num_workers'] = min(2, config['num_workers'])
            config['prefetch_factor'] = 2
        elif self.system_info['ram_gb'] < 32:
            config['num_workers'] = min(4, config['num_workers'])
            config['prefetch_factor'] = 2
        else:
            config['prefetch_factor'] = 4
            
        return config
    
    def _config_a100(self, memory_gb):
        """NVIDIA A100 configuration (40GB/80GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 512 if memory_gb >= 80 else 256,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': True,
            'max_split_size_mb': 512
        }
    
    def _config_a6000(self, memory_gb):
        """NVIDIA A6000 configuration (48GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 256,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': True,
            'max_split_size_mb': 512
        }
    
    def _config_v100(self, memory_gb):
        """NVIDIA V100 configuration (16GB/32GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 128 if memory_gb >= 32 else 64,
            'num_workers': 8,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': True,
            'gradient_checkpointing': memory_gb < 32,
            'compile_model': False,
            'use_flash_attention': False,
            'max_split_size_mb': 256
        }
    
    def _config_rtx4090(self, memory_gb):
        """NVIDIA RTX 4090 configuration (24GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 128,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': True,
            'max_split_size_mb': 256
        }
    
    def _config_rtx4080(self, memory_gb):
        """NVIDIA RTX 4080 configuration (16GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 96,
            'num_workers': 6,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': True,
            'max_split_size_mb': 256
        }
    
    def _config_rtx4070(self, memory_gb):
        """NVIDIA RTX 4070 Ti configuration (12GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 64,
            'num_workers': 6,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': True,
            'compile_model': True,
            'use_flash_attention': False,
            'max_split_size_mb': 128
        }
    
    def _config_rtx3090(self, memory_gb):
        """NVIDIA RTX 3090 configuration (24GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 96,
            'num_workers': 6,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': False,
            'max_split_size_mb': 256
        }
    
    def _config_rtx3080(self, memory_gb):
        """NVIDIA RTX 3080 configuration (10GB/12GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 48 if memory_gb >= 12 else 32,
            'num_workers': 6,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': memory_gb < 12,
            'compile_model': True,
            'use_flash_attention': False,
            'max_split_size_mb': 128
        }
    
    def _config_rtx3070(self, memory_gb):
        """NVIDIA RTX 3070 configuration (8GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 32,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': False,
            'gradient_checkpointing': True,
            'compile_model': True,
            'use_flash_attention': False,
            'max_split_size_mb': 128
        }
    
    def _config_rtx3060(self, memory_gb):
        """NVIDIA RTX 3060 configuration (12GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 48,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': True,
            'use_channels_last': False,
            'gradient_checkpointing': False,
            'compile_model': False,
            'use_flash_attention': False,
            'max_split_size_mb': 128
        }
    
    def _config_rtx3050(self, memory_gb):
        """NVIDIA RTX 3050 configuration (4GB/8GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 16 if memory_gb <= 4 else 24,
            'num_workers': 2,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': False,
            'gradient_checkpointing': True,
            'compile_model': False,
            'use_flash_attention': False,
            'max_split_size_mb': 64
        }
    
    def _config_t4(self, memory_gb):
        """NVIDIA T4 configuration (16GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 32,
            'num_workers': 4,
            'optimization_level': 'O1',
            'use_tf32': False,
            'use_channels_last': True,
            'gradient_checkpointing': True,
            'compile_model': False,
            'use_flash_attention': False,
            'max_split_size_mb': 128
        }
    
    def _config_a10(self, memory_gb):
        """NVIDIA A10 configuration (24GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 64,
            'num_workers': 6,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': False,
            'max_split_size_mb': 256
        }
    
    def _config_a40(self, memory_gb):
        """NVIDIA A40 configuration (48GB)"""
        return {
            'mixed_precision': True,
            'batch_size': 128,
            'num_workers': 8,
            'optimization_level': 'O2',
            'use_tf32': True,
            'use_channels_last': True,
            'gradient_checkpointing': False,
            'compile_model': True,
            'use_flash_attention': True,
            'max_split_size_mb': 512
        }
    
    def _config_generic(self, memory_gb, compute_capability):
        """Generic GPU configuration"""
        # Parse compute capability
        major, minor = map(int, compute_capability.split('.'))
        
        # Determine features based on compute capability
        supports_tf32 = major >= 8  # Ampere and newer
        supports_flash_attention = major >= 8
        supports_compile = major >= 7  # Volta and newer
        
        if memory_gb >= 24:
            batch_size = 64
            gradient_checkpointing = False
            num_workers = 6
        elif memory_gb >= 16:
            batch_size = 48
            gradient_checkpointing = False
            num_workers = 4
        elif memory_gb >= 8:
            batch_size = 32
            gradient_checkpointing = True
            num_workers = 4
        else:
            batch_size = 16
            gradient_checkpointing = True
            num_workers = 2
            
        return {
            'mixed_precision': memory_gb >= 8,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'optimization_level': 'O1' if memory_gb >= 8 else 'O0',
            'use_tf32': supports_tf32,
            'use_channels_last': memory_gb >= 16,
            'gradient_checkpointing': gradient_checkpointing,
            'compile_model': supports_compile and memory_gb >= 16,
            'use_flash_attention': supports_flash_attention and memory_gb >= 16,
            'max_split_size_mb': min(128, int(memory_gb * 1024 / 16))
        }
    
    def save_config(self, path='gpu_config.json'):
        """Save configuration to file"""
        config_data = {
            'timestamp': str(torch.cuda.Event(enable_timing=True).record()),
            'system_info': self.system_info,
            'gpu_info': self.gpu_info,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"GPU configuration saved to {path}")
        
    def print_info(self):
        """Print GPU information"""
        print("="*80)
        print("GPU Environment Detection Report")
        print("="*80)
        
        # System info
        print("\nSystem Information:")
        print(f"  Platform: {self.system_info['platform']} {self.system_info['platform_release']}")
        print(f"  Architecture: {self.system_info['architecture']}")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  RAM: {self.system_info['ram_gb']:.1f} GB")
        print(f"  Python: {self.system_info['python_version']}")
        
        if not self.gpu_info['available']:
            print("\n❌ No GPU detected. Using CPU configuration.")
            return
            
        print(f"\n✅ CUDA Available: {self.gpu_info['available']}")
        print(f"  CUDA Version: {self.gpu_info['cuda_version']}")
        print(f"  cuDNN Version: {self.gpu_info['cudnn_version']}")
        print(f"  Number of GPUs: {self.gpu_info['count']}")
        
        for device in self.gpu_info['devices']:
            print(f"\nGPU {device['index']}: {device['name']}")
            print(f"  Compute Capability: {device['compute_capability']}")
            print(f"  Memory: {device['memory_gb']} GB ({device['memory_mb']} MB)")
            print(f"  Multiprocessors: {device['multi_processor_count']}")
            if 'max_threads_per_block' in device:
                print(f"  Max Threads/Block: {device['max_threads_per_block']}")
            if 'clock_rate' in device:
                print(f"  Clock Rate: {device['clock_rate'] / 1000:.1f} MHz")
            if 'memory_clock_rate' in device:
                print(f"  Memory Clock: {device['memory_clock_rate'] / 1000:.1f} MHz")
            if 'memory_bus_width' in device:
                print(f"  Memory Bus Width: {device['memory_bus_width']} bits")
            
            if 'current_utilization' in device:
                util = device['current_utilization']
                print(f"\n  Current Utilization:")
                print(f"    GPU: {util['gpu_util_percent']}%")
                print(f"    Memory: {util['memory_util_percent']}% ({util['memory_used_mb']} MB used)")
                print(f"    Temperature: {util['temperature_c']}°C")
                print(f"    Power: {util['power_watts']:.1f}W")
        
        print("\n" + "-"*80)
        print("Optimal Configuration:")
        print("-"*80)
        for key, value in sorted(self.config.items()):
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)

    def get_environment_variables(self):
        """Get recommended environment variables"""
        env_vars = {}
        
        if self.gpu_info['available']:
            gpu = self.gpu_info['devices'][0]
            gpu_name = gpu['name'].lower()
            
            # Memory fraction based on GPU
            if gpu['memory_gb'] <= 4:
                env_vars['CUDA_MEMORY_FRACTION'] = '0.75'
            elif gpu['memory_gb'] <= 8:
                env_vars['CUDA_MEMORY_FRACTION'] = '0.80'
            elif gpu['memory_gb'] <= 16:
                env_vars['CUDA_MEMORY_FRACTION'] = '0.85'
            else:
                env_vars['CUDA_MEMORY_FRACTION'] = '0.95'
            
            # TF32 for Ampere GPUs
            if self.config.get('use_tf32', False):
                env_vars['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
            
            # Common settings
            env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            env_vars['OMP_NUM_THREADS'] = str(min(4, self.system_info['cpu_count'] // 2))
            env_vars['MKL_NUM_THREADS'] = str(min(4, self.system_info['cpu_count'] // 2))
            
            # NCCL settings for multi-GPU
            if self.gpu_info['count'] > 1:
                env_vars['NCCL_IB_DISABLE'] = '1'
                env_vars['NCCL_P2P_DISABLE'] = '1'
        
        return env_vars

if __name__ == "__main__":
    import sys
    
    # Check for psutil
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Run detection
    detector = GPUEnvironmentDetector()
    detector.print_info()
    detector.save_config()
    
    # Print environment variables
    print("\nRecommended Environment Variables:")
    print("-"*80)
    env_vars = detector.get_environment_variables()
    for key, value in env_vars.items():
        print(f"export {key}={value}")
    
    # Save to shell script
    with open('set_gpu_env.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated GPU environment settings\n\n")
        for key, value in env_vars.items():
            f.write(f"export {key}={value}\n")
        f.write("\necho 'GPU environment variables set successfully'\n")
    
    os.chmod('set_gpu_env.sh', 0o755)
    print(f"\nEnvironment script saved to: set_gpu_env.sh")
    print("Run: source set_gpu_env.sh")