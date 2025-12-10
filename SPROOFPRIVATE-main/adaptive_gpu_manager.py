#!/usr/bin/env python3
"""
Adaptive GPU Manager for Different Environments
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
from contextlib import nullcontext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveGPUManager:
    """Manages GPU resources adaptively based on detected hardware"""
    
    def __init__(self, config_path: str = 'gpu_config.json', force_cpu: bool = False):
        """
        Initialize adaptive GPU manager
        
        Args:
            config_path: Path to GPU configuration file
            force_cpu: Force CPU usage even if GPU is available
        """
        self.config_path = config_path
        self.force_cpu = force_cpu
        self.config = self._load_config()
        self.device = self._setup_device()
        self._apply_optimizations()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load GPU configuration from file or detect if not exists"""
        if os.path.exists(self.config_path):
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return data['config']
        else:
            logger.info("Configuration not found, running GPU detection...")
            from detect_gpu_environment import GPUEnvironmentDetector
            detector = GPUEnvironmentDetector()
            detector.save_config(self.config_path)
            return detector.config
    
    def _setup_device(self) -> torch.device:
        """Setup PyTorch device based on configuration"""
        if self.force_cpu:
            logger.info("Forced CPU mode enabled")
            return torch.device('cpu')
            
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            # Select GPU
            gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
            device = torch.device(f'cuda:{gpu_id}')
            
            # Set memory fraction if specified
            if os.environ.get('CUDA_MEMORY_FRACTION'):
                fraction = float(os.environ['CUDA_MEMORY_FRACTION'])
                torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)
                logger.info(f"Set GPU memory fraction to {fraction}")
            
            # Set memory allocator settings
            if 'max_split_size_mb' in self.config:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{self.config['max_split_size_mb']}"
                
            logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f}GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU (GPU not available or not configured)")
            
        return device
    
    def _apply_optimizations(self):
        """Apply GPU-specific optimizations"""
        if self.device.type == 'cuda':
            # Enable TF32 on Ampere GPUs
            if self.config.get('use_tf32', False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for Ampere GPU")
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            
            # Enable cudnn benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmarking enabled")
            
            # Set cudnn deterministic if needed
            if os.environ.get('CUDNN_DETERMINISTIC', '0') == '1':
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("cuDNN deterministic mode enabled")
    
    def get_device(self) -> torch.device:
        """Get the configured device"""
        return self.device
    
    def get_data_loader_kwargs(self, dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get optimal DataLoader arguments
        
        Args:
            dataset_size: Size of dataset to optimize batch size
            
        Returns:
            Dictionary of DataLoader kwargs
        """
        kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'] and self.device.type == 'cuda',
            'persistent_workers': self.config.get('persistent_workers', False) and self.config['num_workers'] > 0,
            'prefetch_factor': self.config.get('prefetch_factor', 2)
        }
        
        # Adjust batch size for small datasets
        if dataset_size and dataset_size < kwargs['batch_size']:
            kwargs['batch_size'] = max(1, dataset_size // 4)
            logger.info(f"Adjusted batch size to {kwargs['batch_size']} for small dataset")
        
        return kwargs
    
    def wrap_model(self, model: nn.Module, 
                   distributed: bool = False,
                   find_unused_parameters: bool = False) -> nn.Module:
        """
        Wrap model with GPU optimizations
        
        Args:
            model: PyTorch model to wrap
            distributed: Whether to use DistributedDataParallel
            find_unused_parameters: For DDP, whether to find unused parameters
            
        Returns:
            Wrapped model
        """
        # Move model to device
        model = model.to(self.device)
        
        # Apply optimizations
        if self.device.type == 'cuda':
            # Compile model if supported (PyTorch 2.0+)
            if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Use channels last memory format for vision models
            if self.config.get('use_channels_last', False):
                try:
                    model = model.to(memory_format=torch.channels_last)
                    logger.info("Using channels_last memory format")
                except Exception:
                    pass
            
            # Enable gradient checkpointing if configured
            if self.config.get('gradient_checkpointing', False):
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
            
            # Distributed training
            if distributed and torch.distributed.is_initialized():
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, 
                           device_ids=[self.device.index] if self.device.index is not None else None,
                           find_unused_parameters=find_unused_parameters)
                logger.info("Model wrapped with DistributedDataParallel")
        
        return model
    
    def get_optimizer(self, model_parameters, 
                      optimizer_class=torch.optim.AdamW,
                      lr: float = 1e-3,
                      **kwargs) -> torch.optim.Optimizer:
        """
        Get optimizer with GPU-specific optimizations
        
        Args:
            model_parameters: Model parameters to optimize
            optimizer_class: Optimizer class to use
            lr: Learning rate
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        optimizer_kwargs = kwargs.copy()
        optimizer_kwargs['lr'] = lr
        
        # Use fused optimizers on newer GPUs if available
        if self.device.type == 'cuda' and hasattr(optimizer_class, 'fused'):
            compute_capability = self.config.get('compute_capability', '0.0')
            major, minor = map(int, compute_capability.split('.'))
            
            # Fused optimizers available on compute capability 7.0+
            if major >= 7:
                optimizer_kwargs['fused'] = True
                logger.info(f"Using fused optimizer for {optimizer_class.__name__}")
        
        return optimizer_class(model_parameters, **optimizer_kwargs)
    
    def get_autocast_context(self, enabled: Optional[bool] = None):
        """
        Get autocast context manager for mixed precision
        
        Args:
            enabled: Override config setting for mixed precision
            
        Returns:
            Context manager for autocast
        """
        use_amp = enabled if enabled is not None else self.config.get('mixed_precision', False)
        
        if use_amp and self.device.type == 'cuda':
            # Get optimization level
            opt_level = self.config.get('optimization_level', 'O1')
            
            # Set dtype based on optimization level
            if opt_level == 'O2':
                dtype = torch.float16
            elif opt_level == 'O3':
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float16
                
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            return nullcontext()
    
    def get_grad_scaler(self, enabled: Optional[bool] = None):
        """
        Get gradient scaler for mixed precision training
        
        Args:
            enabled: Override config setting for mixed precision
            
        Returns:
            GradScaler or dummy scaler
        """
        use_amp = enabled if enabled is not None else self.config.get('mixed_precision', False)
        
        if use_amp and self.device.type == 'cuda':
            # Configure scaler based on optimization level
            opt_level = self.config.get('optimization_level', 'O1')
            
            if opt_level == 'O3' and torch.cuda.is_bf16_supported():
                # BFloat16 doesn't need gradient scaling
                return DummyGradScaler()
            else:
                return torch.amp.GradScaler('cuda')
        else:
            return DummyGradScaler()
    
    def optimize_batch_size(self, model: nn.Module, 
                           input_shape: Union[tuple, list],
                           starting_batch: int = 32,
                           max_batch: int = 512,
                           safety_margin: float = 0.9) -> int:
        """
        Find optimal batch size for current GPU
        
        Args:
            model: Model to test
            input_shape: Shape of single input (without batch dimension)
            starting_batch: Starting batch size for search
            max_batch: Maximum batch size to try
            safety_margin: Memory safety margin (0.9 = use 90% of available)
            
        Returns:
            Optimal batch size
        """
        if self.device.type != 'cuda':
            return self.config['batch_size']
        
        import gc
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Binary search for optimal batch size
        low, high = 1, min(max_batch, self.config.get('batch_size', 32) * 4)
        optimal_batch = starting_batch
        
        logger.info(f"Finding optimal batch size for input shape {input_shape}")
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try batch size
                dummy_input = torch.randn(mid, *input_shape, device=self.device)
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    with self.get_autocast_context():
                        _ = model(dummy_input)
                
                # Check memory usage
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                
                memory_usage = reserved / total_memory
                
                if memory_usage < safety_margin:
                    # If successful and under safety margin, try larger
                    optimal_batch = mid
                    low = mid + 1
                else:
                    # Too close to limit, reduce
                    high = mid - 1
                
                del dummy_input
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise e
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information"""
        if self.device.type != 'cuda':
            return {'device': 'cpu', 'memory_available': True}
        
        info = {
            'device': str(self.device),
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'free_mb': (torch.cuda.get_device_properties(self.device).total_memory - 
                       torch.cuda.memory_allocated(self.device)) / 1024**2,
            'total_mb': torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        }
        
        info['usage_percent'] = (info['allocated_mb'] / info['total_mb']) * 100
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")
    
    def set_random_seed(self, seed: int = 42):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Set numpy seed
        import numpy as np
        np.random.seed(seed)
        
        # Set Python seed
        import random
        random.seed(seed)
        
        logger.info(f"Random seed set to {seed}")
    
    def get_summary(self) -> str:
        """Get summary of GPU configuration"""
        lines = [
            "="*60,
            "Adaptive GPU Manager Summary",
            "="*60,
            f"Device: {self.device}",
            f"Configuration: {self.config_path}",
            ""
        ]
        
        if self.device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(self.device)
            lines.extend([
                f"GPU Name: {gpu_props.name}",
                f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB",
                f"Compute Capability: {gpu_props.major}.{gpu_props.minor}",
                ""
            ])
        
        lines.extend([
            "Active Settings:",
            f"  Batch Size: {self.config['batch_size']}",
            f"  Mixed Precision: {self.config.get('mixed_precision', False)}",
            f"  Gradient Checkpointing: {self.config.get('gradient_checkpointing', False)}",
            f"  Model Compilation: {self.config.get('compile_model', False)}",
            f"  TF32: {self.config.get('use_tf32', False)}",
            "="*60
        ])
        
        return "\n".join(lines)


class DummyGradScaler:
    """Dummy gradient scaler for CPU or when AMP is disabled"""
    
    def scale(self, loss):
        return loss
    
    def step(self, optimizer):
        optimizer.step()
    
    def update(self):
        pass
    
    def unscale_(self, optimizer):
        pass
    
    def get_scale(self):
        return 1.0
    
    def set_scale(self, scale):
        pass
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass


def demo():
    """Demonstration of adaptive GPU manager"""
    print("Adaptive GPU Manager Demo")
    print("-"*60)
    
    # Initialize manager
    manager = AdaptiveGPUManager()
    
    # Print summary
    print(manager.get_summary())
    
    # Get DataLoader kwargs
    loader_kwargs = manager.get_data_loader_kwargs(dataset_size=1000)
    print("\nDataLoader Configuration:")
    for key, value in loader_kwargs.items():
        print(f"  {key}: {value}")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x.view(x.size(0), -1))
    
    # Wrap model
    model = SimpleModel()
    model = manager.wrap_model(model)
    
    # Find optimal batch size
    if manager.device.type == 'cuda':
        optimal_batch = manager.optimize_batch_size(model, (1, 28, 28))
        print(f"\nOptimal batch size: {optimal_batch}")
    
    # Get memory info
    mem_info = manager.get_memory_info()
    print("\nMemory Information:")
    for key, value in mem_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo()