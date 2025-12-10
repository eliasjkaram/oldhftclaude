# GPU Integration Developer Guide

## Overview
This guide ensures that ALL parts of the Alpaca trading codebase can leverage GPU acceleration and provides clear instructions for adding GPU support to new code.

## Table of Contents
1. [Current Integration Status](#current-integration-status)
2. [How to Use GPU in Existing Code](#how-to-use-gpu-in-existing-code)
3. [Adding GPU Support to New Code](#adding-gpu-support-to-new-code)
4. [Integration Patterns](#integration-patterns)
5. [Best Practices](#best-practices)
6. [Testing GPU Code](#testing-gpu-code)
7. [Troubleshooting](#troubleshooting)

## Current Integration Status

### ✅ Fully GPU-Integrated Components
- Options pricing models
- Wheel strategy execution
- High-frequency trading engines
- Machine learning models (LSTM, Transformer)
- Backtesting systems
- Risk calculations
- Portfolio optimization

### ⚠️ Partially Integrated Components
- Data preprocessing (MinIO integration needs GPU)
- Real-time market data processing
- Order execution systems
- Paper trading simulators

### ❌ Not Yet GPU-Integrated
- REST API endpoints
- Database operations
- File I/O operations
- Network communications

## How to Use GPU in Existing Code

### Method 1: Direct Adapter Usage (Recommended)
```python
from gpu_adapter_comprehensive import ComprehensiveGPUAdapter

# Initialize once at module level
gpu_adapter = ComprehensiveGPUAdapter()

# In your function/class
def calculate_options_prices(options_data):
    # Get GPU-optimized options pricer
    pricer = gpu_adapter.adapt_options_pricing()
    
    # Use it - automatically optimized for your GPU!
    prices = pricer.price_options(options_data)
    return prices
```

### Method 2: Import with GPU Detection
```python
from adaptive_gpu_manager import AdaptiveGPUManager

class YourTradingSystem:
    def __init__(self):
        # Initialize GPU manager
        self.gpu_manager = AdaptiveGPUManager()
        self.device = self.gpu_manager.device
        
        # Your code automatically uses optimal device
        self.model = YourModel().to(self.device)
```

### Method 3: Retrofit Existing Code
```python
# Before (CPU only)
def calculate_indicators(data):
    import numpy as np
    return np.mean(data, axis=0)

# After (GPU-accelerated)
def calculate_indicators(data):
    from gpu_adapter_comprehensive import ComprehensiveGPUAdapter
    adapter = ComprehensiveGPUAdapter()
    
    import torch
    # Convert to GPU tensor
    data_gpu = torch.tensor(data, device=adapter.device)
    result = torch.mean(data_gpu, dim=0)
    
    # Convert back if needed
    return result.cpu().numpy()
```

## Adding GPU Support to New Code

### Step 1: Create GPU-Aware Module
```python
# new_gpu_module.py
import torch
import torch.nn as nn
from adaptive_gpu_manager import AdaptiveGPUManager

class GPUAcceleratedModule:
    def __init__(self, config_path='gpu_config.json'):
        # Initialize GPU manager
        self.gpu_manager = AdaptiveGPUManager(config_path)
        self.device = self.gpu_manager.device
        self.config = self.gpu_manager.config
        
        # Initialize your components
        self._init_components()
    
    def _init_components(self):
        # Adapt based on GPU capabilities
        if self.config.get('memory_gb', 0) >= 8:
            # Use large model for high-memory GPUs
            self.model = self._create_large_model()
        else:
            # Use small model for low-memory GPUs
            self.model = self._create_small_model()
        
        # Move to GPU and wrap with optimizations
        self.model = self.gpu_manager.wrap_model(self.model)
    
    def _create_large_model(self):
        return nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
    
    def _create_small_model(self):
        return nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).to(self.device)
    
    def process(self, data):
        # Ensure data is on correct device
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        else:
            data = torch.tensor(data, device=self.device)
        
        # Process with GPU
        with torch.no_grad():
            result = self.model(data)
        
        return result
```

### Step 2: Add to GPU Adapter
```python
# In gpu_adapter_comprehensive.py, add:

def adapt_new_module(self):
    """Adapt new GPU module"""
    try:
        from new_gpu_module import GPUAcceleratedModule
        return GPUAcceleratedModule()
    except ImportError:
        return self._create_fallback_new_module()

def _create_fallback_new_module(self):
    """Fallback for new module"""
    class FallbackModule:
        def __init__(self):
            self.device = 'cpu'
        
        def process(self, data):
            # CPU implementation
            return data
    
    return FallbackModule()
```

### Step 3: Register in Adapter
```python
# In get_all_comprehensive_adapters(), add:
adapters['new_module'] = self.adapt_new_module()
```

## Integration Patterns

### Pattern 1: GPU-First Design
```python
class GPUFirstComponent:
    def __init__(self):
        from gpu_adapter_comprehensive import ComprehensiveGPUAdapter
        self.adapter = ComprehensiveGPUAdapter()
        self.device = self.adapter.device
        
        # All operations default to GPU
        self.default_dtype = torch.float32
        self.default_device = self.device
    
    def compute(self, x, y):
        # Automatically on GPU
        x = torch.as_tensor(x, dtype=self.default_dtype, device=self.default_device)
        y = torch.as_tensor(y, dtype=self.default_dtype, device=self.default_device)
        return x @ y
```

### Pattern 2: Hybrid CPU/GPU
```python
class HybridComponent:
    def __init__(self):
        from gpu_adapter_comprehensive import ComprehensiveGPUAdapter
        self.adapter = ComprehensiveGPUAdapter()
        
    def process(self, data):
        # Small data - use CPU
        if len(data) < 1000:
            return self._cpu_process(data)
        
        # Large data - use GPU
        return self._gpu_process(data)
    
    def _cpu_process(self, data):
        import numpy as np
        return np.mean(data)
    
    def _gpu_process(self, data):
        import torch
        data_gpu = torch.tensor(data, device=self.adapter.device)
        return torch.mean(data_gpu).item()
```

### Pattern 3: Async GPU Operations
```python
class AsyncGPUComponent:
    def __init__(self):
        from gpu_adapter_comprehensive import ComprehensiveGPUAdapter
        self.adapter = ComprehensiveGPUAdapter()
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    async def process_async(self, data):
        if self.stream:
            with torch.cuda.stream(self.stream):
                result = self._gpu_compute(data)
                self.stream.synchronize()
        else:
            result = self._cpu_compute(data)
        
        return result
```

## Best Practices

### 1. Always Use the Adapter
```python
# ❌ Bad - hardcoded GPU usage
device = torch.device('cuda:0')

# ✅ Good - adaptive GPU usage
from gpu_adapter_comprehensive import ComprehensiveGPUAdapter
adapter = ComprehensiveGPUAdapter()
device = adapter.device
```

### 2. Handle Memory Gracefully
```python
# ✅ Good - memory-aware processing
def process_large_dataset(data, adapter):
    batch_size = adapter.config['batch_size']
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        try:
            result = gpu_process(batch)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            torch.cuda.empty_cache()
            batch_size //= 2
            result = gpu_process(batch[:batch_size])
            results.append(result)
    
    return results
```

### 3. Profile Before Optimizing
```python
# ✅ Good - profile to ensure GPU is beneficial
def smart_compute(data):
    adapter = ComprehensiveGPUAdapter()
    
    # Profile overhead
    if len(data) < 10000:  # Small data
        return cpu_compute(data)  # CPU is faster
    else:  # Large data
        return gpu_compute(data, adapter)  # GPU is faster
```

### 4. Test Across GPUs
```python
# ✅ Good - test on multiple GPU configurations
def test_my_gpu_code():
    # Test configurations
    test_configs = [
        {'memory_gb': 4, 'name': 'RTX 3050'},
        {'memory_gb': 8, 'name': 'RTX 3070'},
        {'memory_gb': 24, 'name': 'RTX 3090'},
        {'memory_gb': 80, 'name': 'A100'}
    ]
    
    for config in test_configs:
        print(f"Testing on {config['name']}...")
        # Simulate different GPU configs
        test_with_config(config)
```

## Testing GPU Code

### Unit Test Template
```python
# test_gpu_module.py
import unittest
import torch
from gpu_adapter_comprehensive import ComprehensiveGPUAdapter

class TestGPUModule(unittest.TestCase):
    def setUp(self):
        self.adapter = ComprehensiveGPUAdapter()
        self.module = self.adapter.adapt_new_module()
    
    def test_gpu_computation(self):
        # Test data
        data = torch.randn(100, 100)
        
        # Run computation
        result = self.module.process(data)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.device, self.adapter.device)
    
    def test_memory_efficiency(self):
        if self.adapter.device.type == 'cuda':
            # Check memory before
            before = torch.cuda.memory_allocated()
            
            # Process data
            data = torch.randn(1000, 1000)
            result = self.module.process(data)
            
            # Check memory after
            after = torch.cuda.memory_allocated()
            
            # Should not leak memory
            self.assertLess(after - before, 100 * 1024 * 1024)  # <100MB
    
    def test_cpu_fallback(self):
        # Force CPU mode
        self.module.device = torch.device('cpu')
        
        # Should still work
        data = torch.randn(100, 100)
        result = self.module.process(data)
        self.assertIsNotNone(result)
```

### Integration Test Template
```python
# test_integration.py
def test_full_pipeline_gpu():
    adapter = ComprehensiveGPUAdapter()
    
    # Test complete pipeline
    # 1. Data loading
    data_loader = adapter.adapt_data_loader()
    
    # 2. Preprocessing
    preprocessor = adapter.adapt_preprocessor()
    
    # 3. Model inference
    model = adapter.adapt_trading_ai()
    
    # 4. Post-processing
    postprocessor = adapter.adapt_postprocessor()
    
    # Run pipeline
    for batch in data_loader:
        processed = preprocessor(batch)
        predictions = model(processed)
        results = postprocessor(predictions)
    
    print("✅ Full pipeline GPU test passed")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Solution: Adaptive batch sizing
try:
    result = gpu_process(large_batch)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Process in smaller batches
    results = []
    for small_batch in chunk_data(large_batch, size=adapter.config['batch_size']//2):
        results.append(gpu_process(small_batch))
    result = combine_results(results)
```

#### 2. Module Not Found
```python
# Solution: Graceful fallback
try:
    from gpu_accelerated_module import GPUModule
    module = GPUModule()
except ImportError:
    from cpu_fallback_module import CPUModule
    module = CPUModule()
    print("Warning: Using CPU fallback")
```

#### 3. Device Mismatch
```python
# Solution: Ensure consistent device usage
def ensure_same_device(tensor1, tensor2, target_device=None):
    if target_device is None:
        target_device = tensor1.device
    
    tensor1 = tensor1.to(target_device)
    tensor2 = tensor2.to(target_device)
    
    return tensor1, tensor2
```

## Integration Checklist

When adding GPU support to new code:

- [ ] Import ComprehensiveGPUAdapter
- [ ] Initialize adapter at module level
- [ ] Use adapter.device for all tensor operations
- [ ] Implement CPU fallback
- [ ] Add memory management for large operations
- [ ] Test on multiple GPU configurations
- [ ] Update gpu_adapter_comprehensive.py
- [ ] Add unit tests
- [ ] Update documentation
- [ ] Profile performance improvement

## Example: Full Integration

Here's a complete example of adding GPU support to a new trading indicator:

```python
# new_indicator.py
import torch
import numpy as np
from gpu_adapter_comprehensive import ComprehensiveGPUAdapter

class GPUAcceleratedRSI:
    def __init__(self):
        self.adapter = ComprehensiveGPUAdapter()
        self.device = self.adapter.device
        
    def calculate(self, prices, period=14):
        """Calculate RSI with GPU acceleration"""
        # Convert to GPU tensor
        if isinstance(prices, np.ndarray):
            prices = torch.from_numpy(prices).float()
        
        prices = prices.to(self.device)
        
        # Calculate price changes
        deltas = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # Calculate average gains and losses
        avg_gains = self._moving_average(gains, period)
        avg_losses = self._moving_average(losses, period)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Return as numpy if input was numpy
        if isinstance(prices, np.ndarray):
            return rsi.cpu().numpy()
        
        return rsi
    
    def _moving_average(self, data, period):
        """GPU-accelerated moving average"""
        kernel = torch.ones(period, device=self.device) / period
        
        # Use convolution for moving average
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)
        
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # Pad data
        padding = period - 1
        data_padded = torch.nn.functional.pad(data, (padding, 0), mode='constant', value=0)
        
        # Convolve
        result = torch.nn.functional.conv1d(data_padded, kernel)
        
        return result.squeeze()

# Add to adapter
# In gpu_adapter_comprehensive.py:
def adapt_rsi_indicator(self):
    try:
        from new_indicator import GPUAcceleratedRSI
        return GPUAcceleratedRSI()
    except ImportError:
        return self._create_cpu_rsi()
```

## Conclusion

With this guide, ANY part of the Alpaca trading codebase can now leverage GPU acceleration, and developers can easily add GPU support to new code. The key is to always use the ComprehensiveGPUAdapter, which handles all the complexity of GPU detection, configuration, and optimization automatically.