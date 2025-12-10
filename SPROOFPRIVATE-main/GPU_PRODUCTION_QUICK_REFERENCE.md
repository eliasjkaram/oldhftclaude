# GPU Production Quick Reference Guide

## 🚀 Production Deployment Checklist

### Essential Requirements for ANY GPU Script

#### 1. Environment Detection & Setup
```python
# Must have in EVERY GPU script
import torch
import logging

def setup_gpu_environment():
    # Detect GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU Available: {gpu_count} x {gpu_name}")
    else:
        device = torch.device('cpu')
        logging.warning("No GPU available, falling back to CPU")
    
    # Set memory growth for TensorFlow
    if using_tensorflow:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    return device
```

#### 2. Error Handling & Fallbacks
```python
# Wrap ALL GPU operations
try:
    result = gpu_operation()
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Retry with smaller batch or fall back to CPU
    result = cpu_fallback()
except Exception as e:
    logging.error(f"GPU Error: {e}")
    # Graceful degradation
```

#### 3. Resource Management
```python
# Always clean up GPU resources
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### 4. Configuration File
```yaml
# config/gpu_config.yaml
gpu:
  enabled: true
  device_id: 0
  memory_fraction: 0.8
  allow_growth: true
  fallback_to_cpu: true
  
optimization:
  mixed_precision: true
  batch_size: 
    gpu: 1024
    cpu: 32
  
monitoring:
  track_memory: true
  log_utilization: true
  alert_on_oom: true
```

---

## 📋 Critical Production TODOs by Priority

### 🔴 CRITICAL - Must Have Before Production

#### Every GPU Script MUST:
1. **Handle GPU unavailability** → Fall back to CPU
2. **Manage OOM errors** → Reduce batch size or use gradient accumulation
3. **Support multi-GPU** → Use DataParallel or DistributedDataParallel
4. **Log GPU metrics** → Track utilization, memory, temperature
5. **Implement health checks** → `/health` endpoint with GPU status

### 🟡 HIGH PRIORITY - Implement Within First Week

#### Performance Optimizations:
1. **Mixed Precision Training** → Use AMP for 2x speedup
2. **Memory Pooling** → Pre-allocate GPU memory
3. **Batch Size Tuning** → Find optimal batch size per GPU
4. **Data Pipeline Optimization** → GPU-accelerated data loading
5. **Kernel Optimization** → Profile and optimize CUDA kernels

### 🟢 MEDIUM PRIORITY - Implement Within First Month

#### Infrastructure:
1. **Monitoring Dashboards** → Grafana + Prometheus
2. **Auto-scaling** → Based on GPU utilization
3. **Model Versioning** → Track model + GPU config
4. **A/B Testing** → Compare GPU vs CPU performance
5. **Cost Tracking** → Monitor GPU hours and costs

---

## 🛠️ Environment-Specific Configurations

### AWS Deployment
```bash
# Instance Selection
- Development: g4dn.xlarge (T4 GPU, $0.526/hour)
- Production: p3.2xlarge (V100 GPU, $3.06/hour)
- High-Performance: p4d.24xlarge (8x A100, $32.77/hour)

# Required Setup
aws configure set region us-east-1
aws ec2 run-instances --image-id ami-0xxxx --instance-type p3.2xlarge
```

### Docker Configuration
```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Kubernetes Deployment
```yaml
# gpu-deployment.yaml
spec:
  containers:
  - name: gpu-app
    resources:
      limits:
        nvidia.com/gpu: 1
  nodeSelector:
    accelerator: nvidia-tesla-v100
```

---

## 🔍 Quick Debugging Guide

### Common GPU Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size, use gradient accumulation, clear cache |
| CUDA Not Available | Check drivers, CUDA version, use CPU fallback |
| Slow GPU Performance | Check PCIe bandwidth, CPU bottleneck, data loading |
| GPU Not Utilized | Verify operations on GPU, check .to(device) calls |
| Multi-GPU Errors | Check NCCL, use proper distributed backend |

### Essential Debugging Commands
```bash
# Check GPU status
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Profile GPU code
nsys profile python script.py
nvprof python script.py

# Check GPU memory leaks
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

---

## 📊 Performance Benchmarks to Meet

### Minimum Production Requirements

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| GPU Utilization | > 80% | > 60% |
| Memory Efficiency | < 90% used | < 95% used |
| Batch Processing | > 1000/sec | > 100/sec |
| Latency (inference) | < 10ms | < 100ms |
| Throughput | > 10k ops/sec | > 1k ops/sec |

---

## 🚨 Production Go/No-Go Checklist

Before deploying ANY GPU script to production:

- [ ] ✅ GPU detection and CPU fallback implemented
- [ ] ✅ OOM error handling in place
- [ ] ✅ Monitoring and alerting configured
- [ ] ✅ Performance benchmarks met
- [ ] ✅ Multi-GPU support tested
- [ ] ✅ Resource cleanup implemented
- [ ] ✅ Health check endpoint working
- [ ] ✅ Configuration externalized
- [ ] ✅ Logging at appropriate levels
- [ ] ✅ Documentation complete

---

## 📞 Emergency Contacts & Resources

### Quick Links
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch GPU Guide](https://pytorch.org/docs/stable/cuda.html)
- [GPU Monitoring Tools](https://github.com/NVIDIA/gpu-monitoring-tools)

### Troubleshooting Resources
- GPU Driver Issues: `sudo nvidia-bug-report.sh`
- CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/
- Performance Tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

*Use this guide as a quick reference. For detailed implementation, see GPU_PRODUCTION_TODO_LISTS.md*