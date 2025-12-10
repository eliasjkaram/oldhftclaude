# GPU Enhanced Wheel Strategy - Detailed Implementation Guide

## Overview
Transform the existing `gpu_enhanced_wheel.py` into a production-ready, GPU-accelerated wheel strategy that can generate consistent revenue through optimized put selling and covered call strategies.

## Current State Analysis
The script currently has:
- ✅ Basic GPU detection (CuPy)
- ✅ Black-Scholes calculations
- ✅ ML predictor framework
- ✅ Backtesting engine
- ❌ Missing production configuration
- ❌ Missing real-time data integration
- ❌ Missing risk management
- ❌ Missing production monitoring

## Phase 1: Production Configuration (Days 1-2)

### 1.1 Environment Configuration
```python
# File: src/config/wheel_config.yaml
wheel_strategy:
  environment: ${ENVIRONMENT:production}
  
  # Position limits
  max_positions: 10
  max_position_size_pct: 0.1  # 10% of portfolio per position
  max_sector_exposure: 0.3    # 30% max per sector
  
  # Strike selection parameters
  target_delta_put: 0.25      # Target delta for puts
  target_delta_call: 0.30     # Target delta for calls
  min_dte: 7                  # Minimum days to expiration
  max_dte: 45                 # Maximum days to expiration
  preferred_dte: 21           # Preferred DTE
  
  # Risk parameters
  max_drawdown_pct: 0.15      # 15% max drawdown
  stop_loss_pct: 0.5          # 50% loss on premium
  margin_buffer: 1.2          # 20% margin buffer
  
  # GPU settings
  gpu:
    device_ids: [0, 1]
    memory_fraction: 0.8
    batch_size: 10000
    fallback_to_cpu: true
    
  # Data sources
  data:
    primary: alpaca
    fallback: polygon
    options_provider: tradier
    
  # Monitoring
  monitoring:
    prometheus_port: 9090
    health_check_interval: 30
    alert_webhook: ${ALERT_WEBHOOK_URL}
```

### 1.2 Configuration Loader
```python
# Add to gpu_enhanced_wheel.py
import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict
import os

@dataclass
class GPUConfig:
    device_ids: List[int]
    memory_fraction: float
    batch_size: int
    fallback_to_cpu: bool

@dataclass
class WheelConfig:
    environment: str
    max_positions: int
    max_position_size_pct: float
    max_sector_exposure: float
    target_delta_put: float
    target_delta_call: float
    min_dte: int
    max_dte: int
    preferred_dte: int
    max_drawdown_pct: float
    stop_loss_pct: float
    margin_buffer: float
    gpu: GPUConfig
    data: Dict[str, str]
    monitoring: Dict[str, any]
    
    @classmethod
    def from_yaml(cls, path: str) -> 'WheelConfig':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Environment variable substitution
        config_str = yaml.dump(config_dict)
        for match in re.finditer(r'\${(\w+):?([^}]*)}', config_str):
            env_var = match.group(1)
            default = match.group(2)
            value = os.environ.get(env_var, default)
            config_str = config_str.replace(match.group(0), value)
        
        config_dict = yaml.safe_load(config_str)
        wheel_config = config_dict['wheel_strategy']
        
        # Parse GPU config
        gpu_config = GPUConfig(**wheel_config['gpu'])
        wheel_config['gpu'] = gpu_config
        
        return cls(**wheel_config)
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.max_position_size_pct <= 0.2, "Position size must be 0-20%"
        assert 0 < self.target_delta_put <= 0.5, "Put delta must be 0-0.5"
        assert self.min_dte < self.preferred_dte < self.max_dte, "Invalid DTE range"
        assert 0 < self.max_drawdown_pct < 1, "Invalid drawdown limit"
```

## Phase 2: GPU Resource Management (Days 3-4)

### 2.1 Enhanced GPU Manager
```python
# Add to gpu_enhanced_wheel.py
from src.core.gpu_resource_manager_production import GPUResourceManager
import cupy as cp
from numba import cuda
import torch

class WheelGPUManager:
    def __init__(self, config: GPUConfig):
        self.config = config
        self.resource_manager = GPUResourceManager()
        self.memory_pool = None
        self.initialized = False
        
    def initialize(self):
        """Initialize GPU resources"""
        if self.initialized:
            return
            
        try:
            # Set up CuPy
            if cp.cuda.is_available():
                # Use specific GPU devices
                for device_id in self.config.device_ids:
                    with cp.cuda.Device(device_id):
                        # Set memory limit
                        mempool = cp.get_default_memory_pool()
                        mempool.set_limit(
                            fraction=self.config.memory_fraction
                        )
                
                # Create unified memory pool
                self.memory_pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self.memory_pool.malloc)
                
                logger.info(f"GPU initialized: {len(self.config.device_ids)} devices")
                self.initialized = True
                
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            if self.config.fallback_to_cpu:
                logger.info("Falling back to CPU")
                self.initialized = False
            else:
                raise
    
    def allocate_batch_memory(self, size: int, dtype=cp.float32) -> cp.ndarray:
        """Allocate GPU memory with pooling"""
        try:
            return cp.zeros(size, dtype=dtype)
        except cp.cuda.MemoryError:
            # Clear cache and retry
            self.clear_cache()
            try:
                return cp.zeros(size, dtype=dtype)
            except cp.cuda.MemoryError:
                if self.config.fallback_to_cpu:
                    logger.warning("GPU OOM, using CPU")
                    return np.zeros(size, dtype=np.float32)
                raise
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cp.cuda.synchronize()
```

## Phase 3: Real-time Data Pipeline (Days 5-6)

### 3.1 Options Data Integration
```python
# Add to gpu_enhanced_wheel.py
from alpaca.data.live import StockDataStream
from alpaca.data.historical import OptionsHistoricalDataClient
import aiohttp
import asyncio
from typing import AsyncIterator

class OptionsDataPipeline:
    def __init__(self, config: WheelConfig):
        self.config = config
        self.alpaca_key = os.environ['ALPACA_API_KEY']
        self.alpaca_secret = os.environ['ALPACA_API_SECRET']
        
        # Initialize clients
        self.options_client = OptionsHistoricalDataClient(
            self.alpaca_key,
            self.alpaca_secret
        )
        self.stream = StockDataStream(
            self.alpaca_key,
            self.alpaca_secret
        )
        
        # Tradier for options (backup)
        self.tradier_token = os.environ.get('TRADIER_TOKEN')
        
    async def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Get real-time option chain"""
        try:
            # Try Alpaca first
            contracts = await self._get_alpaca_options(symbol)
            if contracts:
                return contracts
                
            # Fallback to Tradier
            if self.tradier_token:
                return await self._get_tradier_options(symbol)
                
            raise Exception("No options data available")
            
        except Exception as e:
            logger.error(f"Failed to get options for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_alpaca_options(self, symbol: str) -> pd.DataFrame:
        """Get options from Alpaca"""
        # Get expiration dates
        expirations = self._get_target_expirations()
        
        options_data = []
        for exp_date in expirations:
            try:
                # Get contracts for expiration
                contracts = self.options_client.get_option_contracts(
                    underlying_symbol=symbol,
                    expiration_date=exp_date,
                    option_type=None  # Get both puts and calls
                )
                
                for contract in contracts:
                    # Get latest quote
                    quote = self.options_client.get_latest_quote(
                        contract.symbol
                    )
                    
                    options_data.append({
                        'symbol': contract.symbol,
                        'underlying': symbol,
                        'strike': contract.strike_price,
                        'expiration': contract.expiration_date,
                        'type': contract.option_type,
                        'bid': quote.bid_price,
                        'ask': quote.ask_price,
                        'mid': (quote.bid_price + quote.ask_price) / 2,
                        'volume': quote.volume,
                        'open_interest': contract.open_interest,
                        'iv': quote.implied_volatility
                    })
                    
            except Exception as e:
                logger.warning(f"Error fetching {symbol} options for {exp_date}: {e}")
                
        return pd.DataFrame(options_data)
    
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict]:
        """Stream real-time quotes"""
        async def handle_quote(data):
            yield {
                'symbol': data.symbol,
                'price': data.price,
                'size': data.size,
                'timestamp': data.timestamp
            }
        
        # Subscribe to symbols
        for symbol in symbols:
            self.stream.subscribe_quotes(handle_quote, symbol)
        
        # Start streaming
        await self.stream.run()
    
    def _get_target_expirations(self) -> List[str]:
        """Get target expiration dates based on DTE preferences"""
        today = datetime.now().date()
        expirations = []
        
        for dte in range(self.config.min_dte, self.config.max_dte + 1, 7):
            exp_date = today + timedelta(days=dte)
            # Adjust to Friday (options expiration)
            days_until_friday = (4 - exp_date.weekday()) % 7
            exp_date += timedelta(days=days_until_friday)
            expirations.append(exp_date.strftime('%Y-%m-%d'))
            
        return expirations
```

## Phase 4: GPU-Accelerated Strike Selection (Days 7-8)

### 4.1 CUDA Kernel for Strike Selection
```python
# Add to gpu_enhanced_wheel.py
@cuda.jit
def optimal_strike_kernel(
    underlying_prices, strikes, ivs, volumes, deltas,
    target_delta, dte_days, scores_out, n_options
):
    """CUDA kernel for parallel strike evaluation"""
    idx = cuda.grid(1)
    
    if idx >= n_options:
        return
    
    # Local variables
    underlying = underlying_prices[idx]
    strike = strikes[idx]
    iv = ivs[idx]
    volume = volumes[idx]
    delta = deltas[idx]
    dte = dte_days[idx]
    
    # Score components
    score = 0.0
    
    # 1. Delta score (40% weight)
    delta_diff = abs(abs(delta) - target_delta)
    delta_score = max(0.0, 1.0 - delta_diff * 4.0)  # Penalize deviation
    score += delta_score * 0.4
    
    # 2. Premium yield score (30% weight)
    moneyness = strike / underlying
    expected_premium = strike * iv * 0.04 * (dte / 365.0) ** 0.5
    annual_yield = (expected_premium / strike) * (365.0 / dte)
    yield_score = min(1.0, annual_yield * 10.0)  # Cap at 10% annual
    score += yield_score * 0.3
    
    # 3. Liquidity score (20% weight)
    volume_score = min(1.0, volume / 1000.0)  # Normalize to 1000
    score += volume_score * 0.2
    
    # 4. Time decay score (10% weight)
    if 14 <= dte <= 28:  # Preferred range
        time_score = 1.0
    else:
        time_score = max(0.0, 1.0 - abs(dte - 21) / 21.0)
    score += time_score * 0.1
    
    # Write final score
    scores_out[idx] = score

class GPUStrikeSelector:
    def __init__(self, gpu_manager: WheelGPUManager):
        self.gpu_manager = gpu_manager
        
    def select_optimal_strikes(
        self,
        options_df: pd.DataFrame,
        target_delta: float,
        option_type: str
    ) -> pd.DataFrame:
        """Select optimal strikes using GPU acceleration"""
        # Filter by option type
        mask = options_df['type'] == option_type
        filtered = options_df[mask].copy()
        
        if filtered.empty:
            return pd.DataFrame()
        
        # Transfer to GPU
        n_options = len(filtered)
        underlying_prices = cp.asarray(filtered['underlying_price'].values)
        strikes = cp.asarray(filtered['strike'].values)
        ivs = cp.asarray(filtered['iv'].fillna(0.25).values)
        volumes = cp.asarray(filtered['volume'].fillna(0).values)
        deltas = cp.asarray(filtered['delta'].fillna(0).values)
        dte_days = cp.asarray(filtered['dte'].values)
        
        # Allocate output
        scores = cp.zeros(n_options, dtype=cp.float32)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (n_options + threads_per_block - 1) // threads_per_block
        
        optimal_strike_kernel[blocks_per_grid, threads_per_block](
            underlying_prices, strikes, ivs, volumes, deltas,
            target_delta, dte_days, scores, n_options
        )
        
        # Get results
        filtered['score'] = cp.asnumpy(scores)
        
        # Return top strikes per symbol
        return (filtered.groupby('underlying')
                       .apply(lambda x: x.nlargest(3, 'score'))
                       .reset_index(drop=True))
```

## Phase 5: Risk Management System (Days 9-10)

### 5.1 Real-time Risk Manager
```python
class WheelRiskManager:
    def __init__(self, config: WheelConfig):
        self.config = config
        self.positions = {}
        self.portfolio_value = 0
        self.margin_used = 0
        
    def check_position_limits(self, symbol: str, quantity: int, option_type: str) -> bool:
        """Check if position is within risk limits"""
        # Check max positions
        active_positions = len([p for p in self.positions.values() 
                               if p['status'] == 'active'])
        if active_positions >= self.config.max_positions:
            logger.warning(f"Max positions ({self.config.max_positions}) reached")
            return False
        
        # Check position size
        position_value = self._calculate_position_value(symbol, quantity, option_type)
        if position_value > self.portfolio_value * self.config.max_position_size_pct:
            logger.warning(f"Position size ${position_value} exceeds limit")
            return False
        
        # Check margin requirements
        margin_required = self._calculate_margin(symbol, quantity, option_type)
        if self.margin_used + margin_required > self.portfolio_value / self.config.margin_buffer:
            logger.warning(f"Insufficient margin: need ${margin_required}")
            return False
        
        # Check sector exposure
        sector = self._get_sector(symbol)
        sector_exposure = sum(p['value'] for p in self.positions.values() 
                            if p['sector'] == sector)
        if sector_exposure + position_value > self.portfolio_value * self.config.max_sector_exposure:
            logger.warning(f"Sector {sector} exposure would exceed limit")
            return False
        
        return True
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate real-time portfolio Greeks on GPU"""
        if not self.positions:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        # Prepare data for GPU
        positions_data = []
        for pos_id, pos in self.positions.items():
            if pos['status'] == 'active':
                positions_data.append({
                    'underlying_price': pos['current_underlying_price'],
                    'strike': pos['strike'],
                    'dte': pos['dte'],
                    'iv': pos['implied_volatility'],
                    'quantity': pos['quantity'],
                    'option_type': pos['option_type']
                })
        
        # Calculate on GPU
        gpu_processor = GPUOptionsProcessor()
        
        # Aggregate Greeks
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        for pos in positions_data:
            greeks = gpu_processor.vectorized_black_scholes(
                S=pos['underlying_price'],
                K=pos['strike'],
                T=pos['dte'] / 365,
                r=0.05,
                sigma=pos['iv'],
                option_type=pos['option_type']
            )
            
            # Scale by position size
            for greek in total_greeks:
                total_greeks[greek] += greeks[greek] * pos['quantity'] * 100
        
        return total_greeks
    
    def check_stop_loss(self):
        """Check positions for stop loss triggers"""
        positions_to_close = []
        
        for pos_id, pos in self.positions.items():
            if pos['status'] != 'active':
                continue
                
            # Calculate current P&L
            current_value = self._get_position_value(pos_id)
            entry_value = pos['entry_value']
            pnl_pct = (current_value - entry_value) / entry_value
            
            # Check stop loss
            if pnl_pct < -self.config.stop_loss_pct:
                logger.warning(f"Stop loss triggered for {pos_id}: {pnl_pct:.2%}")
                positions_to_close.append(pos_id)
        
        return positions_to_close
```

## Phase 6: Production Monitoring (Days 11-12)

### 6.1 Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class WheelMetrics:
    def __init__(self, port: int = 9090):
        # Trade metrics
        self.trades_total = Counter(
            'wheel_trades_total',
            'Total wheel trades executed',
            ['action', 'symbol']
        )
        self.trade_pnl = Gauge(
            'wheel_trade_pnl_dollars',
            'P&L per trade',
            ['symbol']
        )
        
        # Position metrics
        self.active_positions = Gauge(
            'wheel_active_positions',
            'Number of active positions'
        )
        self.assignment_rate = Gauge(
            'wheel_assignment_rate',
            'Assignment rate for puts'
        )
        
        # Performance metrics
        self.win_rate = Gauge(
            'wheel_win_rate',
            'Win rate percentage'
        )
        self.sharpe_ratio = Gauge(
            'wheel_sharpe_ratio',
            'Strategy Sharpe ratio'
        )
        
        # GPU metrics
        self.gpu_inference_time = Histogram(
            'wheel_gpu_inference_seconds',
            'GPU inference time',
            ['operation']
        )
        self.gpu_batch_size = Histogram(
            'wheel_gpu_batch_size',
            'GPU batch sizes processed'
        )
        
        # Start metrics server
        start_http_server(port)
    
    def record_trade(self, action: str, symbol: str, pnl: float):
        """Record trade metrics"""
        self.trades_total.labels(action=action, symbol=symbol).inc()
        self.trade_pnl.labels(symbol=symbol).set(pnl)
    
    def update_performance(self, metrics: Dict):
        """Update performance metrics"""
        self.win_rate.set(metrics.get('win_rate', 0))
        self.sharpe_ratio.set(metrics.get('sharpe_ratio', 0))
        self.active_positions.set(metrics.get('active_positions', 0))
        self.assignment_rate.set(metrics.get('assignment_rate', 0))
```

## Phase 7: Production Deployment (Days 13-14)

### 7.1 Docker Configuration
```dockerfile
# Dockerfile.gpu-wheel
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install GPU libraries
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html \
    numba \
    alpaca-py \
    prometheus-client

# Copy application
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health')"

# Run
CMD ["python3", "src/misc/gpu_enhanced_wheel.py"]
```

### 7.2 Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-wheel-strategy
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gpu-wheel
  template:
    metadata:
      labels:
        app: gpu-wheel
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      containers:
      - name: wheel-strategy
        image: alpaca-trading/gpu-wheel:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 8
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 4
        env:
        - name: ENVIRONMENT
          value: production
        - name: ALPACA_API_KEY
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: api-key
        - name: ALPACA_API_SECRET
          valueFrom:
            secretKeyRef:
              name: alpaca-credentials
              key: api-secret
        volumeMounts:
        - name: config
          mountPath: /app/config
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics
      volumes:
      - name: config
        configMap:
          name: wheel-config
```

## Phase 8: Testing & Validation (Days 15-16)

### 8.1 Comprehensive Tests
```python
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestGPUEnhancedWheel:
    @pytest.fixture
    def wheel_bot(self):
        config = WheelConfig.from_yaml('config/test_config.yaml')
        return GPUEnhancedWheelBot(config)
    
    @pytest.mark.gpu
    async def test_gpu_strike_selection(self, wheel_bot):
        """Test GPU-accelerated strike selection"""
        # Mock options data
        options_data = pd.DataFrame({
            'symbol': ['SPY220520P00400000'] * 100,
            'underlying': ['SPY'] * 100,
            'underlying_price': [420] * 100,
            'strike': np.linspace(400, 440, 100),
            'type': ['put'] * 100,
            'dte': [21] * 100,
            'delta': np.linspace(-0.1, -0.4, 100),
            'iv': [0.25] * 100,
            'volume': np.random.randint(100, 10000, 100)
        })
        
        # Run GPU selection
        selector = GPUStrikeSelector(wheel_bot.gpu_manager)
        selected = selector.select_optimal_strikes(
            options_data, 
            target_delta=0.25,
            option_type='put'
        )
        
        # Verify results
        assert len(selected) > 0
        assert selected['score'].max() > 0.5
        assert abs(selected.iloc[0]['delta'] - (-0.25)) < 0.05
    
    async def test_risk_limits(self, wheel_bot):
        """Test risk management limits"""
        risk_manager = wheel_bot.risk_manager
        risk_manager.portfolio_value = 100000
        
        # Test position size limit
        assert risk_manager.check_position_limits('SPY', 10, 'put')
        assert not risk_manager.check_position_limits('SPY', 100, 'put')
        
        # Test max positions
        for i in range(10):
            risk_manager.positions[f'pos_{i}'] = {
                'status': 'active',
                'value': 5000,
                'sector': 'tech'
            }
        
        assert not risk_manager.check_position_limits('AAPL', 1, 'put')
    
    @pytest.mark.benchmark
    def test_performance(self, wheel_bot, benchmark):
        """Benchmark GPU performance"""
        options_data = generate_large_options_dataset(n=100000)
        
        def run_selection():
            wheel_bot.enhanced_opportunity_finding(options_data)
        
        result = benchmark(run_selection)
        
        # Verify performance targets
        assert result.stats['mean'] < 0.1  # < 100ms average
```

## Production Checklist

### Pre-Production
- [ ] GPU drivers installed (525.60.13+)
- [ ] CUDA 11.8+ configured
- [ ] API credentials secured
- [ ] Configuration validated
- [ ] Risk limits tested

### Deployment
- [ ] Docker image built and scanned
- [ ] Kubernetes manifests applied
- [ ] Health checks passing
- [ ] Metrics flowing to Prometheus
- [ ] Alerts configured

### Post-Deployment
- [ ] GPU utilization 60-80%
- [ ] Latency < 50ms for strike selection
- [ ] Zero memory leaks
- [ ] Position tracking accurate
- [ ] P&L calculations verified

## Performance Targets
- Strike selection: < 50ms for 10,000 options
- Memory usage: < 4GB GPU memory
- Batch processing: 100,000 options/second
- Win rate: > 65%
- Annual return: 15-25%

## Next Steps
1. Deploy to staging environment
2. Run paper trading for 2 weeks
3. Validate all risk checks
4. Graduate to production with small capital
5. Scale up based on performance

---

*This implementation guide transforms the existing GPU Enhanced Wheel script into a production-ready system capable of generating consistent revenue through optimized options strategies.*