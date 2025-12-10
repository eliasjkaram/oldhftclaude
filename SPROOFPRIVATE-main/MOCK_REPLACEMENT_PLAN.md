# Mock Implementation Replacement Plan

## Overview
This document outlines the systematic replacement of mock/stub implementations with real, production-ready code.

## Phase 1: Core Data Provider Replacement

### 1.1 Replace Mock Data Provider
**File**: `src/mock_data_provider.py`
**Replacement Strategy**:
```python
# Instead of mock random prices:
# OLD: return base_price + np.random.uniform(-5, 5)

# NEW: Use real Alpaca API
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest

client = StockHistoricalDataClient(api_key, secret_key)
request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
quote = client.get_stock_latest_quote(request)
return quote[symbol].ask_price
```

### 1.2 Real-time WebSocket Streaming
**Implementation**: Use Alpaca's WebSocket streaming
```python
async def subscribe_to_quotes(symbols):
    wss = StockDataStream(api_key, secret_key)
    
    async def quote_handler(data):
        # Process real-time quotes
        print(f"Quote: {data.symbol} @ {data.ask_price}")
    
    wss.subscribe_quotes(quote_handler, *symbols)
    await wss.run()
```

## Phase 2: Options Greeks Implementation

### 2.1 Replace Mock Greeks Calculations
**Current**: Random/placeholder Greeks
**Replacement**: Use `py_vollib` for Black-Scholes Greeks
```python
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho

def calculate_real_greeks(S, K, T, r, sigma, option_type):
    price = bs(option_type, S, K, T, r, sigma)
    greeks = {
        'delta': delta(option_type, S, K, T, r, sigma),
        'gamma': gamma(option_type, S, K, T, r, sigma),
        'theta': theta(option_type, S, K, T, r, sigma),
        'vega': vega(option_type, S, K, T, r, sigma),
        'rho': rho(option_type, S, K, T, r, sigma)
    }
    return price, greeks
```

### 2.2 Implied Volatility Calculation
```python
from py_vollib.black_scholes.implied_volatility import implied_volatility

def get_real_iv(price, S, K, T, r, option_type):
    try:
        iv = implied_volatility(price, S, K, T, r, option_type)
        return iv
    except:
        return 0.3  # Default IV if calculation fails
```

## Phase 3: ML Model Replacements

### 3.1 Replace Transformer Stub
**File**: `src/ml/model_stubs.py` - EnhancedTransformerV3
**Replacement**: Use actual transformer architecture
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RealEnhancedTransformerV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('bert-base-uncased')
        self.market_projection = nn.Linear(768, config['market_features'])
        self.prediction_head = nn.Linear(config['market_features'], config['num_actions'])
        
    def forward(self, market_data, text_data=None):
        # Real implementation with actual transformer processing
        if text_data:
            encoded = self.transformer(**text_data)
            features = encoded.last_hidden_state.mean(dim=1)
        else:
            features = self.market_projection(market_data)
        
        predictions = self.prediction_head(features)
        return predictions
```

### 3.2 Replace PPO Trading Agent
**Replacement**: Use stable-baselines3 PPO
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_real_ppo_agent(env_config):
    env = TradingEnvironment(env_config)
    env = DummyVecEnv([lambda: env])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    return model
```

## Phase 4: Execution Algorithms

### 4.1 Implement TWAP
**Replace**: NotImplementedError in execution_algorithm_suite.py
```python
class RealTWAPAlgorithm:
    def __init__(self, order_size, duration, interval=60):
        self.order_size = order_size
        self.duration = duration
        self.interval = interval
        self.slices = duration // interval
        self.slice_size = order_size / self.slices
        
    async def execute(self, client, symbol):
        for i in range(self.slices):
            # Place slice order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=self.slice_size,
                side=self.side,
                time_in_force=TimeInForce.DAY
            )
            await client.submit_order(order)
            
            # Wait for next interval
            await asyncio.sleep(self.interval)
```

### 4.2 Implement VWAP
```python
class RealVWAPAlgorithm:
    def __init__(self, order_size, participation_rate=0.1):
        self.order_size = order_size
        self.participation_rate = participation_rate
        
    async def execute(self, client, symbol, market_data_stream):
        remaining = self.order_size
        
        async for bar in market_data_stream:
            # Calculate order size based on volume
            slice_size = min(
                bar.volume * self.participation_rate,
                remaining
            )
            
            if slice_size > 0:
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=slice_size,
                    side=self.side,
                    time_in_force=TimeInForce.DAY
                )
                await client.submit_order(order)
                remaining -= slice_size
                
            if remaining <= 0:
                break
```

## Phase 5: Portfolio Management

### 5.1 Real Position Tracking
```python
class RealPortfolioManager:
    def __init__(self, trading_client):
        self.client = trading_client
        
    def get_positions(self):
        # Get real positions from Alpaca
        positions = self.client.get_all_positions()
        return {
            pos.symbol: {
                'qty': pos.qty,
                'avg_entry_price': pos.avg_entry_price,
                'market_value': pos.market_value,
                'unrealized_pl': pos.unrealized_pl,
                'realized_pl': pos.unrealized_pl
            }
            for pos in positions
        }
    
    def get_account(self):
        # Get real account data
        account = self.client.get_account()
        return {
            'buying_power': account.buying_power,
            'cash': account.cash,
            'portfolio_value': account.portfolio_value,
            'equity': account.equity
        }
```

## Implementation Priority

1. **Week 1**: Replace mock data provider with real Alpaca API
2. **Week 2**: Implement real Greeks calculations using py_vollib
3. **Week 3**: Replace ML model stubs with actual implementations
4. **Week 4**: Implement real execution algorithms
5. **Week 5**: Complete portfolio management integration

## Testing Strategy

1. **Unit Tests**: Test each replacement component individually
2. **Integration Tests**: Test component interactions
3. **Paper Trading**: Run full system on Alpaca paper trading
4. **Gradual Rollout**: Replace one component at a time in production

## Required Dependencies

```bash
pip install alpaca-py
pip install py_vollib
pip install stable-baselines3
pip install torch transformers
pip install pandas numpy scipy
```

## Monitoring

- Log all API calls and responses
- Track execution metrics (slippage, fill rates)
- Monitor model predictions vs actual outcomes
- Alert on any errors or anomalies