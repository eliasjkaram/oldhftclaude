# 🔧 Mock Implementation Replacement Guide

**Purpose**: Step-by-step guide to replace mock/dummy code with real implementations  
**Priority**: Focus on data layer first, then ML models, then execution  
**Time Estimate**: 2-4 weeks for full replacement  

---

## 📊 PRIORITY 1: DATA LAYER (Critical - Week 1)

### 1.1 Fix Alpaca Data Fetching

**File**: `/src/data/market_data/enhanced_data_provider.py`  
**Current Issue**: `_fetch_from_alpaca()` returns None

```python
# CURRENT (Mock):
def _fetch_from_alpaca(self, symbol: str, start_date: datetime, 
                      end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
    """Fetch from Alpaca API"""
    # Placeholder - returns None
    return None

# REPLACEMENT (Real):
def _fetch_from_alpaca(self, symbol: str, start_date: datetime,
                      end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
    """Fetch real data from Alpaca API"""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Map intervals to TimeFrame
        timeframe_map = {
            '1d': TimeFrame.Day,
            '1h': TimeFrame.Hour,
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, 'Min'),
            '15m': TimeFrame(15, 'Min')
        }
        
        timeframe = timeframe_map.get(interval, TimeFrame.Day)
        
        # Create request
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe
        )
        
        # Fetch data
        bars = self.alpaca_client.stock_client().get_stock_bars(request)
        
        # Convert to DataFrame
        if symbol in bars:
            df = bars[symbol].df
            df.index = pd.to_datetime(df.index)
            return df
        
        return None
        
    except Exception as e:
        logger.error(f"Alpaca fetch failed for {symbol}: {e}")
        return None
```

### 1.2 Implement MinIO Connection

**File**: `/src/data/market_data/enhanced_data_provider.py`  
**Current Issue**: MinIO connection not implemented

```python
# CURRENT (Mock):
def _fetch_from_minio(self, symbol: str, start_date: datetime,
                     end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
    """Fetch from MinIO storage"""
    # Placeholder - implement actual MinIO fetching
    return None

# REPLACEMENT (Real):
def _fetch_from_minio(self, symbol: str, start_date: datetime,
                     end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical data from MinIO"""
    try:
        from minio import Minio
        import io
        
        # Initialize MinIO client if not exists
        if not hasattr(self, 'minio_client'):
            self.minio_client = Minio(
                os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY'),
                secret_key=os.getenv('MINIO_SECRET_KEY'),
                secure=False
            )
        
        # Construct object paths
        bucket = 'market-data'
        
        # Try different path patterns
        paths_to_try = [
            f"stocks/{symbol}/daily_{start_date.year}.parquet",
            f"stockdata/{symbol}/{interval}/data.parquet",
            f"historical/{symbol}_{interval}.parquet"
        ]
        
        for object_path in paths_to_try:
            try:
                # Get object
                response = self.minio_client.get_object(bucket, object_path)
                
                # Read parquet
                df = pd.read_parquet(io.BytesIO(response.read()))
                
                # Filter date range
                df.index = pd.to_datetime(df.index)
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df.loc[mask]
                
                if not df.empty:
                    logger.info(f"Loaded {len(df)} rows from MinIO for {symbol}")
                    return df
                    
            except Exception:
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"MinIO fetch failed: {e}")
        return None
```

### 1.3 Fix Options Data Retrieval

**File**: `/src/data/market_data/enhanced_data_provider.py`  
**Current Issue**: Options chain uses random data

```python
# CURRENT (Mock):
def get_options_chain(self, symbol: str, expiry: Optional[datetime] = None) -> Dict:
    """Get options chain"""
    # Generate synthetic options chain
    options_data = {
        'calls': [],
        'puts': []
    }
    
    for i in range(5):
        strike = base_price * (0.9 + i * 0.05)
        options_data['calls'].append({
            'strike': strike,
            'bid': random.uniform(1, 5),
            'ask': random.uniform(1, 5),
            'volume': random.randint(0, 1000),
            'openInterest': random.randint(0, 5000),
            'impliedVolatility': random.uniform(0.15, 0.45)
        })

# REPLACEMENT (Real):
def get_options_chain(self, symbol: str, expiry: Optional[datetime] = None) -> Dict:
    """Get real options chain from Alpaca"""
    try:
        from alpaca.data.requests import OptionChainRequest
        
        # Get options client
        options_client = self.alpaca_client.options_client()
        
        # Create request
        request = OptionChainRequest(
            underlying_symbol=symbol,
            expiration_date=expiry if expiry else None
        )
        
        # Fetch chain
        chain = options_client.get_option_chain(request)
        
        # Organize by calls/puts
        options_data = {
            'calls': [],
            'puts': [],
            'expirations': [],
            'underlying_price': None
        }
        
        for contract in chain:
            option_data = {
                'symbol': contract.symbol,
                'strike': contract.strike_price,
                'expiry': contract.expiration_date,
                'bid': contract.bid_price,
                'ask': contract.ask_price,
                'last': contract.last_price,
                'volume': contract.volume,
                'openInterest': contract.open_interest,
                'impliedVolatility': contract.implied_volatility,
                'delta': contract.delta,
                'gamma': contract.gamma,
                'theta': contract.theta,
                'vega': contract.vega
            }
            
            if contract.option_type == 'call':
                options_data['calls'].append(option_data)
            else:
                options_data['puts'].append(option_data)
        
        # Sort by strike
        options_data['calls'].sort(key=lambda x: x['strike'])
        options_data['puts'].sort(key=lambda x: x['strike'])
        
        return options_data
        
    except Exception as e:
        logger.error(f"Failed to get options chain: {e}")
        # Return empty chain instead of random data
        return {'calls': [], 'puts': []}
```

---

## 🧠 PRIORITY 2: MACHINE LEARNING MODELS (Week 2)

### 2.1 Implement Model Training Pipeline

**Create New File**: `/src/ml/model_training_pipeline.py`

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Automated model training and validation pipeline"""
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def prepare_training_data(self, symbols: list, start_date: datetime, 
                            end_date: datetime) -> tuple:
        """Prepare training data with features and labels"""
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            # Get historical data
            df = self.data_provider.get_data(symbol, start_date, end_date)
            
            if df is None or df.empty:
                continue
                
            # Create features
            features = self.create_features(df)
            
            # Create labels (next day returns)
            labels = df['close'].pct_change().shift(-1).fillna(0)
            
            # Remove last row (no label)
            features = features[:-1]
            labels = labels[:-1]
            
            all_features.append(features)
            all_labels.append(labels)
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        return X, y
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from price data"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['close'])
        features['macd'] = self.calculate_macd(df['close'])
        
        # Price levels
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = df['close'] / df['close'].rolling(period).mean() - 1
            
        # Volatility
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_60'] = features['returns'].rolling(60).std()
        
        # Volume patterns
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
        
        # Fill NaN values
        features = features.fillna(0)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model for direction prediction"""
        # Convert to classification problem
        y_class = (y > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"XGBoost model accuracy: {accuracy:.4f}")
        
        # Store scaler
        self.scalers['xgboost'] = scaler
        
        return model
    
    def train_all_models(self, symbols: list, start_date: datetime, end_date: datetime):
        """Train all ML models"""
        logger.info("Starting model training pipeline...")
        
        # Prepare data
        X, y = self.prepare_training_data(symbols, start_date, end_date)
        
        if X.empty:
            logger.error("No training data available")
            return
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        
        # Train models
        self.models['xgboost'] = self.train_xgboost_model(X, y)
        
        # Save models
        self.save_models()
        
        logger.info("Model training completed")
    
    def save_models(self):
        """Save trained models and scalers"""
        import os
        
        model_dir = "models/trained"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}_model.pkl")
            
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/{name}_scaler.pkl")
            
        # Save feature names
        with open(f"{model_dir}/feature_names.txt", 'w') as f:
            f.write('\n'.join(self.feature_names))
            
        logger.info(f"Models saved to {model_dir}")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        
        return macd
```

### 2.2 Fix ML Predictions in Algorithms

**File**: `/src/ml/advanced_algorithms.py`  
**Current Issue**: All predictions use random values

```python
# CURRENT (Mock):
class MachineLearningPredictor:
    def predict(self, features):
        # MOCK: Random prediction
        return np.random.choice([1, -1], p=[0.55, 0.45])

# REPLACEMENT (Real):
class MachineLearningPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load pre-trained model"""
        try:
            import joblib
            self.model = joblib.load('models/trained/xgboost_model.pkl')
            self.scaler = joblib.load('models/trained/xgboost_scaler.pkl')
            
            with open('models/trained/feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f]
                
            logger.info("Loaded trained ML model")
        except Exception as e:
            logger.warning(f"Could not load trained model: {e}")
            # Initialize empty model
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=10)
            self.scaler = StandardScaler()
    
    def predict(self, features: pd.DataFrame) -> int:
        """Make real prediction using trained model"""
        if self.model is None:
            return 0
            
        try:
            # Ensure features match training
            if isinstance(features, dict):
                features = pd.DataFrame([features])
                
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
                
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()
            
            # Convert to trading signal
            if confidence < 0.6:
                return 0  # No trade
            
            return 1 if prediction == 1 else -1
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0
```

---

## 💰 PRIORITY 3: ORDER EXECUTION (Week 3)

### 3.1 Implement Real Order Execution

**Create New File**: `/src/execution/real_order_executor.py`

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import logging

logger = logging.getLogger(__name__)

class RealOrderExecutor:
    """Production order execution with Alpaca"""
    
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.pending_orders = {}
        self.executed_orders = {}
        
    def submit_order(self, symbol: str, qty: int, side: str, 
                    order_type: str = 'market', limit_price: float = None,
                    stop_price: float = None, time_in_force: str = 'day'):
        """Submit real order to Alpaca"""
        try:
            # Convert side to enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create order request based on type
            if order_type.lower() == 'market':
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
            elif order_type.lower() == 'limit':
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            elif order_type.lower() == 'stop':
                order_data = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price
                )
            else:
                raise ValueError(f"Unknown order type: {order_type}")
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            # Track order
            self.pending_orders[order.id] = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'status': order.status,
                'submitted_at': order.submitted_at
            }
            
            logger.info(f"Order submitted: {order.id} - {side} {qty} {symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            raise
    
    def cancel_order(self, order_id: str):
        """Cancel pending order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise
    
    def get_order_status(self, order_id: str):
        """Get order status"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order.status
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
    
    def update_order_tracking(self):
        """Update status of all pending orders"""
        for order_id in list(self.pending_orders.keys()):
            try:
                order = self.trading_client.get_order_by_id(order_id)
                
                if order.status in ['filled', 'cancelled', 'expired']:
                    # Move to executed
                    self.executed_orders[order_id] = self.pending_orders[order_id]
                    self.executed_orders[order_id]['status'] = order.status
                    self.executed_orders[order_id]['filled_qty'] = order.filled_qty
                    self.executed_orders[order_id]['filled_avg_price'] = order.filled_avg_price
                    
                    del self.pending_orders[order_id]
                    
                    logger.info(f"Order {order_id} completed: {order.status}")
                    
            except Exception as e:
                logger.error(f"Failed to update order {order_id}: {e}")
```

### 3.2 Fix Position Management

**File**: `/src/risk/position_manager.py`  
**Current Issue**: Position tracking is simulated

```python
# REPLACEMENT (Real):
class RealPositionManager:
    """Real position management with Alpaca"""
    
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.positions = {}
        self.update_positions()
        
    def update_positions(self):
        """Update positions from Alpaca"""
        try:
            positions = self.trading_client.get_all_positions()
            
            self.positions = {}
            for position in positions:
                self.positions[position.symbol] = {
                    'qty': int(position.qty),
                    'side': position.side,
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price),
                    'lastday_price': float(position.lastday_price),
                    'change_today': float(position.change_today)
                }
                
            logger.info(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def get_position(self, symbol: str):
        """Get position for symbol"""
        self.update_positions()
        return self.positions.get(symbol)
    
    def calculate_position_size(self, symbol: str, account_value: float, 
                              risk_percent: float = 0.02) -> int:
        """Calculate position size based on Kelly Criterion"""
        try:
            # Get current price
            quote = self.trading_client.get_latest_trade(symbol)
            price = float(quote.price)
            
            # Calculate position value
            position_value = account_value * risk_percent
            
            # Calculate shares
            shares = int(position_value / price)
            
            # Apply maximum position limits
            max_shares = int(account_value * 0.1 / price)  # Max 10% per position
            
            return min(shares, max_shares)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0
```

---

## 🧪 PRIORITY 4: TESTING & VALIDATION (Week 4)

### 4.1 Create Integration Tests

**Create New File**: `/tests/test_real_data_integration.py`

```python
import pytest
import os
from datetime import datetime, timedelta
from src.data.market_data.enhanced_data_provider import EnhancedDataProvider
from src.alpaca_client import AlpacaClient

class TestRealDataIntegration:
    """Test real data fetching"""
    
    @pytest.fixture
    def data_provider(self):
        return EnhancedDataProvider()
    
    def test_alpaca_connection(self):
        """Test Alpaca API connection"""
        client = AlpacaClient()
        account = client.trading_client().get_account()
        
        assert account is not None
        assert float(account.cash) >= 0
        
    def test_fetch_real_data(self, data_provider):
        """Test fetching real market data"""
        symbol = 'SPY'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = data_provider.get_data(symbol, start_date, end_date)
        
        assert df is not None
        assert not df.empty
        assert 'close' in df.columns
        assert len(df) > 20  # Should have at least 20 trading days
        
    def test_options_chain(self, data_provider):
        """Test real options chain fetching"""
        symbol = 'SPY'
        chain = data_provider.get_options_chain(symbol)
        
        assert 'calls' in chain
        assert 'puts' in chain
        
        if len(chain['calls']) > 0:
            call = chain['calls'][0]
            assert 'strike' in call
            assert 'bid' in call
            assert 'ask' in call
            assert call['bid'] <= call['ask']
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Week 1: Data Foundation
- [ ] Implement real Alpaca data fetching
- [ ] Connect MinIO storage
- [ ] Fix options chain retrieval
- [ ] Remove synthetic data dependency
- [ ] Add proper error handling
- [ ] Create data validation tests

### Week 2: ML Models
- [ ] Create model training pipeline
- [ ] Train models on historical data
- [ ] Implement model persistence
- [ ] Fix prediction functions
- [ ] Add feature engineering
- [ ] Create model evaluation metrics

### Week 3: Execution Layer
- [ ] Implement real order submission
- [ ] Add position tracking
- [ ] Create risk management
- [ ] Implement stop loss/take profit
- [ ] Add order status monitoring
- [ ] Create execution tests

### Week 4: Integration & Testing
- [ ] Run full integration tests
- [ ] Paper trade validation
- [ ] Performance benchmarking
- [ ] Fix remaining mock functions
- [ ] Documentation updates
- [ ] Production readiness review

---

## 🎯 SUCCESS CRITERIA

1. **Data Layer**: 
   - Real-time data updates every second
   - Historical data loads in <5 seconds
   - Options data includes all Greeks

2. **ML Models**:
   - Model accuracy >65%
   - Predictions in <100ms
   - Daily retraining capability

3. **Execution**:
   - Orders execute in <1 second
   - Position tracking accurate to the penny
   - Risk limits enforced automatically

4. **Overall System**:
   - Zero mock data in production paths
   - All tests passing
   - Paper trading profitable

---

## 🚨 COMMON PITFALLS

1. **API Rate Limits**: Implement caching and request throttling
2. **Market Hours**: Check if market is open before trading
3. **Data Quality**: Validate all data before using
4. **Error Handling**: Never let exceptions crash the system
5. **Position Sizing**: Always check account balance before ordering

---

*Remember: Replace mocks incrementally and test thoroughly at each step!*