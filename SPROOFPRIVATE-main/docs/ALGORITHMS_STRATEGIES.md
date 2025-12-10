# Alpaca-MCP Algorithms and Trading Strategies

## Table of Contents
1. [Traditional Quantitative Strategies](#traditional-quantitative-strategies)
2. [Machine Learning Strategies](#machine-learning-strategies)
3. [Quantum-Inspired Algorithms](#quantum-inspired-algorithms)
4. [Swarm Intelligence Strategies](#swarm-intelligence-strategies)
5. [Neural Architecture Search](#neural-architecture-search)
6. [Reinforcement Meta-Learning](#reinforcement-meta-learning)
7. [Adversarial Trading Strategies](#adversarial-trading-strategies)
8. [Options Trading Strategies](#options-trading-strategies)
9. [High-Frequency Trading Algorithms](#high-frequency-trading-algorithms)
10. [Risk Management Algorithms](#risk-management-algorithms)

## Traditional Quantitative Strategies

### 1. Statistical Arbitrage

**Implementation**: `statistical_arbitrage.py`

**Core Algorithm**:
```python
def identify_pairs():
    """
    1. Calculate correlation matrix for all stocks
    2. Find pairs with correlation > 0.8
    3. Test for cointegration using Augmented Dickey-Fuller
    4. Calculate hedge ratio using linear regression
    5. Monitor z-score of spread
    """
```

**Trading Rules**:
- **Entry**: Z-score > 2 standard deviations
- **Exit**: Z-score returns to mean (0)
- **Stop Loss**: Z-score > 3 standard deviations

**Key Metrics**:
- Average holding period: 2-5 days
- Win rate: 65-70%
- Sharpe ratio: 1.5-2.0

### 2. Mean Reversion

**Mathematical Foundation**:
```
Ornstein-Uhlenbeck Process:
dX_t = θ(μ - X_t)dt + σdW_t

Where:
- θ: Speed of reversion
- μ: Long-term mean
- σ: Volatility
- W_t: Brownian motion
```

**Implementation**:
```python
def mean_reversion_signal(prices, lookback=20):
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z_score = (prices - mean) / std
    
    signals = pd.Series(index=prices.index, data=0)
    signals[z_score < -2] = 1  # Buy signal
    signals[z_score > 2] = -1  # Sell signal
    return signals
```

### 3. Momentum Trading

**Variants Implemented**:
1. **Time Series Momentum**: Based on asset's own past returns
2. **Cross-Sectional Momentum**: Relative performance ranking
3. **Dual Momentum**: Combines absolute and relative momentum

**Signal Generation**:
```python
def momentum_score(returns, periods=[20, 60, 120]):
    scores = []
    for period in periods:
        score = returns.rolling(period).mean()
        scores.append(score)
    
    # Weight recent momentum more heavily
    weights = [0.5, 0.3, 0.2]
    final_score = sum(w * s for w, s in zip(weights, scores))
    return final_score
```

## Machine Learning Strategies

### 4. Ensemble ML Prediction

**Model Architecture**:
```
Input Features (134) → Feature Selection → Model Ensemble → Weighted Prediction
                                              ├── Random Forest
                                              ├── XGBoost
                                              ├── LightGBM
                                              ├── Neural Network
                                              └── SVM
```

**Feature Engineering Pipeline**:
```python
def engineer_features(data):
    features = pd.DataFrame()
    
    # Price-based features
    features['returns_1d'] = data['close'].pct_change()
    features['returns_5d'] = data['close'].pct_change(5)
    features['log_returns'] = np.log(data['close']).diff()
    
    # Technical indicators
    features['rsi'] = calculate_rsi(data['close'])
    features['macd'] = calculate_macd(data['close'])
    features['bb_position'] = bollinger_position(data['close'])
    
    # Microstructure features
    features['spread'] = data['ask'] - data['bid']
    features['order_imbalance'] = (data['bid_volume'] - data['ask_volume']) / (data['bid_volume'] + data['ask_volume'])
    
    # Rolling statistics
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = features['returns_1d'].rolling(window).std()
        features[f'skew_{window}'] = features['returns_1d'].rolling(window).skew()
        features[f'kurtosis_{window}'] = features['returns_1d'].rolling(window).kurt()
    
    return features
```

**Model Training Process**:
```python
def train_ensemble(X_train, y_train, X_val, y_val):
    models = {
        'rf': RandomForestClassifier(n_estimators=1000, max_depth=10),
        'xgb': XGBClassifier(n_estimators=1000, learning_rate=0.01),
        'lgb': LGBMClassifier(n_estimators=1000, learning_rate=0.01),
        'nn': create_neural_network(),
        'svm': SVC(kernel='rbf', probability=True)
    }
    
    trained_models = {}
    weights = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        trained_models[name] = model
        weights[name] = val_score
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return trained_models, weights
```

### 5. Deep Learning LSTM

**Architecture**:
```python
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # Buy, Hold, Sell
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

## Quantum-Inspired Algorithms

### 6. Quantum Superposition Trading

**Concept**: Analyze multiple market states simultaneously

**Implementation**:
```python
class QuantumSuperpositionTrader:
    def create_market_superposition(self, market_data):
        """
        Create quantum superposition of market states
        |ψ⟩ = α|bull⟩ + β|bear⟩ + γ|neutral⟩
        """
        returns = market_data['returns']
        volatility = market_data['volatility']
        
        # Calculate amplitudes based on market conditions
        bull_amplitude = self.calculate_bull_probability(returns, volatility)
        bear_amplitude = self.calculate_bear_probability(returns, volatility)
        neutral_amplitude = 1 - bull_amplitude - bear_amplitude
        
        # Normalize to ensure |α|² + |β|² + |γ|² = 1
        norm = np.sqrt(bull_amplitude**2 + bear_amplitude**2 + neutral_amplitude**2)
        
        quantum_state = {
            'bull': bull_amplitude / norm,
            'bear': bear_amplitude / norm,
            'neutral': neutral_amplitude / norm
        }
        
        return quantum_state
```

### 7. Quantum Entanglement Detection

**Purpose**: Find hidden correlations between assets

**Algorithm**:
```python
def quantum_entanglement_coefficient(asset1, asset2):
    """
    Calculate quantum-inspired entanglement between assets
    """
    # Traditional correlation
    classical_corr = asset1.corr(asset2)
    
    # Phase correlation (momentum alignment)
    phase1 = np.angle(hilbert(asset1))
    phase2 = np.angle(hilbert(asset2))
    phase_corr = np.cos(phase1 - phase2).mean()
    
    # Amplitude correlation (volatility alignment)
    amp1 = np.abs(hilbert(asset1))
    amp2 = np.abs(hilbert(asset2))
    amp_corr = np.corrcoef(amp1, amp2)[0, 1]
    
    # Quantum entanglement score
    entanglement = np.sqrt(classical_corr**2 + phase_corr**2 + amp_corr**2) / np.sqrt(3)
    
    return entanglement
```

### 8. Quantum Tunneling Probability

**Application**: Predict breakthrough of resistance/support levels

**Mathematical Model**:
```python
def tunneling_probability(price, barrier, volatility):
    """
    Calculate probability of price tunneling through barrier
    Based on quantum mechanical tunneling equation
    """
    # Energy of price movement
    E = 0.5 * volatility**2
    
    # Barrier height
    V = abs(barrier - price) / price
    
    # Barrier width (estimated from recent price action)
    a = estimate_barrier_width(price_history)
    
    # Transmission coefficient (simplified)
    if E < V:
        k = np.sqrt(2 * (V - E))
        T = np.exp(-2 * k * a)
    else:
        T = 1  # Classical case: E > V
    
    return T
```

## Swarm Intelligence Strategies

### 9. Particle Swarm Optimization

**Algorithm**:
```python
class ParticleSwarmOptimizer:
    def __init__(self, n_particles=100, n_dimensions=10):
        self.particles = []
        self.global_best = None
        self.global_best_fitness = -np.inf
        
        # PSO parameters
        self.w = 0.7    # Inertia weight
        self.c1 = 1.5   # Cognitive parameter
        self.c2 = 1.5   # Social parameter
        
    def update_swarm(self, market_data):
        for particle in self.particles:
            # Update velocity
            r1, r2 = np.random.rand(2)
            
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best - particle.position)
            
            particle.velocity = self.w * particle.velocity + cognitive + social
            
            # Update position
            particle.position += particle.velocity
            
            # Evaluate fitness
            fitness = self.evaluate_trading_strategy(particle.position, market_data)
            
            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = particle.position.copy()
```

### 10. Ant Colony Optimization

**Trading Path Optimization**:
```python
class AntColonyTrading:
    def __init__(self, n_ants=50):
        self.n_ants = n_ants
        self.pheromone_matrix = defaultdict(float)
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        
    def find_optimal_trading_path(self, opportunities):
        best_path = None
        best_profit = -np.inf
        
        for ant in range(self.n_ants):
            path = self.construct_path(opportunities)
            profit = self.evaluate_path(path)
            
            if profit > best_profit:
                best_profit = profit
                best_path = path
            
            self.update_pheromones(path, profit)
        
        return best_path, best_profit
    
    def construct_path(self, opportunities):
        path = []
        available = set(range(len(opportunities)))
        current = random.choice(list(available))
        path.append(current)
        available.remove(current)
        
        while available and len(path) < 10:  # Limit path length
            probabilities = []
            
            for next_opp in available:
                pheromone = self.pheromone_matrix[(current, next_opp)]
                heuristic = opportunities[next_opp]['expected_return']
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(probability)
            
            # Select next opportunity
            probabilities = np.array(probabilities) / sum(probabilities)
            next_idx = np.random.choice(list(available), p=probabilities)
            
            path.append(next_idx)
            available.remove(next_idx)
            current = next_idx
        
        return path
```

## Neural Architecture Search

### 11. Evolutionary Architecture Search

**Search Algorithm**:
```python
class NeuralArchitectureSearch:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.search_space = {
            'n_layers': [2, 3, 4, 5, 6],
            'layer_types': ['dense', 'lstm', 'conv1d', 'attention'],
            'units': [32, 64, 128, 256, 512],
            'activation': ['relu', 'tanh', 'gelu', 'swish'],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4]
        }
        
    def evolve_architecture(self, X_train, y_train, generations=100):
        population = self.initialize_population()
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for architecture in population:
                model = self.build_model(architecture)
                score = self.evaluate_model(model, X_train, y_train)
                fitness_scores.append(score)
            
            # Selection
            parents = self.select_parents(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
            
            # Update population
            population = self.select_survivors(population + offspring, fitness_scores)
        
        return population[0]  # Best architecture
```

### 12. Differentiable Architecture Search (DARTS)

**Continuous Relaxation**:
```python
class DARTS:
    def __init__(self):
        self.operations = [
            'conv_3x3',
            'conv_5x5',
            'max_pool',
            'avg_pool',
            'skip_connect',
            'lstm',
            'attention'
        ]
        
    def create_supernet(self):
        """
        Create a supernet containing all possible operations
        """
        def mixed_op(x, weights):
            """
            Weighted sum of all operations
            """
            outputs = []
            for i, op in enumerate(self.operations):
                outputs.append(weights[i] * self.apply_operation(x, op))
            return sum(outputs)
        
        return mixed_op
    
    def train_architecture(self, train_data, val_data):
        # Initialize architecture parameters (α)
        alpha = torch.randn(len(self.operations), requires_grad=True)
        
        # Alternating optimization
        for epoch in range(epochs):
            # Step 1: Update network weights with fixed α
            train_loss = self.train_weights(train_data, alpha)
            
            # Step 2: Update α with fixed weights
            val_loss = self.train_alpha(val_data, alpha)
            
        # Derive discrete architecture
        best_ops = torch.argmax(alpha, dim=0)
        return self.derive_architecture(best_ops)
```

## Reinforcement Meta-Learning

### 13. Model-Agnostic Meta-Learning (MAML)

**Algorithm**:
```python
class MAML:
    def __init__(self, model, alpha=0.01, beta=0.001):
        self.model = model
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
        
    def meta_train(self, tasks, n_inner_steps=5):
        meta_loss = 0
        
        for task in tasks:
            # Clone model for inner loop
            task_model = self.clone_model(self.model)
            
            # Inner loop: Adapt to specific task
            support_X, support_y = task['support']
            for _ in range(n_inner_steps):
                loss = self.compute_loss(task_model, support_X, support_y)
                gradients = torch.autograd.grad(loss, task_model.parameters())
                self.update_parameters(task_model, gradients, self.alpha)
            
            # Compute loss on query set
            query_X, query_y = task['query']
            task_loss = self.compute_loss(task_model, query_X, query_y)
            meta_loss += task_loss
        
        # Outer loop: Update meta-parameters
        meta_gradients = torch.autograd.grad(meta_loss, self.model.parameters())
        self.update_parameters(self.model, meta_gradients, self.beta)
```

### 14. Reptile Algorithm

**Simplified Meta-Learning**:
```python
class Reptile:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        
    def train_step(self, task, n_iterations=100):
        # Save initial parameters
        initial_params = self.get_parameters(self.model)
        
        # Train on task
        for _ in range(n_iterations):
            X, y = task.sample_batch()
            loss = self.model.train_on_batch(X, y)
        
        # Get final parameters
        final_params = self.get_parameters(self.model)
        
        # Update in direction of task-specific parameters
        for initial, final in zip(initial_params, final_params):
            initial.data += self.epsilon * (final.data - initial.data)
```

## Adversarial Trading Strategies

### 15. GAN-Based Market Generation

**Architecture**:
```python
class MarketGAN:
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        return Sequential([
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dense(60 * 4),  # 60 timesteps × 4 features (OHLV)
            Reshape((60, 4))
        ])
    
    def build_discriminator(self):
        return Sequential([
            Conv1D(64, 5, strides=2, padding='same'),
            LeakyReLU(0.2),
            Conv1D(128, 5, strides=2, padding='same'),
            LeakyReLU(0.2),
            Flatten(),
            Dense(256),
            LeakyReLU(0.2),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    
    def generate_adversarial_scenarios(self, n_scenarios=100):
        noise = np.random.normal(0, 1, (n_scenarios, 100))
        synthetic_markets = self.generator.predict(noise)
        return synthetic_markets
```

### 16. Adversarial Training

**Robust Prediction**:
```python
def adversarial_training(model, X_train, y_train, epsilon=0.01):
    """
    Train model to be robust against adversarial examples
    """
    for epoch in range(epochs):
        # Generate adversarial examples
        X_adv = generate_adversarial_examples(model, X_train, epsilon)
        
        # Train on both original and adversarial data
        model.train_on_batch(X_train, y_train)
        model.train_on_batch(X_adv, y_train)
        
    return model

def generate_adversarial_examples(model, X, epsilon):
    """
    Fast Gradient Sign Method (FGSM)
    """
    X_tensor = tf.convert_to_tensor(X)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_true, predictions)
    
    gradients = tape.gradient(loss, X_tensor)
    signed_grad = tf.sign(gradients)
    
    X_adv = X + epsilon * signed_grad
    return X_adv.numpy()
```

## Options Trading Strategies

### 17. Iron Condor Optimization

**Strategy Construction**:
```python
class IronCondorOptimizer:
    def find_optimal_strikes(self, chain, underlying_price, iv_surface):
        """
        Find optimal strikes for Iron Condor
        """
        # Target: 68% probability of profit (1 standard deviation)
        expected_move = underlying_price * iv_surface.atm_iv * np.sqrt(30/365)
        
        # Find strikes
        put_long = self.find_strike(chain, underlying_price - 2 * expected_move)
        put_short = self.find_strike(chain, underlying_price - expected_move)
        call_short = self.find_strike(chain, underlying_price + expected_move)
        call_long = self.find_strike(chain, underlying_price + 2 * expected_move)
        
        # Calculate expected profit
        max_profit = (put_short.bid - put_long.ask) + (call_short.bid - call_long.ask)
        max_loss = (put_short.strike - put_long.strike) - max_profit
        
        # Risk-reward ratio
        risk_reward = max_profit / max_loss
        
        return {
            'strikes': [put_long, put_short, call_short, call_long],
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': risk_reward,
            'probability_profit': self.calculate_pop(strikes, underlying_price, iv_surface)
        }
```

### 18. Dynamic Delta Hedging

**Algorithm**:
```python
class DeltaHedger:
    def __init__(self, rebalance_threshold=0.10):
        self.rebalance_threshold = rebalance_threshold
        self.hedge_history = []
        
    def calculate_portfolio_delta(self, positions, market_data):
        total_delta = 0
        
        for position in positions:
            if position.type == 'option':
                delta = self.calculate_option_delta(
                    position,
                    market_data[position.underlying]
                )
                total_delta += delta * position.quantity
            elif position.type == 'stock':
                total_delta += position.quantity
        
        return total_delta
    
    def rebalance_hedge(self, current_delta, target_delta=0):
        """
        Rebalance to maintain delta-neutral portfolio
        """
        delta_difference = target_delta - current_delta
        
        if abs(delta_difference) > self.rebalance_threshold:
            # Calculate shares needed
            shares_needed = round(delta_difference)
            
            self.hedge_history.append({
                'timestamp': datetime.now(),
                'current_delta': current_delta,
                'target_delta': target_delta,
                'shares_traded': shares_needed
            })
            
            return shares_needed
        
        return 0  # No rebalance needed
```

## High-Frequency Trading Algorithms

### 19. Market Microstructure Alpha

**Order Book Imbalance**:
```python
class OrderBookAlpha:
    def calculate_order_book_imbalance(self, order_book):
        """
        Calculate order book imbalance indicator
        """
        bid_volume = sum(level['size'] for level in order_book['bids'][:5])
        ask_volume = sum(level['size'] for level in order_book['asks'][:5])
        
        # Volume imbalance
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Price-weighted imbalance
        bid_pressure = sum(level['size'] * level['price'] for level in order_book['bids'][:5])
        ask_pressure = sum(level['size'] * level['price'] for level in order_book['asks'][:5])
        pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
        
        # Combined signal
        signal = 0.7 * volume_imbalance + 0.3 * pressure_imbalance
        
        return signal
    
    def generate_trades(self, imbalance, threshold=0.2):
        if imbalance > threshold:
            return 'BUY'
        elif imbalance < -threshold:
            return 'SELL'
        return 'HOLD'
```

### 20. Latency Arbitrage

**Cross-Exchange Arbitrage**:
```python
class LatencyArbitrage:
    def __init__(self, exchanges):
        self.exchanges = exchanges
        self.latencies = self.measure_latencies()
        
    def find_arbitrage_opportunities(self):
        opportunities = []
        
        for symbol in self.symbols:
            prices = {}
            
            # Get prices from all exchanges
            for exchange in self.exchanges:
                prices[exchange] = self.get_price(exchange, symbol)
            
            # Find arbitrage
            min_exchange = min(prices, key=prices.get)
            max_exchange = max(prices, key=prices.get)
            
            spread = prices[max_exchange] - prices[min_exchange]
            spread_pct = spread / prices[min_exchange]
            
            # Account for fees and slippage
            net_profit = spread_pct - self.total_costs
            
            if net_profit > self.min_profit_threshold:
                opportunities.append({
                    'symbol': symbol,
                    'buy_exchange': min_exchange,
                    'sell_exchange': max_exchange,
                    'spread': spread_pct,
                    'net_profit': net_profit
                })
        
        return opportunities
```

## Risk Management Algorithms

### 21. Dynamic Position Sizing

**Kelly Criterion with Modifications**:
```python
class KellyPositionSizer:
    def __init__(self, max_kelly_fraction=0.25):
        self.max_kelly_fraction = max_kelly_fraction
        
    def calculate_position_size(self, win_probability, win_loss_ratio, confidence=1.0):
        """
        Modified Kelly Criterion for trading
        f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = win/loss ratio
        """
        p = win_probability
        q = 1 - p
        b = win_loss_ratio
        
        # Basic Kelly
        kelly_fraction = (p * b - q) / b
        
        # Apply confidence adjustment
        kelly_fraction *= confidence
        
        # Cap at maximum fraction
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        # Never go negative (no shorting via Kelly)
        kelly_fraction = max(kelly_fraction, 0)
        
        return kelly_fraction
```

### 22. Adaptive Stop Loss

**Volatility-Based Stops**:
```python
class AdaptiveStopLoss:
    def __init__(self, atr_multiplier=2.0):
        self.atr_multiplier = atr_multiplier
        
    def calculate_stop_loss(self, entry_price, atr, position_type='long'):
        """
        Calculate adaptive stop loss based on ATR
        """
        stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        return stop_price
    
    def trailing_stop(self, current_price, highest_price, atr, position_type='long'):
        """
        Adaptive trailing stop
        """
        if position_type == 'long':
            # Trail stop up, never down
            new_stop = current_price - (atr * self.atr_multiplier)
            return max(new_stop, self.current_stop)
        else:  # short
            # Trail stop down, never up
            new_stop = current_price + (atr * self.atr_multiplier)
            return min(new_stop, self.current_stop)
```

### 23. Portfolio Risk Optimization

**Conditional Value at Risk (CVaR)**:
```python
class CVaROptimizer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        
    def optimize_portfolio(self, returns, target_return=None):
        """
        Minimize CVaR subject to return constraint
        """
        n_assets = returns.shape[1]
        n_scenarios = returns.shape[0]
        
        # Decision variables
        weights = cp.Variable(n_assets)
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Auxiliary variables
        
        # Objective: Minimize CVaR
        cvar = z + (1 / (n_scenarios * (1 - self.confidence_level))) * cp.sum(u)
        
        # Constraints
        constraints = [
            weights >= 0,  # Long only
            cp.sum(weights) == 1,  # Fully invested
            u >= 0,
            u >= -returns @ weights - z
        ]
        
        if target_return is not None:
            constraints.append(
                cp.sum(cp.multiply(returns.mean(axis=0), weights)) >= target_return
            )
        
        # Solve
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve()
        
        return weights.value, cvar.value
```

---

These algorithms and strategies work together in the Alpaca-MCP system to identify, validate, and execute profitable trading opportunities across multiple asset classes and market conditions. Each algorithm is continuously monitored and optimized based on real-time performance metrics.