#!/usr/bin/env python3
"""
Advanced Ensemble Options AI System
Integrates multiple ML/AI techniques with expanded Greeks and unified prediction
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML/AI imports
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingClassifier, VotingRegressor,
    BaggingRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    
# Advanced libraries
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# Statistical libraries
from scipy.stats import norm, t, chi2, ncx2, skew, kurtosis
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedGreeksCalculator:
    """Calculate expanded Greeks including higher-order and exotic Greeks"""
    
    def __init__(self):
        self.r = 0.05  # Risk-free rate
        
    def calculate_all_greeks(self, S: float, K: float, T: float, sigma: float, 
                           option_type: str = 'call', dividend: float = 0) -> Dict:
        """Calculate all Greeks including higher-order"""
        
        # Prevent division by zero
        T = max(T, 1/365)
        sigma = max(sigma, 0.01)
        
        # Adjusted stock price for dividends
        S_adj = S * np.exp(-dividend * T)
        
        # d1 and d2
        d1 = (np.log(S_adj / K) + (self.r - dividend + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard Greeks
        if option_type == 'call':
            delta = np.exp(-dividend * T) * norm.cdf(d1)
            theta = (-S_adj * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
                    + dividend * S_adj * norm.cdf(d1))
        else:
            delta = np.exp(-dividend * T) * (norm.cdf(d1) - 1)
            theta = (-S_adj * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
                    - dividend * S_adj * norm.cdf(-d1))
        
        gamma = np.exp(-dividend * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S_adj * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-self.r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2)) / 100
        
        # Higher-order Greeks
        vanna = -np.exp(-dividend * T) * norm.pdf(d1) * d2 / sigma  # ∂Delta/∂σ
        charm = -np.exp(-dividend * T) * norm.pdf(d1) * (2 * self.r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))  # ∂Delta/∂T
        vomma = vega * d1 * d2 / sigma  # ∂Vega/∂σ
        veta = -S_adj * norm.pdf(d1) * np.sqrt(T) * (self.r + d1 * sigma / (2 * np.sqrt(T))) / 100  # ∂Vega/∂T
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)  # ∂Gamma/∂S
        zomma = gamma * (d1 * d2 - 1) / sigma  # ∂Gamma/∂σ
        color = -np.exp(-dividend * T) * norm.pdf(d1) / (2 * S * T * sigma * np.sqrt(T)) * (2 * self.r * T - d2 * sigma * np.sqrt(T) + d1 * d2)  # ∂Gamma/∂T
        
        # Ultima (third-order)
        ultima = -vega / (sigma**2) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)  # ∂Vomma/∂σ
        
        # Lambda (leverage)
        option_price = self.black_scholes_price(S, K, T, sigma, option_type, dividend)
        lambda_greek = delta * S / option_price if option_price > 0.01 else 0
        
        return {
            # First-order
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Daily theta
            'vega': vega,
            'rho': rho,
            
            # Second-order
            'vanna': vanna,
            'charm': charm / 365,  # Daily charm
            'vomma': vomma,
            'veta': veta / 365,  # Daily veta
            'speed': speed,
            'zomma': zomma,
            'color': color / 365,  # Daily color
            
            # Third-order
            'ultima': ultima,
            
            # Other useful metrics
            'lambda': lambda_greek,
            'alpha': theta / option_price if option_price > 0.01 else 0,  # Theta decay rate
            'dual_delta': K * np.exp(-self.r * T) * (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)),
            'dual_gamma': K * np.exp(-self.r * T) * norm.pdf(d2) / (S**2 * sigma * np.sqrt(T))
        }
    
    def black_scholes_price(self, S: float, K: float, T: float, sigma: float, 
                           option_type: str = 'call', dividend: float = 0) -> float:
        """Calculate Black-Scholes option price"""
        T = max(T, 1/365)
        S_adj = S * np.exp(-dividend * T)
        
        d1 = (np.log(S_adj / K) + (self.r - dividend + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S_adj * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
            
        return max(0, price)
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, 
                                   T: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson"""
        
        def objective(sigma):
            return self.black_scholes_price(S, K, T, sigma, option_type) - option_price
        
        def vega_func(sigma):
            S_adj = S
            d1 = (np.log(S_adj / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S_adj * norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson iteration
        sigma = 0.25  # Initial guess
        for _ in range(50):
            price_diff = objective(sigma)
            vega = vega_func(sigma)
            
            if abs(price_diff) < 1e-6 or vega < 1e-10:
                break
                
            sigma = sigma - price_diff / vega
            sigma = max(0.01, min(sigma, 5.0))  # Bounds
            
        return sigma

class DeepLearningModels:
    """PyTorch-based deep learning models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
    class LSTMPredictor(nn.Module):
        """LSTM for time series prediction"""
        def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout, bidirectional=True)
            self.fc1 = nn.Linear(hidden_size * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]  # Last time step
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    class TransformerPredictor(nn.Module):
        """Transformer for sequence prediction"""
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=3):
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, 1)
            
        def forward(self, x):
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            x = self.fc(x)
            return x
    
    class CNNPredictor(nn.Module):
        """1D CNN for pattern recognition"""
        def __init__(self, input_channels, sequence_length):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc_input_size = 256 * (sequence_length // 8)
            self.fc1 = nn.Linear(self.fc_input_size, 128)
            self.fc2 = nn.Linear(128, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = x.transpose(1, 2)  # (batch, features, sequence) 
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class AdvancedFeatureEngineering:
    """Advanced feature engineering for options and stocks"""
    
    def __init__(self, greeks_calculator: AdvancedGreeksCalculator):
        self.greeks_calc = greeks_calculator
        
    def engineer_features(self, data: pd.DataFrame, options_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
        
        # Technical indicators
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'distance_from_sma_{period}'] = (data['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_mean = data['close'].rolling(period).mean()
            bb_std = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = bb_mean + 2 * bb_std
            features[f'bb_lower_{period}'] = bb_mean - 2 * bb_std
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / bb_mean
            features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # RSI
        for period in [14, 30]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Stochastic
        for period in [14, 30]:
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        features['obv'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        
        # Money Flow Index
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Statistical features
        for period in [20, 50]:
            features[f'skew_{period}'] = features['returns'].rolling(period).apply(lambda x: skew(x))
            features[f'kurtosis_{period}'] = features['returns'].rolling(period).apply(lambda x: kurtosis(x))
            features[f'max_drawdown_{period}'] = (data['close'] / data['close'].rolling(period).max() - 1)
        
        # Microstructure features
        features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_open_ratio'] = (data['close'] - data['open']) / data['open']
        features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
        features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
        
        # Fourier features (frequency domain)
        close_values = data['close'].values[-100:]  # Last 100 points
        if len(close_values) >= 100:
            fft_values = fft(close_values)
            fft_freq = fftfreq(len(close_values))
            
            # Dominant frequencies
            power_spectrum = np.abs(fft_values)**2
            dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
            
            for i, idx in enumerate(dominant_freq_idx):
                features[f'fft_freq_{i}'] = fft_freq[idx]
                features[f'fft_power_{i}'] = power_spectrum[idx]
        
        # Options-specific features if available
        if options_data is not None:
            # Put-Call ratio
            call_volume = options_data[options_data['type'] == 'call']['volume'].sum()
            put_volume = options_data[options_data['type'] == 'put']['volume'].sum()
            features['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 1
            
            # Implied volatility skew
            atm_strike = data['close'].iloc[-1]
            otm_calls = options_data[(options_data['type'] == 'call') & 
                                    (options_data['strike'] > atm_strike * 1.05)]
            otm_puts = options_data[(options_data['type'] == 'put') & 
                                   (options_data['strike'] < atm_strike * 0.95)]
            
            if len(otm_calls) > 0 and len(otm_puts) > 0:
                features['iv_skew'] = otm_puts['implied_volatility'].mean() - otm_calls['implied_volatility'].mean()
            
            # Term structure
            near_term = options_data[options_data['days_to_expiry'] <= 30]
            far_term = options_data[options_data['days_to_expiry'] > 30]
            
            if len(near_term) > 0 and len(far_term) > 0:
                features['term_structure'] = far_term['implied_volatility'].mean() - near_term['implied_volatility'].mean()
        
        # Time-based features
        features['hour'] = pd.to_datetime(data.index).hour
        features['day_of_week'] = pd.to_datetime(data.index).dayofweek
        features['day_of_month'] = pd.to_datetime(data.index).day
        features['month'] = pd.to_datetime(data.index).month
        
        # Interaction features
        features['rsi_bb_interaction'] = features['rsi_14'] * features['bb_position_20']
        features['volume_volatility_interaction'] = features['volume_ratio'] * features['volatility']
        features['momentum_volume'] = features['returns'] * features['volume_ratio']
        
        return features.fillna(method='ffill').fillna(0)

class EnsemblePredictor:
    """Advanced ensemble predictor combining multiple models"""
    
    def __init__(self, feature_engineer: AdvancedFeatureEngineering):
        self.feature_engineer = feature_engineer
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_regression, k=50)
        
    def build_ensemble(self):
        """Build comprehensive ensemble of models"""
        
        # Tree-based models
        self.models['rf'] = RandomForestRegressor(n_estimators=300, max_depth=15, 
                                                 min_samples_split=5, n_jobs=-1)
        self.models['et'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, 
                                               min_samples_split=5, n_jobs=-1)
        self.models['gb'] = GradientBoostingRegressor(n_estimators=200, max_depth=7, 
                                                      learning_rate=0.05, subsample=0.8)
        self.models['hist_gb'] = HistGradientBoostingRegressor(max_iter=200, max_depth=10,
                                                               learning_rate=0.05)
        
        # Boosting models
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = XGBRegressor(n_estimators=200, max_depth=7, 
                                             learning_rate=0.05, subsample=0.8)
        
        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = LGBMRegressor(n_estimators=200, max_depth=7,
                                              learning_rate=0.05, subsample=0.8)
        
        if CATBOOST_AVAILABLE:
            self.models['cat'] = CatBoostRegressor(iterations=200, depth=7,
                                                  learning_rate=0.05, verbose=False)
        
        # Neural networks
        self.models['mlp'] = MLPRegressor(hidden_layer_sizes=(200, 100, 50),
                                         activation='relu', solver='adam',
                                         max_iter=500, early_stopping=True)
        
        # Linear models
        self.models['elastic'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        self.models['huber'] = HuberRegressor(epsilon=1.35, max_iter=1000)
        
        # Support Vector Machine
        self.models['svr'] = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        
        # Gaussian Process
        kernel = RBF() + Matern() + RationalQuadratic()
        self.models['gp'] = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                                    normalize_y=True)
        
        # K-Nearest Neighbors
        self.models['knn'] = KNeighborsRegressor(n_neighbors=20, weights='distance')
        
        # Initialize ensemble weights
        self.ensemble_weights = {name: 1.0 / len(self.models) for name in self.models}
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                      validation_split: float = 0.2) -> Dict:
        """Train ensemble with dynamic weight optimization"""
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = defaultdict(list)
        
        for train_idx, val_idx in tscv.split(X_selected):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train each model
            for name, model in self.models.items():
                try:
                    # Clone model for this fold
                    if name == 'cat' and CATBOOST_AVAILABLE:
                        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                                verbose=False, early_stopping_rounds=50)
                    else:
                        model.fit(X_train, y_train)
                    
                    # Evaluate
                    pred = model.predict(X_val)
                    score = r2_score(y_val, pred)
                    model_scores[name].append(score)
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    model_scores[name].append(0)
        
        # Calculate ensemble weights based on performance
        avg_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
        total_score = sum(max(0, score) for score in avg_scores.values())
        
        if total_score > 0:
            self.ensemble_weights = {
                name: max(0, score) / total_score 
                for name, score in avg_scores.items()
            }
        
        # Train final models on full data
        for name, model in self.models.items():
            try:
                if name == 'cat' and CATBOOST_AVAILABLE:
                    model.fit(X_selected, y, verbose=False)
                else:
                    model.fit(X_selected, y)
            except Exception as e:
                logger.error(f"Error in final training {name}: {e}")
        
        return {
            'model_scores': dict(avg_scores),
            'ensemble_weights': self.ensemble_weights,
            'selected_features': self.feature_selector.get_support()
        }
    
    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make ensemble prediction with uncertainty estimation"""
        
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        predictions = []
        
        for name, model in self.models.items():
            if self.ensemble_weights.get(name, 0) > 0:
                try:
                    pred = model.predict(X_selected)
                    predictions.append(pred * self.ensemble_weights[name])
                except Exception as e:
                    logger.error(f"Prediction error {name}: {e}")
        
        if not predictions:
            return np.zeros(len(X)) if not return_std else (np.zeros(len(X)), np.ones(len(X)))
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        if return_std:
            # Estimate uncertainty as weighted standard deviation
            pred_array = np.array([p / self.ensemble_weights[name] 
                                  for name, p in zip(self.models.keys(), predictions)
                                  if self.ensemble_weights.get(name, 0) > 0])
            ensemble_std = np.std(pred_array, axis=0)
            return ensemble_pred, ensemble_std
        
        return ensemble_pred

class IntegratedOptionsStockPredictor:
    """Unified predictor for stocks and options at specific time points"""
    
    def __init__(self, paper=True):
        # API setup
        if paper:
            self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
            self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
        else:
            self.api_key = 'AK7LZKPVTPZTOTO9VVPM'
            self.api_secret = '2TjtRymW9aWXkWWQFwTThQGAKQrTbSWwLdz1LGKI'
            
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Initialize components
        self.greeks_calc = AdvancedGreeksCalculator()
        self.feature_engineer = AdvancedFeatureEngineering(self.greeks_calc)
        self.ensemble = EnsemblePredictor(self.feature_engineer)
        self.dl_models = DeepLearningModels() if TORCH_AVAILABLE else None
        
        # Build models
        self.ensemble.build_ensemble()
        
        # Cache
        self.prediction_cache = {}
        
    def predict_at_time_point(self, symbol: str, prediction_time: datetime, 
                             horizons: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Make integrated prediction for stock and options at specific time point
        
        Args:
            symbol: Stock symbol
            prediction_time: Time point for prediction
            horizons: Prediction horizons in days
            
        Returns:
            Comprehensive prediction dictionary
        """
        
        # Get historical data
        end_date = prediction_time
        start_date = end_date - timedelta(days=365)
        
        try:
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
            )
            
            if symbol not in bars.df.index:
                return {'error': f'No data for {symbol}'}
                
            df = bars.df.loc[symbol]
            
            # Engineer features
            features = self.feature_engineer.engineer_features(df)
            
            # Prepare data for prediction
            X = features.iloc[-252:-1]  # Last year excluding last day
            y = df['close'].pct_change().shift(-1).iloc[-252:-1]  # Next day returns
            
            # Train ensemble
            train_results = self.ensemble.train_ensemble(X, y)
            
            # Make predictions for different horizons
            predictions = {}
            current_price = df['close'].iloc[-1]
            
            for horizon in horizons:
                # Feature engineering for prediction
                pred_features = features.iloc[-1:].copy()
                
                # Add horizon-specific features
                pred_features[f'horizon'] = horizon
                pred_features[f'horizon_sqrt'] = np.sqrt(horizon)
                pred_features[f'horizon_log'] = np.log(horizon + 1)
                
                # Predict return
                pred_return, pred_std = self.ensemble.predict(pred_features, return_std=True)
                
                # Convert to price
                pred_price = current_price * (1 + pred_return[0])
                price_std = current_price * pred_std[0]
                
                # Calculate confidence intervals
                ci_lower = pred_price - 1.96 * price_std
                ci_upper = pred_price + 1.96 * price_std
                
                predictions[f'{horizon}d'] = {
                    'predicted_price': pred_price,
                    'predicted_return': pred_return[0],
                    'confidence_interval': (ci_lower, ci_upper),
                    'uncertainty': pred_std[0],
                    'probability_up': norm.cdf(0, loc=pred_return[0], scale=pred_std[0])
                }
                
                # Options predictions
                if horizon <= 30:  # Only for reasonable option horizons
                    options_pred = self.predict_options_for_horizon(
                        symbol, current_price, pred_price, price_std, horizon
                    )
                    predictions[f'{horizon}d']['options'] = options_pred
            
            # Market regime prediction
            regime = self.predict_market_regime(features.iloc[-20:])
            
            # Technical signals
            signals = self.generate_trading_signals(df, features)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'prediction_time': prediction_time.isoformat(),
                'predictions': predictions,
                'market_regime': regime,
                'trading_signals': signals,
                'model_performance': train_results['model_scores'],
                'ensemble_weights': train_results['ensemble_weights']
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {'error': str(e)}
    
    def predict_options_for_horizon(self, symbol: str, current_price: float,
                                   predicted_price: float, price_std: float,
                                   horizon_days: int) -> Dict:
        """Predict options values and optimal strategies"""
        
        # Calculate expected volatility
        annual_vol = price_std / current_price * np.sqrt(252 / horizon_days)
        
        # Generate strike recommendations
        strikes = self.generate_optimal_strikes(current_price, predicted_price, price_std)
        
        options_analysis = {}
        
        for strike in strikes:
            # Calculate Greeks for calls and puts
            T = horizon_days / 365
            
            call_greeks = self.greeks_calc.calculate_all_greeks(
                current_price, strike, T, annual_vol, 'call'
            )
            put_greeks = self.greeks_calc.calculate_all_greeks(
                current_price, strike, T, annual_vol, 'put'
            )
            
            # Expected values at expiration
            call_payoff = max(0, predicted_price - strike)
            put_payoff = max(0, strike - predicted_price)
            
            # Current theoretical values
            call_value = self.greeks_calc.black_scholes_price(
                current_price, strike, T, annual_vol, 'call'
            )
            put_value = self.greeks_calc.black_scholes_price(
                current_price, strike, T, annual_vol, 'put'
            )
            
            options_analysis[f'strike_{strike}'] = {
                'call': {
                    'theoretical_value': call_value,
                    'expected_payoff': call_payoff,
                    'greeks': call_greeks,
                    'breakeven': strike + call_value,
                    'max_profit': float('inf'),
                    'max_loss': call_value
                },
                'put': {
                    'theoretical_value': put_value,
                    'expected_payoff': put_payoff,
                    'greeks': put_greeks,
                    'breakeven': strike - put_value,
                    'max_profit': strike - put_value,
                    'max_loss': put_value
                }
            }
        
        # Recommend strategies
        strategies = self.recommend_options_strategies(
            current_price, predicted_price, price_std, annual_vol, horizon_days
        )
        
        return {
            'strikes_analysis': options_analysis,
            'recommended_strategies': strategies,
            'implied_volatility': annual_vol,
            'volatility_forecast': {
                'current': annual_vol,
                'predicted': annual_vol * (1 + np.random.normal(0, 0.1))  # Simplified
            }
        }
    
    def generate_optimal_strikes(self, current_price: float, predicted_price: float,
                                price_std: float) -> List[float]:
        """Generate optimal strike prices based on prediction"""
        
        strikes = []
        
        # ATM strike
        atm = round(current_price / 5) * 5
        strikes.append(atm)
        
        # Predicted price strike
        pred_strike = round(predicted_price / 5) * 5
        if pred_strike != atm:
            strikes.append(pred_strike)
        
        # One standard deviation strikes
        upper_1sd = round((current_price + price_std) / 5) * 5
        lower_1sd = round((current_price - price_std) / 5) * 5
        strikes.extend([lower_1sd, upper_1sd])
        
        # Two standard deviation strikes
        upper_2sd = round((current_price + 2 * price_std) / 5) * 5
        lower_2sd = round((current_price - 2 * price_std) / 5) * 5
        strikes.extend([lower_2sd, upper_2sd])
        
        return sorted(list(set(strikes)))
    
    def recommend_options_strategies(self, current_price: float, predicted_price: float,
                                   price_std: float, volatility: float, 
                                   horizon_days: int) -> List[Dict]:
        """Recommend optimal options strategies based on prediction"""
        
        strategies = []
        
        # Calculate prediction confidence
        move_size = abs(predicted_price - current_price) / current_price
        confidence = 1 - (price_std / current_price)
        
        # Directional strategies
        if predicted_price > current_price * 1.01:  # Bullish
            if confidence > 0.7:
                # High confidence - aggressive
                strategies.append({
                    'name': 'Long Call',
                    'confidence': confidence,
                    'description': 'Buy ATM or slightly OTM call',
                    'risk_level': 'High',
                    'max_profit': 'Unlimited',
                    'max_loss': 'Premium paid'
                })
                
                strategies.append({
                    'name': 'Bull Call Spread',
                    'confidence': confidence * 0.9,
                    'description': 'Buy ATM call, sell OTM call',
                    'risk_level': 'Medium',
                    'max_profit': 'Strike difference - net debit',
                    'max_loss': 'Net debit'
                })
            
            strategies.append({
                'name': 'Bull Put Spread',
                'confidence': confidence * 0.8,
                'description': 'Sell OTM put, buy further OTM put',
                'risk_level': 'Medium',
                'max_profit': 'Net credit',
                'max_loss': 'Strike difference - net credit'
            })
            
        elif predicted_price < current_price * 0.99:  # Bearish
            if confidence > 0.7:
                strategies.append({
                    'name': 'Long Put',
                    'confidence': confidence,
                    'description': 'Buy ATM or slightly OTM put',
                    'risk_level': 'High',
                    'max_profit': 'Strike price - premium',
                    'max_loss': 'Premium paid'
                })
            
            strategies.append({
                'name': 'Bear Put Spread',
                'confidence': confidence * 0.9,
                'description': 'Buy ATM put, sell OTM put',
                'risk_level': 'Medium',
                'max_profit': 'Strike difference - net debit',
                'max_loss': 'Net debit'
            })
        
        # Neutral strategies
        if move_size < 0.02:  # Small expected move
            strategies.append({
                'name': 'Iron Condor',
                'confidence': 0.7,
                'description': 'Sell OTM call spread and put spread',
                'risk_level': 'Medium',
                'max_profit': 'Net credit',
                'max_loss': 'Strike width - net credit'
            })
            
            strategies.append({
                'name': 'Iron Butterfly',
                'confidence': 0.65,
                'description': 'Sell ATM straddle, buy OTM strangle',
                'risk_level': 'Medium-High',
                'max_profit': 'Net credit',
                'max_loss': 'Strike width - net credit'
            })
        
        # Volatility strategies
        if volatility > 0.3:  # High volatility
            strategies.append({
                'name': 'Long Straddle',
                'confidence': 0.6,
                'description': 'Buy ATM call and put',
                'risk_level': 'High',
                'max_profit': 'Unlimited',
                'max_loss': 'Total premium paid'
            })
            
            strategies.append({
                'name': 'Long Strangle',
                'confidence': 0.55,
                'description': 'Buy OTM call and put',
                'risk_level': 'High',
                'max_profit': 'Unlimited',
                'max_loss': 'Total premium paid'
            })
        
        # Advanced strategies for experienced traders
        if confidence > 0.6 and volatility < 0.25:
            strategies.append({
                'name': 'Jade Lizard',
                'confidence': confidence * 0.7,
                'description': 'Sell OTM call, sell OTM put spread',
                'risk_level': 'Advanced',
                'max_profit': 'Net credit',
                'max_loss': 'Put spread width - net credit (downside only)'
            })
            
            strategies.append({
                'name': 'Broken Wing Butterfly',
                'confidence': confidence * 0.65,
                'description': 'Asymmetric butterfly with skewed risk',
                'risk_level': 'Advanced',
                'max_profit': 'Net credit + spread width',
                'max_loss': 'Varies by construction'
            })
        
        # Sort by confidence
        strategies.sort(key=lambda x: x['confidence'], reverse=True)
        
        return strategies[:5]  # Top 5 recommendations
    
    def predict_market_regime(self, recent_features: pd.DataFrame) -> Dict:
        """Predict current market regime"""
        
        # Simple regime detection based on features
        volatility = recent_features['volatility'].iloc[-1]
        trend = recent_features['returns'].mean()
        volume_trend = recent_features['volume_ratio'].mean()
        
        regime = {
            'volatility_regime': 'High' if volatility > 0.25 else 'Low',
            'trend_regime': 'Bullish' if trend > 0.001 else 'Bearish' if trend < -0.001 else 'Neutral',
            'volume_regime': 'High' if volume_trend > 1.2 else 'Normal',
            'vix_proxy': volatility * 100,
            'market_state': 'Risk-On' if trend > 0 and volatility < 0.2 else 'Risk-Off'
        }
        
        # Regime probabilities
        regime['regime_probabilities'] = {
            'bull_market': 0.4 if trend > 0 else 0.2,
            'bear_market': 0.4 if trend < 0 else 0.2,
            'ranging_market': 0.4 if abs(trend) < 0.001 else 0.2,
            'volatile_market': 0.6 if volatility > 0.25 else 0.2
        }
        
        return regime
    
    def generate_trading_signals(self, price_data: pd.DataFrame, 
                               features: pd.DataFrame) -> Dict:
        """Generate comprehensive trading signals"""
        
        signals = {
            'technical': {},
            'statistical': {},
            'ml_based': {},
            'options_flow': {}
        }
        
        # Technical signals
        rsi = features['rsi_14'].iloc[-1]
        bb_position = features['bb_position_20'].iloc[-1]
        macd_hist = features['macd_histogram'].iloc[-1]
        
        signals['technical'] = {
            'rsi_signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral',
            'bollinger_signal': 'Oversold' if bb_position < 0.2 else 'Overbought' if bb_position > 0.8 else 'Neutral',
            'macd_signal': 'Bullish' if macd_hist > 0 else 'Bearish',
            'overall': self._combine_technical_signals(rsi, bb_position, macd_hist)
        }
        
        # Statistical signals
        recent_returns = features['returns'].iloc[-20:]
        signals['statistical'] = {
            'mean_reversion': 'Buy' if recent_returns.mean() < -0.02 else 'Sell' if recent_returns.mean() > 0.02 else 'Hold',
            'momentum': 'Buy' if recent_returns.iloc[-5:].mean() > recent_returns.iloc[-20:-5].mean() else 'Sell',
            'volatility_signal': 'Reduce' if features['volatility'].iloc[-1] > features['volatility'].iloc[-20:].mean() * 1.5 else 'Normal'
        }
        
        return signals
    
    def _combine_technical_signals(self, rsi: float, bb_pos: float, macd: float) -> str:
        """Combine technical signals into overall signal"""
        
        score = 0
        
        # RSI contribution
        if rsi < 30:
            score += 2
        elif rsi < 40:
            score += 1
        elif rsi > 70:
            score -= 2
        elif rsi > 60:
            score -= 1
            
        # Bollinger Bands contribution
        if bb_pos < 0.2:
            score += 2
        elif bb_pos < 0.4:
            score += 1
        elif bb_pos > 0.8:
            score -= 2
        elif bb_pos > 0.6:
            score -= 1
            
        # MACD contribution
        if macd > 0:
            score += 1
        else:
            score -= 1
            
        # Overall signal
        if score >= 3:
            return 'Strong Buy'
        elif score >= 1:
            return 'Buy'
        elif score <= -3:
            return 'Strong Sell'
        elif score <= -1:
            return 'Sell'
        else:
            return 'Neutral'

def main():
    """Demonstration of integrated prediction system"""
    
    logger.info("🚀 ADVANCED ENSEMBLE OPTIONS AI SYSTEM")
    logger.info("=" * 70)
    
    # Initialize predictor
    predictor = IntegratedOptionsStockPredictor(paper=True)
    
    # Test symbols
    symbols = ['SPY', 'AAPL', 'TSLA']
    
    for symbol in symbols:
        logger.info(f"\n📊 Analyzing {symbol}...")
        
        # Make prediction
        prediction = predictor.predict_at_time_point(
            symbol=symbol,
            prediction_time=datetime.now(),
            horizons=[1, 5, 10, 30]
        )
        
        if 'error' in prediction:
            logger.error(f"Error: {prediction['error']}")
            continue
            
        # Display results
        logger.info(f"\nCurrent Price: ${prediction['current_price']:.2f}")
        logger.info(f"Market Regime: {prediction['market_regime']['market_state']}")
        logger.info(f"Technical Signal: {prediction['trading_signals']['technical']['overall']}")
        
        logger.info("\n📈 Price Predictions:")
        for horizon, pred in prediction['predictions'].items():
            logger.info(f"\n{horizon} Prediction:")
            logger.info(f"  Price: ${pred['predicted_price']:.2f} ({pred['predicted_return']:+.2%})")
            logger.info(f"  95% CI: ${pred['confidence_interval'][0]:.2f} - ${pred['confidence_interval'][1]:.2f}")
            logger.info(f"  Probability Up: {pred['probability_up']:.1%}")
            
            if 'options' in pred:
                logger.info(f"\n  Top Options Strategies:")
                for i, strategy in enumerate(pred['options']['recommended_strategies'][:3]):
                    logger.info(f"    {i+1}. {strategy['name']} (Confidence: {strategy['confidence']:.1%})")
        
        logger.info("\n" + "-" * 70)

if __name__ == "__main__":
    main()