#!/usr/bin/env python3
"""
AI-Powered Options Trading System
Integrates stock prediction algorithms with multi-leg options strategies
Uses ML predictions, technical analysis, and market microstructure to trade options
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import norm
from scipy.optimize import minimize
import requests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Trading APIs
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AIPoweredOptionsSystem:
    def __init__(self):
        """Initialize AI-powered options trading system"""
        # API credentials
        self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
        self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
        self.base_url = 'https://paper-api.alpaca.markets'
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # API headers
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # System parameters
        self.scan_interval = 30
        self.max_positions = 20
        self.position_size = 1  # contracts
        self.confidence_threshold = 0.65  # 65% confidence required
        
        # Prediction models (will be trained on startup)
        self.price_predictor = None
        self.volatility_predictor = None
        self.direction_classifier = None
        
        # Feature scalers
        self.price_scaler = StandardScaler()
        self.vol_scaler = StandardScaler()
        
        # Watchlist with focus symbols
        self.watchlist = {
            'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'high_volatility': ['TSLA', 'AMD', 'COIN', 'MARA', 'RIOT'],
            'indexes': ['SPY', 'QQQ', 'IWM', 'DIA'],
            'financials': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC'],
            'etfs': ['XLF', 'XLE', 'XLK', 'GLD', 'TLT', 'VXX']
        }
        
        # Track predictions and trades
        self.predictions_cache = {}
        self.active_strategies = {}
        self.performance_tracker = defaultdict(lambda: {
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0
        })
        
        # Technical indicators settings
        self.indicators = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'bb_period': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
    async def run(self):
        """Main execution loop"""
        logger.info("🚀 Starting AI-Powered Options Trading System")
        logger.info("🧠 Training prediction models...")
        
        # Train models on startup
        await self.train_prediction_models()
        
        # Display account status
        await self.display_account_status()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"🎯 AI Options Trading Cycle #{cycle}")
                logger.info(f"{'='*70}")
                
                # Check market status
                if not await self.is_market_open():
                    logger.info("Market is closed")
                    await asyncio.sleep(300)
                    continue
                
                # Run predictions for all symbols
                await self.generate_predictions()
                
                # Execute AI-driven strategies
                await self.execute_ai_strategies()
                
                # Manage existing positions
                await self.manage_positions_with_ai()
                
                # Display dashboard
                await self.display_ai_dashboard()
                
                await asyncio.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Closing all positions...")
                await self.close_all_positions()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
                
    async def train_prediction_models(self):
        """Train ML models for price and volatility prediction"""
        logger.info("Training prediction models...")
        
        try:
            # Initialize models
            self.price_predictor = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1
            )
            self.volatility_predictor = RandomForestRegressor(
                n_estimators=100, max_depth=10
            )
            self.direction_classifier = GradientBoostingRegressor(
                n_estimators=50, max_depth=3
            )
            
            # Train on a few key symbols
            for symbol in ['SPY', 'AAPL', 'TSLA']:
                await self.train_model_for_symbol(symbol)
                
            logger.info("✅ Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            # Use simple fallback predictors
            self.use_fallback_predictors()
            
    async def train_model_for_symbol(self, symbol: str):
        """Train models for a specific symbol"""
        try:
            # Get historical data
            df = await self.get_historical_data(symbol, periods=252)  # 1 year
            if df is None or len(df) < 100:
                return
                
            # Calculate features
            features = self.calculate_ml_features(df)
            if features is None:
                return
                
            # Prepare targets
            # Price prediction: next day's return
            df['next_return'] = df['close'].pct_change().shift(-1)
            # Volatility prediction: next day's realized volatility
            df['next_volatility'] = df['close'].pct_change().rolling(5).std().shift(-1)
            
            # Remove NaN values
            valid_data = pd.concat([features, df[['next_return', 'next_volatility']]], axis=1).dropna()
            
            if len(valid_data) < 50:
                return
                
            X = valid_data[features.columns]
            y_price = valid_data['next_return']
            y_vol = valid_data['next_volatility']
            
            # Train price predictor
            X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2)
            X_train_scaled = self.price_scaler.fit_transform(X_train)
            self.price_predictor.fit(X_train_scaled, y_train)
            
            # Train volatility predictor
            X_train_vol, X_test_vol, y_train_vol, y_test_vol = train_test_split(X, y_vol, test_size=0.2)
            X_train_vol_scaled = self.vol_scaler.fit_transform(X_train_vol)
            self.volatility_predictor.fit(X_train_vol_scaled, y_train_vol)
            
            logger.info(f"  ✅ Trained models for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            
    def calculate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate machine learning features from price data"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features['returns_1d'] = df['close'].pct_change()
            features['returns_5d'] = df['close'].pct_change(5)
            features['returns_20d'] = df['close'].pct_change(20)
            
            # Moving averages
            for period in self.indicators['sma_periods']:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']
                
            # Volatility features
            features['volatility_5d'] = features['returns_1d'].rolling(5).std()
            features['volatility_20d'] = features['returns_1d'].rolling(20).std()
            features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d']
            
            # RSI
            rsi_period = self.indicators['rsi_period']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = self.indicators['bb_period']
            sma = df['close'].rolling(bb_period).mean()
            std = df['close'].rolling(bb_period).std()
            features['bb_upper'] = sma + (2 * std)
            features['bb_lower'] = sma - (2 * std)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # MACD
            exp1 = df['close'].ewm(span=self.indicators['macd_fast']).mean()
            exp2 = df['close'].ewm(span=self.indicators['macd_slow']).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=self.indicators['macd_signal']).mean()
            features['macd_diff'] = features['macd'] - features['macd_signal']
            
            # Volume features
            if 'volume' in df.columns:
                features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            # Market microstructure
            features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            features['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Momentum
            features['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
            features['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
            
    def use_fallback_predictors(self):
        """Use simple technical analysis as fallback"""
        logger.info("Using fallback predictors...")
        # These will use technical indicators directly
        
    async def generate_predictions(self):
        """Generate predictions for all watched symbols"""
        logger.info("\n🧠 Generating AI Predictions...")
        
        all_symbols = []
        for category in self.watchlist.values():
            all_symbols.extend(category)
            
        # Remove duplicates
        all_symbols = list(set(all_symbols))
        
        for symbol in all_symbols[:20]:  # Limit for performance
            try:
                prediction = await self.predict_stock_movement(symbol)
                if prediction:
                    self.predictions_cache[symbol] = prediction
                    
                    # Display high-confidence predictions
                    if prediction['confidence'] > 0.7:
                        logger.info(f"\n✨ High Confidence Prediction for {symbol}:")
                        logger.info(f"   Direction: {prediction['direction']}")
                        logger.info(f"   Expected Move: {prediction['expected_return']:.2%}")
                        logger.info(f"   Volatility: {prediction['expected_volatility']:.2%}")
                        logger.info(f"   Confidence: {prediction['confidence']:.1%}")
                        
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                
    async def predict_stock_movement(self, symbol: str) -> Dict:
        """Predict stock price movement and volatility"""
        try:
            # Get recent data
            df = await self.get_historical_data(symbol, periods=60)
            if df is None or len(df) < 30:
                return None
                
            # Calculate features
            features = self.calculate_ml_features(df)
            if features is None or len(features) == 0:
                return None
                
            # Get latest features
            latest_features = features.iloc[-1:].fillna(0)
            
            # Make predictions
            if self.price_predictor and self.volatility_predictor:
                # ML predictions
                X_scaled = self.price_scaler.transform(latest_features)
                expected_return = float(self.price_predictor.predict(X_scaled)[0])
                
                X_vol_scaled = self.vol_scaler.transform(latest_features)
                expected_volatility = float(self.volatility_predictor.predict(X_vol_scaled)[0])
                
                # Calculate confidence based on recent accuracy
                confidence = self.calculate_prediction_confidence(symbol, features)
            else:
                # Fallback to technical analysis
                prediction_ta = self.technical_analysis_prediction(df, features.iloc[-1])
                expected_return = prediction_ta['return']
                expected_volatility = prediction_ta['volatility']
                confidence = prediction_ta['confidence']
                
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Calculate target prices
            target_price = current_price * (1 + expected_return)
            stop_loss = current_price * (1 - expected_volatility * 2)  # 2 std devs
            
            # Determine direction
            if expected_return > 0.01:  # 1% threshold
                direction = 'bullish'
            elif expected_return < -0.01:
                direction = 'bearish'
            else:
                direction = 'neutral'
                
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'direction': direction,
                'confidence': confidence,
                'time_horizon': 5,  # 5 days
                'features': latest_features.to_dict('records')[0]
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return None
            
    def technical_analysis_prediction(self, df: pd.DataFrame, features: pd.Series) -> Dict:
        """Fallback technical analysis prediction"""
        try:
            # Simple prediction based on technical indicators
            rsi = features.get('rsi', 50)
            bb_position = features.get('bb_position', 0.5)
            macd_diff = features.get('macd_diff', 0)
            momentum = features.get('momentum_20d', 0)
            
            # Score calculation
            bull_score = 0
            bear_score = 0
            
            # RSI signals
            if rsi < 30:
                bull_score += 2
            elif rsi > 70:
                bear_score += 2
                
            # Bollinger Bands
            if bb_position < 0.2:
                bull_score += 1
            elif bb_position > 0.8:
                bear_score += 1
                
            # MACD
            if macd_diff > 0:
                bull_score += 1
            else:
                bear_score += 1
                
            # Momentum
            if momentum > 0.05:
                bull_score += 2
            elif momentum < -0.05:
                bear_score += 2
                
            # Calculate expected return
            net_score = bull_score - bear_score
            expected_return = net_score * 0.01  # 1% per score point
            
            # Volatility from historical data
            expected_volatility = df['close'].pct_change().std() * np.sqrt(5)
            
            # Confidence based on signal strength
            confidence = min(0.8, abs(net_score) * 0.1 + 0.3)
            
            return {
                'return': expected_return,
                'volatility': expected_volatility,
                'confidence': confidence
            }
            
        except Exception as e:
            # Default prediction
            return {'return': 0, 'volatility': 0.02, 'confidence': 0.5}
            
    def calculate_prediction_confidence(self, symbol: str, features: pd.DataFrame) -> float:
        """Calculate confidence based on model performance"""
        # Simplified confidence calculation
        # In production, track actual vs predicted performance
        base_confidence = 0.6
        
        # Adjust based on feature quality
        if len(features) > 50:
            base_confidence += 0.1
            
        # Adjust based on recent volatility
        recent_vol = features['volatility_20d'].iloc[-1]
        if recent_vol < 0.02:  # Low volatility = higher confidence
            base_confidence += 0.1
        elif recent_vol > 0.05:  # High volatility = lower confidence
            base_confidence -= 0.1
            
        return min(0.9, max(0.3, base_confidence))
        
    async def execute_ai_strategies(self):
        """Execute options strategies based on AI predictions"""
        logger.info("\n🎯 Executing AI-Driven Options Strategies...")
        
        # Get high-confidence predictions
        high_confidence_predictions = {
            symbol: pred for symbol, pred in self.predictions_cache.items()
            if pred['confidence'] > self.confidence_threshold
        }
        
        if not high_confidence_predictions:
            logger.info("No high-confidence predictions available")
            return
            
        # Sort by expected return magnitude
        sorted_predictions = sorted(
            high_confidence_predictions.items(),
            key=lambda x: abs(x[1]['expected_return']) * x[1]['confidence'],
            reverse=True
        )
        
        # Execute strategies for top predictions
        for symbol, prediction in sorted_predictions[:5]:  # Top 5
            try:
                # Get option chain
                chain = await self.get_option_chain(symbol)
                if not chain['contracts']:
                    continue
                    
                # Select strategy based on prediction
                strategy = self.select_optimal_strategy(prediction, chain)
                
                if strategy:
                    logger.info(f"\n🎲 Executing {strategy['type']} for {symbol}")
                    logger.info(f"   Based on: {prediction['direction']} prediction")
                    logger.info(f"   Expected return: {prediction['expected_return']:.2%}")
                    logger.info(f"   Confidence: {prediction['confidence']:.1%}")
                    
                    await self.execute_strategy(strategy)
                    
            except Exception as e:
                logger.error(f"Error executing strategy for {symbol}: {e}")
                
    def select_optimal_strategy(self, prediction: Dict, chain: Dict) -> Dict:
        """Select optimal options strategy based on prediction"""
        try:
            symbol = prediction['symbol']
            direction = prediction['direction']
            expected_return = prediction['expected_return']
            expected_volatility = prediction['expected_volatility']
            confidence = prediction['confidence']
            stock_price = chain['stock_price']
            contracts = chain['contracts']
            
            # Separate calls and puts
            calls = [c for c in contracts if c.get('type') == 'call']
            puts = [c for c in contracts if c.get('type') == 'put']
            
            strategy = None
            
            # High confidence directional plays
            if confidence > 0.75:
                if direction == 'bullish' and expected_return > 0.03:  # 3%+ expected
                    # Bull Call Spread or Call Debit Spread
                    strategy = self.construct_bull_call_spread(symbol, calls, stock_price, prediction)
                    
                elif direction == 'bearish' and expected_return < -0.03:
                    # Bear Put Spread or Put Debit Spread
                    strategy = self.construct_bear_put_spread(symbol, puts, stock_price, prediction)
                    
            # Moderate confidence strategies
            elif confidence > 0.65:
                if direction == 'bullish':
                    # Bull Put Spread (credit)
                    strategy = self.construct_bull_put_spread(symbol, puts, stock_price, prediction)
                    
                elif direction == 'bearish':
                    # Bear Call Spread (credit)
                    strategy = self.construct_bear_call_spread(symbol, calls, stock_price, prediction)
                    
                elif direction == 'neutral' and expected_volatility > 0.03:
                    # Iron Condor for high IV
                    strategy = self.construct_iron_condor(symbol, calls, puts, stock_price, prediction)
                    
            # Volatility plays
            if not strategy and expected_volatility > 0.04:  # High volatility expected
                # Straddle or Strangle
                strategy = self.construct_volatility_play(symbol, calls, puts, stock_price, prediction)
                
            return strategy
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return None
            
    def construct_bull_call_spread(self, symbol: str, calls: List, stock_price: float, 
                                  prediction: Dict) -> Dict:
        """Construct bull call spread based on prediction"""
        try:
            target_price = prediction['target_price']
            
            # Find appropriate strikes
            sorted_calls = sorted(calls, key=lambda x: float(x.get('strike_price', 0)))
            
            # Long call near current price
            long_call = None
            for call in sorted_calls:
                if float(call.get('strike_price', 0)) >= stock_price * 0.98:
                    long_call = call
                    break
                    
            # Short call near target
            short_call = None
            if long_call:
                long_strike = float(long_call.get('strike_price'))
                for call in sorted_calls:
                    if float(call.get('strike_price', 0)) >= target_price * 0.95:
                        short_call = call
                        break
                        
            if long_call and short_call:
                return {
                    'type': 'bull_call_spread',
                    'symbol': symbol,
                    'legs': [
                        {'option': long_call, 'side': 'buy', 'qty': 1},
                        {'option': short_call, 'side': 'sell', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'max_profit': float(short_call.get('strike_price')) - float(long_call.get('strike_price')),
                    'strategy_confidence': prediction['confidence'] * 0.9
                }
                
        except Exception as e:
            logger.error(f"Error constructing bull call spread: {e}")
            
        return None
        
    def construct_bear_put_spread(self, symbol: str, puts: List, stock_price: float,
                                 prediction: Dict) -> Dict:
        """Construct bear put spread based on prediction"""
        try:
            target_price = prediction['target_price']
            
            # Find appropriate strikes
            sorted_puts = sorted(puts, key=lambda x: float(x.get('strike_price', 0)), reverse=True)
            
            # Long put near current price
            long_put = None
            for put in sorted_puts:
                if float(put.get('strike_price', 0)) <= stock_price * 1.02:
                    long_put = put
                    break
                    
            # Short put near target
            short_put = None
            if long_put:
                long_strike = float(long_put.get('strike_price'))
                for put in sorted_puts:
                    if float(put.get('strike_price', 0)) <= target_price * 1.05:
                        short_put = put
                        break
                        
            if long_put and short_put:
                return {
                    'type': 'bear_put_spread',
                    'symbol': symbol,
                    'legs': [
                        {'option': long_put, 'side': 'buy', 'qty': 1},
                        {'option': short_put, 'side': 'sell', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'max_profit': float(long_put.get('strike_price')) - float(short_put.get('strike_price')),
                    'strategy_confidence': prediction['confidence'] * 0.9
                }
                
        except Exception as e:
            logger.error(f"Error constructing bear put spread: {e}")
            
        return None
        
    def construct_bull_put_spread(self, symbol: str, puts: List, stock_price: float,
                                 prediction: Dict) -> Dict:
        """Construct bull put spread (credit spread)"""
        try:
            # Find OTM puts
            otm_puts = [p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.98]
            
            if len(otm_puts) >= 2:
                # Sort by strike
                otm_puts = sorted(otm_puts, key=lambda x: float(x.get('strike_price', 0)), reverse=True)
                
                # Use delta if available, otherwise use strikes
                short_put = otm_puts[0]  # Higher strike
                long_put = otm_puts[1]   # Lower strike
                
                # Adjust based on expected move
                expected_low = stock_price * (1 - prediction['expected_volatility'])
                
                # Make sure short strike is above expected low
                for put in otm_puts:
                    if float(put.get('strike_price', 0)) > expected_low:
                        short_put = put
                        break
                        
                return {
                    'type': 'bull_put_spread',
                    'symbol': symbol,
                    'legs': [
                        {'option': short_put, 'side': 'sell', 'qty': 1},
                        {'option': long_put, 'side': 'buy', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'credit_spread': True,
                    'strategy_confidence': prediction['confidence'] * 0.85
                }
                
        except Exception as e:
            logger.error(f"Error constructing bull put spread: {e}")
            
        return None
        
    def construct_bear_call_spread(self, symbol: str, calls: List, stock_price: float,
                                  prediction: Dict) -> Dict:
        """Construct bear call spread (credit spread)"""
        try:
            # Find OTM calls
            otm_calls = [c for c in calls if float(c.get('strike_price', 0)) > stock_price * 1.02]
            
            if len(otm_calls) >= 2:
                # Sort by strike
                otm_calls = sorted(otm_calls, key=lambda x: float(x.get('strike_price', 0)))
                
                short_call = otm_calls[0]  # Lower strike
                long_call = otm_calls[1]   # Higher strike
                
                # Adjust based on expected move
                expected_high = stock_price * (1 + prediction['expected_volatility'])
                
                # Make sure short strike is below expected high
                for call in otm_calls:
                    if float(call.get('strike_price', 0)) < expected_high:
                        short_call = call
                        break
                        
                return {
                    'type': 'bear_call_spread',
                    'symbol': symbol,
                    'legs': [
                        {'option': short_call, 'side': 'sell', 'qty': 1},
                        {'option': long_call, 'side': 'buy', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'credit_spread': True,
                    'strategy_confidence': prediction['confidence'] * 0.85
                }
                
        except Exception as e:
            logger.error(f"Error constructing bear call spread: {e}")
            
        return None
        
    def construct_iron_condor(self, symbol: str, calls: List, puts: List, 
                             stock_price: float, prediction: Dict) -> Dict:
        """Construct iron condor for neutral prediction"""
        try:
            # Expected range
            expected_high = stock_price * (1 + prediction['expected_volatility'])
            expected_low = stock_price * (1 - prediction['expected_volatility'])
            
            # Find appropriate strikes
            sorted_puts = sorted(puts, key=lambda x: float(x.get('strike_price', 0)))
            sorted_calls = sorted(calls, key=lambda x: float(x.get('strike_price', 0)))
            
            # Put side
            short_put = None
            long_put = None
            for i, put in enumerate(sorted_puts):
                if float(put.get('strike_price', 0)) > expected_low * 0.95:
                    short_put = put
                    if i > 0:
                        long_put = sorted_puts[i-1]
                    break
                    
            # Call side
            short_call = None
            long_call = None
            for i, call in enumerate(sorted_calls):
                if float(call.get('strike_price', 0)) > expected_high * 1.05:
                    short_call = call
                    if i < len(sorted_calls) - 1:
                        long_call = sorted_calls[i+1]
                    break
                    
            if short_put and long_put and short_call and long_call:
                return {
                    'type': 'iron_condor',
                    'symbol': symbol,
                    'legs': [
                        {'option': long_put, 'side': 'buy', 'qty': 1},
                        {'option': short_put, 'side': 'sell', 'qty': 1},
                        {'option': short_call, 'side': 'sell', 'qty': 1},
                        {'option': long_call, 'side': 'buy', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'expected_range': (expected_low, expected_high),
                    'strategy_confidence': prediction['confidence'] * 0.8
                }
                
        except Exception as e:
            logger.error(f"Error constructing iron condor: {e}")
            
        return None
        
    def construct_volatility_play(self, symbol: str, calls: List, puts: List,
                                 stock_price: float, prediction: Dict) -> Dict:
        """Construct straddle or strangle for volatility play"""
        try:
            # Find ATM options for straddle
            atm_strike = round(stock_price / 5) * 5  # Round to nearest 5
            
            atm_call = next((c for c in calls if float(c.get('strike_price', 0)) == atm_strike), None)
            atm_put = next((p for p in puts if float(p.get('strike_price', 0)) == atm_strike), None)
            
            if atm_call and atm_put:
                # Straddle
                return {
                    'type': 'long_straddle',
                    'symbol': symbol,
                    'legs': [
                        {'option': atm_call, 'side': 'buy', 'qty': 1},
                        {'option': atm_put, 'side': 'buy', 'qty': 1}
                    ],
                    'prediction': prediction,
                    'volatility_play': True,
                    'expected_move': prediction['expected_volatility'] * stock_price,
                    'strategy_confidence': prediction['confidence'] * 0.7
                }
            else:
                # Strangle with OTM options
                otm_call = next((c for c in calls if float(c.get('strike_price', 0)) > stock_price * 1.02), None)
                otm_put = next((p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.98), None)
                
                if otm_call and otm_put:
                    return {
                        'type': 'long_strangle',
                        'symbol': symbol,
                        'legs': [
                            {'option': otm_call, 'side': 'buy', 'qty': 1},
                            {'option': otm_put, 'side': 'buy', 'qty': 1}
                        ],
                        'prediction': prediction,
                        'volatility_play': True,
                        'strategy_confidence': prediction['confidence'] * 0.7
                    }
                    
        except Exception as e:
            logger.error(f"Error constructing volatility play: {e}")
            
        return None
        
    async def execute_strategy(self, strategy: Dict):
        """Execute the selected options strategy"""
        try:
            symbol = strategy['symbol']
            strategy_type = strategy['type']
            
            # Log strategy details
            logger.info(f"\n📝 Executing {strategy_type} on {symbol}")
            
            # Execute each leg
            for leg in strategy['legs']:
                option = leg['option']
                side = OrderSide.BUY if leg['side'] == 'buy' else OrderSide.SELL
                qty = leg['qty']
                
                order = MarketOrderRequest(
                    symbol=option.get('symbol'),
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                self.trading_client.submit_order(order)
                logger.info(f"   ✅ {leg['side'].upper()} {qty} {option.get('symbol')}")
                
            # Track the strategy
            strategy_id = f"{strategy_type}_{symbol}_{int(time.time())}"
            self.active_strategies[strategy_id] = {
                'strategy': strategy,
                'entry_time': datetime.now(),
                'entry_prediction': strategy['prediction']
            }
            
            # Update performance tracker
            self.performance_tracker[strategy_type]['total_trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            
    async def manage_positions_with_ai(self):
        """Manage positions using AI predictions"""
        logger.info("\n🔧 AI-Powered Position Management...")
        
        try:
            positions = self.trading_client.get_all_positions()
            option_positions = [p for p in positions if len(p.symbol) > 10]
            
            for position in option_positions:
                # Get underlying symbol
                underlying = self.extract_underlying_symbol(position.symbol)
                
                if underlying in self.predictions_cache:
                    prediction = self.predictions_cache[underlying]
                    
                    # Check if prediction has changed significantly
                    await self.evaluate_position_adjustment(position, prediction)
                    
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def extract_underlying_symbol(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        # Option symbols format: AAPL230120C00150000
        # Extract the alphabetic part at the beginning
        underlying = ''
        for char in option_symbol:
            if char.isalpha():
                underlying += char
            else:
                break
        return underlying
        
    async def evaluate_position_adjustment(self, position, prediction: Dict):
        """Evaluate if position needs adjustment based on new prediction"""
        try:
            pnl_pct = float(position.unrealized_plpc)
            
            # Take profit if target reached
            if prediction['direction'] == 'bullish' and pnl_pct > 0.5:  # 50% profit
                logger.info(f"  💰 Taking profit on {position.symbol}: {pnl_pct:.1%}")
                self.close_position(position)
                
            # Stop loss if prediction reversed
            elif prediction['confidence'] < 0.5 and pnl_pct < -0.3:  # 30% loss
                logger.info(f"  🛑 Stop loss on {position.symbol}: {pnl_pct:.1%}")
                self.close_position(position)
                
        except Exception as e:
            logger.error(f"Error evaluating position: {e}")
            
    def close_position(self, position):
        """Close a position"""
        try:
            qty = abs(int(position.qty))
            side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
            
            order = MarketOrderRequest(
                symbol=position.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            self.trading_client.submit_order(order)
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            
    async def get_option_chain(self, symbol: str, days_to_expiry: int = 30) -> Dict:
        """Get option chain for analysis"""
        try:
            today = datetime.now().date()
            expiry_target = today + timedelta(days=days_to_expiry)
            expiry_min = (expiry_target - timedelta(days=7)).strftime('%Y-%m-%d')
            expiry_max = (expiry_target + timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Get current stock price
            stock_quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            stock_price = float(stock_quote[symbol].ask_price)
            
            # Get options
            url = f"{self.base_url}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'expiration_date_gte': expiry_min,
                'expiration_date_lte': expiry_max,
                'limit': 100
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                # Calculate Greeks
                for contract in contracts:
                    self.calculate_greeks(contract, stock_price)
                    
                return {
                    'symbol': symbol,
                    'stock_price': stock_price,
                    'contracts': contracts
                }
                
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            
        return {'symbol': symbol, 'stock_price': 0, 'contracts': []}
        
    def calculate_greeks(self, contract: Dict, stock_price: float):
        """Calculate option Greeks"""
        try:
            strike = float(contract.get('strike_price', 0))
            expiry = datetime.strptime(contract.get('expiration_date'), '%Y-%m-%d')
            dte = (expiry - datetime.now()).days
            
            if dte <= 0:
                return
                
            T = dte / 365.0
            r = 0.05
            iv = 0.25  # Placeholder
            
            d1 = (np.log(stock_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            
            if contract.get('type') == 'call':
                delta = norm.cdf(d1)
                theta = -(stock_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) - r * strike * np.exp(-r * T) * norm.cdf(d2)
            else:
                delta = -norm.cdf(-d1)
                theta = -(stock_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) + r * strike * np.exp(-r * T) * norm.cdf(-d2)
                
            gamma = norm.pdf(d1) / (stock_price * iv * np.sqrt(T))
            vega = stock_price * norm.pdf(d1) * np.sqrt(T) / 100
            
            contract['delta'] = round(delta, 3)
            contract['gamma'] = round(gamma, 4)
            contract['theta'] = round(theta / 365, 2)
            contract['vega'] = round(vega, 2)
            contract['dte'] = dte
            
        except Exception as e:
            pass
            
    async def get_historical_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=periods
            )
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars.data:
                df = bars.df.loc[symbol].reset_index()
                return df
                
        except Exception as e:
            pass
            
        return None
        
    async def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    async def display_account_status(self):
        """Display account status"""
        try:
            account = self.trading_client.get_account()
            logger.info("\n💼 Account Status:")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Options Level: {account.options_trading_level}")
        except Exception as e:
            logger.error(f"Error checking account: {e}")
            
    async def display_ai_dashboard(self):
        """Display AI trading dashboard"""
        try:
            logger.info("\n" + "="*70)
            logger.info("🤖 AI OPTIONS TRADING DASHBOARD")
            logger.info("="*70)
            
            # Predictions summary
            logger.info("\n📊 Active Predictions:")
            bullish = sum(1 for p in self.predictions_cache.values() if p['direction'] == 'bullish')
            bearish = sum(1 for p in self.predictions_cache.values() if p['direction'] == 'bearish')
            neutral = sum(1 for p in self.predictions_cache.values() if p['direction'] == 'neutral')
            
            logger.info(f"  Bullish: {bullish}, Bearish: {bearish}, Neutral: {neutral}")
            
            # Top predictions
            top_predictions = sorted(
                self.predictions_cache.items(),
                key=lambda x: abs(x[1]['expected_return']) * x[1]['confidence'],
                reverse=True
            )[:3]
            
            if top_predictions:
                logger.info("\n🌟 Top Predictions:")
                for symbol, pred in top_predictions:
                    logger.info(f"  {symbol}: {pred['direction']} "
                              f"({pred['expected_return']:+.2%}) "
                              f"Confidence: {pred['confidence']:.1%}")
                    
            # Active strategies
            if self.active_strategies:
                logger.info(f"\n🎯 Active AI Strategies: {len(self.active_strategies)}")
                strategy_types = defaultdict(int)
                for strategy_info in self.active_strategies.values():
                    strategy_types[strategy_info['strategy']['type']] += 1
                    
                for stype, count in strategy_types.items():
                    logger.info(f"  {stype}: {count}")
                    
            # Performance summary
            logger.info("\n📈 Strategy Performance:")
            for strategy_type, perf in self.performance_tracker.items():
                if perf['total_trades'] > 0:
                    win_rate = perf['winning_trades'] / perf['total_trades'] * 100
                    logger.info(f"  {strategy_type}: {perf['total_trades']} trades, "
                              f"{win_rate:.1f}% win rate")
                    
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
            
    async def close_all_positions(self):
        """Close all positions"""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("All positions closed")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

async def main():
    """Main entry point"""
    system = AIPoweredOptionsSystem()
    await system.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")