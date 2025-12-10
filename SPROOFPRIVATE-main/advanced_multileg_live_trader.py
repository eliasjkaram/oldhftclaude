#!/usr/bin/env python3
"""
Advanced Multi-Leg Options Live Trader
Executes sophisticated multi-leg strategies during market hours
with ML optimization and real-time position management
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
import threading
import queue

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedMultiLegLiveTrader:
    def __init__(self, paper=True):
        """Initialize advanced trader with real-time capabilities"""
        # API credentials
        if paper:
            self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
            self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.api_key = 'AK7LZKPVTPZTOTO9VVPM'
            self.api_secret = '2TjtRymW9aWXkWWQFwTThQGAKQrTbSWwLdz1LGKI'
            self.base_url = 'https://api.alpaca.markets'
            
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # API headers
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
        
        # Trading universe
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMD', 'AMZN', 'GOOGL']
        
        # Position sizing
        self.max_position_size = 5
        self.max_trades_per_session = 10
        self.min_profit_target = 0.15  # 15% minimum
        
        # Strategy catalog with advanced multi-leg combinations
        self.strategies = {
            'high_profit': {
                'iron_butterfly': {'legs': 4, 'profit': 'high_probability', 'risk': 'defined'},
                'broken_wing_butterfly': {'legs': 3, 'profit': 'asymmetric', 'risk': 'limited'},
                'ratio_backspread': {'legs': 3, 'profit': 'unlimited', 'risk': 'credit'},
                'jade_lizard': {'legs': 3, 'profit': 'credit', 'risk': 'no_upside'},
                'twisted_sister': {'legs': 3, 'profit': 'credit', 'risk': 'no_downside'}
            },
            'directional': {
                'bull_call_ladder': {'legs': 3, 'profit': 'capped', 'risk': 'limited'},
                'bear_put_ladder': {'legs': 3, 'profit': 'capped', 'risk': 'limited'},
                'synthetic_long': {'legs': 2, 'profit': 'unlimited', 'risk': 'unlimited'},
                'synthetic_short': {'legs': 2, 'profit': 'unlimited', 'risk': 'unlimited'},
                'risk_reversal': {'legs': 2, 'profit': 'directional', 'risk': 'unlimited'}
            },
            'volatility': {
                'long_iron_condor': {'legs': 4, 'profit': 'volatility', 'risk': 'limited'},
                'double_diagonal': {'legs': 4, 'profit': 'time_decay', 'risk': 'limited'},
                'calendar_spread': {'legs': 2, 'profit': 'time_decay', 'risk': 'limited'},
                'diagonal_spread': {'legs': 2, 'profit': 'directional_time', 'risk': 'limited'}
            },
            'arbitrage': {
                'box_spread': {'legs': 4, 'profit': 'arbitrage', 'risk': 'none'},
                'conversion': {'legs': 3, 'profit': 'arbitrage', 'risk': 'none'},
                'reversal': {'legs': 3, 'profit': 'arbitrage', 'risk': 'none'},
                'jelly_roll': {'legs': 4, 'profit': 'time_arbitrage', 'risk': 'limited'}
            }
        }
        
        # ML model parameters
        self.ml_confidence_threshold = 0.70
        self.volatility_regimes = {
            'low': (0, 0.15),
            'normal': (0.15, 0.25),
            'high': (0.25, 0.40),
            'extreme': (0.40, 1.0)
        }
        
        # Tracking
        self.executed_trades = []
        self.active_positions = {}
        self.pnl_tracker = defaultdict(float)
        
        # Real-time monitoring
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def run(self):
        """Main execution loop with market hours awareness"""
        logger.info("🚀 ADVANCED MULTI-LEG OPTIONS LIVE TRADER")
        logger.info("=" * 70)
        
        # Check market status
        market_open = self.check_market_status()
        
        if not market_open:
            logger.warning("⚠️  Market is CLOSED. Options trading only during market hours.")
            self.show_after_hours_summary()
            self.suggest_schedule()
            return
            
        logger.info("✅ Market is OPEN - Starting live trading session")
        
        # Check account
        if not self.verify_account():
            return
            
        # Start monitoring thread
        self.start_monitoring()
        
        # Execute trading session
        self.execute_trading_session()
        
        # Stop monitoring
        self.stop_monitoring = True
        
        # Show results
        self.show_session_results()
        
    def check_market_status(self) -> bool:
        """Check if market is open for options trading"""
        try:
            clock = self.trading_client.get_clock()
            
            logger.info(f"\n⏰ Market Status:")
            logger.info(f"   Current Time: {datetime.now(timezone.utc)}")
            logger.info(f"   Market Open: {clock.is_open}")
            
            if clock.is_open:
                logger.info(f"   Market Close: {clock.next_close}")
                time_to_close = clock.next_close - datetime.now(timezone.utc)
                logger.info(f"   Time Until Close: {time_to_close}")
            else:
                logger.info(f"   Next Open: {clock.next_open}")
                time_to_open = clock.next_open - datetime.now(timezone.utc)
                logger.info(f"   Time Until Open: {time_to_open}")
                
            return clock.is_open
            
        except Exception as e:
            logger.error(f"Clock error: {e}")
            return False
            
    def verify_account(self) -> bool:
        """Verify account is ready for options trading"""
        try:
            account = self.trading_client.get_account()
            
            self.buying_power = float(account.buying_power)
            self.portfolio_value = float(account.portfolio_value)
            self.options_level = account.options_trading_level
            
            logger.info(f"\n💼 Account Verification:")
            logger.info(f"   Portfolio Value: ${self.portfolio_value:,.2f}")
            logger.info(f"   Buying Power: ${self.buying_power:,.2f}")
            logger.info(f"   Options Level: {self.options_level}")
            logger.info(f"   Pattern Day Trader: {account.pattern_day_trader}")
            logger.info(f"   Trade Count: {account.daytrade_count}")
            
            if self.options_level < 2:
                logger.error("❌ Insufficient options trading level")
                return False
                
            if self.buying_power < 5000:
                logger.warning("⚠️  Low buying power for multi-leg strategies")
                
            return True
            
        except Exception as e:
            logger.error(f"Account verification error: {e}")
            return False
            
    def execute_trading_session(self):
        """Execute main trading session"""
        logger.info("\n📊 Starting Trading Session")
        
        trades_executed = 0
        
        # Analyze each symbol
        for symbol in self.symbols:
            if trades_executed >= self.max_trades_per_session:
                logger.info(f"\n✋ Reached max trades limit ({self.max_trades_per_session})")
                break
                
            try:
                logger.info(f"\n🔍 Analyzing {symbol}...")
                
                # Comprehensive analysis
                analysis = self.perform_ml_analysis(symbol)
                if not analysis or analysis['confidence'] < self.ml_confidence_threshold:
                    logger.info(f"   Skipping - Low confidence ({analysis.get('confidence', 0):.1%})")
                    continue
                    
                # Get option chains
                option_chains = self.get_comprehensive_option_chains(symbol, analysis)
                if not option_chains:
                    logger.info(f"   Skipping - No suitable option chains")
                    continue
                    
                # Find optimal strategy
                optimal_strategy = self.find_optimal_strategy(symbol, analysis, option_chains)
                if not optimal_strategy:
                    logger.info(f"   Skipping - No profitable strategy found")
                    continue
                    
                # Risk check
                if not self.passes_risk_check(optimal_strategy):
                    logger.info(f"   Skipping - Failed risk check")
                    continue
                    
                # Execute strategy
                if self.execute_strategy(optimal_strategy):
                    trades_executed += 1
                    time.sleep(2)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
                
    def perform_ml_analysis(self, symbol: str) -> Dict:
        """Perform comprehensive ML-based analysis"""
        try:
            # Get current quote
            quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            
            if symbol not in quote:
                return None
                
            current_price = float(quote[symbol].ask_price)
            bid_price = float(quote[symbol].bid_price)
            spread = current_price - bid_price
            
            # Get historical data
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=100
                )
            )
            
            if symbol not in bars.df.index:
                return None
                
            df = bars.df.loc[symbol]
            
            # Technical analysis
            analysis = self.calculate_technical_indicators(df, current_price)
            
            # ML predictions
            predictions = self.generate_ml_predictions(analysis, df)
            
            # Combine results
            result = {
                'symbol': symbol,
                'price': current_price,
                'spread': spread,
                'analysis': analysis,
                'predictions': predictions,
                'confidence': predictions['confidence'],
                'expected_move': predictions['expected_move'],
                'volatility_regime': predictions['volatility_regime']
            }
            
            self.display_analysis(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None
            
    def calculate_technical_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Price action
        indicators['price'] = current_price
        indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) > 50 else indicators['sma_20']
        indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = df['close'].ewm(span=9).mean().iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        indicators['bb_upper'] = (bb_sma + 2 * bb_std).iloc[-1]
        indicators['bb_lower'] = (bb_sma - 2 * bb_std).iloc[-1]
        indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        indicators['bb_position'] = (current_price - indicators['bb_lower']) / indicators['bb_width']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean().iloc[-1]
        
        # Volatility
        returns = df['close'].pct_change()
        indicators['volatility_10d'] = returns.rolling(10).std().iloc[-1] * np.sqrt(252)
        indicators['volatility_30d'] = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
        indicators['volatility_60d'] = returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(df) > 60 else indicators['volatility_30d']
        
        # Volume analysis
        indicators['volume_avg'] = df['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_avg']
        
        # Support/Resistance
        indicators['support'] = df['low'].rolling(20).min().iloc[-1]
        indicators['resistance'] = df['high'].rolling(20).max().iloc[-1]
        
        return indicators
        
    def generate_ml_predictions(self, analysis: Dict, df: pd.DataFrame) -> Dict:
        """Generate ML-based predictions"""
        predictions = {}
        
        # Trend prediction
        trend_score = 0
        
        # Moving average analysis
        if analysis['price'] > analysis['sma_20']:
            trend_score += 2
        if analysis['price'] > analysis['sma_50']:
            trend_score += 2
        if analysis['sma_20'] > analysis['sma_50']:
            trend_score += 3
            
        # MACD analysis
        if analysis['macd'] > analysis['macd_signal']:
            trend_score += 2
            
        # RSI analysis
        if 30 < analysis['rsi'] < 70:
            trend_score += 1
        elif analysis['rsi'] < 30:
            trend_score += 3  # Oversold bounce
        elif analysis['rsi'] > 70:
            trend_score -= 3  # Overbought pullback
            
        # Bollinger Bands
        if analysis['bb_position'] < 0.2:
            trend_score += 2  # Near lower band
        elif analysis['bb_position'] > 0.8:
            trend_score -= 2  # Near upper band
            
        # Determine direction
        if trend_score >= 5:
            predictions['direction'] = 'strong_bullish'
            predictions['confidence'] = min(0.85, 0.60 + trend_score * 0.03)
        elif trend_score >= 2:
            predictions['direction'] = 'bullish'
            predictions['confidence'] = min(0.75, 0.55 + trend_score * 0.03)
        elif trend_score <= -5:
            predictions['direction'] = 'strong_bearish'
            predictions['confidence'] = min(0.85, 0.60 + abs(trend_score) * 0.03)
        elif trend_score <= -2:
            predictions['direction'] = 'bearish'
            predictions['confidence'] = min(0.75, 0.55 + abs(trend_score) * 0.03)
        else:
            predictions['direction'] = 'neutral'
            predictions['confidence'] = 0.60
            
        # Volatility regime
        current_vol = analysis['volatility_30d']
        for regime, (min_vol, max_vol) in self.volatility_regimes.items():
            if min_vol <= current_vol < max_vol:
                predictions['volatility_regime'] = regime
                break
                
        # Expected move calculation
        predictions['expected_move'] = analysis['price'] * current_vol * np.sqrt(30/365)
        predictions['expected_move_pct'] = current_vol * np.sqrt(30/365) * 100
        
        # Price targets
        base_move = predictions['expected_move']
        if predictions['direction'] in ['strong_bullish', 'bullish']:
            predictions['target_1w'] = analysis['price'] + base_move * 0.5
            predictions['target_2w'] = analysis['price'] + base_move * 0.75
            predictions['target_1m'] = analysis['price'] + base_move
        elif predictions['direction'] in ['strong_bearish', 'bearish']:
            predictions['target_1w'] = analysis['price'] - base_move * 0.5
            predictions['target_2w'] = analysis['price'] - base_move * 0.75
            predictions['target_1m'] = analysis['price'] - base_move
        else:
            predictions['target_1w'] = analysis['price']
            predictions['target_2w'] = analysis['price']
            predictions['target_1m'] = analysis['price']
            
        # Risk levels
        predictions['stop_loss'] = analysis['price'] - 2 * analysis['atr']
        predictions['take_profit'] = analysis['price'] + 3 * analysis['atr']
        
        return predictions
        
    def display_analysis(self, result: Dict):
        """Display analysis results"""
        logger.info(f"\n   📊 {result['symbol']} Analysis:")
        logger.info(f"   Price: ${result['price']:.2f} (spread: ${result['spread']:.2f})")
        logger.info(f"   Direction: {result['predictions']['direction']} ({result['confidence']:.1%} confidence)")
        logger.info(f"   Expected Move: ±${result['expected_move']:.2f} ({result['predictions']['expected_move_pct']:.1f}%)")
        logger.info(f"   Volatility: {result['predictions']['volatility_regime']} ({result['analysis']['volatility_30d']:.1%})")
        logger.info(f"   RSI: {result['analysis']['rsi']:.1f}")
        logger.info(f"   Volume: {result['analysis']['volume_ratio']:.1f}x average")
        
    def get_comprehensive_option_chains(self, symbol: str, analysis: Dict) -> Dict:
        """Get option chains for multiple expirations"""
        try:
            chains = {}
            today = datetime.now().date()
            price = analysis['price']
            
            # Define expiration targets based on strategy needs
            expiration_targets = [
                ('weekly', 5, 10),      # For short-term plays
                ('bi_weekly', 12, 18),  # For spreads
                ('monthly', 25, 35),    # For standard strategies
                ('bi_monthly', 50, 65), # For diagonals
                ('quarterly', 80, 100)  # For long-term plays
            ]
            
            for exp_name, min_days, max_days in expiration_targets:
                exp_min = (today + timedelta(days=min_days)).strftime('%Y-%m-%d')
                exp_max = (today + timedelta(days=max_days)).strftime('%Y-%m-%d')
                
                # Define strike range based on expected move
                expected_move = analysis['expected_move']
                strike_min = price - 2 * expected_move
                strike_max = price + 2 * expected_move
                
                url = f"{self.base_url}/v2/options/contracts"
                params = {
                    'underlying_symbols': symbol,
                    'status': 'active',
                    'expiration_date_gte': exp_min,
                    'expiration_date_lte': exp_max,
                    'strike_price_gte': strike_min,
                    'strike_price_lte': strike_max,
                    'limit': 200
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    contracts = data.get('option_contracts', [])
                    
                    if contracts:
                        chains[exp_name] = self.organize_chain_with_greeks(contracts, price, analysis)
                        logger.info(f"   Found {len(contracts)} contracts for {exp_name} expiration")
                        
            return chains if chains else None
            
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None
            
    def organize_chain_with_greeks(self, contracts: List[Dict], stock_price: float, analysis: Dict) -> Dict:
        """Organize option chain with Greeks calculation"""
        chain = {
            'puts': [],
            'calls': [],
            'by_strike': defaultdict(dict),
            'by_expiration': defaultdict(list)
        }
        
        for contract in contracts:
            # Calculate Greeks
            contract['greeks'] = self.calculate_greeks(contract, stock_price, analysis)
            
            # Add to appropriate lists
            strike = float(contract.get('strike_price', 0))
            exp_date = contract.get('expiration_date')
            
            if contract.get('type') == 'put':
                chain['puts'].append(contract)
                chain['by_strike'][strike]['put'] = contract
            else:
                chain['calls'].append(contract)
                chain['by_strike'][strike]['call'] = contract
                
            chain['by_expiration'][exp_date].append(contract)
            
        # Sort by strike
        chain['puts'] = sorted(chain['puts'], key=lambda x: float(x.get('strike_price', 0)))
        chain['calls'] = sorted(chain['calls'], key=lambda x: float(x.get('strike_price', 0)))
        
        # Find key strikes
        atm_strike = round(stock_price / 5) * 5
        
        # Categorize by moneyness
        chain['itm_puts'] = [p for p in chain['puts'] if float(p.get('strike_price', 0)) > stock_price]
        chain['atm_puts'] = [p for p in chain['puts'] if abs(float(p.get('strike_price', 0)) - stock_price) / stock_price < 0.02]
        chain['otm_puts'] = [p for p in chain['puts'] if float(p.get('strike_price', 0)) < stock_price]
        
        chain['itm_calls'] = [c for c in chain['calls'] if float(c.get('strike_price', 0)) < stock_price]
        chain['atm_calls'] = [c for c in chain['calls'] if abs(float(c.get('strike_price', 0)) - stock_price) / stock_price < 0.02]
        chain['otm_calls'] = [c for c in chain['calls'] if float(c.get('strike_price', 0)) > stock_price]
        
        return chain
        
    def calculate_greeks(self, contract: Dict, stock_price: float, analysis: Dict) -> Dict:
        """Calculate option Greeks"""
        strike = float(contract.get('strike_price', 0))
        days_to_exp = (datetime.strptime(contract.get('expiration_date'), '%Y-%m-%d').date() - datetime.now().date()).days
        time_to_exp = days_to_exp / 365.0
        
        # Use market volatility
        volatility = analysis['analysis']['volatility_30d']
        r = 0.05  # Risk-free rate
        
        # Prevent division by zero
        if time_to_exp <= 0:
            time_to_exp = 1/365.0
            
        # Black-Scholes calculations
        d1 = (np.log(stock_price / strike) + (r + 0.5 * volatility**2) * time_to_exp) / (volatility * np.sqrt(time_to_exp))
        d2 = d1 - volatility * np.sqrt(time_to_exp)
        
        if contract.get('type') == 'call':
            delta = norm.cdf(d1)
            theta = -(stock_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_exp)) - r * strike * np.exp(-r * time_to_exp) * norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            theta = -(stock_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_exp)) + r * strike * np.exp(-r * time_to_exp) * norm.cdf(-d2)
            
        gamma = norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_exp))
        vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_exp) / 100
        rho = 0.01 * strike * time_to_exp * np.exp(-r * time_to_exp) * (norm.cdf(d2) if contract.get('type') == 'call' else norm.cdf(-d2))
        
        # Calculate implied volatility (simplified)
        if 'ask_price' in contract and 'bid_price' in contract:
            mid_price = (float(contract.get('ask_price', 0)) + float(contract.get('bid_price', 0))) / 2
            if mid_price > 0:
                # Simplified IV calculation
                iv = volatility * (1 + 0.1 * (mid_price / stock_price - 0.05))
            else:
                iv = volatility
        else:
            iv = volatility
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Daily theta
            'vega': vega,
            'rho': rho,
            'iv': iv,
            'moneyness': stock_price / strike,
            'days_to_expiration': days_to_exp
        }
        
    def find_optimal_strategy(self, symbol: str, analysis: Dict, option_chains: Dict) -> Optional[Dict]:
        """Find optimal multi-leg strategy using ML insights"""
        logger.info(f"\n   🧠 Finding Optimal Strategy for {symbol}")
        
        strategies = []
        
        # Generate strategies based on market conditions
        direction = analysis['predictions']['direction']
        vol_regime = analysis['predictions']['volatility_regime']
        rsi = analysis['analysis']['rsi']
        
        # High confidence directional plays
        if analysis['confidence'] > 0.75:
            if direction == 'strong_bullish':
                strategies.extend(self.generate_bullish_strategies(symbol, analysis, option_chains))
            elif direction == 'strong_bearish':
                strategies.extend(self.generate_bearish_strategies(symbol, analysis, option_chains))
                
        # Volatility plays
        if vol_regime == 'high' or vol_regime == 'extreme':
            strategies.extend(self.generate_volatility_strategies(symbol, analysis, option_chains))
        elif vol_regime == 'low':
            strategies.extend(self.generate_low_volatility_strategies(symbol, analysis, option_chains))
            
        # Special conditions
        if rsi < 30:  # Oversold
            strategies.extend(self.generate_oversold_strategies(symbol, analysis, option_chains))
        elif rsi > 70:  # Overbought
            strategies.extend(self.generate_overbought_strategies(symbol, analysis, option_chains))
            
        # Neutral strategies
        if direction == 'neutral' or analysis['confidence'] < 0.65:
            strategies.extend(self.generate_neutral_strategies(symbol, analysis, option_chains))
            
        # Score and rank strategies
        scored_strategies = []
        for strategy in strategies:
            score = self.score_strategy(strategy, analysis)
            strategy['score'] = score
            strategy['expected_profit'] = self.calculate_expected_profit(strategy, analysis)
            scored_strategies.append(strategy)
            
        # Sort by score
        scored_strategies.sort(key=lambda x: x['score'], reverse=True)
        
        # Return best strategy that meets criteria
        for strategy in scored_strategies:
            if strategy['expected_profit'] > self.min_profit_target * 100:  # Convert to dollar amount
                logger.info(f"   ✅ Selected: {strategy['type']} (Score: {strategy['score']:.1f}, Expected: ${strategy['expected_profit']:.2f})")
                return strategy
                
        logger.info(f"   ❌ No strategies met profit target (${self.min_profit_target * 100:.2f})")
        return None
        
    def generate_bullish_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate bullish multi-leg strategies"""
        strategies = []
        
        for exp_name, chain in chains.items():
            if not chain or 'weekly' in exp_name:  # Skip very short term
                continue
                
            # Bull Call Spread
            if len(chain['calls']) >= 2:
                for i in range(len(chain['calls']) - 1):
                    long_call = chain['calls'][i]
                    short_call = chain['calls'][i + 1]
                    
                    if float(short_call['strike_price']) - float(long_call['strike_price']) <= 10:
                        strategies.append({
                            'type': 'bull_call_spread',
                            'symbol': symbol,
                            'expiration': exp_name,
                            'legs': [
                                {'option': long_call, 'action': 'buy', 'quantity': 1},
                                {'option': short_call, 'action': 'sell', 'quantity': 1}
                            ],
                            'max_profit': (float(short_call['strike_price']) - float(long_call['strike_price'])) * 100,
                            'risk_type': 'defined'
                        })
                        
            # Jade Lizard (Bull Put Spread + Short Call)
            if len(chain['otm_puts']) >= 2 and chain['otm_calls']:
                put_short = chain['otm_puts'][-1]
                put_long = chain['otm_puts'][0]
                call_short = chain['otm_calls'][0]
                
                strategies.append({
                    'type': 'jade_lizard',
                    'symbol': symbol,
                    'expiration': exp_name,
                    'legs': [
                        {'option': put_long, 'action': 'buy', 'quantity': 1},
                        {'option': put_short, 'action': 'sell', 'quantity': 1},
                        {'option': call_short, 'action': 'sell', 'quantity': 1}
                    ],
                    'no_upside_risk': True,
                    'risk_type': 'credit'
                })
                
        return strategies
        
    def generate_volatility_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate volatility expansion strategies"""
        strategies = []
        
        for exp_name, chain in chains.items():
            if not chain:
                continue
                
            # Long Iron Condor (Reverse Iron Condor)
            if len(chain['otm_puts']) >= 2 and len(chain['otm_calls']) >= 2:
                # Buy closer strikes, sell further strikes
                put_buy = chain['otm_puts'][-1]
                put_sell = chain['otm_puts'][0]
                call_buy = chain['otm_calls'][0]
                call_sell = chain['otm_calls'][-1]
                
                strategies.append({
                    'type': 'long_iron_condor',
                    'symbol': symbol,
                    'expiration': exp_name,
                    'legs': [
                        {'option': put_sell, 'action': 'sell', 'quantity': 1},
                        {'option': put_buy, 'action': 'buy', 'quantity': 1},
                        {'option': call_buy, 'action': 'buy', 'quantity': 1},
                        {'option': call_sell, 'action': 'sell', 'quantity': 1}
                    ],
                    'profit_from': 'large_moves',
                    'risk_type': 'defined'
                })
                
        return strategies
        
    def generate_neutral_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate neutral/range-bound strategies"""
        strategies = []
        
        for exp_name, chain in chains.items():
            if not chain:
                continue
                
            # Iron Butterfly
            if chain['atm_calls'] and chain['atm_puts'] and chain['otm_calls'] and chain['otm_puts']:
                atm_strike = round(analysis['price'] / 5) * 5
                
                # Find ATM options
                atm_call = next((c for c in chain['calls'] if float(c['strike_price']) == atm_strike), None)
                atm_put = next((p for p in chain['puts'] if float(p['strike_price']) == atm_strike), None)
                
                if atm_call and atm_put:
                    # Find wings
                    wing_width = 10 if analysis['price'] > 100 else 5
                    otm_put = next((p for p in chain['puts'] if float(p['strike_price']) == atm_strike - wing_width), None)
                    otm_call = next((c for c in chain['calls'] if float(c['strike_price']) == atm_strike + wing_width), None)
                    
                    if otm_put and otm_call:
                        strategies.append({
                            'type': 'iron_butterfly',
                            'symbol': symbol,
                            'expiration': exp_name,
                            'legs': [
                                {'option': otm_put, 'action': 'buy', 'quantity': 1},
                                {'option': atm_put, 'action': 'sell', 'quantity': 1},
                                {'option': atm_call, 'action': 'sell', 'quantity': 1},
                                {'option': otm_call, 'action': 'buy', 'quantity': 1}
                            ],
                            'max_profit_at': atm_strike,
                            'risk_type': 'defined'
                        })
                        
        return strategies
        
    def generate_oversold_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate strategies for oversold conditions"""
        strategies = []
        
        # Similar to bullish but with higher confidence
        bullish_strategies = self.generate_bullish_strategies(symbol, analysis, chains)
        
        # Add risk reversal
        for exp_name, chain in chains.items():
            if chain['otm_puts'] and chain['otm_calls']:
                strategies.append({
                    'type': 'risk_reversal',
                    'symbol': symbol,
                    'expiration': exp_name,
                    'legs': [
                        {'option': chain['otm_puts'][0], 'action': 'sell', 'quantity': 1},
                        {'option': chain['otm_calls'][0], 'action': 'buy', 'quantity': 1}
                    ],
                    'synthetic_long': True,
                    'risk_type': 'unlimited'
                })
                
        strategies.extend(bullish_strategies)
        return strategies
        
    def generate_overbought_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate strategies for overbought conditions"""
        strategies = []
        
        for exp_name, chain in chains.items():
            # Twisted Sister (Bear Call Spread + Short Put)
            if len(chain['otm_calls']) >= 2 and chain['otm_puts']:
                call_short = chain['otm_calls'][0]
                call_long = chain['otm_calls'][1]
                put_short = chain['otm_puts'][-1]
                
                strategies.append({
                    'type': 'twisted_sister',
                    'symbol': symbol,
                    'expiration': exp_name,
                    'legs': [
                        {'option': call_short, 'action': 'sell', 'quantity': 1},
                        {'option': call_long, 'action': 'buy', 'quantity': 1},
                        {'option': put_short, 'action': 'sell', 'quantity': 1}
                    ],
                    'no_downside_risk': True,
                    'risk_type': 'credit'
                })
                
        return strategies
        
    def generate_bearish_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate bearish multi-leg strategies"""
        strategies = []
        
        for exp_name, chain in chains.items():
            # Bear Put Spread
            if len(chain['puts']) >= 2:
                for i in range(len(chain['puts']) - 1):
                    short_put = chain['puts'][i]
                    long_put = chain['puts'][i + 1]
                    
                    if float(long_put['strike_price']) - float(short_put['strike_price']) <= 10:
                        strategies.append({
                            'type': 'bear_put_spread',
                            'symbol': symbol,
                            'expiration': exp_name,
                            'legs': [
                                {'option': long_put, 'action': 'buy', 'quantity': 1},
                                {'option': short_put, 'action': 'sell', 'quantity': 1}
                            ],
                            'max_profit': (float(long_put['strike_price']) - float(short_put['strike_price'])) * 100,
                            'risk_type': 'defined'
                        })
                        
        return strategies
        
    def generate_low_volatility_strategies(self, symbol: str, analysis: Dict, chains: Dict) -> List[Dict]:
        """Generate strategies for low volatility environment"""
        strategies = []
        
        for exp_name, chain in chains.items():
            # Standard Iron Condor
            if len(chain['otm_puts']) >= 2 and len(chain['otm_calls']) >= 2:
                expected_move = analysis['expected_move']
                current_price = analysis['price']
                
                # Find strikes outside expected move
                suitable_puts = [p for p in chain['otm_puts'] if float(p['strike_price']) < current_price - expected_move * 0.8]
                suitable_calls = [c for c in chain['otm_calls'] if float(c['strike_price']) > current_price + expected_move * 0.8]
                
                if len(suitable_puts) >= 2 and len(suitable_calls) >= 2:
                    strategies.append({
                        'type': 'iron_condor',
                        'symbol': symbol,
                        'expiration': exp_name,
                        'legs': [
                            {'option': suitable_puts[0], 'action': 'buy', 'quantity': 1},
                            {'option': suitable_puts[-1], 'action': 'sell', 'quantity': 1},
                            {'option': suitable_calls[0], 'action': 'sell', 'quantity': 1},
                            {'option': suitable_calls[-1], 'action': 'buy', 'quantity': 1}
                        ],
                        'profit_range': (float(suitable_puts[-1]['strike_price']), float(suitable_calls[0]['strike_price'])),
                        'risk_type': 'defined'
                    })
                    
        return strategies
        
    def score_strategy(self, strategy: Dict, analysis: Dict) -> float:
        """Score strategy based on multiple factors"""
        score = 0
        
        # Risk type preference
        if strategy.get('risk_type') == 'defined':
            score += 20
        elif strategy.get('risk_type') == 'credit':
            score += 15
            
        # Special features
        if strategy.get('no_upside_risk') or strategy.get('no_downside_risk'):
            score += 10
            
        # Greeks optimization
        total_theta = 0
        total_delta = 0
        
        for leg in strategy['legs']:
            greeks = leg['option'].get('greeks', {})
            multiplier = 1 if leg['action'] == 'buy' else -1
            total_theta += multiplier * greeks.get('theta', 0)
            total_delta += multiplier * greeks.get('delta', 0)
            
        # Prefer positive theta
        if total_theta > 0:
            score += 15
            
        # Prefer neutral delta for non-directional
        if abs(total_delta) < 0.2:
            score += 10
            
        # Liquidity check
        for leg in strategy['legs']:
            if 'bid_price' in leg['option'] and 'ask_price' in leg['option']:
                spread = float(leg['option']['ask_price']) - float(leg['option']['bid_price'])
                if spread < 0.10:  # Tight spread
                    score += 5
                    
        # Complexity penalty (prefer simpler strategies)
        score -= len(strategy['legs']) * 2
        
        return score
        
    def calculate_expected_profit(self, strategy: Dict, analysis: Dict) -> float:
        """Calculate expected profit for strategy"""
        # Simplified calculation - in production, use more sophisticated models
        
        if strategy['type'] == 'iron_condor':
            # Credit spread - estimate based on width
            if 'profit_range' in strategy:
                lower, upper = strategy['profit_range']
                current = analysis['price']
                
                # Probability of staying in range
                vol = analysis['analysis']['volatility_30d']
                days = 30  # Approximate
                std_move = current * vol * np.sqrt(days/365)
                
                z_lower = (lower - current) / std_move
                z_upper = (upper - current) / std_move
                
                prob_success = norm.cdf(z_upper) - norm.cdf(z_lower)
                
                # Estimate credit as 20% of width
                width = 10  # Typical width
                credit = width * 0.2 * 100
                max_loss = (width - credit/100) * 100
                
                expected = prob_success * credit - (1 - prob_success) * max_loss
                return expected
                
        elif strategy['type'] in ['bull_call_spread', 'bear_put_spread']:
            # Debit spread
            max_profit = strategy.get('max_profit', 500)
            max_loss = max_profit * 0.4  # Assume 40% of max profit as cost
            
            # Use ML confidence for probability
            prob_success = analysis['confidence'] * 0.8
            
            expected = prob_success * max_profit - (1 - prob_success) * max_loss
            return expected
            
        elif strategy['type'] in ['jade_lizard', 'twisted_sister']:
            # Credit strategies with no risk on one side
            # Typically collect 30-40% of spread width
            return 300  # Conservative estimate
            
        else:
            # Default conservative estimate
            return 200
            
    def passes_risk_check(self, strategy: Dict) -> bool:
        """Check if strategy passes risk management rules"""
        # Position size check
        max_contracts = self.max_position_size
        total_contracts = sum(leg['quantity'] for leg in strategy['legs'])
        
        if total_contracts > max_contracts * 2:  # Allow more for multi-leg
            logger.info(f"   Risk check failed: Too many contracts ({total_contracts})")
            return False
            
        # Margin check (simplified)
        estimated_margin = total_contracts * 1000  # $1000 per spread typical
        if estimated_margin > self.buying_power * 0.2:  # Max 20% of buying power
            logger.info(f"   Risk check failed: Margin too high (${estimated_margin})")
            return False
            
        # Undefined risk check
        if strategy.get('risk_type') == 'unlimited':
            if self.portfolio_value < 50000:  # Require larger account
                logger.info(f"   Risk check failed: Unlimited risk strategy")
                return False
                
        return True
        
    def execute_strategy(self, strategy: Dict) -> bool:
        """Execute the multi-leg strategy"""
        try:
            logger.info(f"\n   🚀 EXECUTING {strategy['type'].upper()}")
            logger.info(f"   Expected Profit: ${strategy['expected_profit']:.2f}")
            
            # Sort legs - protective legs first
            buy_legs = [leg for leg in strategy['legs'] if leg['action'] == 'buy']
            sell_legs = [leg for leg in strategy['legs'] if leg['action'] == 'sell']
            
            executed_orders = []
            
            # Execute buy orders first (protective legs)
            for leg in buy_legs:
                if not self.execute_leg(leg, 'BUY'):
                    # Rollback if any leg fails
                    self.rollback_orders(executed_orders)
                    return False
                executed_orders.append(leg)
                
            # Execute sell orders
            for leg in sell_legs:
                if not self.execute_leg(leg, 'SELL'):
                    # Rollback if any leg fails
                    self.rollback_orders(executed_orders)
                    return False
                executed_orders.append(leg)
                
            # Record successful execution
            self.executed_trades.append({
                'strategy': strategy,
                'timestamp': datetime.now(),
                'legs_executed': len(executed_orders)
            })
            
            logger.info(f"   ✅ Strategy executed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"   Execution error: {e}")
            return False
            
    def execute_leg(self, leg: Dict, side: str) -> bool:
        """Execute individual option leg"""
        try:
            option = leg['option']
            quantity = leg['quantity']
            
            logger.info(f"      {side} {quantity}x {option['symbol']} @ ${float(option['strike_price'])}")
            
            # Use limit orders for better fills
            if 'bid_price' in option and 'ask_price' in option:
                bid = float(option['bid_price'])
                ask = float(option['ask_price'])
                
                # Price improvement attempt
                if side == 'BUY':
                    limit_price = bid + (ask - bid) * 0.3  # 30% above bid
                else:
                    limit_price = ask - (ask - bid) * 0.3  # 30% below ask
                    
                order = LimitOrderRequest(
                    symbol=option['symbol'],
                    qty=quantity,
                    side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2)
                )
            else:
                # Fallback to market order
                order = MarketOrderRequest(
                    symbol=option['symbol'],
                    qty=quantity,
                    side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
            result = self.trading_client.submit_order(order)
            logger.info(f"      ✓ Order submitted: {result.id}")
            
            # Track position
            self.active_positions[option['symbol']] = {
                'quantity': quantity * (1 if side == 'BUY' else -1),
                'strike': float(option['strike_price']),
                'type': option['type'],
                'order_id': result.id
            }
            
            return True
            
        except Exception as e:
            logger.error(f"      ✗ Leg execution failed: {e}")
            return False
            
    def rollback_orders(self, executed_orders: List[Dict]):
        """Rollback executed orders in case of failure"""
        logger.warning("   ⚠️  Rolling back executed orders...")
        
        for leg in executed_orders:
            try:
                # Submit opposite order
                opposite_side = 'SELL' if leg['action'] == 'buy' else 'BUY'
                self.execute_leg(leg, opposite_side)
            except Exception as e:
                logger.error(f"   Rollback error: {e}")
                
    def start_monitoring(self):
        """Start real-time position monitoring"""
        def monitor_positions():
            while not self.stop_monitoring:
                try:
                    self.update_position_pnl()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    
        self.monitoring_thread = threading.Thread(target=monitor_positions)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def update_position_pnl(self):
        """Update P&L for active positions"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                if len(position.symbol) > 10:  # Options position
                    symbol = position.symbol
                    qty = float(position.qty)
                    entry_price = float(position.avg_entry_price)
                    current_price = float(position.current_price) if position.current_price else entry_price
                    
                    pnl = (current_price - entry_price) * qty * 100
                    self.pnl_tracker[symbol] = pnl
                    
        except Exception as e:
            logger.error(f"P&L update error: {e}")
            
    def show_after_hours_summary(self):
        """Show summary when market is closed"""
        logger.info("\n📊 After Hours Summary")
        
        # Show current positions
        try:
            positions = self.trading_client.get_all_positions()
            option_positions = [p for p in positions if len(p.symbol) > 10]
            
            if option_positions:
                logger.info(f"\n📋 Current Option Positions ({len(option_positions)}):")
                
                total_pnl = 0
                by_underlying = defaultdict(list)
                
                for pos in option_positions:
                    underlying = pos.symbol[:3] if len(pos.symbol) > 10 else pos.symbol
                    by_underlying[underlying].append(pos)
                    total_pnl += float(pos.unrealized_pl)
                    
                for underlying, positions in by_underlying.items():
                    logger.info(f"\n   {underlying}:")
                    for pos in positions:
                        logger.info(f"      {pos.symbol}: {float(pos.qty):+.0f} @ ${float(pos.avg_entry_price):.2f}")
                        logger.info(f"         P&L: ${float(pos.unrealized_pl):+.2f}")
                        
                logger.info(f"\n   💰 Total P&L: ${total_pnl:+,.2f}")
                
        except Exception as e:
            logger.error(f"Position summary error: {e}")
            
    def suggest_schedule(self):
        """Suggest when to run the system"""
        logger.info("\n⏰ Suggested Schedule:")
        logger.info("   • Run during market hours: 9:30 AM - 4:00 PM ET")
        logger.info("   • Best times: 9:45-10:30 AM, 2:30-3:30 PM")
        logger.info("   • Avoid: First/last 15 minutes (volatility)")
        logger.info("   • Check positions: Every 30-60 minutes")
        
    def show_session_results(self):
        """Show comprehensive session results"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 TRADING SESSION RESULTS")
        logger.info("=" * 70)
        
        if self.executed_trades:
            logger.info(f"\n✅ Executed {len(self.executed_trades)} strategies")
            
            for i, trade in enumerate(self.executed_trades, 1):
                strategy = trade['strategy']
                logger.info(f"\n{i}. {strategy['type'].upper()} on {strategy['symbol']}")
                logger.info(f"   Legs: {len(strategy['legs'])}")
                logger.info(f"   Expected Profit: ${strategy['expected_profit']:.2f}")
                logger.info(f"   Time: {trade['timestamp'].strftime('%H:%M:%S')}")
                
            # Show total P&L
            total_pnl = sum(self.pnl_tracker.values())
            logger.info(f"\n💰 Session P&L: ${total_pnl:+,.2f}")
            
        else:
            logger.info("\n❌ No trades executed this session")
            
        # Show account status
        try:
            account = self.trading_client.get_account()
            logger.info(f"\n📈 Account Status:")
            logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"   Day's P&L: ${float(account.portfolio_value) - self.portfolio_value:+,.2f}")
            
        except Exception as e:
            logger.error(f"Account status error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Multi-Leg Options Trader')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    args = parser.parse_args()
    
    trader = AdvancedMultiLegLiveTrader(paper=not args.live)
    
    if args.symbols:
        trader.symbols = args.symbols
        
    trader.run()

if __name__ == "__main__":
    main()