#!/usr/bin/env python3
"""
Continuous ML Options Strategy Trainer
Runs 24/7 to find opportunities, backtest strategies, and train models
Works during both market and non-market hours
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
import pickle
import joblib
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousMLOptionsTrainer:
    def __init__(self, paper=True):
        """Initialize continuous ML trainer"""
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
        
        # Universe of symbols
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMD', 'AMZN', 'GOOGL']
        
        # ML models storage
        self.models = {
            'strategy_selector': None,
            'strike_optimizer': None,
            'profit_predictor': None,
            'risk_analyzer': None,
            'regime_detector': None
        }
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'sharpe': 0,
            'max_drawdown': 0
        })
        
        # Historical data cache
        self.data_cache = {}
        self.options_cache = {}
        
        # Model paths
        self.model_dir = 'ml_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Backtesting parameters
        self.backtest_days = 365  # 1 year of data
        self.min_sample_size = 1000
        
        # Training parameters
        self.retrain_interval = 86400  # 24 hours
        self.last_train_time = 0
        
        # Control flags
        self.running = True
        self.market_open = False
        
    def run(self):
        """Main continuous loop - runs 24/7"""
        logger.info("🤖 CONTINUOUS ML OPTIONS STRATEGY TRAINER")
        logger.info("=" * 70)
        logger.info("Running 24/7 - Training during off-hours, trading during market hours")
        
        # Load existing models
        self.load_models()
        
        try:
            while self.running:
                current_time = datetime.now(timezone.utc)
                
                # Check market status
                self.market_open = self.check_market_status()
                
                if self.market_open:
                    logger.info(f"\n📈 Market is OPEN - Looking for live opportunities")
                    self.market_hours_operations()
                else:
                    logger.info(f"\n📚 Market is CLOSED - Training and backtesting")
                    self.off_hours_operations()
                    
                # Sleep interval based on market status
                sleep_time = 300 if self.market_open else 1800  # 5 min vs 30 min
                logger.info(f"\n⏳ Sleeping for {sleep_time/60:.0f} minutes...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down...")
            self.shutdown()
            
    def check_market_status(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            # Fallback
            now = datetime.now(timezone(timedelta(hours=-5)))  # ET
            if now.weekday() >= 5:
                return False
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)
            return market_open <= now <= market_close
            
    def market_hours_operations(self):
        """Operations during market hours"""
        logger.info("\n🔄 MARKET HOURS OPERATIONS")
        
        # 1. Find live trading opportunities
        opportunities = self.find_live_opportunities()
        
        if opportunities:
            logger.info(f"   Found {len(opportunities)} opportunities")
            
            # 2. Execute best opportunities
            for opp in opportunities[:3]:  # Top 3
                if self.validate_opportunity(opp):
                    self.execute_opportunity(opp)
                    
        # 3. Monitor existing positions
        self.monitor_positions()
        
        # 4. Collect real-time data for training
        self.collect_real_time_data()
        
    def off_hours_operations(self):
        """Operations during off-hours"""
        logger.info("\n📊 OFF-HOURS OPERATIONS")
        
        # 1. Download historical data
        self.download_historical_data()
        
        # 2. Backtest strategies
        backtest_results = self.backtest_all_strategies()
        
        # 3. Train/retrain ML models
        if time.time() - self.last_train_time > self.retrain_interval:
            self.train_ml_models(backtest_results)
            self.last_train_time = time.time()
            
        # 4. Optimize strategy parameters
        self.optimize_strategies()
        
        # 5. Generate performance report
        self.generate_performance_report()
        
    def download_historical_data(self):
        """Download historical options and stock data"""
        logger.info("\n📥 Downloading historical data...")
        
        for symbol in self.symbols:
            try:
                # Get stock data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.backtest_days)
                
                bars = self.data_client.get_stock_bars(
                    StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                )
                
                if symbol in bars.df.index:
                    self.data_cache[symbol] = bars.df.loc[symbol]
                    logger.info(f"   ✓ Downloaded {len(self.data_cache[symbol])} days for {symbol}")
                    
                # Get options data (simulated for backtesting)
                self.generate_historical_options(symbol)
                
            except Exception as e:
                logger.error(f"   ✗ Error downloading {symbol}: {e}")
                
    def generate_historical_options(self, symbol: str):
        """Generate historical options data for backtesting"""
        if symbol not in self.data_cache:
            return
            
        df = self.data_cache[symbol]
        options_data = []
        
        # Generate options for each historical date
        for date, row in df.iterrows():
            price = row['close']
            volatility = self.calculate_historical_volatility(df, date)
            
            # Generate strikes around the price
            strikes = self.generate_strikes(price)
            
            # Generate expirations
            expirations = [7, 14, 30, 45, 60]  # Days to expiration
            
            for strike in strikes:
                for dte in expirations:
                    # Calculate theoretical option prices
                    call_price = self.black_scholes(price, strike, dte/365, 0.05, volatility, 'call')
                    put_price = self.black_scholes(price, strike, dte/365, 0.05, volatility, 'put')
                    
                    # Add some realistic spread
                    spread = 0.05 * min(call_price, put_price) if min(call_price, put_price) > 0 else 0.01
                    
                    options_data.append({
                        'date': date,
                        'symbol': symbol,
                        'underlying_price': price,
                        'strike': strike,
                        'dte': dte,
                        'call_bid': call_price - spread,
                        'call_ask': call_price + spread,
                        'call_mid': call_price,
                        'put_bid': put_price - spread,
                        'put_ask': put_price + spread,
                        'put_mid': put_price,
                        'volatility': volatility
                    })
                    
        self.options_cache[symbol] = pd.DataFrame(options_data)
        
    def calculate_historical_volatility(self, df: pd.DataFrame, date) -> float:
        """Calculate historical volatility up to a given date"""
        # Get data up to the date
        hist_data = df[df.index <= date].tail(30)
        
        if len(hist_data) < 10:
            return 0.25  # Default volatility
            
        returns = hist_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)
        
    def generate_strikes(self, price: float) -> List[float]:
        """Generate option strikes around current price"""
        # Round to nearest 5 for most stocks
        if price > 100:
            base = round(price / 5) * 5
            increment = 5
        else:
            base = round(price / 2.5) * 2.5
            increment = 2.5
            
        strikes = []
        for i in range(-10, 11):  # 20 strikes total
            strike = base + i * increment
            if strike > 0:
                strikes.append(strike)
                
        return strikes
        
    def black_scholes(self, S: float, K: float, T: float, r: float, 
                     sigma: float, option_type: str) -> float:
        """Black-Scholes option pricing"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return max(0, price)
        
    def backtest_all_strategies(self) -> Dict:
        """Backtest all multi-leg strategies"""
        logger.info("\n📊 Backtesting strategies...")
        
        results = {}
        strategies = [
            'iron_condor',
            'iron_butterfly',
            'bull_call_spread',
            'bear_put_spread',
            'bull_put_spread',
            'bear_call_spread',
            'straddle',
            'strangle',
            'jade_lizard',
            'broken_wing_butterfly',
            'calendar_spread',
            'diagonal_spread',
            'ratio_spread',
            'double_diagonal'
        ]
        
        for strategy in strategies:
            logger.info(f"\n   Testing {strategy}...")
            strategy_results = []
            
            for symbol in self.symbols:
                if symbol in self.data_cache and symbol in self.options_cache:
                    trades = self.backtest_strategy(symbol, strategy)
                    strategy_results.extend(trades)
                    
            # Calculate performance metrics
            if strategy_results:
                results[strategy] = self.calculate_performance_metrics(strategy_results)
                logger.info(f"   • Trades: {results[strategy]['num_trades']}")
                logger.info(f"   • Win Rate: {results[strategy]['win_rate']:.1%}")
                logger.info(f"   • Avg P&L: ${results[strategy]['avg_pnl']:.2f}")
                logger.info(f"   • Sharpe: {results[strategy]['sharpe']:.2f}")
                
        return results
        
    def backtest_strategy(self, symbol: str, strategy: str) -> List[Dict]:
        """Backtest a specific strategy on historical data"""
        trades = []
        
        stock_data = self.data_cache[symbol]
        options_data = self.options_cache[symbol]
        
        # Iterate through historical dates
        for i in range(30, len(stock_data) - 30):  # Need future data for P&L
            date = stock_data.index[i]
            price = stock_data.iloc[i]['close']
            
            # Get options available on this date
            day_options = options_data[options_data['date'] == date]
            
            if len(day_options) < 10:  # Need enough options
                continue
                
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions(stock_data, i)
            
            # Select appropriate strategy
            if self.should_trade_strategy(strategy, market_conditions):
                # Find optimal strikes and expiration
                trade_setup = self.setup_strategy(strategy, day_options, price, market_conditions)
                
                if trade_setup:
                    # Simulate trade execution and calculate P&L
                    trade_result = self.simulate_trade(
                        symbol, date, trade_setup, stock_data, options_data, i
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        
        return trades
        
    def analyze_market_conditions(self, data: pd.DataFrame, idx: int) -> Dict:
        """Analyze market conditions at a point in time"""
        # Get recent data
        lookback = 20
        recent_data = data.iloc[max(0, idx-lookback):idx+1]
        
        conditions = {}
        
        # Price trend
        sma_20 = recent_data['close'].mean()
        sma_50 = data.iloc[max(0, idx-50):idx+1]['close'].mean()
        conditions['trend'] = 'bullish' if data.iloc[idx]['close'] > sma_20 > sma_50 else 'bearish'
        
        # Volatility
        returns = recent_data['close'].pct_change().dropna()
        conditions['volatility'] = returns.std() * np.sqrt(252)
        conditions['vol_regime'] = 'high' if conditions['volatility'] > 0.30 else 'low'
        
        # RSI
        delta = recent_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        conditions['rsi'] = 100 - (100 / (1 + rs))
        
        # Support/Resistance
        conditions['support'] = recent_data['low'].min()
        conditions['resistance'] = recent_data['high'].max()
        
        # Volume
        conditions['volume_ratio'] = data.iloc[idx]['volume'] / recent_data['volume'].mean()
        
        return conditions
        
    def should_trade_strategy(self, strategy: str, conditions: Dict) -> bool:
        """Determine if conditions are right for a strategy"""
        # Strategy-specific rules
        if strategy == 'iron_condor':
            return conditions['vol_regime'] == 'low' and 30 < conditions['rsi'] < 70
            
        elif strategy == 'bull_call_spread':
            return conditions['trend'] == 'bullish' and conditions['rsi'] < 70
            
        elif strategy == 'bear_put_spread':
            return conditions['trend'] == 'bearish' and conditions['rsi'] > 30
            
        elif strategy == 'straddle':
            return conditions['vol_regime'] == 'high'
            
        elif strategy == 'jade_lizard':
            return conditions['rsi'] < 35 and conditions['trend'] == 'bullish'
            
        elif strategy == 'iron_butterfly':
            return conditions['vol_regime'] == 'low' and conditions['volume_ratio'] < 0.8
            
        # Add more strategy rules...
        
        return True  # Default to allowing trade
        
    def setup_strategy(self, strategy: str, options: pd.DataFrame, 
                      price: float, conditions: Dict) -> Optional[Dict]:
        """Setup specific strategy with optimal parameters"""
        if strategy == 'iron_condor':
            return self.setup_iron_condor(options, price, conditions)
        elif strategy == 'bull_call_spread':
            return self.setup_bull_call_spread(options, price, conditions)
        elif strategy == 'jade_lizard':
            return self.setup_jade_lizard(options, price, conditions)
        elif strategy == 'iron_butterfly':
            return self.setup_iron_butterfly(options, price, conditions)
        elif strategy == 'straddle':
            return self.setup_straddle(options, price, conditions)
        # Add more strategies...
        
        return None
        
    def setup_iron_condor(self, options: pd.DataFrame, price: float, 
                         conditions: Dict) -> Optional[Dict]:
        """Setup iron condor trade"""
        # Select expiration (30-45 days ideal)
        target_dte = 30
        exp_options = options[
            (options['dte'] >= target_dte - 5) & 
            (options['dte'] <= target_dte + 15)
        ]
        
        if len(exp_options) < 20:
            return None
            
        # Use expected move for strike selection
        expected_move = price * conditions['volatility'] * np.sqrt(target_dte/365)
        
        # Find strikes
        put_short_strike = price - expected_move * 0.8
        put_long_strike = put_short_strike - 10
        call_short_strike = price + expected_move * 0.8
        call_long_strike = call_short_strike + 10
        
        # Find closest available strikes
        strikes = exp_options['strike'].unique()
        
        put_short = min(strikes, key=lambda x: abs(x - put_short_strike))
        put_long = min(strikes, key=lambda x: abs(x - put_long_strike))
        call_short = min(strikes, key=lambda x: abs(x - call_short_strike))
        call_long = min(strikes, key=lambda x: abs(x - call_long_strike))
        
        # Get option prices
        legs = []
        
        # Put spread
        put_long_opt = exp_options[(exp_options['strike'] == put_long)].iloc[0]
        legs.append({
            'action': 'buy',
            'type': 'put',
            'strike': put_long,
            'price': put_long_opt['put_ask'],
            'quantity': 1
        })
        
        put_short_opt = exp_options[(exp_options['strike'] == put_short)].iloc[0]
        legs.append({
            'action': 'sell',
            'type': 'put',
            'strike': put_short,
            'price': put_short_opt['put_bid'],
            'quantity': 1
        })
        
        # Call spread
        call_short_opt = exp_options[(exp_options['strike'] == call_short)].iloc[0]
        legs.append({
            'action': 'sell',
            'type': 'call',
            'strike': call_short,
            'price': call_short_opt['call_bid'],
            'quantity': 1
        })
        
        call_long_opt = exp_options[(exp_options['strike'] == call_long)].iloc[0]
        legs.append({
            'action': 'buy',
            'type': 'call',
            'strike': call_long,
            'price': call_long_opt['call_ask'],
            'quantity': 1
        })
        
        # Calculate net credit
        net_credit = sum(
            leg['price'] * leg['quantity'] * (1 if leg['action'] == 'sell' else -1)
            for leg in legs
        )
        
        return {
            'strategy': 'iron_condor',
            'legs': legs,
            'net_credit': net_credit,
            'max_loss': 10 - net_credit,  # Width of spread minus credit
            'breakeven_low': put_short - net_credit,
            'breakeven_high': call_short + net_credit,
            'dte': target_dte
        }
        
    def setup_bull_call_spread(self, options: pd.DataFrame, price: float,
                              conditions: Dict) -> Optional[Dict]:
        """Setup bull call spread"""
        # Select expiration
        target_dte = 30
        exp_options = options[
            (options['dte'] >= target_dte - 5) & 
            (options['dte'] <= target_dte + 15)
        ]
        
        if len(exp_options) < 10:
            return None
            
        # Find strikes
        long_strike = price * 0.98  # Slightly ITM
        short_strike = price * 1.02  # OTM
        
        strikes = exp_options['strike'].unique()
        long_strike_actual = min(strikes, key=lambda x: abs(x - long_strike))
        short_strike_actual = min(strikes, key=lambda x: abs(x - short_strike))
        
        # Get options
        long_opt = exp_options[(exp_options['strike'] == long_strike_actual)].iloc[0]
        short_opt = exp_options[(exp_options['strike'] == short_strike_actual)].iloc[0]
        
        legs = [
            {
                'action': 'buy',
                'type': 'call',
                'strike': long_strike_actual,
                'price': long_opt['call_ask'],
                'quantity': 1
            },
            {
                'action': 'sell',
                'type': 'call',
                'strike': short_strike_actual,
                'price': short_opt['call_bid'],
                'quantity': 1
            }
        ]
        
        net_debit = legs[0]['price'] - legs[1]['price']
        
        return {
            'strategy': 'bull_call_spread',
            'legs': legs,
            'net_debit': net_debit,
            'max_profit': short_strike_actual - long_strike_actual - net_debit,
            'max_loss': net_debit,
            'breakeven': long_strike_actual + net_debit,
            'dte': target_dte
        }
        
    def setup_jade_lizard(self, options: pd.DataFrame, price: float,
                         conditions: Dict) -> Optional[Dict]:
        """Setup jade lizard (bull put spread + short call)"""
        target_dte = 30
        exp_options = options[
            (options['dte'] >= target_dte - 5) & 
            (options['dte'] <= target_dte + 15)
        ]
        
        if len(exp_options) < 15:
            return None
            
        # Strikes
        put_short_strike = price * 0.95
        put_long_strike = price * 0.90
        call_short_strike = price * 1.05
        
        strikes = exp_options['strike'].unique()
        put_short = min(strikes, key=lambda x: abs(x - put_short_strike))
        put_long = min(strikes, key=lambda x: abs(x - put_long_strike))
        call_short = min(strikes, key=lambda x: abs(x - call_short_strike))
        
        # Get options
        put_long_opt = exp_options[(exp_options['strike'] == put_long)].iloc[0]
        put_short_opt = exp_options[(exp_options['strike'] == put_short)].iloc[0]
        call_short_opt = exp_options[(exp_options['strike'] == call_short)].iloc[0]
        
        legs = [
            {
                'action': 'buy',
                'type': 'put',
                'strike': put_long,
                'price': put_long_opt['put_ask'],
                'quantity': 1
            },
            {
                'action': 'sell',
                'type': 'put',
                'strike': put_short,
                'price': put_short_opt['put_bid'],
                'quantity': 1
            },
            {
                'action': 'sell',
                'type': 'call',
                'strike': call_short,
                'price': call_short_opt['call_bid'],
                'quantity': 1
            }
        ]
        
        net_credit = (
            legs[1]['price'] + legs[2]['price'] - legs[0]['price']
        )
        
        return {
            'strategy': 'jade_lizard',
            'legs': legs,
            'net_credit': net_credit,
            'max_loss_downside': put_short - put_long - net_credit,
            'no_upside_risk': net_credit > (put_short - put_long),
            'dte': target_dte
        }
        
    def setup_iron_butterfly(self, options: pd.DataFrame, price: float,
                           conditions: Dict) -> Optional[Dict]:
        """Setup iron butterfly"""
        target_dte = 30
        exp_options = options[
            (options['dte'] >= target_dte - 5) & 
            (options['dte'] <= target_dte + 15)
        ]
        
        if len(exp_options) < 10:
            return None
            
        # ATM strike
        atm_strike = round(price / 5) * 5
        wing_width = 10 if price > 100 else 5
        
        strikes = exp_options['strike'].unique()
        atm = min(strikes, key=lambda x: abs(x - atm_strike))
        put_long = min(strikes, key=lambda x: abs(x - (atm - wing_width)))
        call_long = min(strikes, key=lambda x: abs(x - (atm + wing_width)))
        
        # Get options
        atm_opt = exp_options[(exp_options['strike'] == atm)].iloc[0]
        put_long_opt = exp_options[(exp_options['strike'] == put_long)].iloc[0]
        call_long_opt = exp_options[(exp_options['strike'] == call_long)].iloc[0]
        
        legs = [
            {
                'action': 'buy',
                'type': 'put',
                'strike': put_long,
                'price': put_long_opt['put_ask'],
                'quantity': 1
            },
            {
                'action': 'sell',
                'type': 'put',
                'strike': atm,
                'price': atm_opt['put_bid'],
                'quantity': 1
            },
            {
                'action': 'sell',
                'type': 'call',
                'strike': atm,
                'price': atm_opt['call_bid'],
                'quantity': 1
            },
            {
                'action': 'buy',
                'type': 'call',
                'strike': call_long,
                'price': call_long_opt['call_ask'],
                'quantity': 1
            }
        ]
        
        net_credit = sum(
            leg['price'] * (1 if leg['action'] == 'sell' else -1)
            for leg in legs
        )
        
        return {
            'strategy': 'iron_butterfly',
            'legs': legs,
            'net_credit': net_credit,
            'max_profit': net_credit,
            'max_loss': wing_width - net_credit,
            'profit_zone': (atm - net_credit, atm + net_credit),
            'dte': target_dte
        }
        
    def setup_straddle(self, options: pd.DataFrame, price: float,
                      conditions: Dict) -> Optional[Dict]:
        """Setup long straddle for volatility play"""
        target_dte = 30
        exp_options = options[
            (options['dte'] >= target_dte - 5) & 
            (options['dte'] <= target_dte + 15)
        ]
        
        if len(exp_options) < 5:
            return None
            
        # ATM strike
        atm_strike = round(price / 5) * 5
        strikes = exp_options['strike'].unique()
        atm = min(strikes, key=lambda x: abs(x - atm_strike))
        
        # Get options
        atm_opt = exp_options[(exp_options['strike'] == atm)].iloc[0]
        
        legs = [
            {
                'action': 'buy',
                'type': 'put',
                'strike': atm,
                'price': atm_opt['put_ask'],
                'quantity': 1
            },
            {
                'action': 'buy',
                'type': 'call',
                'strike': atm,
                'price': atm_opt['call_ask'],
                'quantity': 1
            }
        ]
        
        net_debit = legs[0]['price'] + legs[1]['price']
        
        return {
            'strategy': 'straddle',
            'legs': legs,
            'net_debit': net_debit,
            'max_loss': net_debit,
            'breakeven_low': atm - net_debit,
            'breakeven_high': atm + net_debit,
            'dte': target_dte
        }
        
    def simulate_trade(self, symbol: str, entry_date, trade_setup: Dict,
                      stock_data: pd.DataFrame, options_data: pd.DataFrame,
                      entry_idx: int) -> Optional[Dict]:
        """Simulate trade execution and calculate P&L"""
        # Find exit date (hold until expiration or exit early)
        dte = trade_setup['dte']
        exit_date = entry_date + timedelta(days=min(dte - 5, 20))  # Exit 5 days before exp
        
        # Find exit index
        exit_idx = entry_idx + min(dte - 5, 20)
        if exit_idx >= len(stock_data):
            return None
            
        exit_price = stock_data.iloc[exit_idx]['close']
        
        # Calculate P&L based on strategy
        if trade_setup['strategy'] == 'iron_condor':
            pnl = self.calculate_iron_condor_pnl(trade_setup, exit_price)
        elif trade_setup['strategy'] == 'bull_call_spread':
            pnl = self.calculate_vertical_spread_pnl(trade_setup, exit_price, 'call')
        elif trade_setup['strategy'] == 'jade_lizard':
            pnl = self.calculate_jade_lizard_pnl(trade_setup, exit_price)
        elif trade_setup['strategy'] == 'iron_butterfly':
            pnl = self.calculate_iron_butterfly_pnl(trade_setup, exit_price)
        elif trade_setup['strategy'] == 'straddle':
            pnl = self.calculate_straddle_pnl(trade_setup, exit_price, stock_data.iloc[entry_idx]['close'])
        else:
            pnl = 0
            
        return {
            'symbol': symbol,
            'strategy': trade_setup['strategy'],
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': stock_data.iloc[entry_idx]['close'],
            'exit_price': exit_price,
            'pnl': pnl * 100,  # Per contract
            'return': pnl / abs(trade_setup.get('max_loss', 1)),
            'holding_days': (exit_date - entry_date).days
        }
        
    def calculate_iron_condor_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for iron condor"""
        # Extract strikes
        put_short_strike = trade['legs'][1]['strike']
        call_short_strike = trade['legs'][2]['strike']
        
        net_credit = trade['net_credit']
        
        # Check if price is within profit zone
        if put_short_strike <= exit_price <= call_short_strike:
            return net_credit  # Keep full credit
        elif exit_price < put_short_strike:
            loss = put_short_strike - exit_price
            return net_credit - min(loss, 10)  # Max loss is width
        else:
            loss = exit_price - call_short_strike
            return net_credit - min(loss, 10)
            
    def calculate_vertical_spread_pnl(self, trade: Dict, exit_price: float, 
                                    spread_type: str) -> float:
        """Calculate P&L for vertical spreads"""
        long_strike = trade['legs'][0]['strike']
        short_strike = trade['legs'][1]['strike']
        
        if spread_type == 'call':
            if exit_price <= long_strike:
                return -trade.get('net_debit', 0)
            elif exit_price >= short_strike:
                return short_strike - long_strike - trade.get('net_debit', 0)
            else:
                return exit_price - long_strike - trade.get('net_debit', 0)
        else:  # Put spread
            if exit_price >= short_strike:
                return -trade.get('net_debit', 0)
            elif exit_price <= long_strike:
                return short_strike - long_strike - trade.get('net_debit', 0)
            else:
                return short_strike - exit_price - trade.get('net_debit', 0)
                
    def calculate_jade_lizard_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for jade lizard"""
        put_long_strike = trade['legs'][0]['strike']
        put_short_strike = trade['legs'][1]['strike']
        call_short_strike = trade['legs'][2]['strike']
        net_credit = trade['net_credit']
        
        pnl = net_credit
        
        # Put spread component
        if exit_price < put_short_strike:
            put_loss = min(put_short_strike - exit_price, put_short_strike - put_long_strike)
            pnl -= put_loss
            
        # Short call component
        if exit_price > call_short_strike:
            call_loss = exit_price - call_short_strike
            pnl -= call_loss
            
        return pnl
        
    def calculate_iron_butterfly_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for iron butterfly"""
        atm_strike = trade['legs'][1]['strike']  # Both short strikes are same
        net_credit = trade['net_credit']
        wing_width = trade['legs'][3]['strike'] - atm_strike
        
        # Calculate intrinsic value at expiration
        if exit_price == atm_strike:
            return net_credit  # Maximum profit
        else:
            loss = abs(exit_price - atm_strike)
            return net_credit - min(loss, wing_width)
            
    def calculate_straddle_pnl(self, trade: Dict, exit_price: float, 
                              entry_price: float) -> float:
        """Calculate P&L for straddle"""
        strike = trade['legs'][0]['strike']
        net_debit = trade['net_debit']
        
        # Calculate value at exit
        put_value = max(0, strike - exit_price)
        call_value = max(0, exit_price - strike)
        total_value = put_value + call_value
        
        return total_value - net_debit
        
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate strategy performance metrics"""
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'sharpe': 0,
                'max_drawdown': 0
            }
            
        df = pd.DataFrame(trades)
        
        metrics = {
            'num_trades': len(trades),
            'win_rate': len(df[df['pnl'] > 0]) / len(df),
            'avg_pnl': df['pnl'].mean(),
            'total_pnl': df['pnl'].sum(),
            'std_pnl': df['pnl'].std()
        }
        
        # Sharpe ratio (simplified)
        if metrics['std_pnl'] > 0:
            metrics['sharpe'] = metrics['avg_pnl'] / metrics['std_pnl'] * np.sqrt(252/20)
        else:
            metrics['sharpe'] = 0
            
        # Max drawdown
        cumulative = df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.abs()
        metrics['max_drawdown'] = drawdown.min()
        
        # Additional metrics
        metrics['avg_win'] = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
        metrics['avg_loss'] = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
        
        return metrics
        
    def train_ml_models(self, backtest_results: Dict):
        """Train ML models on backtest results"""
        logger.info("\n🧠 Training ML models...")
        
        # Prepare training data
        X, y_strategy, y_profit = self.prepare_training_data(backtest_results)
        
        if len(X) < self.min_sample_size:
            logger.warning(f"   Insufficient data for training ({len(X)} samples)")
            return
            
        # Split data
        X_train, X_test, y_strat_train, y_strat_test, y_prof_train, y_prof_test = train_test_split(
            X, y_strategy, y_profit, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Train strategy selector
        logger.info("   Training strategy selector...")
        self.models['strategy_selector'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models['strategy_selector'].fit(X_train_scaled, y_strat_train)
        
        accuracy = accuracy_score(y_strat_test, self.models['strategy_selector'].predict(X_test_scaled))
        logger.info(f"   • Strategy selector accuracy: {accuracy:.2%}")
        
        # 2. Train profit predictor
        logger.info("   Training profit predictor...")
        self.models['profit_predictor'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['profit_predictor'].fit(X_train_scaled, y_prof_train)
        
        mse = mean_squared_error(y_prof_test, self.models['profit_predictor'].predict(X_test_scaled))
        logger.info(f"   • Profit predictor RMSE: ${np.sqrt(mse):.2f}")
        
        # 3. Train strike optimizer
        logger.info("   Training strike optimizer...")
        self.models['strike_optimizer'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        self.models['strike_optimizer'].fit(X_train_scaled, y_prof_train)
        
        # Save models
        self.save_models()
        
    def prepare_training_data(self, backtest_results: Dict) -> Tuple:
        """Prepare data for ML training"""
        X = []
        y_strategy = []
        y_profit = []
        
        # Collect all trades from backtest
        all_trades = []
        for strategy, results in backtest_results.items():
            if 'trades' in results:
                for trade in results['trades']:
                    trade['strategy_name'] = strategy
                    all_trades.append(trade)
                    
        # Extract features from each trade
        for trade in all_trades:
            features = self.extract_trade_features(trade)
            if features:
                X.append(features)
                y_strategy.append(trade['strategy_name'])
                y_profit.append(trade['pnl'])
                
        return np.array(X), np.array(y_strategy), np.array(y_profit)
        
    def extract_trade_features(self, trade: Dict) -> Optional[List[float]]:
        """Extract ML features from a trade"""
        try:
            features = [
                trade.get('entry_price', 0),
                trade.get('volatility', 0.25),
                trade.get('rsi', 50),
                trade.get('volume_ratio', 1),
                trade.get('trend_strength', 0),
                trade.get('dte', 30),
                trade.get('moneyness', 1),
                trade.get('spread_width', 10),
                trade.get('vol_regime_encoded', 0),
                trade.get('time_of_day', 12),
                trade.get('day_of_week', 3)
            ]
            return features
        except:
            return None
            
    def optimize_strategies(self):
        """Optimize strategy parameters using ML insights"""
        logger.info("\n🔧 Optimizing strategy parameters...")
        
        if not self.models['profit_predictor']:
            logger.warning("   No trained models available")
            return
            
        # Optimize each strategy
        optimizations = {}
        
        for strategy in self.strategy_performance.keys():
            logger.info(f"\n   Optimizing {strategy}...")
            
            # Grid search for optimal parameters
            best_params = self.grid_search_strategy(strategy)
            optimizations[strategy] = best_params
            
            logger.info(f"   • Best parameters: {best_params}")
            
        # Save optimizations
        with open(os.path.join(self.model_dir, 'strategy_optimizations.json'), 'w') as f:
            json.dump(optimizations, f, indent=2)
            
    def grid_search_strategy(self, strategy: str) -> Dict:
        """Grid search for optimal strategy parameters"""
        # Define parameter ranges
        param_grid = {
            'dte': [15, 30, 45, 60],
            'moneyness': [0.90, 0.95, 1.0, 1.05, 1.10],
            'spread_width': [5, 10, 15, 20],
            'vol_threshold': [0.15, 0.25, 0.35]
        }
        
        best_score = -float('inf')
        best_params = {}
        
        # Simple grid search (in production, use more sophisticated optimization)
        for dte in param_grid['dte']:
            for moneyness in param_grid['moneyness']:
                for width in param_grid['spread_width']:
                    for vol in param_grid['vol_threshold']:
                        # Simulate performance with these parameters
                        score = self.evaluate_parameters(
                            strategy, dte, moneyness, width, vol
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'dte': dte,
                                'moneyness': moneyness,
                                'spread_width': width,
                                'vol_threshold': vol,
                                'expected_return': score
                            }
                            
        return best_params
        
    def evaluate_parameters(self, strategy: str, dte: int, moneyness: float,
                          width: float, vol: float) -> float:
        """Evaluate strategy parameters using ML model"""
        if not self.models['profit_predictor']:
            return 0
            
        # Create feature vector
        features = np.array([[
            100,  # Dummy price
            vol,
            50,   # RSI
            1,    # Volume ratio
            0,    # Trend
            dte,
            moneyness,
            width,
            1,    # Vol regime
            12,   # Time
            3     # Day
        ]])
        
        # Predict profit
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            profit = self.models['profit_predictor'].predict(features_scaled)[0]
            return profit
        except:
            return 0
            
    def find_live_opportunities(self) -> List[Dict]:
        """Find trading opportunities using ML models"""
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Get current data
                quote = self.data_client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=symbol)
                )
                
                if symbol not in quote:
                    continue
                    
                price = float(quote[symbol].ask_price)
                
                # Get recent history
                bars = self.data_client.get_stock_bars(
                    StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Hour,
                        limit=50
                    )
                )
                
                if symbol not in bars.df.index:
                    continue
                    
                # Analyze conditions
                conditions = self.analyze_market_conditions(bars.df.loc[symbol], -1)
                
                # Use ML to predict best strategy
                if self.models['strategy_selector']:
                    features = self.extract_current_features(symbol, price, conditions)
                    if features:
                        strategy = self.predict_best_strategy(features)
                        expected_profit = self.predict_profit(features)
                        
                        if expected_profit > 50:  # $50 minimum
                            opportunities.append({
                                'symbol': symbol,
                                'strategy': strategy,
                                'expected_profit': expected_profit,
                                'confidence': 0.8,  # Placeholder
                                'conditions': conditions
                            })
                            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                
        # Sort by expected profit
        opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        return opportunities
        
    def extract_current_features(self, symbol: str, price: float, 
                                conditions: Dict) -> Optional[np.ndarray]:
        """Extract features for current market conditions"""
        try:
            features = np.array([[
                price,
                conditions.get('volatility', 0.25),
                conditions.get('rsi', 50),
                conditions.get('volume_ratio', 1),
                1 if conditions.get('trend') == 'bullish' else -1,
                30,  # Default DTE
                1.0,  # Default moneyness
                10,   # Default width
                1 if conditions.get('vol_regime') == 'high' else 0,
                datetime.now().hour,
                datetime.now().weekday()
            ]])
            
            return features
            
        except:
            return None
            
    def predict_best_strategy(self, features: np.ndarray) -> str:
        """Predict best strategy using ML"""
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            strategy = self.models['strategy_selector'].predict(features_scaled)[0]
            return strategy
        except:
            return 'iron_condor'  # Default
            
    def predict_profit(self, features: np.ndarray) -> float:
        """Predict expected profit using ML"""
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            profit = self.models['profit_predictor'].predict(features_scaled)[0]
            return profit
        except:
            return 0
            
    def validate_opportunity(self, opportunity: Dict) -> bool:
        """Validate trading opportunity"""
        # Risk checks
        if opportunity['expected_profit'] < 50:
            return False
            
        # Volatility check
        if opportunity['conditions'].get('volatility', 0) > 0.50:
            return False
            
        # Additional validation...
        
        return True
        
    def execute_opportunity(self, opportunity: Dict):
        """Execute trading opportunity"""
        logger.info(f"\n   Executing {opportunity['strategy']} on {opportunity['symbol']}")
        logger.info(f"   Expected profit: ${opportunity['expected_profit']:.2f}")
        
        # In live mode, would execute actual trades
        # For now, just log
        
    def monitor_positions(self):
        """Monitor existing positions"""
        try:
            positions = self.trading_client.get_all_positions()
            option_positions = [p for p in positions if len(p.symbol) > 10]
            
            if option_positions:
                logger.info(f"\n   Monitoring {len(option_positions)} positions")
                
                for pos in option_positions:
                    pnl = float(pos.unrealized_pl)
                    
                    # Check exit conditions
                    if pnl > 100:  # Take profit
                        logger.info(f"   • {pos.symbol}: ${pnl:+.2f} - Consider taking profit")
                    elif pnl < -50:  # Stop loss
                        logger.warning(f"   • {pos.symbol}: ${pnl:+.2f} - Consider stop loss")
                        
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
            
    def collect_real_time_data(self):
        """Collect real-time data for future training"""
        # Store current market conditions and trades for future training
        pass
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("\n📈 Performance Report")
        logger.info("-" * 50)
        
        # Sort strategies by performance
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1]['sharpe'],
            reverse=True
        )
        
        for strategy, metrics in sorted_strategies:
            if metrics['trades'] > 0:
                logger.info(f"\n{strategy.upper()}:")
                logger.info(f"   Trades: {metrics['trades']}")
                logger.info(f"   Win Rate: {metrics['wins']/metrics['trades']*100:.1f}%")
                logger.info(f"   Avg P&L: ${metrics['avg_pnl']:.2f}")
                logger.info(f"   Total P&L: ${metrics['total_pnl']:.2f}")
                logger.info(f"   Sharpe: {metrics['sharpe']:.2f}")
                logger.info(f"   Max DD: {metrics['max_drawdown']:.1%}")
                
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            if model:
                path = os.path.join(self.model_dir, f'{name}_model.pkl')
                joblib.dump(model, path)
                logger.info(f"   Saved {name} model")
                
    def load_models(self):
        """Load existing models"""
        for name in self.models.keys():
            path = os.path.join(self.model_dir, f'{name}_model.pkl')
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                logger.info(f"   Loaded {name} model")
                
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        # Save final state
        self.save_models()
        
        # Save performance metrics
        with open(os.path.join(self.model_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(dict(self.strategy_performance), f, indent=2)
            
        logger.info("\n✅ Shutdown complete")

def main():
    """Main entry point"""
    trainer = ContinuousMLOptionsTrainer()
    trainer.run()

if __name__ == "__main__":
    main()