#!/usr/bin/env python3
"""
Advanced Options Trading Bot with Multiple Algorithms
Implements sophisticated options strategies with Greeks, volatility analysis, and ML
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import norm
from scipy.optimize import minimize
import requests
from collections import defaultdict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('advanced_options_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedOptionsAlgorithmsBot:
    def __init__(self):
        """Initialize the advanced options trading bot"""
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
        
        # Trading parameters
        self.scan_interval = 20  # Fast scanning
        self.max_positions = 20
        self.position_size = 1  # contracts per trade
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Greeks thresholds
        self.min_delta = 0.15  # Minimum delta for OTM options
        self.max_delta = 0.85  # Maximum delta
        self.min_theta = -0.50  # Minimum theta (time decay)
        self.max_vega = 0.30   # Maximum vega exposure
        
        # Volatility parameters
        self.iv_percentile_threshold = 70  # High IV percentile
        self.hv_lookback = 20  # Historical volatility lookback
        self.iv_skew_threshold = 0.05  # 5% IV skew
        
        # Advanced watchlist
        self.watchlist = {
            'high_volume': ['SPY', 'QQQ', 'IWM', 'DIA'],
            'tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL', 'META'],
            'financials': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC'],
            'volatility': ['TSLA', 'COIN', 'MARA', 'RIOT'],
            'etfs': ['XLF', 'XLE', 'XLK', 'GLD', 'TLT', 'VXX']
        }
        
        # Track strategies and performance
        self.active_positions = {}
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'win_rate': 0
        })
        
        # Market regime
        self.market_regime = 'neutral'
        self.vix_level = 20
        
    async def run(self):
        """Main trading loop with multiple algorithms"""
        logger.info("🚀 Starting Advanced Options Algorithms Bot")
        logger.info("Strategies: Delta-Neutral, Volatility Arbitrage, Gamma Scalping, ML-Based")
        
        await self.check_account_status()
        
        # Start background threads
        self.start_background_tasks()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"🎯 Advanced Options Trading Cycle #{cycle}")
                logger.info(f"{'='*70}")
                
                # Check market
                if not await self.is_market_open():
                    logger.info("Market is closed")
                    await asyncio.sleep(300)
                    continue
                
                # Update market regime
                await self.analyze_market_regime()
                
                # Run all algorithms concurrently
                await asyncio.gather(
                    # Core options algorithms
                    self.volatility_premium_harvesting(),
                    self.delta_neutral_strategies(),
                    self.gamma_scalping_algorithm(),
                    self.theta_decay_optimization(),
                    
                    # Advanced algorithms
                    self.iv_rank_mean_reversion(),
                    self.skew_trading_algorithm(),
                    self.term_structure_arbitrage(),
                    self.earnings_volatility_crush(),
                    
                    # Machine learning based
                    self.ml_options_prediction(),
                    self.dynamic_hedging_algorithm(),
                    
                    # Spread strategies
                    self.optimal_spread_selection(),
                    self.multi_leg_optimization()
                )
                
                # Portfolio management
                await self.manage_portfolio_greeks()
                await self.dynamic_position_sizing()
                
                # Display comprehensive dashboard
                await self.display_trading_dashboard()
                
                await asyncio.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Closing all option positions...")
                await self.close_all_positions()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
                
    def start_background_tasks(self):
        """Start background monitoring threads"""
        # Real-time Greeks monitor
        greeks_thread = threading.Thread(target=self.monitor_portfolio_greeks)
        greeks_thread.daemon = True
        greeks_thread.start()
        
        # Volatility surface updater
        vol_thread = threading.Thread(target=self.update_volatility_surface)
        vol_thread.daemon = True
        vol_thread.start()
        
    def monitor_portfolio_greeks(self):
        """Monitor portfolio Greeks in real-time"""
        while True:
            try:
                # Calculate aggregate Greeks every 30 seconds
                time.sleep(30)
                self.calculate_portfolio_greeks()
            except Exception as e:
                logger.error(f"Greeks monitor error: {e}")
                
    def update_volatility_surface(self):
        """Update implied volatility surface"""
        while True:
            try:
                # Update vol surface every minute
                time.sleep(60)
                self.build_volatility_surface()
            except Exception as e:
                logger.error(f"Vol surface error: {e}")
                
    async def check_account_status(self):
        """Check account and options trading status"""
        try:
            account = self.trading_client.get_account()
            logger.info("\n💼 Account Status:")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Options Level: {account.options_trading_level}")
            
            self.buying_power = float(account.buying_power)
            self.portfolio_value = float(account.portfolio_value)
            
        except Exception as e:
            logger.error(f"Error checking account: {e}")
            
    async def is_market_open(self) -> bool:
        """Check if options market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    async def analyze_market_regime(self):
        """Analyze current market regime for strategy selection"""
        try:
            # Get VIX proxy (VXX)
            vxx_quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols='VXX')
            )
            if 'VXX' in vxx_quote:
                self.vix_level = float(vxx_quote['VXX'].ask_price)
                
            # Get SPY trend
            spy_bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols='SPY',
                    timeframe=TimeFrame.Day,
                    limit=20
                )
            )
            
            if 'SPY' in spy_bars.data:
                spy_df = spy_bars.df.loc['SPY']
                sma_20 = spy_df['close'].mean()
                current = spy_df['close'].iloc[-1]
                
                # Determine regime
                if self.vix_level > 30:
                    self.market_regime = 'high_volatility'
                elif self.vix_level < 15:
                    self.market_regime = 'low_volatility'
                elif current > sma_20 * 1.02:
                    self.market_regime = 'trending_up'
                elif current < sma_20 * 0.98:
                    self.market_regime = 'trending_down'
                else:
                    self.market_regime = 'neutral'
                    
            logger.info(f"📊 Market Regime: {self.market_regime}, VIX: {self.vix_level:.1f}")
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            
    async def get_option_chain(self, symbol: str, min_dte: int = 20, max_dte: int = 45) -> Dict:
        """Get option chain with Greeks calculation"""
        try:
            # Get current stock price
            stock_quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            stock_price = float(stock_quote[symbol].ask_price)
            
            # Get options from API
            today = datetime.now().date()
            expiry_min = (today + timedelta(days=min_dte)).strftime('%Y-%m-%d')
            expiry_max = (today + timedelta(days=max_dte)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'expiration_date_gte': expiry_min,
                'expiration_date_lte': expiry_max
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                # Calculate Greeks for each option
                for contract in contracts:
                    self.calculate_option_greeks(contract, stock_price)
                    
                return {
                    'underlying_price': stock_price,
                    'contracts': contracts
                }
                
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            
        return {'underlying_price': 0, 'contracts': []}
        
    def calculate_option_greeks(self, contract: Dict, stock_price: float):
        """Calculate Greeks for an option contract"""
        try:
            strike = float(contract.get('strike_price', 0))
            expiry = datetime.strptime(contract.get('expiration_date'), '%Y-%m-%d')
            dte = (expiry - datetime.now()).days
            
            if dte <= 0:
                return
                
            # Estimate IV (simplified - in production use actual IV)
            iv = 0.25  # 25% implied volatility
            r = 0.05   # Risk-free rate
            
            # Time to expiration in years
            T = dte / 365.0
            
            # Black-Scholes calculations
            d1 = (np.log(stock_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            
            if contract.get('type') == 'call':
                delta = norm.cdf(d1)
                theta = -(stock_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) - r * strike * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                delta = -norm.cdf(-d1)
                theta = -(stock_price * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) + r * strike * np.exp(-r * T) * norm.cdf(-d2)
                
            gamma = norm.pdf(d1) / (stock_price * iv * np.sqrt(T))
            vega = stock_price * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Add Greeks to contract
            contract['delta'] = delta
            contract['gamma'] = gamma
            contract['theta'] = theta / 365  # Daily theta
            contract['vega'] = vega
            contract['iv'] = iv
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            
    # ALGORITHM 1: Volatility Premium Harvesting
    async def volatility_premium_harvesting(self):
        """Harvest volatility risk premium through short volatility strategies"""
        logger.info("📊 Algorithm 1: Volatility Premium Harvesting")
        
        for symbol in self.watchlist['high_volume']:
            try:
                chain = await self.get_option_chain(symbol)
                if not chain['contracts']:
                    continue
                    
                stock_price = chain['underlying_price']
                
                # Calculate IV rank
                iv_rank = await self.calculate_iv_rank(symbol)
                
                if iv_rank > self.iv_percentile_threshold:
                    logger.info(f"✅ High IV Rank on {symbol}: {iv_rank:.0f}%")
                    
                    # Find optimal short volatility trade
                    if self.market_regime in ['neutral', 'low_volatility']:
                        # Iron Condor for high IV
                        await self.execute_iron_condor(symbol, chain)
                    else:
                        # Credit spreads for directional bias
                        await self.execute_credit_spread(symbol, chain)
                        
            except Exception as e:
                logger.error(f"Error in volatility harvesting {symbol}: {e}")
                
    # ALGORITHM 2: Delta-Neutral Strategies
    async def delta_neutral_strategies(self):
        """Implement delta-neutral strategies for consistent income"""
        logger.info("📊 Algorithm 2: Delta-Neutral Strategies")
        
        for symbol in self.watchlist['tech'][:3]:
            try:
                chain = await self.get_option_chain(symbol)
                if not chain['contracts']:
                    continue
                    
                # Find ATM straddle/strangle opportunities
                atm_options = self.find_atm_options(chain)
                
                if atm_options:
                    # Calculate expected move
                    expected_move = await self.calculate_expected_move(symbol, atm_options)
                    
                    if expected_move > 0.03:  # 3% expected move
                        logger.info(f"✅ Delta-neutral opportunity on {symbol}")
                        logger.info(f"   Expected move: {expected_move:.1%}")
                        
                        # Execute market-neutral strategy
                        await self.execute_delta_neutral_trade(symbol, atm_options)
                        
            except Exception as e:
                logger.error(f"Error in delta-neutral {symbol}: {e}")
                
    # ALGORITHM 3: Gamma Scalping
    async def gamma_scalping_algorithm(self):
        """Gamma scalping for volatile stocks"""
        logger.info("📊 Algorithm 3: Gamma Scalping")
        
        for symbol in self.watchlist['volatility']:
            try:
                # Check intraday volatility
                intraday_vol = await self.calculate_intraday_volatility(symbol)
                
                if intraday_vol > 0.02:  # 2% intraday moves
                    chain = await self.get_option_chain(symbol, min_dte=7, max_dte=30)
                    
                    # Find high gamma options
                    high_gamma = self.find_high_gamma_options(chain)
                    
                    if high_gamma:
                        logger.info(f"✅ Gamma scalping setup on {symbol}")
                        logger.info(f"   Intraday vol: {intraday_vol:.1%}")
                        
                        await self.execute_gamma_scalp(symbol, high_gamma)
                        
            except Exception as e:
                logger.error(f"Error in gamma scalping {symbol}: {e}")
                
    # ALGORITHM 4: Theta Decay Optimization
    async def theta_decay_optimization(self):
        """Optimize theta decay through calendar spreads"""
        logger.info("📊 Algorithm 4: Theta Decay Optimization")
        
        for symbol in self.watchlist['etfs']:
            try:
                # Get near and far term options
                near_chain = await self.get_option_chain(symbol, min_dte=7, max_dte=21)
                far_chain = await self.get_option_chain(symbol, min_dte=30, max_dte=60)
                
                if near_chain['contracts'] and far_chain['contracts']:
                    # Find calendar spread opportunities
                    calendar_opps = self.find_calendar_spreads(near_chain, far_chain)
                    
                    for opp in calendar_opps:
                        if opp['theta_edge'] > 0.20:  # 20 cents daily theta edge
                            logger.info(f"✅ Calendar spread on {symbol}")
                            logger.info(f"   Theta edge: ${opp['theta_edge']:.2f}/day")
                            
                            await self.execute_calendar_spread(symbol, opp)
                            
            except Exception as e:
                logger.error(f"Error in theta optimization {symbol}: {e}")
                
    # ALGORITHM 5: IV Rank Mean Reversion
    async def iv_rank_mean_reversion(self):
        """Trade IV rank mean reversion"""
        logger.info("📊 Algorithm 5: IV Rank Mean Reversion")
        
        for symbol in self.watchlist['high_volume']:
            try:
                iv_rank = await self.calculate_iv_rank(symbol)
                
                # Extreme IV conditions
                if iv_rank > 80:
                    logger.info(f"✅ IV Rank extremely high on {symbol}: {iv_rank:.0f}%")
                    # Sell volatility
                    chain = await self.get_option_chain(symbol)
                    await self.execute_short_volatility(symbol, chain)
                    
                elif iv_rank < 20:
                    logger.info(f"✅ IV Rank extremely low on {symbol}: {iv_rank:.0f}%")
                    # Buy volatility
                    chain = await self.get_option_chain(symbol)
                    await self.execute_long_volatility(symbol, chain)
                    
            except Exception as e:
                logger.error(f"Error in IV mean reversion {symbol}: {e}")
                
    # ALGORITHM 6: Skew Trading
    async def skew_trading_algorithm(self):
        """Trade volatility skew anomalies"""
        logger.info("📊 Algorithm 6: Skew Trading")
        
        for symbol in self.watchlist['tech']:
            try:
                chain = await self.get_option_chain(symbol)
                if not chain['contracts']:
                    continue
                    
                # Calculate put/call skew
                skew = self.calculate_volatility_skew(chain)
                
                if abs(skew) > self.iv_skew_threshold:
                    logger.info(f"✅ Skew opportunity on {symbol}: {skew:.2%}")
                    
                    if skew > 0:  # Put skew high
                        await self.execute_put_spread_sale(symbol, chain)
                    else:  # Call skew high
                        await self.execute_call_spread_sale(symbol, chain)
                        
            except Exception as e:
                logger.error(f"Error in skew trading {symbol}: {e}")
                
    # ALGORITHM 7: Term Structure Arbitrage
    async def term_structure_arbitrage(self):
        """Arbitrage term structure anomalies"""
        logger.info("📊 Algorithm 7: Term Structure Arbitrage")
        
        for symbol in self.watchlist['etfs']:
            try:
                # Get multiple expirations
                expirations = await self.get_term_structure(symbol)
                
                if len(expirations) >= 3:
                    # Find term structure anomalies
                    anomalies = self.find_term_structure_anomalies(expirations)
                    
                    for anomaly in anomalies:
                        if anomaly['edge'] > 0.02:  # 2% edge
                            logger.info(f"✅ Term structure arb on {symbol}")
                            logger.info(f"   Edge: {anomaly['edge']:.2%}")
                            
                            await self.execute_term_structure_trade(symbol, anomaly)
                            
            except Exception as e:
                logger.error(f"Error in term structure arb {symbol}: {e}")
                
    # ALGORITHM 8: Earnings Volatility Crush
    async def earnings_volatility_crush(self):
        """Trade pre/post earnings volatility crush"""
        logger.info("📊 Algorithm 8: Earnings Volatility Crush")
        
        # Check for upcoming earnings
        earnings_stocks = await self.get_upcoming_earnings()
        
        for symbol in earnings_stocks:
            try:
                chain = await self.get_option_chain(symbol, min_dte=0, max_dte=30)
                
                # Find pre-earnings high IV
                pre_earnings_iv = self.find_pre_earnings_options(chain)
                
                if pre_earnings_iv:
                    logger.info(f"✅ Earnings vol crush setup on {symbol}")
                    
                    # Short volatility before earnings
                    await self.execute_earnings_trade(symbol, pre_earnings_iv)
                    
            except Exception as e:
                logger.error(f"Error in earnings trade {symbol}: {e}")
                
    # ALGORITHM 9: ML Options Prediction
    async def ml_options_prediction(self):
        """Machine learning based options trading"""
        logger.info("📊 Algorithm 9: ML Options Prediction")
        
        for symbol in self.watchlist['high_volume']:
            try:
                # Get features for ML model
                features = await self.extract_ml_features(symbol)
                
                if features is not None:
                    # Predict optimal strategy
                    prediction = self.ml_predict_strategy(features)
                    
                    if prediction['confidence'] > 0.7:
                        logger.info(f"✅ ML prediction for {symbol}")
                        logger.info(f"   Strategy: {prediction['strategy']}")
                        logger.info(f"   Confidence: {prediction['confidence']:.1%}")
                        
                        chain = await self.get_option_chain(symbol)
                        await self.execute_ml_strategy(symbol, chain, prediction)
                        
            except Exception as e:
                logger.error(f"Error in ML prediction {symbol}: {e}")
                
    # ALGORITHM 10: Dynamic Hedging
    async def dynamic_hedging_algorithm(self):
        """Dynamic portfolio hedging based on market conditions"""
        logger.info("📊 Algorithm 10: Dynamic Hedging")
        
        try:
            # Calculate portfolio beta
            portfolio_beta = await self.calculate_portfolio_beta()
            
            # Determine hedge requirement
            if abs(portfolio_beta) > 0.5:
                hedge_delta = -portfolio_beta * self.portfolio_value / 100
                
                logger.info(f"Portfolio beta: {portfolio_beta:.2f}")
                logger.info(f"Hedge delta needed: {hedge_delta:.0f}")
                
                # Execute hedge
                await self.execute_portfolio_hedge(hedge_delta)
                
        except Exception as e:
            logger.error(f"Error in dynamic hedging: {e}")
            
    # ALGORITHM 11: Optimal Spread Selection
    async def optimal_spread_selection(self):
        """Select optimal spread width using Kelly Criterion"""
        logger.info("📊 Algorithm 11: Optimal Spread Selection")
        
        for symbol in self.watchlist['high_volume']:
            try:
                chain = await self.get_option_chain(symbol)
                
                # Calculate optimal spread parameters
                optimal = self.calculate_optimal_spread(chain)
                
                if optimal['expected_value'] > 0:
                    logger.info(f"✅ Optimal spread for {symbol}")
                    logger.info(f"   Type: {optimal['type']}")
                    logger.info(f"   Width: ${optimal['width']:.0f}")
                    logger.info(f"   Expected value: ${optimal['expected_value']:.2f}")
                    
                    await self.execute_optimal_spread(symbol, optimal)
                    
            except Exception as e:
                logger.error(f"Error in spread selection {symbol}: {e}")
                
    # ALGORITHM 12: Multi-Leg Optimization
    async def multi_leg_optimization(self):
        """Optimize complex multi-leg strategies"""
        logger.info("📊 Algorithm 12: Multi-Leg Optimization")
        
        for symbol in self.watchlist['tech'][:2]:
            try:
                chain = await self.get_option_chain(symbol)
                
                # Optimize butterfly/condor parameters
                optimal_structure = self.optimize_multi_leg(chain)
                
                if optimal_structure['sharpe_ratio'] > 1.5:
                    logger.info(f"✅ Optimized structure for {symbol}")
                    logger.info(f"   Type: {optimal_structure['type']}")
                    logger.info(f"   Sharpe ratio: {optimal_structure['sharpe_ratio']:.2f}")
                    
                    await self.execute_multi_leg(symbol, optimal_structure)
                    
            except Exception as e:
                logger.error(f"Error in multi-leg optimization {symbol}: {e}")
                
    # Helper methods for calculations and execution
    async def calculate_iv_rank(self, symbol: str) -> float:
        """Calculate IV rank (0-100)"""
        try:
            # Simplified IV rank calculation
            # In production, use historical IV data
            current_iv = 0.25  # Placeholder
            return min(100, max(0, (current_iv - 0.15) / (0.40 - 0.15) * 100))
        except:
            return 50
            
    async def calculate_expected_move(self, symbol: str, options: List) -> float:
        """Calculate expected move from options prices"""
        try:
            # ATM straddle price as % of stock
            atm_call = next((o for o in options if o.get('type') == 'call'), None)
            atm_put = next((o for o in options if o.get('type') == 'put'), None)
            
            if atm_call and atm_put:
                # Simplified - use mid prices
                straddle_price = 2.0  # Placeholder
                stock_price = options[0].get('underlying_price', 100)
                return straddle_price / stock_price
                
        except:
            return 0
            
    async def calculate_intraday_volatility(self, symbol: str) -> float:
        """Calculate intraday volatility"""
        try:
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=390  # Full trading day
                )
            )
            
            if symbol in bars.data:
                df = bars.df.loc[symbol]
                returns = df['close'].pct_change()
                return returns.std() * np.sqrt(390)  # Annualized
                
        except:
            return 0
            
    def find_atm_options(self, chain: Dict) -> List:
        """Find ATM options"""
        contracts = chain['contracts']
        stock_price = chain['underlying_price']
        
        # Find closest strike to current price
        atm_strike = min(contracts, key=lambda x: abs(float(x.get('strike_price', 0)) - stock_price))
        
        return [c for c in contracts if c.get('strike_price') == atm_strike.get('strike_price')]
        
    def find_high_gamma_options(self, chain: Dict) -> List:
        """Find options with high gamma"""
        contracts = chain['contracts']
        
        # Sort by gamma
        high_gamma = sorted(
            [c for c in contracts if c.get('gamma', 0) > 0.01],
            key=lambda x: x.get('gamma', 0),
            reverse=True
        )
        
        return high_gamma[:5]  # Top 5
        
    def calculate_volatility_skew(self, chain: Dict) -> float:
        """Calculate put/call volatility skew"""
        try:
            contracts = chain['contracts']
            stock_price = chain['underlying_price']
            
            # Get OTM puts and calls
            otm_puts = [c for c in contracts if c.get('type') == 'put' and 
                       float(c.get('strike_price', 0)) < stock_price * 0.95]
            otm_calls = [c for c in contracts if c.get('type') == 'call' and 
                        float(c.get('strike_price', 0)) > stock_price * 1.05]
                        
            if otm_puts and otm_calls:
                avg_put_iv = np.mean([p.get('iv', 0.25) for p in otm_puts])
                avg_call_iv = np.mean([c.get('iv', 0.25) for c in otm_calls])
                return (avg_put_iv - avg_call_iv) / avg_call_iv
                
        except:
            return 0
            
    # Execution methods
    async def execute_iron_condor(self, symbol: str, chain: Dict):
        """Execute Iron Condor trade"""
        try:
            contracts = chain['contracts']
            stock_price = chain['underlying_price']
            
            # Find strikes
            puts = sorted([c for c in contracts if c.get('type') == 'put'], 
                         key=lambda x: float(x.get('strike_price', 0)))
            calls = sorted([c for c in contracts if c.get('type') == 'call'], 
                          key=lambda x: float(x.get('strike_price', 0)))
                          
            if len(puts) >= 4 and len(calls) >= 4:
                # Select strikes ~5% OTM
                put_short_idx = len([p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.95])
                call_short_idx = len([c for c in calls if float(c.get('strike_price', 0)) < stock_price * 1.05])
                
                if put_short_idx > 1 and call_short_idx < len(calls) - 1:
                    # Iron Condor legs
                    legs = [
                        (puts[put_short_idx - 2], OrderSide.BUY, "Long Put"),
                        (puts[put_short_idx - 1], OrderSide.SELL, "Short Put"),
                        (calls[call_short_idx], OrderSide.SELL, "Short Call"),
                        (calls[call_short_idx + 1], OrderSide.BUY, "Long Call")
                    ]
                    
                    logger.info(f"  🦅 Executing Iron Condor on {symbol}")
                    
                    for option, side, desc in legs:
                        order = MarketOrderRequest(
                            symbol=option.get('symbol'),
                            qty=self.position_size,
                            side=side,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        self.trading_client.submit_order(order)
                        logger.info(f"    ✅ {desc}: {option.get('symbol')}")
                        
                    # Track position
                    self.active_positions[f"IC_{symbol}_{int(time.time())}"] = {
                        'strategy': 'iron_condor',
                        'legs': [leg[0].get('symbol') for leg in legs],
                        'entry_time': datetime.now()
                    }
                    
                    self.strategy_performance['iron_condor']['trades'] += 1
                    
        except Exception as e:
            logger.error(f"Error executing Iron Condor: {e}")
            
    async def execute_credit_spread(self, symbol: str, chain: Dict):
        """Execute credit spread based on market direction"""
        try:
            contracts = chain['contracts']
            stock_price = chain['underlying_price']
            
            # Determine direction
            direction = await self.determine_trend(symbol)
            
            if direction == 'bullish':
                # Bull Put Spread
                puts = sorted([c for c in contracts if c.get('type') == 'put'], 
                             key=lambda x: float(x.get('strike_price', 0)))
                             
                otm_puts = [p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.95]
                
                if len(otm_puts) >= 2:
                    short_put = otm_puts[-1]
                    long_put = otm_puts[-2]
                    
                    logger.info(f"  📈 Executing Bull Put Spread on {symbol}")
                    
                    # Sell put
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=short_put.get('symbol'),
                        qty=self.position_size,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    # Buy put
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=long_put.get('symbol'),
                        qty=self.position_size,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    self.strategy_performance['bull_put_spread']['trades'] += 1
                    
            else:
                # Bear Call Spread
                calls = sorted([c for c in contracts if c.get('type') == 'call'], 
                              key=lambda x: float(x.get('strike_price', 0)))
                              
                otm_calls = [c for c in calls if float(c.get('strike_price', 0)) > stock_price * 1.05]
                
                if len(otm_calls) >= 2:
                    short_call = otm_calls[0]
                    long_call = otm_calls[1]
                    
                    logger.info(f"  📉 Executing Bear Call Spread on {symbol}")
                    
                    # Sell call
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=short_call.get('symbol'),
                        qty=self.position_size,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    # Buy call
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=long_call.get('symbol'),
                        qty=self.position_size,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    
                    self.strategy_performance['bear_call_spread']['trades'] += 1
                    
        except Exception as e:
            logger.error(f"Error executing credit spread: {e}")
            
    async def determine_trend(self, symbol: str) -> str:
        """Determine stock trend"""
        try:
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=20
                )
            )
            
            if symbol in bars.data:
                df = bars.df.loc[symbol]
                sma_20 = df['close'].mean()
                current = df['close'].iloc[-1]
                
                return 'bullish' if current > sma_20 else 'bearish'
                
        except:
            return 'neutral'
            
    async def manage_portfolio_greeks(self):
        """Manage aggregate portfolio Greeks"""
        try:
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            
            positions = self.trading_client.get_all_positions()
            
            # Calculate aggregate Greeks
            for position in positions:
                if len(position.symbol) > 10:  # Options
                    # Estimate Greeks (simplified)
                    qty = int(position.qty)
                    total_delta += qty * 0.5 * 100  # Placeholder
                    total_theta += qty * -0.1 * 100  # Placeholder
                    
            logger.info(f"\n📊 Portfolio Greeks:")
            logger.info(f"  Delta: {total_delta:+.0f}")
            logger.info(f"  Theta: ${total_theta:+.2f}/day")
            
            # Hedge if needed
            if abs(total_delta) > 500:
                logger.info(f"  ⚠️ Delta hedge needed!")
                
        except Exception as e:
            logger.error(f"Error managing Greeks: {e}")
            
    async def display_trading_dashboard(self):
        """Display comprehensive trading dashboard"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n" + "="*70)
            logger.info("📊 ADVANCED OPTIONS TRADING DASHBOARD")
            logger.info("="*70)
            
            # Account summary
            logger.info(f"\n💼 Account Summary:")
            logger.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"  Active Positions: {len(positions)}")
            
            # Strategy performance
            logger.info(f"\n📈 Strategy Performance:")
            for strategy, stats in self.strategy_performance.items():
                if stats['trades'] > 0:
                    win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                    logger.info(f"  {strategy}:")
                    logger.info(f"    Trades: {stats['trades']}, Win Rate: {win_rate:.1f}%")
                    
            # Active option positions
            option_positions = [p for p in positions if len(p.symbol) > 10]
            if option_positions:
                logger.info(f"\n🎯 Active Option Positions: {len(option_positions)}")
                for pos in option_positions[:5]:
                    logger.info(f"  {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
                    logger.info(f"    P&L: ${float(pos.unrealized_pl):,.2f}")
                    
            # Market regime
            logger.info(f"\n🌐 Market Regime: {self.market_regime}")
            logger.info(f"  VIX Level: {self.vix_level:.1f}")
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
            
    async def close_all_positions(self):
        """Close all option positions"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                if len(position.symbol) > 10:  # Options
                    qty = abs(int(position.qty))
                    side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
                    
                    order = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    self.trading_client.submit_order(order)
                    logger.info(f"Closed: {position.symbol}")
                    
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            
    # Additional helper methods
    async def get_upcoming_earnings(self) -> List[str]:
        """Get stocks with upcoming earnings"""
        # Placeholder - integrate with earnings calendar API
        return ['AAPL', 'MSFT', 'GOOGL']
        
    async def get_term_structure(self, symbol: str) -> Dict:
        """Get full term structure"""
        # Get multiple expirations
        expirations = {}
        
        for days in [7, 14, 21, 30, 45, 60]:
            chain = await self.get_option_chain(symbol, min_dte=days-3, max_dte=days+3)
            if chain['contracts']:
                exp_date = chain['contracts'][0].get('expiration_date')
                expirations[exp_date] = chain
                
        return expirations
        
    def ml_predict_strategy(self, features: np.ndarray) -> Dict:
        """ML model prediction (placeholder)"""
        # In production, use trained model
        strategies = ['iron_condor', 'bull_put_spread', 'calendar_spread']
        
        return {
            'strategy': np.random.choice(strategies),
            'confidence': np.random.uniform(0.6, 0.9)
        }
        
    async def dynamic_position_sizing(self):
        """Dynamic position sizing based on Kelly Criterion"""
        # Implement Kelly Criterion for optimal position sizing
        pass

async def main():
    """Main entry point"""
    bot = AdvancedOptionsAlgorithmsBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")