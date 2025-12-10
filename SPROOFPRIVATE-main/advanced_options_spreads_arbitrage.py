#!/usr/bin/env python3
"""
Advanced Options, Spreads & Arbitrage Trading System
===================================================
Implements:
- Black-Scholes option pricing
- Synthetic options strategies
- Spread trading (calendar, vertical, diagonal)
- Statistical arbitrage
- Momentum algorithms
- Greeks calculation and hedging
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set Alpaca credentials
os.environ['ALPACA_API_KEY'] = 'PKEP9PIBDKOSUGHHY44Z'
os.environ['ALPACA_SECRET_KEY'] = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class OptionsSignal:
    strategy: str
    underlying: str
    action: str
    strikes: Dict
    expiry: int  # days to expiry
    confidence: float
    expected_profit: float
    max_loss: float
    greeks: Optional[OptionGreeks]
    metadata: Dict

class AdvancedOptionsArbitrageSystem:
    """Advanced options and arbitrage trading system"""
    
    def __init__(self):
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY'],
            paper=True
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY']
        )
        
        # Trading universe
        self.option_underlyings = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']
        self.arbitrage_pairs = [
            ('GOOGL', 'GOOG'),  # Alphabet Class A vs C
            ('XOM', 'CVX'),     # Energy sector
            ('JPM', 'BAC'),     # Banking sector
            ('MSFT', 'AAPL'),   # Tech giants
        ]
        
        # Market parameters
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.trading_days = 252
        
        logger.info("✅ Advanced Options & Arbitrage System initialized")
    
    # ========== BLACK-SCHOLES MODEL ==========
    def black_scholes(self, S: float, K: float, T: float, r: float, 
                     sigma: float, option_type: str = 'call') -> float:
        """Calculate option price using Black-Scholes model"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> OptionGreeks:
        """Calculate option Greeks"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = -stats.norm.cdf(-d1)
        
        # Gamma
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (- (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
        else:
            theta = (- (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
        theta = theta / 365  # Convert to daily
        
        # Vega
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100  # Per 1% change in rate
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return OptionGreeks(delta, gamma, theta, vega, rho)
    
    def implied_volatility(self, option_price: float, S: float, K: float, 
                          T: float, r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        def objective(sigma):
            return self.black_scholes(S, K, T, r, sigma, option_type) - option_price
        
        def vega(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return S * stats.norm.pdf(d1) * np.sqrt(T)
        
        # Initial guess
        sigma = 0.3
        
        for _ in range(100):
            price_diff = objective(sigma)
            if abs(price_diff) < 1e-5:
                break
            
            v = vega(sigma)
            if v < 1e-10:
                break
                
            sigma = sigma - price_diff / v
            sigma = max(0.001, min(5, sigma))  # Keep within reasonable bounds
        
        return sigma
    
    # ========== DATA FETCHING ==========
    async def get_market_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Get market data with volatility calculation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            if symbol in bars.data:
                df = bars.data[symbol]
                data = pd.DataFrame({
                    'Open': [bar.open for bar in df],
                    'High': [bar.high for bar in df],
                    'Low': [bar.low for bar in df],
                    'Close': [bar.close for bar in df],
                    'Volume': [bar.volume for bar in df]
                }, index=[bar.timestamp for bar in df])
                
                # Calculate returns and volatility
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
                
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    # ========== OPTIONS STRATEGIES ==========
    async def find_options_opportunities(self) -> List[OptionsSignal]:
        """Find options trading opportunities"""
        signals = []
        
        for symbol in self.option_underlyings:
            try:
                data = await self.get_market_data(symbol)
                if len(data) < 30:
                    continue
                
                current_price = data['Close'].iloc[-1]
                volatility = data['Volatility'].iloc[-1]
                
                if pd.isna(volatility) or volatility <= 0:
                    volatility = 0.25  # Default volatility
                
                # 1. Iron Condor (high IV strategy)
                if volatility > 0.3:
                    signal = self._create_iron_condor(symbol, current_price, volatility)
                    if signal:
                        signals.append(signal)
                
                # 2. Bull Call Spread (momentum strategy)
                momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
                if momentum > 0.05:
                    signal = self._create_bull_call_spread(symbol, current_price, volatility)
                    if signal:
                        signals.append(signal)
                
                # 3. Calendar Spread (volatility play)
                vol_change = (volatility - data['Volatility'].rolling(20).mean().iloc[-1])
                if abs(vol_change) > 0.05:
                    signal = self._create_calendar_spread(symbol, current_price, volatility)
                    if signal:
                        signals.append(signal)
                
                # 4. Straddle (earnings/event play)
                if volatility > 0.4:
                    signal = self._create_straddle(symbol, current_price, volatility)
                    if signal:
                        signals.append(signal)
                
            except Exception as e:
                logger.error(f"Options analysis error for {symbol}: {e}")
        
        return signals
    
    def _create_iron_condor(self, symbol: str, price: float, volatility: float) -> Optional[OptionsSignal]:
        """Create iron condor signal"""
        try:
            # Calculate strikes
            otm_percent = 0.05  # 5% out of the money
            put_sell = round(price * (1 - otm_percent), 0)
            put_buy = round(price * (1 - otm_percent * 2), 0)
            call_sell = round(price * (1 + otm_percent), 0)
            call_buy = round(price * (1 + otm_percent * 2), 0)
            
            days_to_expiry = 30
            T = days_to_expiry / 365
            
            # Calculate option prices
            put_sell_price = self.black_scholes(price, put_sell, T, self.risk_free_rate, volatility, 'put')
            put_buy_price = self.black_scholes(price, put_buy, T, self.risk_free_rate, volatility, 'put')
            call_sell_price = self.black_scholes(price, call_sell, T, self.risk_free_rate, volatility, 'call')
            call_buy_price = self.black_scholes(price, call_buy, T, self.risk_free_rate, volatility, 'call')
            
            # Calculate P&L
            credit = put_sell_price - put_buy_price + call_sell_price - call_buy_price
            max_profit = credit * 100  # Per contract
            max_loss = ((put_sell - put_buy) - credit) * 100
            
            # Calculate net Greeks
            put_sell_greeks = self.calculate_greeks(price, put_sell, T, self.risk_free_rate, volatility, 'put')
            net_delta = -put_sell_greeks.delta  # Short put delta
            
            signal = OptionsSignal(
                strategy="Iron_Condor",
                underlying=symbol,
                action="SELL",
                strikes={
                    'put_sell': put_sell,
                    'put_buy': put_buy,
                    'call_sell': call_sell,
                    'call_buy': call_buy
                },
                expiry=days_to_expiry,
                confidence=0.75,
                expected_profit=max_profit * 0.7,  # 70% probability estimate
                max_loss=max_loss,
                greeks=OptionGreeks(
                    delta=net_delta,
                    gamma=-put_sell_greeks.gamma,
                    theta=put_sell_greeks.theta * 4,  # All 4 legs
                    vega=-put_sell_greeks.vega * 2,  # Net short vega
                    rho=put_sell_greeks.rho * 2
                ),
                metadata={
                    'volatility': volatility,
                    'credit': credit,
                    'breakeven_low': put_sell - credit,
                    'breakeven_high': call_sell + credit
                }
            )
            
            logger.info(f"📊 Iron Condor: {symbol} Credit: ${credit:.2f} Max Profit: ${max_profit:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Iron condor error for {symbol}: {e}")
            return None
    
    def _create_bull_call_spread(self, symbol: str, price: float, volatility: float) -> Optional[OptionsSignal]:
        """Create bull call spread signal"""
        try:
            # Calculate strikes
            long_strike = round(price * 0.98, 0)  # Slightly ITM
            short_strike = round(price * 1.05, 0)  # OTM
            
            days_to_expiry = 30
            T = days_to_expiry / 365
            
            # Calculate option prices
            long_call_price = self.black_scholes(price, long_strike, T, self.risk_free_rate, volatility, 'call')
            short_call_price = self.black_scholes(price, short_strike, T, self.risk_free_rate, volatility, 'call')
            
            # Calculate P&L
            debit = long_call_price - short_call_price
            max_profit = (short_strike - long_strike - debit) * 100
            max_loss = debit * 100
            
            # Calculate Greeks
            long_greeks = self.calculate_greeks(price, long_strike, T, self.risk_free_rate, volatility, 'call')
            short_greeks = self.calculate_greeks(price, short_strike, T, self.risk_free_rate, volatility, 'call')
            
            signal = OptionsSignal(
                strategy="Bull_Call_Spread",
                underlying=symbol,
                action="BUY",
                strikes={
                    'long_call': long_strike,
                    'short_call': short_strike
                },
                expiry=days_to_expiry,
                confidence=0.7,
                expected_profit=max_profit * 0.6,
                max_loss=max_loss,
                greeks=OptionGreeks(
                    delta=long_greeks.delta - short_greeks.delta,
                    gamma=long_greeks.gamma - short_greeks.gamma,
                    theta=long_greeks.theta - short_greeks.theta,
                    vega=long_greeks.vega - short_greeks.vega,
                    rho=long_greeks.rho - short_greeks.rho
                ),
                metadata={
                    'volatility': volatility,
                    'debit': debit,
                    'breakeven': long_strike + debit
                }
            )
            
            logger.info(f"📊 Bull Call Spread: {symbol} Debit: ${debit:.2f} Max Profit: ${max_profit:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Bull call spread error for {symbol}: {e}")
            return None
    
    def _create_calendar_spread(self, symbol: str, price: float, volatility: float) -> Optional[OptionsSignal]:
        """Create calendar spread signal"""
        try:
            # ATM strike
            strike = round(price, 0)
            
            # Different expiries
            near_expiry = 30
            far_expiry = 60
            T1 = near_expiry / 365
            T2 = far_expiry / 365
            
            # Calculate option prices
            near_call = self.black_scholes(price, strike, T1, self.risk_free_rate, volatility, 'call')
            far_call = self.black_scholes(price, strike, T2, self.risk_free_rate, volatility, 'call')
            
            # Calculate P&L
            debit = far_call - near_call
            
            # Expected profit from theta decay
            near_greeks = self.calculate_greeks(price, strike, T1, self.risk_free_rate, volatility, 'call')
            far_greeks = self.calculate_greeks(price, strike, T2, self.risk_free_rate, volatility, 'call')
            
            daily_theta = near_greeks.theta - far_greeks.theta
            expected_profit = daily_theta * 20 * 100  # 20 days of theta
            
            signal = OptionsSignal(
                strategy="Calendar_Spread",
                underlying=symbol,
                action="BUY",
                strikes={
                    'strike': strike,
                    'near_expiry': near_expiry,
                    'far_expiry': far_expiry
                },
                expiry=near_expiry,
                confidence=0.65,
                expected_profit=expected_profit,
                max_loss=debit * 100,
                greeks=OptionGreeks(
                    delta=far_greeks.delta - near_greeks.delta,
                    gamma=far_greeks.gamma - near_greeks.gamma,
                    theta=daily_theta,
                    vega=far_greeks.vega - near_greeks.vega,
                    rho=far_greeks.rho - near_greeks.rho
                ),
                metadata={
                    'volatility': volatility,
                    'debit': debit,
                    'daily_theta': daily_theta
                }
            )
            
            logger.info(f"📊 Calendar Spread: {symbol} Debit: ${debit:.2f} Daily Theta: ${daily_theta:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Calendar spread error for {symbol}: {e}")
            return None
    
    def _create_straddle(self, symbol: str, price: float, volatility: float) -> Optional[OptionsSignal]:
        """Create straddle signal for high volatility"""
        try:
            # ATM strike
            strike = round(price, 0)
            
            days_to_expiry = 30
            T = days_to_expiry / 365
            
            # Calculate option prices
            call_price = self.black_scholes(price, strike, T, self.risk_free_rate, volatility, 'call')
            put_price = self.black_scholes(price, strike, T, self.risk_free_rate, volatility, 'put')
            
            # Calculate P&L
            debit = (call_price + put_price) * 100
            
            # Breakeven points
            breakeven_up = strike + call_price + put_price
            breakeven_down = strike - call_price - put_price
            
            # Expected move based on volatility
            expected_move = price * volatility * np.sqrt(T)
            
            # Calculate Greeks
            call_greeks = self.calculate_greeks(price, strike, T, self.risk_free_rate, volatility, 'call')
            put_greeks = self.calculate_greeks(price, strike, T, self.risk_free_rate, volatility, 'put')
            
            signal = OptionsSignal(
                strategy="Long_Straddle",
                underlying=symbol,
                action="BUY",
                strikes={
                    'strike': strike
                },
                expiry=days_to_expiry,
                confidence=0.6,
                expected_profit=expected_move * 100 - debit,
                max_loss=debit,
                greeks=OptionGreeks(
                    delta=call_greeks.delta + put_greeks.delta,
                    gamma=call_greeks.gamma + put_greeks.gamma,
                    theta=call_greeks.theta + put_greeks.theta,
                    vega=call_greeks.vega + put_greeks.vega,
                    rho=call_greeks.rho + put_greeks.rho
                ),
                metadata={
                    'volatility': volatility,
                    'debit': debit,
                    'breakeven_up': breakeven_up,
                    'breakeven_down': breakeven_down,
                    'expected_move': expected_move
                }
            )
            
            logger.info(f"📊 Long Straddle: {symbol} Debit: ${debit:.2f} Expected Move: ${expected_move:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Straddle error for {symbol}: {e}")
            return None
    
    # ========== ARBITRAGE STRATEGIES ==========
    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """Find statistical arbitrage opportunities"""
        opportunities = []
        
        for pair in self.arbitrage_pairs:
            try:
                # Get data for both symbols
                data1 = await self.get_market_data(pair[0], days=90)
                data2 = await self.get_market_data(pair[1], days=90)
                
                if len(data1) < 60 or len(data2) < 60:
                    continue
                
                # Align data
                common_dates = data1.index.intersection(data2.index)
                if len(common_dates) < 60:
                    continue
                
                prices1 = data1.loc[common_dates, 'Close']
                prices2 = data2.loc[common_dates, 'Close']
                
                # Calculate spread
                spread = self._calculate_spread(prices1, prices2)
                
                # Check for trading opportunity
                signal = self._analyze_spread(spread, pair, prices1.iloc[-1], prices2.iloc[-1])
                if signal:
                    opportunities.append(signal)
                    
            except Exception as e:
                logger.error(f"Arbitrage analysis error for {pair}: {e}")
        
        return opportunities
    
    def _calculate_spread(self, prices1: pd.Series, prices2: pd.Series) -> pd.DataFrame:
        """Calculate spread metrics"""
        # Calculate hedge ratio using rolling regression
        window = 30
        
        spread_data = pd.DataFrame(index=prices1.index)
        spread_data['price1'] = prices1
        spread_data['price2'] = prices2
        
        # Rolling hedge ratio
        spread_data['hedge_ratio'] = prices1.rolling(window).apply(
            lambda x: np.polyfit(prices2[-len(x):], x, 1)[0]
        )
        
        # Calculate spread
        spread_data['spread'] = prices1 - spread_data['hedge_ratio'] * prices2
        
        # Calculate z-score
        spread_data['spread_mean'] = spread_data['spread'].rolling(window).mean()
        spread_data['spread_std'] = spread_data['spread'].rolling(window).std()
        spread_data['z_score'] = (spread_data['spread'] - spread_data['spread_mean']) / spread_data['spread_std']
        
        return spread_data
    
    def _analyze_spread(self, spread_data: pd.DataFrame, pair: Tuple[str, str], 
                       price1: float, price2: float) -> Optional[Dict]:
        """Analyze spread for trading signals"""
        if len(spread_data) < 30:
            return None
        
        current_z = spread_data['z_score'].iloc[-1]
        hedge_ratio = spread_data['hedge_ratio'].iloc[-1]
        
        if pd.isna(current_z) or pd.isna(hedge_ratio):
            return None
        
        # Generate signal based on z-score
        if abs(current_z) > 2:
            # Calculate position sizes
            position1_size = 10000  # $10k base position
            position2_size = position1_size * hedge_ratio
            
            if current_z > 2:
                # Spread too high - sell spread
                action = "SELL_SPREAD"
                trades = [
                    {"symbol": pair[0], "action": "SELL", "qty": int(position1_size / price1)},
                    {"symbol": pair[1], "action": "BUY", "qty": int(position2_size / price2)}
                ]
            else:
                # Spread too low - buy spread
                action = "BUY_SPREAD"
                trades = [
                    {"symbol": pair[0], "action": "BUY", "qty": int(position1_size / price1)},
                    {"symbol": pair[1], "action": "SELL", "qty": int(position2_size / price2)}
                ]
            
            # Calculate expected profit
            mean_reversion_target = spread_data['spread_mean'].iloc[-1]
            current_spread = spread_data['spread'].iloc[-1]
            expected_profit = abs(current_spread - mean_reversion_target) * int(position1_size / price1)
            
            signal = {
                'type': 'STATISTICAL_ARBITRAGE',
                'pair': f"{pair[0]}/{pair[1]}",
                'action': action,
                'z_score': current_z,
                'hedge_ratio': hedge_ratio,
                'expected_profit': expected_profit,
                'confidence': min(0.85, abs(current_z) / 3),
                'trades': trades,
                'metadata': {
                    'spread': current_spread,
                    'spread_mean': mean_reversion_target,
                    'spread_std': spread_data['spread_std'].iloc[-1],
                    'prices': {pair[0]: price1, pair[1]: price2}
                }
            }
            
            logger.info(f"🎯 Arbitrage: {pair[0]}/{pair[1]} Z-score: {current_z:.2f} Action: {action}")
            return signal
        
        return None
    
    # ========== MOMENTUM ALGORITHM ==========
    async def momentum_algorithm(self) -> List[Dict]:
        """Advanced momentum trading algorithm"""
        signals = []
        
        for symbol in self.option_underlyings:
            try:
                data = await self.get_market_data(symbol, days=90)
                if len(data) < 60:
                    continue
                
                # Calculate multiple momentum indicators
                momentum_scores = {}
                
                # 1. Price momentum
                momentum_scores['price_5d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                momentum_scores['price_20d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
                momentum_scores['price_60d'] = (data['Close'].iloc[-1] - data['Close'].iloc[-60]) / data['Close'].iloc[-60]
                
                # 2. Volume momentum
                vol_ratio = data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:].mean()
                momentum_scores['volume'] = vol_ratio - 1
                
                # 3. Volatility momentum
                recent_vol = data['Volatility'].iloc[-5:].mean()
                hist_vol = data['Volatility'].iloc[-20:].mean()
                momentum_scores['volatility'] = (recent_vol - hist_vol) / hist_vol if hist_vol > 0 else 0
                
                # 4. RSI momentum
                rsi = self._calculate_rsi(data['Close'])
                momentum_scores['rsi'] = (rsi - 50) / 50
                
                # Composite score
                weights = {'price_5d': 0.3, 'price_20d': 0.3, 'price_60d': 0.2, 
                          'volume': 0.1, 'volatility': 0.05, 'rsi': 0.05}
                
                composite_score = sum(momentum_scores[k] * weights[k] for k in weights)
                
                # Generate signal
                if abs(composite_score) > 0.05:
                    signal = {
                        'type': 'MOMENTUM',
                        'symbol': symbol,
                        'action': 'BUY' if composite_score > 0 else 'SELL',
                        'score': composite_score,
                        'confidence': min(0.85, abs(composite_score) * 5),
                        'expected_return': composite_score * 0.5,  # Conservative estimate
                        'metadata': momentum_scores
                    }
                    
                    signals.append(signal)
                    logger.info(f"📈 Momentum: {symbol} Score: {composite_score:.3f} Action: {signal['action']}")
                    
            except Exception as e:
                logger.error(f"Momentum algorithm error for {symbol}: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ========== EXECUTION ==========
    async def execute_strategies(self, options_signals: List[OptionsSignal], 
                               arb_signals: List[Dict], momentum_signals: List[Dict]):
        """Execute trading strategies"""
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        
        logger.info(f"\n💰 Buying Power: ${buying_power:,.2f}")
        
        # Since Alpaca doesn't support options, we'll create synthetic positions
        executed = 0
        
        # Execute momentum signals (these we can actually trade)
        for signal in momentum_signals[:3]:  # Top 3
            try:
                symbol = signal['symbol']
                
                # Get current price
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self.data_client.get_stock_latest_quote(request)
                
                if symbol not in quote:
                    continue
                
                current_price = float(quote[symbol].ask_price)
                
                # Position sizing based on Kelly Criterion
                kelly_fraction = signal['confidence'] * signal['expected_return'] / 0.02  # Assume 2% risk
                position_size = min(buying_power * 0.1 * kelly_fraction, 10000)  # Max 10% or $10k
                shares = int(position_size / current_price)
                
                if shares < 1:
                    continue
                
                logger.info(f"\n📈 Executing Momentum Trade:")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Action: {signal['action']}")
                logger.info(f"   Shares: {shares} @ ${current_price:.2f}")
                logger.info(f"   Confidence: {signal['confidence']:.2%}")
                
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY if signal['action'] == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_data)
                logger.info(f"✅ Order placed: {order.id}")
                
                executed += 1
                buying_power -= position_size
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Execution error for {signal['symbol']}: {e}")
        
        # Log options signals (can't execute directly)
        logger.info(f"\n📊 Options Signals (Synthetic Positions):")
        for signal in options_signals[:3]:
            logger.info(f"\n   Strategy: {signal.strategy}")
            logger.info(f"   Underlying: {signal.underlying}")
            logger.info(f"   Expected Profit: ${signal.expected_profit:.2f}")
            logger.info(f"   Max Loss: ${signal.max_loss:.2f}")
            logger.info(f"   Greeks - Delta: {signal.greeks.delta:.3f}, Theta: ${signal.greeks.theta:.2f}/day")
        
        # Log arbitrage signals
        logger.info(f"\n📊 Arbitrage Opportunities:")
        for signal in arb_signals[:3]:
            logger.info(f"\n   Pair: {signal['pair']}")
            logger.info(f"   Z-Score: {signal['z_score']:.2f}")
            logger.info(f"   Expected Profit: ${signal['expected_profit']:.2f}")
        
        logger.info(f"\n📊 Total Executed: {executed} trades")
    
    async def run(self):
        """Main execution"""
        logger.info("="*80)
        logger.info("ADVANCED OPTIONS, SPREADS & ARBITRAGE SYSTEM")
        logger.info("="*80)
        
        # Check account
        account = self.trading_client.get_account()
        logger.info(f"\n💰 Account Status:")
        logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        # Run all strategies
        logger.info(f"\n🔍 Analyzing Markets...")
        
        # 1. Options strategies
        options_signals = await self.find_options_opportunities()
        logger.info(f"\n📊 Options Signals: {len(options_signals)}")
        
        # 2. Arbitrage opportunities
        arb_signals = await self.find_arbitrage_opportunities()
        logger.info(f"📊 Arbitrage Opportunities: {len(arb_signals)}")
        
        # 3. Momentum algorithm
        momentum_signals = await self.momentum_algorithm()
        logger.info(f"📊 Momentum Signals: {len(momentum_signals)}")
        
        # Execute strategies
        await self.execute_strategies(options_signals, arb_signals, momentum_signals)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'account_value': float(account.portfolio_value),
            'signals': {
                'options': [
                    {
                        'strategy': s.strategy,
                        'underlying': s.underlying,
                        'strikes': s.strikes,
                        'expected_profit': s.expected_profit,
                        'max_loss': s.max_loss,
                        'confidence': s.confidence,
                        'greeks': {
                            'delta': s.greeks.delta,
                            'gamma': s.greeks.gamma,
                            'theta': s.greeks.theta,
                            'vega': s.greeks.vega,
                            'rho': s.greeks.rho
                        } if s.greeks else None
                    }
                    for s in options_signals
                ],
                'arbitrage': arb_signals,
                'momentum': momentum_signals
            }
        }
        
        with open('advanced_options_arbitrage_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Report saved to: advanced_options_arbitrage_report.json")
        logger.info("\n" + "="*80)
        logger.info("SYSTEM COMPLETE")
        logger.info("="*80)


async def main():
    """Entry point"""
    system = AdvancedOptionsArbitrageSystem()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())