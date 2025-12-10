#!/usr/bin/env python3
"""
Advanced Options Opportunity Scanner
Finds opportunities across all option positions and complex spreads
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import requests
from scipy.stats import norm
from scipy.optimize import minimize
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Import our modules
from advanced_ensemble_options_ai import AdvancedGreeksCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedOpportunityScanner:
    """Scan for opportunities across all option strategies"""
    
    def __init__(self, paper=True):
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
        
        # Greeks calculator
        self.greeks_calc = AdvancedGreeksCalculator()
        
        # Universe
        self.symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'META', 
                       'MSFT', 'AMD', 'AMZN', 'GOOGL', 'NFLX', 'DIS', 'BA', 
                       'JPM', 'GS', 'XOM', 'CVX', 'GLD', 'TLT']
        
        # Strategy definitions
        self.strategies = {
            # Volatility strategies
            'long_straddle': self.scan_long_straddle,
            'long_strangle': self.scan_long_strangle,
            'short_straddle': self.scan_short_straddle,
            'short_strangle': self.scan_short_strangle,
            
            # Directional spreads
            'bull_call_spread': self.scan_bull_call_spread,
            'bear_put_spread': self.scan_bear_put_spread,
            'bull_put_spread': self.scan_bull_put_spread,
            'bear_call_spread': self.scan_bear_call_spread,
            
            # Neutral strategies
            'iron_condor': self.scan_iron_condor,
            'iron_butterfly': self.scan_iron_butterfly,
            'short_butterfly': self.scan_short_butterfly,
            'long_butterfly': self.scan_long_butterfly,
            
            # Time spreads
            'calendar_spread': self.scan_calendar_spread,
            'diagonal_spread': self.scan_diagonal_spread,
            'double_calendar': self.scan_double_calendar,
            'double_diagonal': self.scan_double_diagonal,
            
            # Advanced strategies
            'jade_lizard': self.scan_jade_lizard,
            'twisted_sister': self.scan_twisted_sister,
            'broken_wing_butterfly': self.scan_broken_wing_butterfly,
            'ratio_spread': self.scan_ratio_spread,
            'backspread': self.scan_backspread,
            'christmas_tree': self.scan_christmas_tree,
            
            # Arbitrage opportunities
            'box_spread': self.scan_box_spread,
            'conversion_reversal': self.scan_conversion_reversal,
            'jelly_roll': self.scan_jelly_roll,
            'synthetic_arbitrage': self.scan_synthetic_arbitrage,
            
            # Earnings plays
            'earnings_straddle': self.scan_earnings_straddle,
            'earnings_condor': self.scan_earnings_condor,
            
            # Dividend strategies
            'dividend_capture': self.scan_dividend_capture,
            'dividend_arbitrage': self.scan_dividend_arbitrage
        }
        
        # Cache
        self.options_chain_cache = {}
        self.market_data_cache = {}
        self.iv_rank_cache = {}
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def scan_all_opportunities(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Scan all symbols and strategies for opportunities"""
        
        logger.info("🔍 SCANNING FOR OPTIONS OPPORTUNITIES")
        logger.info("=" * 70)
        
        # Default filters
        if filters is None:
            filters = {
                'min_volume': 100,
                'min_open_interest': 50,
                'max_spread_pct': 0.10,  # 10% of mid price
                'min_profit_probability': 0.50,
                'min_expected_return': 0.10,  # 10%
                'max_days_to_expiry': 90,
                'min_iv_rank': 20,  # For volatility strategies
                'max_iv_rank': 80   # For theta strategies
            }
        
        all_opportunities = []
        
        # Get market data for all symbols
        logger.info("📊 Fetching market data...")
        self._update_market_data()
        
        # Scan each symbol
        for symbol in self.symbols:
            logger.info(f"\n🔍 Scanning {symbol}...")
            
            try:
                # Get options chain
                chain = self._get_options_chain(symbol)
                if not chain:
                    continue
                
                # Get current market conditions
                conditions = self._analyze_market_conditions(symbol)
                
                # Scan each strategy
                symbol_opportunities = []
                
                for strategy_name, scanner_func in self.strategies.items():
                    try:
                        opportunities = scanner_func(symbol, chain, conditions, filters)
                        
                        for opp in opportunities:
                            opp['symbol'] = symbol
                            opp['strategy'] = strategy_name
                            opp['timestamp'] = datetime.now().isoformat()
                            
                        symbol_opportunities.extend(opportunities)
                        
                    except Exception as e:
                        logger.error(f"Error scanning {strategy_name} for {symbol}: {e}")
                
                # Rank opportunities for this symbol
                ranked = self._rank_opportunities(symbol_opportunities)
                all_opportunities.extend(ranked[:10])  # Top 10 per symbol
                
                # Log summary
                if ranked:
                    logger.info(f"  Found {len(ranked)} opportunities")
                    logger.info(f"  Best: {ranked[0]['strategy']} - Score: {ranked[0]['score']:.2f}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Final ranking across all symbols
        all_opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Display top opportunities
        self._display_top_opportunities(all_opportunities[:20])
        
        return all_opportunities
    
    def _get_options_chain(self, symbol: str) -> Optional[Dict]:
        """Get full options chain for symbol"""
        
        if symbol in self.options_chain_cache:
            return self.options_chain_cache[symbol]
        
        try:
            # Get current price
            price = self.market_data_cache[symbol]['price']
            
            # Get options for multiple expirations
            chain = {'puts': [], 'calls': [], 'expirations': set()}
            
            # Look for options 0-180 days out
            today = datetime.now().date()
            
            for days in [7, 14, 21, 30, 45, 60, 90, 120, 150, 180]:
                exp_date = today + timedelta(days=days)
                exp_min = (exp_date - timedelta(days=3)).strftime('%Y-%m-%d')
                exp_max = (exp_date + timedelta(days=3)).strftime('%Y-%m-%d')
                
                # Define strike range
                strike_min = price * 0.70
                strike_max = price * 1.30
                
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
                    
                    for contract in contracts:
                        # Add additional data
                        contract['days_to_expiry'] = (
                            datetime.strptime(contract['expiration_date'], '%Y-%m-%d').date() - today
                        ).days
                        
                        # Calculate implied volatility (simplified)
                        contract['implied_volatility'] = self._estimate_iv(contract, price)
                        
                        # Categorize
                        if contract['type'] == 'put':
                            chain['puts'].append(contract)
                        else:
                            chain['calls'].append(contract)
                            
                        chain['expirations'].add(contract['expiration_date'])
            
            # Sort by expiration and strike
            chain['puts'].sort(key=lambda x: (x['expiration_date'], x['strike_price']))
            chain['calls'].sort(key=lambda x: (x['expiration_date'], x['strike_price']))
            chain['expirations'] = sorted(list(chain['expirations']))
            
            # Cache it
            self.options_chain_cache[symbol] = chain
            
            return chain
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return None
    
    def _analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions for symbol"""
        
        conditions = {
            'price': self.market_data_cache[symbol]['price'],
            'trend': 'neutral',
            'volatility': 0.25,
            'iv_rank': 50,
            'iv_percentile': 50,
            'volume_ratio': 1.0,
            'put_call_ratio': 1.0,
            'technical_signal': 'neutral',
            'support_levels': [],
            'resistance_levels': [],
            'earnings_date': None,
            'dividend_date': None
        }
        
        try:
            # Get historical data
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=100
                )
            )
            
            if symbol in bars.df.index:
                df = bars.df.loc[symbol]
                
                # Calculate trend
                sma20 = df['close'].rolling(20).mean().iloc[-1]
                sma50 = df['close'].rolling(50).mean().iloc[-1]
                current = df['close'].iloc[-1]
                
                if current > sma20 > sma50:
                    conditions['trend'] = 'bullish'
                elif current < sma20 < sma50:
                    conditions['trend'] = 'bearish'
                
                # Calculate volatility
                returns = df['close'].pct_change()
                conditions['volatility'] = returns.std() * np.sqrt(252)
                
                # Volume analysis
                conditions['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
                # Support/Resistance (simplified)
                conditions['support_levels'] = [
                    df['low'].rolling(20).min().iloc[-1],
                    df['low'].rolling(50).min().iloc[-1]
                ]
                conditions['resistance_levels'] = [
                    df['high'].rolling(20).max().iloc[-1],
                    df['high'].rolling(50).max().iloc[-1]
                ]
                
                # IV Rank (if we have historical IV data)
                if symbol in self.iv_rank_cache:
                    conditions['iv_rank'] = self.iv_rank_cache[symbol]
                else:
                    # Estimate based on current volatility vs historical
                    hist_vol = returns.rolling(252).std() * np.sqrt(252)
                    percentile = (hist_vol < conditions['volatility']).sum() / len(hist_vol) * 100
                    conditions['iv_rank'] = percentile
                    self.iv_rank_cache[symbol] = percentile
                
        except Exception as e:
            logger.error(f"Error analyzing conditions for {symbol}: {e}")
        
        return conditions
    
    # Strategy scanners
    
    def scan_iron_condor(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for iron condor opportunities"""
        
        opportunities = []
        price = conditions['price']
        
        # Group by expiration
        expirations = {}
        for contract in chain['puts'] + chain['calls']:
            exp = contract['expiration_date']
            if exp not in expirations:
                expirations[exp] = {'puts': [], 'calls': []}
            
            if contract['type'] == 'put':
                expirations[exp]['puts'].append(contract)
            else:
                expirations[exp]['calls'].append(contract)
        
        # Check each expiration
        for exp_date, contracts in expirations.items():
            if len(contracts['puts']) < 4 or len(contracts['calls']) < 4:
                continue
                
            # Days to expiry
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            if dte < 20 or dte > 60:  # Ideal range for iron condors
                continue
            
            # Find optimal strikes
            # Target: 15-20 delta short strikes
            put_short_strike = None
            put_long_strike = None
            call_short_strike = None
            call_long_strike = None
            
            # Find put strikes
            for put in sorted(contracts['puts'], key=lambda x: x['strike_price'], reverse=True):
                if put['strike_price'] < price * 0.95:  # OTM puts
                    delta = self._calculate_delta(put, price, dte)
                    if abs(delta) < 0.20 and put_short_strike is None:
                        put_short_strike = put
                    elif put_short_strike and put['strike_price'] < put_short_strike['strike_price'] - 5:
                        put_long_strike = put
                        break
            
            # Find call strikes
            for call in sorted(contracts['calls'], key=lambda x: x['strike_price']):
                if call['strike_price'] > price * 1.05:  # OTM calls
                    delta = self._calculate_delta(call, price, dte)
                    if delta < 0.20 and call_short_strike is None:
                        call_short_strike = call
                    elif call_short_strike and call['strike_price'] > call_short_strike['strike_price'] + 5:
                        call_long_strike = call
                        break
            
            # Validate we have all legs
            if not all([put_long_strike, put_short_strike, call_short_strike, call_long_strike]):
                continue
            
            # Calculate metrics
            setup = {
                'put_long': put_long_strike,
                'put_short': put_short_strike,
                'call_short': call_short_strike,
                'call_long': call_long_strike
            }
            
            metrics = self._calculate_iron_condor_metrics(setup, price, dte, conditions)
            
            # Apply filters
            if (metrics['profit_probability'] >= filters['min_profit_probability'] and
                metrics['expected_return'] >= filters['min_expected_return'] and
                conditions['iv_rank'] >= filters['min_iv_rank']):
                
                opportunities.append({
                    'setup': setup,
                    'metrics': metrics,
                    'expiration': exp_date,
                    'days_to_expiry': dte,
                    'score': self._score_opportunity(metrics, conditions)
                })
        
        return opportunities
    
    def scan_calendar_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for calendar spread opportunities"""
        
        opportunities = []
        price = conditions['price']
        
        # Group by strike
        strikes = defaultdict(list)
        for contract in chain['calls'] + chain['puts']:
            strikes[contract['strike_price']].append(contract)
        
        # Look for calendar opportunities
        for strike, contracts in strikes.items():
            # Skip if too far from current price
            if abs(strike - price) / price > 0.10:
                continue
            
            # Group by type and expiration
            calls = [c for c in contracts if c['type'] == 'call']
            puts = [c for c in contracts if c['type'] == 'put']
            
            # Check calls
            if len(calls) >= 2:
                calls_sorted = sorted(calls, key=lambda x: x['days_to_expiry'])
                
                for i in range(len(calls_sorted) - 1):
                    near = calls_sorted[i]
                    far = calls_sorted[i + 1]
                    
                    # Check criteria
                    if (near['days_to_expiry'] >= 20 and 
                        far['days_to_expiry'] - near['days_to_expiry'] >= 30):
                        
                        metrics = self._calculate_calendar_metrics(near, far, price, conditions)
                        
                        if metrics['expected_return'] >= filters['min_expected_return']:
                            opportunities.append({
                                'type': 'call_calendar',
                                'strike': strike,
                                'near': near,
                                'far': far,
                                'metrics': metrics,
                                'score': self._score_opportunity(metrics, conditions)
                            })
            
            # Check puts (similar logic)
            if len(puts) >= 2:
                puts_sorted = sorted(puts, key=lambda x: x['days_to_expiry'])
                
                for i in range(len(puts_sorted) - 1):
                    near = puts_sorted[i]
                    far = puts_sorted[i + 1]
                    
                    if (near['days_to_expiry'] >= 20 and 
                        far['days_to_expiry'] - near['days_to_expiry'] >= 30):
                        
                        metrics = self._calculate_calendar_metrics(near, far, price, conditions)
                        
                        if metrics['expected_return'] >= filters['min_expected_return']:
                            opportunities.append({
                                'type': 'put_calendar',
                                'strike': strike,
                                'near': near,
                                'far': far,
                                'metrics': metrics,
                                'score': self._score_opportunity(metrics, conditions)
                            })
        
        return opportunities
    
    def scan_diagonal_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for diagonal spread opportunities"""
        
        opportunities = []
        price = conditions['price']
        
        # Diagonal = calendar + vertical
        # Buy far expiry, sell near expiry at different strikes
        
        # Group by expiration
        exp_groups = defaultdict(list)
        for contract in chain['calls'] + chain['puts']:
            exp_groups[contract['expiration_date']].append(contract)
        
        expirations = sorted(exp_groups.keys())
        
        for i in range(len(expirations) - 1):
            near_exp = expirations[i]
            far_exp = expirations[i + 1]
            
            near_contracts = exp_groups[near_exp]
            far_contracts = exp_groups[far_exp]
            
            # Bullish diagonal (calls)
            near_calls = [c for c in near_contracts if c['type'] == 'call']
            far_calls = [c for c in far_contracts if c['type'] == 'call']
            
            for far_call in far_calls:
                # Buy lower strike far expiry
                if far_call['strike_price'] < price * 1.05:
                    
                    for near_call in near_calls:
                        # Sell higher strike near expiry
                        if (near_call['strike_price'] > far_call['strike_price'] and
                            near_call['strike_price'] < price * 1.10):
                            
                            metrics = self._calculate_diagonal_metrics(
                                near_call, far_call, price, conditions, 'bullish'
                            )
                            
                            if metrics['expected_return'] >= filters['min_expected_return']:
                                opportunities.append({
                                    'type': 'bullish_diagonal',
                                    'buy': far_call,
                                    'sell': near_call,
                                    'metrics': metrics,
                                    'score': self._score_opportunity(metrics, conditions)
                                })
        
        return opportunities
    
    def scan_jade_lizard(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for jade lizard opportunities (short call + bull put spread)"""
        
        opportunities = []
        price = conditions['price']
        
        # Jade lizard works best in high IV environments
        if conditions['iv_rank'] < 50:
            return opportunities
        
        # Group by expiration
        exp_groups = defaultdict(lambda: {'puts': [], 'calls': []})
        for contract in chain['puts'] + chain['calls']:
            exp = contract['expiration_date']
            if contract['type'] == 'put':
                exp_groups[exp]['puts'].append(contract)
            else:
                exp_groups[exp]['calls'].append(contract)
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            if dte < 25 or dte > 60:
                continue
            
            puts = sorted(contracts['puts'], key=lambda x: x['strike_price'])
            calls = sorted(contracts['calls'], key=lambda x: x['strike_price'])
            
            if len(puts) < 2 or len(calls) < 1:
                continue
            
            # Find short call (10-20 delta)
            short_call = None
            for call in calls:
                if call['strike_price'] > price * 1.05:
                    delta = self._calculate_delta(call, price, dte)
                    if 0.10 <= delta <= 0.20:
                        short_call = call
                        break
            
            if not short_call:
                continue
            
            # Find bull put spread
            short_put = None
            long_put = None
            
            for put in reversed(puts):
                if put['strike_price'] < price * 0.95:
                    delta = self._calculate_delta(put, price, dte)
                    if abs(delta) <= 0.30 and not short_put:
                        short_put = put
                    elif short_put and put['strike_price'] < short_put['strike_price'] - 5:
                        long_put = put
                        break
            
            if not short_put or not long_put:
                continue
            
            # Calculate metrics
            setup = {
                'short_call': short_call,
                'short_put': short_put,
                'long_put': long_put
            }
            
            metrics = self._calculate_jade_lizard_metrics(setup, price, dte, conditions)
            
            # Jade lizard should collect more credit than put spread width
            if (metrics['no_upside_risk'] and
                metrics['expected_return'] >= filters['min_expected_return']):
                
                opportunities.append({
                    'setup': setup,
                    'metrics': metrics,
                    'expiration': exp_date,
                    'days_to_expiry': dte,
                    'score': self._score_opportunity(metrics, conditions)
                })
        
        return opportunities
    
    def scan_broken_wing_butterfly(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for broken wing butterfly opportunities"""
        
        opportunities = []
        price = conditions['price']
        
        # BWB is an asymmetric butterfly with different wing widths
        # Used for directional bias with limited risk
        
        exp_groups = defaultdict(lambda: {'puts': [], 'calls': []})
        for contract in chain['puts'] + chain['calls']:
            exp = contract['expiration_date']
            if contract['type'] == 'put':
                exp_groups[exp]['puts'].append(contract)
            else:
                exp_groups[exp]['calls'].append(contract)
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            if dte < 30 or dte > 60:
                continue
            
            # Bullish BWB with calls
            calls = sorted(contracts['calls'], key=lambda x: x['strike_price'])
            
            if len(calls) >= 3:
                # Find ATM strike
                atm_idx = min(range(len(calls)), key=lambda i: abs(calls[i]['strike_price'] - price))
                
                if atm_idx > 0 and atm_idx < len(calls) - 2:
                    # Standard butterfly: long lower, short 2x middle, long upper
                    # Broken wing: upper strike is further away
                    
                    lower = calls[atm_idx - 1]
                    middle = calls[atm_idx]
                    
                    # Try different upper strikes for broken wing
                    for upper_idx in range(atm_idx + 1, min(atm_idx + 4, len(calls))):
                        upper = calls[upper_idx]
                        
                        setup = {
                            'long_lower': lower,
                            'short_middle': middle,  # x2
                            'long_upper': upper
                        }
                        
                        metrics = self._calculate_bwb_metrics(setup, price, dte, conditions, 'bullish')
                        
                        if (metrics['risk_reward_ratio'] > 2 and
                            metrics['profit_probability'] >= filters['min_profit_probability']):
                            
                            opportunities.append({
                                'type': 'bullish_bwb',
                                'setup': setup,
                                'metrics': metrics,
                                'expiration': exp_date,
                                'score': self._score_opportunity(metrics, conditions)
                            })
        
        return opportunities
    
    def scan_ratio_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for ratio spread opportunities"""
        
        opportunities = []
        price = conditions['price']
        
        # Ratio spreads involve selling more options than buying
        # High profit potential but unlimited risk on one side
        
        exp_groups = defaultdict(list)
        for contract in chain['calls'] + chain['puts']:
            exp_groups[contract['expiration_date']].append(contract)
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            if dte < 30 or dte > 90:
                continue
            
            calls = sorted([c for c in contracts if c['type'] == 'call'], key=lambda x: x['strike_price'])
            puts = sorted([c for c in contracts if c['type'] == 'put'], key=lambda x: x['strike_price'])
            
            # Call ratio spread (1x2)
            for i in range(len(calls) - 1):
                long_call = calls[i]
                short_call = calls[i + 1]
                
                if (long_call['strike_price'] < price and
                    short_call['strike_price'] > price * 1.02):
                    
                    # Calculate metrics for 1x2 ratio
                    metrics = self._calculate_ratio_metrics(
                        long_call, short_call, price, dte, conditions, 'call', ratio=2
                    )
                    
                    if metrics['max_profit'] / metrics['margin_required'] > 0.20:
                        opportunities.append({
                            'type': 'call_ratio_1x2',
                            'buy': long_call,
                            'sell': short_call,
                            'ratio': '1x2',
                            'metrics': metrics,
                            'score': self._score_opportunity(metrics, conditions)
                        })
        
        return opportunities
    
    def scan_box_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for box spread arbitrage opportunities"""
        
        opportunities = []
        
        # Box spread = bull call spread + bear put spread at same strikes
        # Should theoretically equal the difference in strikes
        
        exp_groups = defaultdict(lambda: {'puts': {}, 'calls': {}})
        
        for contract in chain['puts'] + chain['calls']:
            exp = contract['expiration_date']
            strike = contract['strike_price']
            
            if contract['type'] == 'put':
                exp_groups[exp]['puts'][strike] = contract
            else:
                exp_groups[exp]['calls'][strike] = contract
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            # Box spreads work best with longer expiration
            if dte < 60:
                continue
            
            strikes = sorted(set(contracts['puts'].keys()) & set(contracts['calls'].keys()))
            
            # Need at least 2 strikes with both calls and puts
            for i in range(len(strikes) - 1):
                lower_strike = strikes[i]
                upper_strike = strikes[i + 1]
                
                if (lower_strike in contracts['calls'] and upper_strike in contracts['calls'] and
                    lower_strike in contracts['puts'] and upper_strike in contracts['puts']):
                    
                    # Calculate box value
                    theoretical_value = upper_strike - lower_strike
                    
                    # Calculate actual cost
                    call_spread_cost = (
                        self._get_mid_price(contracts['calls'][lower_strike]) -
                        self._get_mid_price(contracts['calls'][upper_strike])
                    )
                    
                    put_spread_cost = (
                        self._get_mid_price(contracts['puts'][upper_strike]) -
                        self._get_mid_price(contracts['puts'][lower_strike])
                    )
                    
                    actual_cost = call_spread_cost + put_spread_cost
                    
                    # Check for arbitrage
                    discount = (theoretical_value - actual_cost) / theoretical_value
                    
                    # Account for transaction costs
                    if discount > 0.01:  # 1% arbitrage after costs
                        opportunities.append({
                            'type': 'box_spread_arbitrage',
                            'lower_strike': lower_strike,
                            'upper_strike': upper_strike,
                            'theoretical_value': theoretical_value,
                            'actual_cost': actual_cost,
                            'arbitrage': discount * 100,
                            'expiration': exp_date,
                            'contracts': {
                                'buy_call': contracts['calls'][lower_strike],
                                'sell_call': contracts['calls'][upper_strike],
                                'buy_put': contracts['puts'][upper_strike],
                                'sell_put': contracts['puts'][lower_strike]
                            },
                            'score': discount * 100  # Arbitrage percentage as score
                        })
        
        return opportunities
    
    def scan_earnings_straddle(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for earnings play opportunities"""
        
        opportunities = []
        
        # Check if earnings are coming up (would need earnings calendar API)
        # For now, we'll look for high IV options that might indicate earnings
        
        # Find options with unusually high IV
        high_iv_threshold = conditions['volatility'] * 1.5
        
        exp_groups = defaultdict(list)
        for contract in chain['calls'] + chain['puts']:
            if contract.get('implied_volatility', 0) > high_iv_threshold:
                exp_groups[contract['expiration_date']].append(contract)
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            # Earnings plays typically use weekly options
            if dte > 14:
                continue
            
            # Find ATM straddle
            price = conditions['price']
            
            calls = [c for c in contracts if c['type'] == 'call']
            puts = [c for c in contracts if c['type'] == 'put']
            
            # Find closest strikes
            if calls and puts:
                atm_call = min(calls, key=lambda x: abs(x['strike_price'] - price))
                atm_put = min(puts, key=lambda x: abs(x['strike_price'] - price))
                
                if atm_call['strike_price'] == atm_put['strike_price']:
                    # Calculate expected move
                    straddle_price = (
                        self._get_mid_price(atm_call) + 
                        self._get_mid_price(atm_put)
                    )
                    
                    expected_move_pct = straddle_price / price
                    
                    # Historical earnings moves (would need historical data)
                    historical_avg_move = 0.05  # Placeholder
                    
                    if expected_move_pct < historical_avg_move * 0.8:
                        # Straddle might be underpriced
                        opportunities.append({
                            'type': 'earnings_long_straddle',
                            'call': atm_call,
                            'put': atm_put,
                            'strike': atm_call['strike_price'],
                            'expected_move': expected_move_pct,
                            'historical_move': historical_avg_move,
                            'edge': (historical_avg_move - expected_move_pct) / expected_move_pct,
                            'expiration': exp_date,
                            'score': 50 * (historical_avg_move / expected_move_pct)
                        })
        
        return opportunities
    
    # More strategy scanners...
    
    def scan_long_straddle(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for long straddle opportunities"""
        opportunities = []
        
        # Long straddles work best when IV is low but expected to increase
        if conditions['iv_rank'] > 30:
            return opportunities
            
        price = conditions['price']
        
        # Group by expiration
        exp_groups = defaultdict(lambda: {'puts': [], 'calls': []})
        for contract in chain['puts'] + chain['calls']:
            exp = contract['expiration_date']
            if contract['type'] == 'put':
                exp_groups[exp]['puts'].append(contract)
            else:
                exp_groups[exp]['calls'].append(contract)
        
        for exp_date, contracts in exp_groups.items():
            dte = (datetime.strptime(exp_date, '%Y-%m-%d').date() - datetime.now().date()).days
            
            if dte < 30 or dte > 60:
                continue
                
            # Find ATM options
            atm_call = min(contracts['calls'], key=lambda x: abs(x['strike_price'] - price))
            atm_put = min(contracts['puts'], key=lambda x: abs(x['strike_price'] - price))
            
            if atm_call['strike_price'] == atm_put['strike_price']:
                metrics = self._calculate_straddle_metrics(atm_call, atm_put, price, dte, conditions)
                
                if metrics['expected_return'] > filters['min_expected_return']:
                    opportunities.append({
                        'type': 'long_straddle',
                        'call': atm_call,
                        'put': atm_put,
                        'metrics': metrics,
                        'score': self._score_opportunity(metrics, conditions)
                    })
                    
        return opportunities
    
    def scan_long_strangle(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for long strangle opportunities"""
        # Similar to straddle but with OTM options
        return []
    
    def scan_short_straddle(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for short straddle opportunities"""
        # High IV environments
        return []
    
    def scan_short_strangle(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for short strangle opportunities"""
        # High IV, range-bound markets
        return []
    
    def scan_bull_call_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for bull call spread opportunities"""
        # Already implemented in previous code
        return []
    
    def scan_bear_put_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for bear put spread opportunities"""
        return []
    
    def scan_bull_put_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for bull put spread opportunities"""
        return []
    
    def scan_bear_call_spread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for bear call spread opportunities"""
        return []
    
    def scan_iron_butterfly(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for iron butterfly opportunities"""
        return []
    
    def scan_short_butterfly(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for short butterfly opportunities"""
        return []
    
    def scan_long_butterfly(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for long butterfly opportunities"""
        return []
    
    def scan_double_calendar(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for double calendar opportunities"""
        return []
    
    def scan_double_diagonal(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for double diagonal opportunities"""
        return []
    
    def scan_twisted_sister(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for twisted sister opportunities"""
        return []
    
    def scan_backspread(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for backspread opportunities"""
        return []
    
    def scan_christmas_tree(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for christmas tree opportunities"""
        return []
    
    def scan_conversion_reversal(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for conversion/reversal arbitrage"""
        return []
    
    def scan_jelly_roll(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for jelly roll opportunities"""
        return []
    
    def scan_synthetic_arbitrage(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for synthetic arbitrage opportunities"""
        return []
    
    def scan_earnings_condor(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for earnings iron condor opportunities"""
        return []
    
    def scan_dividend_capture(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for dividend capture with options"""
        return []
    
    def scan_dividend_arbitrage(self, symbol: str, chain: Dict, conditions: Dict, filters: Dict) -> List[Dict]:
        """Scan for dividend arbitrage opportunities"""
        return []
    
    # Metrics calculation methods
    
    def _calculate_iron_condor_metrics(self, setup: Dict, price: float, dte: int, conditions: Dict) -> Dict:
        """Calculate metrics for iron condor"""
        
        # Calculate net credit
        net_credit = (
            self._get_mid_price(setup['put_short']) -
            self._get_mid_price(setup['put_long']) +
            self._get_mid_price(setup['call_short']) -
            self._get_mid_price(setup['call_long'])
        )
        
        # Calculate max loss
        put_spread_width = setup['put_short']['strike_price'] - setup['put_long']['strike_price']
        max_loss = put_spread_width - net_credit
        
        # Calculate breakevens
        lower_breakeven = setup['put_short']['strike_price'] - net_credit
        upper_breakeven = setup['call_short']['strike_price'] + net_credit
        
        # Calculate probability of profit
        T = dte / 365
        vol = conditions['volatility']
        
        prob_below_lower = norm.cdf(np.log(lower_breakeven / price) / (vol * np.sqrt(T)))
        prob_above_upper = 1 - norm.cdf(np.log(upper_breakeven / price) / (vol * np.sqrt(T)))
        profit_probability = 1 - prob_below_lower - prob_above_upper
        
        # Expected value
        expected_value = profit_probability * net_credit - (1 - profit_probability) * max_loss
        
        return {
            'net_credit': net_credit,
            'max_loss': max_loss,
            'max_profit': net_credit,
            'lower_breakeven': lower_breakeven,
            'upper_breakeven': upper_breakeven,
            'profit_probability': profit_probability,
            'expected_value': expected_value,
            'expected_return': expected_value / max_loss if max_loss > 0 else 0,
            'risk_reward_ratio': net_credit / max_loss if max_loss > 0 else 0,
            'margin_required': put_spread_width * 100  # Per contract
        }
    
    def _calculate_calendar_metrics(self, near: Dict, far: Dict, price: float, conditions: Dict) -> Dict:
        """Calculate metrics for calendar spread"""
        
        # Calendar spreads profit from time decay differential
        near_price = self._get_mid_price(near)
        far_price = self._get_mid_price(far)
        net_debit = far_price - near_price
        
        # Estimate profit (simplified)
        # Maximum profit occurs when stock is at strike at near expiration
        near_iv = near.get('implied_volatility', conditions['volatility'])
        far_iv = far.get('implied_volatility', conditions['volatility'])
        
        # Vega exposure
        vega_near = self._calculate_vega(near, price)
        vega_far = self._calculate_vega(far, price)
        net_vega = vega_far - vega_near
        
        # Theta exposure
        theta_near = self._calculate_theta(near, price)
        theta_far = self._calculate_theta(far, price)
        net_theta = theta_far - theta_near  # Should be positive
        
        # Estimate max profit (when near expires worthless)
        remaining_value = self._estimate_option_value(
            far['strike_price'], price, far['days_to_expiry'] - near['days_to_expiry'], 
            far_iv, far['type']
        )
        
        max_profit = remaining_value - net_debit
        
        return {
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': net_debit,
            'breakeven': 'Complex',
            'net_theta': net_theta,
            'net_vega': net_vega,
            'iv_differential': far_iv - near_iv,
            'expected_return': max_profit / net_debit if net_debit > 0 else 0,
            'profit_zone': f"{near['strike_price'] * 0.95:.2f} - {near['strike_price'] * 1.05:.2f}"
        }
    
    def _calculate_diagonal_metrics(self, near: Dict, far: Dict, price: float, 
                                   conditions: Dict, direction: str) -> Dict:
        """Calculate metrics for diagonal spread"""
        
        near_price = self._get_mid_price(near)
        far_price = self._get_mid_price(far)
        net_debit = far_price - near_price
        
        # Diagonal specific calculations
        strike_diff = abs(far['strike_price'] - near['strike_price'])
        
        if direction == 'bullish':
            # Bullish diagonal: buy lower strike far, sell higher strike near
            max_profit = near['strike_price'] - far['strike_price'] + near_price
            directional_edge = (near['strike_price'] - price) / price
        else:
            # Bearish diagonal
            max_profit = far['strike_price'] - near['strike_price'] + near_price
            directional_edge = (price - near['strike_price']) / price
        
        return {
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': net_debit,
            'strike_differential': strike_diff,
            'directional_edge': directional_edge,
            'expected_return': max_profit / net_debit if net_debit > 0 else 0,
            'margin_required': max(net_debit, strike_diff) * 100
        }
    
    def _calculate_jade_lizard_metrics(self, setup: Dict, price: float, dte: int, conditions: Dict) -> Dict:
        """Calculate metrics for jade lizard"""
        
        # Net credit from all legs
        call_credit = self._get_mid_price(setup['short_call'])
        put_spread_credit = (
            self._get_mid_price(setup['short_put']) -
            self._get_mid_price(setup['long_put'])
        )
        
        total_credit = call_credit + put_spread_credit
        
        # Put spread width
        put_spread_width = setup['short_put']['strike_price'] - setup['long_put']['strike_price']
        
        # Max loss on downside
        max_loss_down = put_spread_width - total_credit
        
        # No upside risk if credit > put spread width
        no_upside_risk = total_credit >= put_spread_width
        
        # Profit probability
        T = dte / 365
        vol = conditions['volatility']
        
        # Probability of finishing below short put
        prob_below_put = norm.cdf(
            np.log(setup['short_put']['strike_price'] / price) / (vol * np.sqrt(T))
        )
        
        # Probability of finishing above short call
        prob_above_call = 1 - norm.cdf(
            np.log(setup['short_call']['strike_price'] / price) / (vol * np.sqrt(T))
        )
        
        profit_probability = 1 - prob_below_put - (0 if no_upside_risk else prob_above_call)
        
        return {
            'total_credit': total_credit,
            'max_loss_down': max_loss_down,
            'no_upside_risk': no_upside_risk,
            'put_spread_width': put_spread_width,
            'profit_probability': profit_probability,
            'expected_return': total_credit / max_loss_down if max_loss_down > 0 else float('inf'),
            'margin_required': put_spread_width * 100
        }
    
    def _calculate_bwb_metrics(self, setup: Dict, price: float, dte: int, 
                              conditions: Dict, direction: str) -> Dict:
        """Calculate metrics for broken wing butterfly"""
        
        # Net debit/credit
        lower_price = self._get_mid_price(setup['long_lower'])
        middle_price = self._get_mid_price(setup['short_middle'])
        upper_price = self._get_mid_price(setup['long_upper'])
        
        net_cost = lower_price - 2 * middle_price + upper_price
        
        # Wing widths
        lower_wing = setup['short_middle']['strike_price'] - setup['long_lower']['strike_price']
        upper_wing = setup['long_upper']['strike_price'] - setup['short_middle']['strike_price']
        
        # Max profit at middle strike
        max_profit = lower_wing - abs(net_cost)
        
        # Max loss depends on which wing is wider
        if upper_wing > lower_wing:
            # Risk to upside
            max_loss = abs(net_cost) if net_cost < 0 else upper_wing - lower_wing + net_cost
        else:
            # Risk to downside
            max_loss = abs(net_cost)
        
        # Profit zone
        if net_cost < 0:  # Net debit
            lower_be = setup['long_lower']['strike_price'] + abs(net_cost)
            upper_be = setup['long_upper']['strike_price'] - abs(net_cost)
        else:  # Net credit
            lower_be = setup['long_lower']['strike_price'] - net_cost
            upper_be = setup['long_upper']['strike_price'] + net_cost
        
        # Probability of profit
        T = dte / 365
        vol = conditions['volatility']
        
        prob_in_profit_zone = (
            norm.cdf(np.log(upper_be / price) / (vol * np.sqrt(T))) -
            norm.cdf(np.log(lower_be / price) / (vol * np.sqrt(T)))
        )
        
        return {
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else float('inf'),
            'profit_zone': (lower_be, upper_be),
            'profit_probability': prob_in_profit_zone,
            'wing_ratio': upper_wing / lower_wing,
            'expected_return': (prob_in_profit_zone * max_profit - 
                              (1 - prob_in_profit_zone) * max_loss) / max_loss
        }
    
    def _calculate_ratio_metrics(self, long_option: Dict, short_option: Dict, price: float,
                                dte: int, conditions: Dict, option_type: str, ratio: int) -> Dict:
        """Calculate metrics for ratio spreads"""
        
        long_price = self._get_mid_price(long_option)
        short_price = self._get_mid_price(short_option)
        
        # Net cost/credit
        net_cost = long_price - ratio * short_price
        
        # Breakeven calculations
        if option_type == 'call':
            if net_cost < 0:  # Net credit
                lower_be = long_option['strike_price'] - abs(net_cost)
                upper_be = short_option['strike_price'] + abs(net_cost) / (ratio - 1)
            else:  # Net debit
                lower_be = long_option['strike_price'] + net_cost
                upper_be = float('inf')  # Unlimited risk to upside
        else:  # Put ratio
            if net_cost < 0:  # Net credit
                upper_be = long_option['strike_price'] + abs(net_cost)
                lower_be = short_option['strike_price'] - abs(net_cost) / (ratio - 1)
            else:
                upper_be = long_option['strike_price'] - net_cost
                lower_be = 0  # Risk to downside
        
        # Max profit at short strike
        max_profit = abs(short_option['strike_price'] - long_option['strike_price']) - net_cost
        
        # Margin requirement (simplified)
        margin_required = short_option['strike_price'] * 0.20 * (ratio - 1) * 100
        
        return {
            'net_cost': net_cost,
            'max_profit': max_profit,
            'breakeven_points': (lower_be, upper_be),
            'ratio': f"1x{ratio}",
            'margin_required': margin_required,
            'unlimited_risk': True,
            'profit_zone': f"{long_option['strike_price']} - {short_option['strike_price']}"
        }
    
    def _calculate_straddle_metrics(self, call: Dict, put: Dict, price: float, 
                                   dte: int, conditions: Dict) -> Dict:
        """Calculate metrics for straddle"""
        
        call_price = self._get_mid_price(call)
        put_price = self._get_mid_price(put)
        total_cost = call_price + put_price
        
        # Breakeven points
        upper_be = call['strike_price'] + total_cost
        lower_be = put['strike_price'] - total_cost
        
        # Expected move based on IV
        iv = (call.get('implied_volatility', 0.25) + put.get('implied_volatility', 0.25)) / 2
        expected_move = price * iv * np.sqrt(dte / 365)
        
        # Probability of profit (price moving beyond breakevens)
        T = dte / 365
        prob_above_upper = 1 - norm.cdf(np.log(upper_be / price) / (iv * np.sqrt(T)))
        prob_below_lower = norm.cdf(np.log(lower_be / price) / (iv * np.sqrt(T)))
        profit_probability = prob_above_upper + prob_below_lower
        
        # Expected return based on historical volatility
        hist_vol = conditions['volatility']
        expected_move_hist = price * hist_vol * np.sqrt(T)
        
        if expected_move_hist > total_cost:
            expected_return = (expected_move_hist - total_cost) / total_cost
        else:
            expected_return = -0.5  # Expect to lose half on average
        
        return {
            'total_cost': total_cost,
            'upper_breakeven': upper_be,
            'lower_breakeven': lower_be,
            'expected_move': expected_move,
            'profit_probability': profit_probability,
            'expected_return': expected_return,
            'iv_rank': conditions['iv_rank'],
            'hist_vol_edge': (hist_vol - iv) / iv if iv > 0 else 0
        }
    
    # Helper methods
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        
        for symbol in self.symbols:
            try:
                quote = self.data_client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=symbol)
                )
                
                if symbol in quote:
                    self.market_data_cache[symbol] = {
                        'price': float(quote[symbol].ask_price),
                        'bid': float(quote[symbol].bid_price),
                        'ask': float(quote[symbol].ask_price),
                        'spread': float(quote[symbol].ask_price) - float(quote[symbol].bid_price)
                    }
                    
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
    
    def _estimate_iv(self, contract: Dict, underlying_price: float) -> float:
        """Estimate implied volatility for contract"""
        
        # This is simplified - in production would calculate from option prices
        base_vol = 0.20  # Base volatility
        
        # Adjust for moneyness
        moneyness = contract['strike_price'] / underlying_price
        if contract['type'] == 'put':
            if moneyness < 0.90:  # Deep OTM put
                vol_adjustment = 0.05 * (0.90 - moneyness)
            else:
                vol_adjustment = 0
        else:  # Call
            if moneyness > 1.10:  # Deep OTM call
                vol_adjustment = 0.05 * (moneyness - 1.10)
            else:
                vol_adjustment = 0
        
        # Adjust for time
        dte = contract.get('days_to_expiry', 30)
        if dte < 7:
            time_adjustment = 0.10
        elif dte < 30:
            time_adjustment = 0.05
        else:
            time_adjustment = 0
        
        return base_vol + vol_adjustment + time_adjustment
    
    def _calculate_delta(self, contract: Dict, underlying_price: float, dte: int) -> float:
        """Calculate option delta"""
        
        K = contract['strike_price']
        S = underlying_price
        T = dte / 365
        sigma = contract.get('implied_volatility', 0.25)
        r = 0.05
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if contract['type'] == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def _calculate_vega(self, contract: Dict, underlying_price: float) -> float:
        """Calculate option vega"""
        
        K = contract['strike_price']
        S = underlying_price
        T = contract['days_to_expiry'] / 365
        sigma = contract.get('implied_volatility', 0.25)
        r = 0.05
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    def _calculate_theta(self, contract: Dict, underlying_price: float) -> float:
        """Calculate option theta"""
        
        greeks = self.greeks_calc.calculate_all_greeks(
            underlying_price,
            contract['strike_price'],
            contract['days_to_expiry'] / 365,
            contract.get('implied_volatility', 0.25),
            contract['type']
        )
        
        return greeks['theta']
    
    def _get_mid_price(self, contract: Dict) -> float:
        """Get mid price for contract"""
        
        bid = contract.get('bid_price', 0)
        ask = contract.get('ask_price', 0)
        
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        elif ask > 0:
            return ask * 0.95  # Estimate
        else:
            # Estimate from Black-Scholes
            return self.greeks_calc.black_scholes_price(
                self.market_data_cache.get(contract['underlying_symbol'], {}).get('price', 100),
                contract['strike_price'],
                contract['days_to_expiry'] / 365,
                contract.get('implied_volatility', 0.25),
                contract['type']
            )
    
    def _estimate_option_value(self, strike: float, underlying: float, dte: int, 
                              vol: float, option_type: str) -> float:
        """Estimate option value using Black-Scholes"""
        
        return self.greeks_calc.black_scholes_price(
            underlying, strike, dte / 365, vol, option_type
        )
    
    def _score_opportunity(self, metrics: Dict, conditions: Dict) -> float:
        """Score opportunity based on multiple factors"""
        
        score = 0
        
        # Expected return component (40%)
        expected_return = metrics.get('expected_return', 0)
        score += 40 * min(expected_return / 0.20, 1)  # Cap at 20% return
        
        # Probability component (30%)
        prob = metrics.get('profit_probability', 0.5)
        score += 30 * prob
        
        # Risk-reward component (20%)
        rr = metrics.get('risk_reward_ratio', 1)
        score += 20 * min(rr / 3, 1)  # Cap at 3:1
        
        # Market conditions component (10%)
        if conditions['trend'] == 'bullish' and 'bull' in str(metrics):
            score += 5
        elif conditions['trend'] == 'bearish' and 'bear' in str(metrics):
            score += 5
        
        # IV rank bonus for certain strategies
        if conditions['iv_rank'] > 70 and metrics.get('net_credit', 0) > 0:
            score += 5
        
        return score
    
    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by score and other factors"""
        
        # Sort by score
        opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Additional filtering
        # Remove similar opportunities (same strategy, similar strikes)
        filtered = []
        seen = set()
        
        for opp in opportunities:
            key = f"{opp['strategy']}_{opp['symbol']}"
            if 'expiration' in opp:
                key += f"_{opp['expiration']}"
                
            if key not in seen:
                filtered.append(opp)
                seen.add(key)
        
        return filtered
    
    def _display_top_opportunities(self, opportunities: List[Dict]):
        """Display top opportunities in a formatted way"""
        
        if not opportunities:
            logger.info("\n❌ No opportunities found matching criteria")
            return
        
        logger.info("\n" + "=" * 90)
        logger.info("🏆 TOP OPTIONS OPPORTUNITIES")
        logger.info("=" * 90)
        
        for i, opp in enumerate(opportunities[:10], 1):
            logger.info(f"\n#{i}. {opp['symbol']} - {opp['strategy'].upper()}")
            logger.info(f"    Score: {opp.get('score', 0):.1f}/100")
            
            if 'expiration' in opp:
                logger.info(f"    Expiration: {opp['expiration']} ({opp.get('days_to_expiry', 0)} days)")
            
            # Display strategy-specific details
            if opp['strategy'] == 'iron_condor':
                setup = opp['setup']
                metrics = opp['metrics']
                logger.info(f"    Strikes: {setup['put_long']['strike_price']}/{setup['put_short']['strike_price']} - "
                          f"{setup['call_short']['strike_price']}/{setup['call_long']['strike_price']}")
                logger.info(f"    Net Credit: ${metrics['net_credit']:.2f}")
                logger.info(f"    Max Loss: ${metrics['max_loss']:.2f}")
                logger.info(f"    Probability: {metrics['profit_probability']:.1%}")
                logger.info(f"    Expected Return: {metrics['expected_return']:.1%}")
                
            elif opp['strategy'] == 'calendar_spread':
                logger.info(f"    Type: {opp['type']}")
                logger.info(f"    Strike: ${opp['strike']}")
                logger.info(f"    Near: {opp['near']['expiration_date']} ({opp['near']['days_to_expiry']}d)")
                logger.info(f"    Far: {opp['far']['expiration_date']} ({opp['far']['days_to_expiry']}d)")
                logger.info(f"    Net Debit: ${opp['metrics']['net_debit']:.2f}")
                logger.info(f"    Expected Return: {opp['metrics']['expected_return']:.1%}")
                
            elif opp['strategy'] == 'jade_lizard':
                setup = opp['setup']
                metrics = opp['metrics']
                logger.info(f"    Short Call: ${setup['short_call']['strike_price']}")
                logger.info(f"    Put Spread: ${setup['long_put']['strike_price']}-${setup['short_put']['strike_price']}")
                logger.info(f"    Total Credit: ${metrics['total_credit']:.2f}")
                logger.info(f"    No Upside Risk: {metrics['no_upside_risk']}")
                logger.info(f"    Probability: {metrics['profit_probability']:.1%}")
                
            elif opp['strategy'] == 'box_spread_arbitrage':
                logger.info(f"    Strikes: ${opp['lower_strike']}-${opp['upper_strike']}")
                logger.info(f"    Theoretical Value: ${opp['theoretical_value']:.2f}")
                logger.info(f"    Actual Cost: ${opp['actual_cost']:.2f}")
                logger.info(f"    Arbitrage: {opp['arbitrage']:.2%}")
                
            elif 'earnings' in opp['strategy']:
                logger.info(f"    Strike: ${opp.get('strike', 0)}")
                logger.info(f"    Expected Move: {opp.get('expected_move', 0):.1%}")
                logger.info(f"    Historical Move: {opp.get('historical_move', 0):.1%}")
                logger.info(f"    Edge: {opp.get('edge', 0):.1%}")

def main():
    """Run opportunity scanner"""
    
    scanner = AdvancedOpportunityScanner(paper=True)
    
    # Custom filters
    filters = {
        'min_volume': 50,
        'min_open_interest': 25,
        'max_spread_pct': 0.15,
        'min_profit_probability': 0.60,
        'min_expected_return': 0.15,
        'max_days_to_expiry': 90,
        'min_iv_rank': 30,
        'max_iv_rank': 80
    }
    
    # Scan all opportunities
    opportunities = scanner.scan_all_opportunities(filters)
    
    # Save results
    with open('options_opportunities.json', 'w') as f:
        json.dump(opportunities, f, indent=2, default=str)
    
    logger.info(f"\n✅ Scan complete. Found {len(opportunities)} opportunities.")
    logger.info("Results saved to options_opportunities.json")

if __name__ == "__main__":
    main()