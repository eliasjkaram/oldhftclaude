
#!/usr/bin/env python3
"""
Comprehensive Spread Strategies Module
All possible spread variations including exotic and advanced strategies
"""

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, AssetClass
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetOrdersRequest
from alpaca.common.exceptions import APIError


import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class SpreadType(Enum):
    """All spread strategy types"""
    # Basic Spreads
    VERTICAL_CALL_SPREAD = "vertical_call_spread"
    VERTICAL_PUT_SPREAD = "vertical_put_spread"
    
    # Time Spreads
    CALENDAR_CALL = "calendar_call"
    CALENDAR_PUT = "calendar_put"
    DIAGONAL_CALL = "diagonal_call" 
    DIAGONAL_PUT = "diagonal_put"
    DOUBLE_DIAGONAL = "double_diagonal"
    
    # Volatility Spreads
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    STRAP = "strap"                    # 2 calls + 1 put
    STRIP = "strip"                    # 1 call + 2 puts
    
    # Butterfly Spreads
    LONG_CALL_BUTTERFLY = "long_call_butterfly"
    SHORT_CALL_BUTTERFLY = "short_call_butterfly"
    LONG_PUT_BUTTERFLY = "long_put_butterfly" 
    SHORT_PUT_BUTTERFLY = "short_put_butterfly"
    IRON_BUTTERFLY = "iron_butterfly"
    
    # Condor Spreads
    LONG_CALL_CONDOR = "long_call_condor"
    SHORT_CALL_CONDOR = "short_call_condor"
    LONG_PUT_CONDOR = "long_put_condor"
    SHORT_PUT_CONDOR = "short_put_condor"
    IRON_CONDOR = "iron_condor"
    
    # Ratio Spreads
    CALL_RATIO_SPREAD = "call_ratio_spread"
    PUT_RATIO_SPREAD = "put_ratio_spread" 
    CALL_RATIO_BACKSPREAD = "call_ratio_backspread"
    PUT_RATIO_BACKSPREAD = "put_ratio_backspread"
    
    # Exotic Spreads
    JADE_LIZARD = "jade_lizard"
    BIG_LIZARD = "big_lizard"
    REVERSE_JADE_LIZARD = "reverse_jade_lizard"
    CHRISTMAS_TREE = "christmas_tree"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    SKIP_STRIKE_BUTTERFLY = "skip_strike_butterfly"
    
    # Synthetic Strategies
    SYNTHETIC_LONG_STOCK = "synthetic_long_stock"
    SYNTHETIC_SHORT_STOCK = "synthetic_short_stock"
    SYNTHETIC_LONG_CALL = "synthetic_long_call"
    SYNTHETIC_SHORT_CALL = "synthetic_short_call"
    SYNTHETIC_LONG_PUT = "synthetic_long_put"
    SYNTHETIC_SHORT_PUT = "synthetic_short_put"
    
    # Multi-Strike Exotics
    CONDOR_SPREAD_5_STRIKES = "condor_spread_5_strikes"
    BUTTERFLY_SPREAD_5_STRIKES = "butterfly_spread_5_strikes"
    LADDER_SPREAD = "ladder_spread"
    RATIO_DIAGONAL = "ratio_diagonal"
    
    # Income Strategies
    COVERED_CALL = "covered_call"
    COVERED_PUT = "covered_put"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    CASH_SECURED_PUT = "cash_secured_put"
    POOR_MANS_COVERED_CALL = "poor_mans_covered_call"
    
    # Advanced Arbitrage
    CONVERSION = "conversion"
    REVERSAL = "reversal"
    BOX_SPREAD = "box_spread"
    JELLY_ROLL = "jelly_roll"
    
    # Multi-Expiration Exotics
    ZEBRA_SPREAD = "zebra_spread"
    SEAGULL_SPREAD = "seagull_spread"
    ALBATROSS_SPREAD = "albatross_spread"

@dataclass
class SpreadLeg:
    """Individual spread leg"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    action: str      # 'buy' or 'sell'
    quantity: int
    price: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class SpreadOpportunity:
    """Complete spread opportunity"""
    spread_type: SpreadType
    underlying: str
    legs: List[SpreadLeg]
    
    # Financial metrics
    net_debit: float
    net_credit: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    
    # Probability metrics
    probability_profit: float
    probability_max_profit: float
    expected_return: float
    
    # Risk metrics
    profit_loss_ratio: float
    risk_reward_ratio: float
    margin_requirement: float
    
    # Greeks portfolio
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    
    # Execution metrics
    liquidity_score: float
    execution_difficulty: str
    commission_cost: float
    bid_ask_spread_cost: float
    
    # Strategy specific
    ideal_volatility_environment: str
    ideal_market_outlook: str
    time_decay_impact: str
    
    # Confidence and scoring
    overall_score: float
    confidence_level: float
    detection_method: str

class ComprehensiveSpreadDetector:
    """Detects all possible spread opportunities"""
    
    def __init__(self):
        self.strategies_detected = {strategy: 0 for strategy in SpreadType}
        self.total_opportunities = 0
        
    def detect_all_spreads(self, options_chain: List[Dict], market_data: Dict) -> List[SpreadOpportunity]:
        """Detect all possible spread opportunities"""
        all_opportunities = []
        
        # Group options by expiry and type
        options_by_expiry = self._group_options_by_expiry(options_chain)
        
        # Basic spreads (same expiry)
        for expiry, options in options_by_expiry.items():
            all_opportunities.extend(self._detect_vertical_spreads(options, market_data)
            all_opportunities.extend(self._detect_volatility_spreads(options, market_data)
            all_opportunities.extend(self._detect_butterfly_spreads(options, market_data)
            all_opportunities.extend(self._detect_condor_spreads(options, market_data)
            all_opportunities.extend(self._detect_ratio_spreads(options, market_data)
            all_opportunities.extend(self._detect_exotic_spreads(options, market_data)
        
        # Time spreads (different expiries)
        all_opportunities.extend(self._detect_calendar_spreads(options_by_expiry, market_data)
        all_opportunities.extend(self._detect_diagonal_spreads(options_by_expiry, market_data)
        
        # Synthetic strategies
        all_opportunities.extend(self._detect_synthetic_strategies(options_by_expiry, market_data)
        
        # Arbitrage opportunities
        all_opportunities.extend(self._detect_arbitrage_spreads(options_by_expiry, market_data)
        
        # Advanced multi-expiry exotics
        all_opportunities.extend(self._detect_multi_expiry_exotics(options_by_expiry, market_data)
        
        # Filter and score opportunities
        filtered_opportunities = self._filter_and_score_opportunities(all_opportunities, market_data)
        
        self.total_opportunities += len(filtered_opportunities)
        
        return filtered_opportunities
    
    def _group_options_by_expiry(self, options_chain: List[Dict]) -> Dict[str, Dict]:
        """Group options by expiry date"""
        by_expiry = {}
        
        for option in options_chain:
            expiry_key = option['expiry'].strftime('%Y-%m-%d')
            
            if expiry_key not in by_expiry:
                by_expiry[expiry_key] = {
                    'calls': [],
                    'puts': [],
                    'dte': option.get('dte', 0)
                }
            
            if option['option_type'] == 'call':
                by_expiry[expiry_key]['calls'].append(option)
            else:
                by_expiry[expiry_key]['puts'].append(option)
        
        # Sort by strike for each expiry
        for expiry_data in by_expiry.values():
            expiry_data['calls'].sort(key=lambda x: x['strike'])
            expiry_data['puts'].sort(key=lambda x: x['strike'])
        
        return by_expiry
    
    def _detect_vertical_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect vertical spreads (bull/bear call/put spreads)"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        
        # Bull call spreads
        for i in range(len(calls) - 1):
            for j in range(i + 1, min(i + 4, len(calls)):  # Check next 3 strikes
                long_call = calls[i]
                short_call = calls[j]
                
                if (long_call['volume'] > 5 and short_call['volume'] > 5 and
                    long_call['strike'] < current_price * 1.1):  # Near ATM
                    
                    net_debit = long_call['ask'] - short_call['bid']
                    max_profit = (short_call['strike'] - long_call['strike']) - net_debit
                    max_loss = net_debit
                    
                    if max_profit > 0 and max_loss > 0 and max_profit / max_loss > 0.5:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.VERTICAL_CALL_SPREAD,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(long_call, 'buy', 1),
                                self._create_spread_leg(short_call, 'sell', 1)
                            ],
                            net_debit, 0, max_profit, max_loss,
                            [long_call['strike'] + net_debit],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        # Bear put spreads
        for i in range(len(puts) - 1):
            for j in range(i + 1, min(i + 4, len(puts)):
                long_put = puts[j]  # Higher strike
                short_put = puts[i]  # Lower strike
                
                if (long_put['volume'] > 5 and short_put['volume'] > 5 and
                    long_put['strike'] > current_price * 0.9):
                    
                    net_debit = long_put['ask'] - short_put['bid']
                    max_profit = (long_put['strike'] - short_put['strike']) - net_debit
                    max_loss = net_debit
                    
                    if max_profit > 0 and max_loss > 0 and max_profit / max_loss > 0.5:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.VERTICAL_PUT_SPREAD,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(long_put, 'buy', 1),
                                self._create_spread_leg(short_put, 'sell', 1)
                            ],
                            net_debit, 0, max_profit, max_loss,
                            [long_put['strike'] - net_debit],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_volatility_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect volatility-based spreads"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        iv_rank = market_data.get('iv_rank', 0.5)
        
        # Find ATM options
        atm_call = min(calls, key=lambda x: abs(x['strike'] - current_price)
        atm_put = min(puts, key=lambda x: abs(x['strike'] - current_price)
        
        # Long Straddle (low IV environment)
        if (iv_rank < 0.4 and atm_call['strike'] == atm_put['strike'] and
            atm_call['volume'] > 10 and atm_put['volume'] > 10):
            
            net_debit = atm_call['ask'] + atm_put['ask']
            
            opportunity = self._create_spread_opportunity(
                SpreadType.LONG_STRADDLE,
                market_data['symbol'],
                [
                    self._create_spread_leg(atm_call, 'buy', 1),
                    self._create_spread_leg(atm_put, 'buy', 1)
                ],
                net_debit, 0, float('inf'), net_debit,
                [atm_call['strike'] - net_debit, atm_call['strike'] + net_debit],
                market_data
            )
            opportunities.append(opportunity)
        
        # Short Straddle (high IV environment)
        elif (iv_rank > 0.7 and atm_call['strike'] == atm_put['strike'] and
              atm_call['volume'] > 10 and atm_put['volume'] > 10):
            
            net_credit = atm_call['bid'] + atm_put['bid']
            
            opportunity = self._create_spread_opportunity(
                SpreadType.SHORT_STRADDLE,
                market_data['symbol'],
                [
                    self._create_spread_leg(atm_call, 'sell', 1),
                    self._create_spread_leg(atm_put, 'sell', 1)
                ],
                0, net_credit, net_credit, float('inf'),
                [atm_call['strike'] - net_credit, atm_call['strike'] + net_credit],
                market_data
            )
            opportunities.append(opportunity)
        
        # Long Strangle
        otm_call = next((c for c in calls if c['strike'] > current_price * 1.05), None)
        otm_put = next((p for p in puts if p['strike'] < current_price * 0.95), None)
        
        if (otm_call and otm_put and iv_rank < 0.5 and
            otm_call['volume'] > 5 and otm_put['volume'] > 5):
            
            net_debit = otm_call['ask'] + otm_put['ask']
            
            opportunity = self._create_spread_opportunity(
                SpreadType.LONG_STRANGLE,
                market_data['symbol'],
                [
                    self._create_spread_leg(otm_call, 'buy', 1),
                    self._create_spread_leg(otm_put, 'buy', 1)
                ],
                net_debit, 0, float('inf'), net_debit,
                [otm_put['strike'] - net_debit, otm_call['strike'] + net_debit],
                market_data
            )
            opportunities.append(opportunity)
        
        # Strap (bullish volatility play: 2 calls + 1 put)
        if atm_call and atm_put and iv_rank < 0.4:
            net_debit = 2 * atm_call['ask'] + atm_put['ask']
            
            opportunity = self._create_spread_opportunity(
                SpreadType.STRAP,
                market_data['symbol'],
                [
                    self._create_spread_leg(atm_call, 'buy', 2),
                    self._create_spread_leg(atm_put, 'buy', 1)
                ],
                net_debit, 0, float('inf'), net_debit,
                [atm_call['strike'] - net_debit/3, atm_call['strike'] + net_debit/2],
                market_data
            )
            opportunities.append(opportunity)
        
        # Strip (bearish volatility play: 1 call + 2 puts)
        if atm_call and atm_put and iv_rank < 0.4:
            net_debit = atm_call['ask'] + 2 * atm_put['ask']
            
            opportunity = self._create_spread_opportunity(
                SpreadType.STRIP,
                market_data['symbol'],
                [
                    self._create_spread_leg(atm_call, 'buy', 1),
                    self._create_spread_leg(atm_put, 'buy', 2)
                ],
                net_debit, 0, float('inf'), net_debit,
                [atm_put['strike'] - net_debit/2, atm_put['strike'] + net_debit/3],
                market_data
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_butterfly_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect butterfly spread opportunities"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        
        # Long Call Butterfly
        for i in range(len(calls) - 2):
            if i + 2 < len(calls):
                wing1 = calls[i]      # Lower strike (buy)
                body = calls[i + 1]   # Middle strike (sell 2)
                wing2 = calls[i + 2]  # Higher strike (buy)
                
                # Check if strikes are evenly spaced and liquid
                strike_diff1 = body['strike'] - wing1['strike']
                strike_diff2 = wing2['strike'] - body['strike']
                
                if (abs(strike_diff1 - strike_diff2) < 2.5 and  # Evenly spaced
                    all(opt['volume'] > 3 for opt in [wing1, body, wing2]) and
                    abs(body['strike'] - current_price) < current_price * 0.1):  # Near ATM
                    
                    net_debit = wing1['ask'] + wing2['ask'] - 2 * body['bid']
                    max_profit = strike_diff1 - net_debit
                    max_loss = net_debit
                    
                    if net_debit > 0 and max_profit > 0 and max_profit / max_loss > 1.5:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.LONG_CALL_BUTTERFLY,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(wing1, 'buy', 1),
                                self._create_spread_leg(body, 'sell', 2),
                                self._create_spread_leg(wing2, 'buy', 1)
                            ],
                            net_debit, 0, max_profit, max_loss,
                            [wing1['strike'] + net_debit, wing2['strike'] - net_debit],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        # Iron Butterfly (ATM butterfly with calls and puts)
        atm_strike = min(calls + puts, key=lambda x: abs(x['strike'] - current_price)['strike']
        
        atm_call = next((c for c in calls if c['strike'] == atm_strike), None)
        atm_put = next((p for p in puts if p['strike'] == atm_strike), None)
        
        if atm_call and atm_put:
            # Find wing strikes
            lower_put = next((p for p in puts if p['strike'] < atm_strike), None)
            higher_call = next((c for c in calls if c['strike'] > atm_strike), None)
            
            if (lower_put and higher_call and
                all(opt['volume'] > 5 for opt in [atm_call, atm_put, lower_put, higher_call]):
                
                net_credit = atm_call['bid'] + atm_put['bid'] - lower_put['ask'] - higher_call['ask']
                
                if net_credit > 0:
                    strike_width = min(atm_strike - lower_put['strike'], 
                                     higher_call['strike'] - atm_strike)
                    max_profit = net_credit
                    max_loss = strike_width - net_credit
                    
                    if max_loss > 0 and max_profit / max_loss > 0.3:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.IRON_BUTTERFLY,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(lower_put, 'buy', 1),
                                self._create_spread_leg(atm_put, 'sell', 1),
                                self._create_spread_leg(atm_call, 'sell', 1),
                                self._create_spread_leg(higher_call, 'buy', 1)
                            ],
                            0, net_credit, max_profit, max_loss,
                            [atm_strike - net_credit, atm_strike + net_credit],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_condor_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect condor spread opportunities"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        iv_rank = market_data.get('iv_rank', 0.5)
        
        # Iron Condor (high IV environment)
        if iv_rank > 0.6 and len(calls) >= 4 and len(puts) >= 4:
            
            # Find optimal strikes
            put_short_strike = current_price * 0.92   # OTM put to sell
            call_short_strike = current_price * 1.08  # OTM call to sell
            
            put_short = min(puts, key=lambda p: abs(p['strike'] - put_short_strike)
            call_short = min(calls, key=lambda c: abs(c['strike'] - call_short_strike)
            
            # Find protective wings (further OTM)
            put_long = next((p for p in puts if p['strike'] < put_short['strike'] - 5), None)
            call_long = next((c for c in calls if c['strike'] > call_short['strike'] + 5), None)
            
            if (put_long and call_long and
                all(opt['volume'] > 3 for opt in [put_short, call_short, put_long, call_long]):
                
                net_credit = (put_short['bid'] + call_short['bid'] - 
                            put_long['ask'] - call_long['ask'])
                
                if net_credit > 0:
                    put_width = put_short['strike'] - put_long['strike']
                    call_width = call_long['strike'] - call_short['strike']
                    max_loss = min(put_width, call_width) - net_credit
                    
                    if max_loss > 0 and net_credit / max_loss > 0.25:
                        # Calculate probability of profit
                        prob_profit = self._calculate_iron_condor_probability(
                            current_price, put_short['strike'], call_short['strike'], 
                            market_data, options['dte']
                        )
                        
                        opportunity = self._create_spread_opportunity(
                            SpreadType.IRON_CONDOR,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(put_long, 'buy', 1),
                                self._create_spread_leg(put_short, 'sell', 1),
                                self._create_spread_leg(call_short, 'sell', 1),
                                self._create_spread_leg(call_long, 'buy', 1)
                            ],
                            0, net_credit, net_credit, max_loss,
                            [put_short['strike'] - net_credit, call_short['strike'] + net_credit],
                            market_data,
                            probability_profit=prob_profit
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_ratio_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect ratio spread opportunities"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        
        # Call Ratio Spread (1 buy : 2 sell)
        for i in range(len(calls) - 1):
            if i + 1 < len(calls):
                long_call = calls[i]    # Lower strike (buy 1)
                short_calls = calls[i + 1]  # Higher strike (sell 2)
                
                if (long_call['volume'] > 5 and short_calls['volume'] > 10 and
                    long_call['strike'] < current_price * 1.05):
                    
                    net_credit = 2 * short_calls['bid'] - long_call['ask']
                    
                    if net_credit > 0:  # Credit spread
                        max_profit_strike = short_calls['strike']
                        max_profit = (max_profit_strike - long_call['strike']) + net_credit
                        
                        opportunity = self._create_spread_opportunity(
                            SpreadType.CALL_RATIO_SPREAD,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(long_call, 'buy', 1),
                                self._create_spread_leg(short_calls, 'sell', 2)
                            ],
                            0, net_credit, max_profit, float('inf'),  # Unlimited upside risk
                            [long_call['strike'] + net_credit, 
                             short_calls['strike'] + (short_calls['strike'] - long_call['strike'])],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_exotic_spreads(self, options: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect exotic spread opportunities"""
        opportunities = []
        calls = options['calls']
        puts = options['puts']
        current_price = market_data['current_price']
        
        # Jade Lizard (Short put + Short call spread)
        if len(calls) >= 2 and len(puts) >= 1:
            # Short an OTM put
            otm_put = next((p for p in puts if p['strike'] < current_price * 0.9), None)
            
            # Short call spread (sell lower, buy higher)
            if len(calls) >= 2:
                short_call = next((c for c in calls if c['strike'] > current_price * 1.05), None)
                long_call = next((c for c in calls if c['strike'] > short_call['strike'] + 5), None) if short_call else None
                
                if (otm_put and short_call and long_call and
                    all(opt['volume'] > 3 for opt in [otm_put, short_call, long_call]):
                    
                    net_credit = otm_put['bid'] + short_call['bid'] - long_call['ask']
                    
                    if net_credit > 0:
                        call_spread_width = long_call['strike'] - short_call['strike']
                        
                        # Jade Lizard is profitable if underlying stays above put strike
                        # and below short call strike
                        if net_credit >= call_spread_width:  # No upside risk
                            opportunity = self._create_spread_opportunity(
                                SpreadType.JADE_LIZARD,
                                market_data['symbol'],
                                [
                                    self._create_spread_leg(otm_put, 'sell', 1),
                                    self._create_spread_leg(short_call, 'sell', 1),
                                    self._create_spread_leg(long_call, 'buy', 1)
                                ],
                                0, net_credit, net_credit, otm_put['strike'],
                                [otm_put['strike'] - net_credit, short_call['strike']],
                                market_data
                            )
                            opportunities.append(opportunity)
        
        # Broken Wing Butterfly (Asymmetric butterfly)
        if len(calls) >= 3:
            for i in range(len(calls) - 2):
                wing1 = calls[i]
                body = calls[i + 1]
                wing2 = calls[i + 2]
                
                # Check for uneven spacing (broken wing)
                left_width = body['strike'] - wing1['strike']
                right_width = wing2['strike'] - body['strike']
                
                if (abs(right_width - left_width) > 5 and  # Asymmetric
                    all(opt['volume'] > 3 for opt in [wing1, body, wing2]):
                    
                    net_cost = wing1['ask'] + wing2['ask'] - 2 * body['bid']
                    
                    if net_cost < 0:  # Credit spread
                        opportunity = self._create_spread_opportunity(
                            SpreadType.BROKEN_WING_BUTTERFLY,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(wing1, 'buy', 1),
                                self._create_spread_leg(body, 'sell', 2),
                                self._create_spread_leg(wing2, 'buy', 1)
                            ],
                            0, abs(net_cost), abs(net_cost), max(left_width, right_width),
                            [wing1['strike'], wing2['strike']],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_calendar_spreads(self, options_by_expiry: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect calendar spread opportunities"""
        opportunities = []
        current_price = market_data['current_price']
        
        expiry_keys = list(options_by_expiry.keys()
        
        # Calendar spreads (same strike, different expiries)
        for i in range(len(expiry_keys) - 1):
            near_expiry = expiry_keys[i]
            far_expiry = expiry_keys[i + 1]
            
            near_options = options_by_expiry[near_expiry]
            far_options = options_by_expiry[far_expiry]
            
            # Calendar call spreads
            for near_call in near_options['calls']:
                # Find matching strike in far expiry
                far_call = next((c for c in far_options['calls'] 
                               if abs(c['strike'] - near_call['strike']) < 1), None)
                
                if (far_call and near_call['volume'] > 3 and far_call['volume'] > 3 and
                    abs(near_call['strike'] - current_price) < current_price * 0.1):
                    
                    # Buy far expiry, sell near expiry
                    net_debit = far_call['ask'] - near_call['bid']
                    
                    if 0 < net_debit < far_call['ask'] * 0.5:  # Reasonable cost
                        # Calendar spreads profit from time decay and low volatility
                        opportunity = self._create_spread_opportunity(
                            SpreadType.CALENDAR_CALL,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(near_call, 'sell', 1),
                                self._create_spread_leg(far_call, 'buy', 1)
                            ],
                            net_debit, 0, net_debit * 2, net_debit,  # Estimated max profit
                            [near_call['strike']],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_diagonal_spreads(self, options_by_expiry: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect diagonal spread opportunities"""
        opportunities = []
        current_price = market_data['current_price']
        
        expiry_keys = list(options_by_expiry.keys()
        
        # Diagonal spreads (different strikes AND expiries)
        for i in range(len(expiry_keys) - 1):
            near_expiry = expiry_keys[i]
            far_expiry = expiry_keys[i + 1]
            
            near_options = options_by_expiry[near_expiry]
            far_options = options_by_expiry[far_expiry]
            
            # Diagonal call spreads
            for near_call in near_options['calls']:
                for far_call in far_options['calls']:
                    # Different strikes: sell lower strike near-term, buy higher strike long-term
                    if (far_call['strike'] > near_call['strike'] and
                        near_call['volume'] > 3 and far_call['volume'] > 3 and
                        near_call['strike'] < current_price * 1.1 and
                        far_call['strike'] < current_price * 1.2):
                        
                        net_debit = far_call['ask'] - near_call['bid']
                        
                        if 0 < net_debit < far_call['ask'] * 0.6:
                            opportunity = self._create_spread_opportunity(
                                SpreadType.DIAGONAL_CALL,
                                market_data['symbol'],
                                [
                                    self._create_spread_leg(near_call, 'sell', 1),
                                    self._create_spread_leg(far_call, 'buy', 1)
                                ],
                                net_debit, 0, 
                                (far_call['strike'] - near_call['strike']) - net_debit,
                                net_debit,
                                [near_call['strike'], far_call['strike']],
                                market_data
                            )
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_synthetic_strategies(self, options_by_expiry: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect synthetic strategies"""
        opportunities = []
        current_price = market_data['current_price']
        
        for expiry_key, options in options_by_expiry.items():
            calls = options['calls']
            puts = options['puts']
            
            # Synthetic long stock (long call + short put, same strike)
            for call in calls:
                matching_put = next((p for p in puts if abs(p['strike'] - call['strike']) < 1), None)
                
                if (matching_put and call['volume'] > 5 and matching_put['volume'] > 5 and
                    abs(call['strike'] - current_price) < current_price * 0.05):
                    
                    net_cost = call['ask'] - matching_put['bid']
                    synthetic_stock_price = call['strike'] + net_cost
                    
                    # Check for arbitrage opportunity
                    if abs(synthetic_stock_price - current_price) > 1.0:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.SYNTHETIC_LONG_STOCK,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(call, 'buy', 1),
                                self._create_spread_leg(matching_put, 'sell', 1)
                            ],
                            max(0, net_cost), max(0, -net_cost), 
                            float('inf'), float('inf'),
                            [],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_arbitrage_spreads(self, options_by_expiry: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect arbitrage opportunities"""
        opportunities = []
        current_price = market_data['current_price']
        
        for expiry_key, options in options_by_expiry.items():
            calls = options['calls']
            puts = options['puts']
            
            # Box spread arbitrage
            if len(calls) >= 2 and len(puts) >= 2:
                for i in range(len(calls) - 1):
                    for j in range(i + 1, len(calls):
                        call1 = calls[i]  # Lower strike
                        call2 = calls[j]  # Higher strike
                        
                        # Find matching puts
                        put1 = next((p for p in puts if abs(p['strike'] - call1['strike']) < 1), None)
                        put2 = next((p for p in puts if abs(p['strike'] - call2['strike']) < 1), None)
                        
                        if (put1 and put2 and 
                            all(opt['volume'] > 3 for opt in [call1, call2, put1, put2]):
                            
                            # Box spread: Long call spread + Short put spread
                            box_cost = (call1['ask'] - call2['bid']) + (put2['ask'] - put1['bid'])
                            theoretical_value = call2['strike'] - call1['strike']
                            
                            arbitrage_profit = theoretical_value - box_cost - 0.10  # Account for commissions
                            
                            if arbitrage_profit > 0.25:  # Profitable arbitrage
                                opportunity = self._create_spread_opportunity(
                                    SpreadType.BOX_SPREAD,
                                    market_data['symbol'],
                                    [
                                        self._create_spread_leg(call1, 'buy', 1),
                                        self._create_spread_leg(call2, 'sell', 1),
                                        self._create_spread_leg(put1, 'sell', 1),
                                        self._create_spread_leg(put2, 'buy', 1)
                                    ],
                                    box_cost, 0, arbitrage_profit, 0,  # Risk-free
                                    [],
                                    market_data
                                )
                                opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_multi_expiry_exotics(self, options_by_expiry: Dict, market_data: Dict) -> List[SpreadOpportunity]:
        """Detect advanced multi-expiry exotic strategies"""
        opportunities = []
        
        if len(options_by_expiry) >= 3:
            expiry_keys = sorted(options_by_expiry.keys()
            
            # Double Diagonal (diagonal on both sides)
            if len(expiry_keys) >= 2:
                near_exp = expiry_keys[0]
                far_exp = expiry_keys[1]
                
                near_options = options_by_expiry[near_exp]
                far_options = options_by_expiry[far_exp]
                
                current_price = market_data['current_price']
                
                # Find optimal strikes
                near_call = next((c for c in near_options['calls'] 
                                if c['strike'] > current_price * 1.05), None)
                near_put = next((p for p in near_options['puts'] 
                               if p['strike'] < current_price * 0.95), None)
                far_call = next((c for c in far_options['calls'] 
                               if c['strike'] > current_price * 1.1), None)
                far_put = next((p for p in far_options['puts'] 
                              if p['strike'] < current_price * 0.9), None)
                
                if (near_call and near_put and far_call and far_put and
                    all(opt['volume'] > 3 for opt in [near_call, near_put, far_call, far_put]):
                    
                    # Sell near-term options, buy far-term options
                    net_cost = (far_call['ask'] + far_put['ask'] - 
                              near_call['bid'] - near_put['bid'])
                    
                    if 0 < net_cost < (far_call['ask'] + far_put['ask']) * 0.4:
                        opportunity = self._create_spread_opportunity(
                            SpreadType.DOUBLE_DIAGONAL,
                            market_data['symbol'],
                            [
                                self._create_spread_leg(near_call, 'sell', 1),
                                self._create_spread_leg(near_put, 'sell', 1),
                                self._create_spread_leg(far_call, 'buy', 1),
                                self._create_spread_leg(far_put, 'buy', 1)
                            ],
                            net_cost, 0, net_cost * 3, net_cost,  # Estimated
                            [near_put['strike'], near_call['strike']],
                            market_data
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _create_spread_leg(self, option: Dict, action: str, quantity: int) -> SpreadLeg:
        """Create a spread leg from option data"""
        return SpreadLeg(
            symbol=option['symbol'],
            option_type=option['option_type'],
            strike=option['strike'],
            expiry=option['expiry'],
            action=action,
            quantity=quantity,
            price=option['ask'] if action == 'buy' else option['bid'],
            iv=option.get('iv', 0.25),
            delta=option.get('delta', 0.5),
            gamma=option.get('gamma', 0.01),
            theta=option.get('theta', -0.05),
            vega=option.get('vega', 0.1)
        )
    
    def _create_spread_opportunity(self, spread_type: SpreadType, underlying: str, 
                                 legs: List[SpreadLeg], net_debit: float, net_credit: float,
                                 max_profit: float, max_loss: float, breakeven_points: List[float],
                                 market_data: Dict, **kwargs) -> SpreadOpportunity:
        """Create a complete spread opportunity"""
        
        # Calculate portfolio Greeks
        total_delta = sum(leg.delta * leg.quantity * (1 if leg.action == 'buy' else -1) for leg in legs)
        total_gamma = sum(leg.gamma * leg.quantity * (1 if leg.action == 'buy' else -1) for leg in legs)
        total_theta = sum(leg.theta * leg.quantity * (1 if leg.action == 'buy' else -1) for leg in legs)
        total_vega = sum(leg.vega * leg.quantity * (1 if leg.action == 'buy' else -1) for leg in legs)
        
        # Calculate liquidity score
        liquidity_score = min(1.0, sum(self._get_volume_score(leg) for leg in legs) / len(legs)
        
        # Calculate execution difficulty
        execution_difficulty = self._assess_execution_difficulty(legs)
        
        # Estimate commission costs
        commission_cost = len(legs) * 0.65  # $0.65 per contract per leg
        
        # Calculate bid-ask spread cost
        bid_ask_spread_cost = sum(
            abs(self._get_ask_price(leg) - self._get_bid_price(leg) * leg.quantity 
            for leg in legs
        ) * 100
        
        # Risk metrics
        profit_loss_ratio = max_profit / max_loss if max_loss > 0 else float('inf')
        risk_reward_ratio = max_loss / max_profit if max_profit > 0 else float('inf')
        
        # Probability estimation
        probability_profit = kwargs.get('probability_profit', 
                                      self._estimate_probability_profit(spread_type, breakeven_points, market_data)
        
        # Expected return
        expected_return = (max_profit * probability_profit - 
                         max_loss * (1 - probability_profit) * 100
        
        # Overall scoring
        overall_score = self._calculate_overall_score(
            profit_loss_ratio, liquidity_score, probability_profit, execution_difficulty
        )
        
        return SpreadOpportunity(
            spread_type=spread_type,
            underlying=underlying,
            legs=legs,
            net_debit=net_debit,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            probability_profit=probability_profit,
            probability_max_profit=probability_profit * 0.3,  # Estimated
            expected_return=expected_return,
            profit_loss_ratio=profit_loss_ratio,
            risk_reward_ratio=risk_reward_ratio,
            margin_requirement=self._calculate_margin_requirement(legs),
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_rho=0.0,  # Simplified
            liquidity_score=liquidity_score,
            execution_difficulty=execution_difficulty,
            commission_cost=commission_cost,
            bid_ask_spread_cost=bid_ask_spread_cost,
            ideal_volatility_environment=self._get_ideal_vol_environment(spread_type),
            ideal_market_outlook=self._get_ideal_market_outlook(spread_type),
            time_decay_impact=self._get_time_decay_impact(spread_type),
            overall_score=overall_score,
            confidence_level=liquidity_score * probability_profit,
            detection_method="Comprehensive_Spread_Detector"
        )
    
    def _get_volume_score(self, leg: SpreadLeg) -> float:
        """Get volume-based liquidity score"""
        # This would need to be implemented based on actual option data
        return 0.8  # Placeholder
    
    def _get_ask_price(self, leg: SpreadLeg) -> float:
        """Get ask price for leg"""
        return leg.price * 1.02  # Simplified bid-ask spread
    
    def _get_bid_price(self, leg: SpreadLeg) -> float:
        """Get bid price for leg"""
        return leg.price * 0.98  # Simplified bid-ask spread
    
    def _assess_execution_difficulty(self, legs: List[SpreadLeg]) -> str:
        """Assess execution difficulty based on leg count and liquidity"""
        if len(legs) <= 2:
            return "Easy"
        elif len(legs) <= 4:
            return "Medium"
        else:
            return "Hard"
    
    def _calculate_margin_requirement(self, legs: List[SpreadLeg]) -> float:
        """Calculate estimated margin requirement"""
        # Simplified margin calculation
        total_short_value = sum(leg.price * leg.quantity * 100 
                              for leg in legs if leg.action == 'sell')
        return total_short_value * 0.2  # 20% of short value
    
    def _estimate_probability_profit(self, spread_type: SpreadType, 
                                   breakeven_points: List[float], market_data: Dict) -> float:
        """Estimate probability of profit"""
        current_price = market_data['current_price']
        volatility = market_data.get('realized_vol_20d', 0.25)
        
        if not breakeven_points:
            return 0.6  # Default for complex strategies
        
        if len(breakeven_points) == 1:
            # Single breakeven point
            distance = abs(breakeven_points[0] - current_price) / current_price
            return max(0.3, min(0.9, 0.7 - distance * 2)
        
        elif len(breakeven_points) == 2:
            # Range strategies (straddles, strangles, condors)
            lower_be, upper_be = sorted(breakeven_points)
            range_width = upper_be - lower_be
            price_position = (current_price - lower_be) / range_width
            
            if 0.2 <= price_position <= 0.8:  # In the profitable range
                return max(0.5, min(0.8, 0.7 + (0.5 - abs(price_position - 0.5) * 0.4)
            else:
                return max(0.2, 0.5 - abs(price_position - 0.5) * 0.6)
        
        return 0.5  # Default
    
    def _calculate_iron_condor_probability(self, current_price: float, put_strike: float,
                                         call_strike: float, market_data: Dict, dte: int) -> float:
        """Calculate Iron Condor success probability"""
        range_width = call_strike - put_strike
        volatility = market_data.get('realized_vol_20d', 0.25)
        
        # Expected move based on volatility and time
        expected_move = current_price * volatility * np.sqrt(dte / 365)
        
        # Probability that price stays within the range
        prob = min(0.85, max(0.4, (range_width / 2) / expected_move * 0.7)
        return prob
    
    def _get_ideal_vol_environment(self, spread_type: SpreadType) -> str:
        """Get ideal volatility environment for strategy"""
        high_vol_strategies = [
            SpreadType.SHORT_STRADDLE, SpreadType.SHORT_STRANGLE, 
            SpreadType.IRON_CONDOR, SpreadType.IRON_BUTTERFLY
        ]
        
        low_vol_strategies = [
            SpreadType.LONG_STRADDLE, SpreadType.LONG_STRANGLE,
            SpreadType.CALENDAR_CALL, SpreadType.CALENDAR_PUT
        ]
        
        if spread_type in high_vol_strategies:
            return "High IV"
        elif spread_type in low_vol_strategies:
            return "Low IV"
        else:
            return "Neutral"
    
    def _get_ideal_market_outlook(self, spread_type: SpreadType) -> str:
        """Get ideal market outlook for strategy"""
        bullish_strategies = [
            SpreadType.VERTICAL_CALL_SPREAD, SpreadType.STRAP
        ]
        
        bearish_strategies = [
            SpreadType.VERTICAL_PUT_SPREAD, SpreadType.STRIP
        ]
        
        neutral_strategies = [
            SpreadType.IRON_CONDOR, SpreadType.IRON_BUTTERFLY,
            SpreadType.SHORT_STRADDLE, SpreadType.CALENDAR_CALL
        ]
        
        if spread_type in bullish_strategies:
            return "Bullish"
        elif spread_type in bearish_strategies:
            return "Bearish"
        else:
            return "Neutral"
    
    def _get_time_decay_impact(self, spread_type: SpreadType) -> str:
        """Get time decay impact for strategy"""
        benefits_from_decay = [
            SpreadType.SHORT_STRADDLE, SpreadType.SHORT_STRANGLE,
            SpreadType.IRON_CONDOR, SpreadType.COVERED_CALL
        ]
        
        hurt_by_decay = [
            SpreadType.LONG_STRADDLE, SpreadType.LONG_STRANGLE,
            SpreadType.VERTICAL_CALL_SPREAD, SpreadType.VERTICAL_PUT_SPREAD
        ]
        
        if spread_type in benefits_from_decay:
            return "Benefits"
        elif spread_type in hurt_by_decay:
            return "Hurts"
        else:
            return "Neutral"
    
    def _calculate_overall_score(self, profit_loss_ratio: float, liquidity_score: float,
                               probability_profit: float, execution_difficulty: str) -> float:
        """Calculate overall opportunity score"""
        # Base score from profit/loss ratio
        ratio_score = min(1.0, profit_loss_ratio / 3.0)
        
        # Execution difficulty penalty
        difficulty_multiplier = {"Easy": 1.0, "Medium": 0.9, "Hard": 0.8}[execution_difficulty]
        
        # Combined score
        overall_score = (ratio_score * 0.4 + 
                        liquidity_score * 0.3 + 
                        probability_profit * 0.3) * difficulty_multiplier
        
        return min(1.0, overall_score)
    
    def _filter_and_score_opportunities(self, opportunities: List[SpreadOpportunity], 
                                      market_data: Dict) -> List[SpreadOpportunity]:
        """Filter and score opportunities"""
        filtered = []
        
        for opp in opportunities:
            # Basic filters
            if (opp.overall_score >= 0.3 and 
                opp.liquidity_score >= 0.4 and
                opp.probability_profit >= 0.3):
                
                # Update strategy count
                if opp.spread_type in self.strategies_detected:
                    self.strategies_detected[opp.spread_type] += 1
                
                filtered.append(opp)
        
        # Sort by overall score
        filtered.sort(key=lambda x: x.overall_score, reverse=True)
        
        return filtered[:50]  # Return top 50 opportunities

# Test the comprehensive spread detector
async def test_comprehensive_spreads():
    """Test comprehensive spread detection"""
    print("🚀 TESTING COMPREHENSIVE SPREAD STRATEGIES")
    print("=" * 80)
    
    detector = ComprehensiveSpreadDetector()
    
    # Generate sample options chain
    current_price = 100.0
    options_chain = []
    
    # Generate comprehensive options chain
    for dte in [7, 14, 21, 30, 45]:
        for strike in np.arange(80, 121, 2.5):
            for option_type in ['call', 'put']:
                option = {
                    'symbol': f"SPY{datetime.now().strftime('%y%m%d')}{option_type[0].upper()}{int(strike*1000):08d}",
                    'underlying': 'SPY',
                    'strike': strike,
                    'expiry': datetime.now() + timedelta(days=dte),
                    'option_type': option_type,
                    'mark': max(0.05, abs(strike - current_price) * 0.1 + np.random.uniform(0.5, 3.0),
                    'ask': 0,  # Will be calculated
                    'bid': 0,  # Will be calculated
                    'iv': 0.2 + np.random.uniform(-0.05, 0.1),
                    'volume': max(0, int(np.random.exponential(20)),
                    'dte': dte,
                    'delta': 0.5 if option_type == 'call' else -0.5,
                    'gamma': 0.01,
                    'theta': -0.05,
                    'vega': 0.1
                }
                
                # Set bid/ask based on mark
                spread = option['mark'] * 0.05  # 5% spread
                option['ask'] = option['mark'] + spread/2
                option['bid'] = option['mark'] - spread/2
                
                options_chain.append(option)
    
    market_data = {
        'symbol': 'SPY',
        'current_price': current_price,
        'iv_rank': 0.6,
        'realized_vol_20d': 0.18,
        'volume': 50000000
    }
    
    print(f"📊 Generated {len(options_chain)} option contracts")
    print(f"🔍 Detecting all possible spread strategies...")
    
    start_time = time.time()
    opportunities = detector.detect_all_spreads(options_chain, market_data)
    detection_time = time.time() - start_time
    
    print(f"\n✅ DETECTION COMPLETE:")
    print(f"   ⚡ Time: {detection_time:.2f}s")
    print(f"   💰 Opportunities Found: {len(opportunities)}")
    print(f"   📊 Contracts Analyzed: {len(options_chain)}")
    print(f"   🚀 Detection Rate: {len(opportunities)/detection_time:.1f} opps/sec")
    
    # Show strategy breakdown
    print(f"\n📈 STRATEGY TYPE BREAKDOWN:")
    print("-" * 60)
    
    strategy_counts = {}
    for opp in opportunities:
        strategy_name = opp.spread_type.value.replace('_', ' ').title()
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"   {strategy:<30} {count:3d}")
    
    # Show top opportunities
    print(f"\n🏆 TOP 10 OPPORTUNITIES:")
    print("-" * 80)
    
    for i, opp in enumerate(opportunities[:10], 1):
        strategy_name = opp.spread_type.value.replace('_', ' ').title()
        print(f"{i:2d}. {strategy_name}")
        print(f"    💰 Max Profit: ${opp.max_profit:.0f} | Max Loss: ${opp.max_loss:.0f}")
        print(f"    🎯 Prob Profit: {opp.probability_profit:.1%} | Score: {opp.overall_score:.2f}")
        print(f"    ⚡ Legs: {len(opp.legs)} | Liquidity: {opp.liquidity_score:.2f}")
        print()
    
    print(f"🎊 COMPREHENSIVE SPREAD DETECTION COMPLETE!")
    print(f"   📊 Total Strategies Available: {len(SpreadType)} types")
    print(f"   ✅ Strategies Detected: {len([s for s in detector.strategies_detected.values() if s > 0])}")
    print(f"   💎 Success Rate: {len(opportunities)/len(options_chain)*100:.1f}% opportunity detection")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_comprehensive_spreads()