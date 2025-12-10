#!/usr/bin/env python3
"""
Alpaca Real Options Trading Bot
===============================
Uses Alpaca's actual options trading API for live execution
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
from dataclasses import dataclass
from scipy import stats
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
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, StockLatestQuoteRequest,
    OptionChainRequest, OptionLatestQuoteRequest,
    OptionBarsRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOptionContractsRequest, OptionContractRequest
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, AssetClass,
    ContractType, ExerciseStyle
)

@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    type: str  # 'call' or 'put'
    style: str  # 'american' or 'european'
    contract_id: str
    multiplier: int = 100
    # Market data
    bid: float = 0
    ask: float = 0
    last: float = 0
    volume: int = 0
    open_interest: int = 0
    iv: float = 0  # Implied volatility
    # Greeks
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0
    rho: float = 0

@dataclass
class OptionStrategy:
    """Options trading strategy"""
    name: str
    legs: List[Dict]  # Each leg has contract, side, quantity
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_profit: float
    net_debit: float  # Positive for debit, negative for credit
    confidence: float
    metadata: Dict

class AlpacaRealOptionsBot:
    """Real options trading bot using Alpaca's options API"""
    
    def __init__(self):
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY'],
            paper=True
        )
        
        self.stock_data_client = StockHistoricalDataClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY']
        )
        
        self.options_data_client = OptionHistoricalDataClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY']
        )
        
        # Trading parameters
        self.risk_free_rate = 0.05
        self.max_position_size = 10000  # $10k max per position
        self.max_contracts = 10  # Max contracts per trade
        self.min_volume = 10  # Min option volume
        self.min_open_interest = 50  # Min open interest
        self.max_bid_ask_spread = 0.20  # Max 20% spread
        
        # Target symbols (high liquidity options)
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META']
        
        # Track positions
        self.active_positions = {}
        self.executed_trades = []
        
        logger.info("✅ Alpaca Real Options Bot initialized")
    
    async def check_options_trading_status(self):
        """Check if account is approved for options trading"""
        try:
            account = self.trading_client.get_account()
            
            logger.info(f"\n📊 Account Options Status:")
            logger.info(f"   Options Approved: {account.options_approved_level}")
            logger.info(f"   Options Trading Level: {account.options_trading_level}")
            
            if not account.options_approved_level or account.options_approved_level == 0:
                logger.warning("⚠️  Account not approved for options trading!")
                logger.warning("   Continuing in analysis mode...")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking options status: {e}")
            return False
    
    async def get_option_chain(self, symbol: str, 
                              min_days: int = 20, 
                              max_days: int = 45) -> List[OptionContract]:
        """Get option chain for a symbol"""
        try:
            # Get current stock price
            stock_quote = await self._get_stock_quote(symbol)
            if not stock_quote:
                return []
            
            current_price = stock_quote['price']
            
            # Calculate date range
            min_expiry = datetime.now() + timedelta(days=min_days)
            max_expiry = datetime.now() + timedelta(days=max_days)
            
            # Get option contracts
            request = GetOptionContractsRequest(
                underlying_symbols=symbol,
                expiration_date_gte=min_expiry.strftime('%Y-%m-%d'),
                expiration_date_lte=max_expiry.strftime('%Y-%m-%d'),
                strike_price_gte=current_price * 0.85,  # 15% OTM
                strike_price_lte=current_price * 1.15   # 15% OTM
            )
            
            response = self.trading_client.get_option_contracts(request)
            
            option_contracts = []
            
            # Get quotes for each contract
            for contract in response.option_contracts:
                try:
                    # Get option quote
                    quote_request = OptionLatestQuoteRequest(
                        symbol_or_symbols=contract.symbol
                    )
                    quote_response = self.options_data_client.get_option_latest_quote(quote_request)
                    
                    if contract.symbol in quote_response:
                        quote = quote_response[contract.symbol]
                        
                        # Create option contract object
                        option = OptionContract(
                            symbol=contract.symbol,
                            underlying=contract.underlying_symbol,
                            strike=float(contract.strike_price),
                            expiration=contract.expiration_date,
                            type='call' if contract.type == ContractType.CALL else 'put',
                            style='american' if contract.style == ExerciseStyle.AMERICAN else 'european',
                            contract_id=contract.id,
                            multiplier=contract.size,
                            bid=float(quote.bid_price) if quote.bid_price else 0,
                            ask=float(quote.ask_price) if quote.ask_price else 0,
                            last=float(quote.last_price) if quote.last_price else 0,
                            volume=quote.bid_size + quote.ask_size,  # Approximate volume
                        )
                        
                        # Calculate mid price and IV
                        if option.bid > 0 and option.ask > 0:
                            mid_price = (option.bid + option.ask) / 2
                            option.iv = self._calculate_implied_volatility(
                                mid_price, current_price, option.strike,
                                self._days_to_expiry(option.expiration) / 365,
                                self.risk_free_rate, option.type
                            )
                            
                            # Calculate Greeks
                            greeks = self._calculate_greeks(
                                current_price, option.strike,
                                self._days_to_expiry(option.expiration) / 365,
                                self.risk_free_rate, option.iv, option.type
                            )
                            option.delta = greeks['delta']
                            option.gamma = greeks['gamma']
                            option.theta = greeks['theta']
                            option.vega = greeks['vega']
                            option.rho = greeks['rho']
                        
                        option_contracts.append(option)
                        
                except Exception as e:
                    logger.debug(f"Error getting quote for {contract.symbol}: {e}")
            
            # Filter by liquidity
            filtered_contracts = []
            for contract in option_contracts:
                spread_pct = (contract.ask - contract.bid) / contract.ask if contract.ask > 0 else 1
                
                if (contract.volume >= self.min_volume and 
                    spread_pct <= self.max_bid_ask_spread):
                    filtered_contracts.append(contract)
            
            logger.info(f"Found {len(filtered_contracts)} liquid option contracts for {symbol}")
            return filtered_contracts
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return []
    
    async def _get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """Get current stock quote"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            response = self.stock_data_client.get_stock_latest_quote(request)
            
            if symbol in response:
                quote = response[symbol]
                return {
                    'price': float(quote.ask_price),
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'volume': quote.ask_size + quote.bid_size
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock quote for {symbol}: {e}")
            return None
    
    def _days_to_expiry(self, expiration: datetime) -> int:
        """Calculate days to expiry"""
        return max(0, (expiration - datetime.now()).days)
    
    def _calculate_implied_volatility(self, option_price: float, S: float, K: float,
                                    T: float, r: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Initial guess
            sigma = 0.3
            
            for _ in range(100):
                # Calculate option price and vega
                d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                
                if option_type == 'call':
                    price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
                else:
                    price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
                
                vega = S * stats.norm.pdf(d1) * np.sqrt(T)
                
                price_diff = price - option_price
                
                if abs(price_diff) < 0.001:
                    break
                
                if vega < 0.001:
                    break
                
                sigma = sigma - price_diff / vega
                sigma = max(0.01, min(3, sigma))  # Keep within bounds
            
            return sigma
            
        except Exception:
            return 0.25  # Default IV
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = stats.norm.cdf(d1)
            else:
                delta = -stats.norm.cdf(-d1)
            
            # Gamma
            gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta (per day)
            if option_type == 'call':
                theta = (- (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
            else:
                theta = (- (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
            
            # Vega (per 1% change)
            vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho (per 1% change)
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    # ========== OPTIONS STRATEGIES ==========
    
    async def find_covered_calls(self, symbol: str, shares_owned: int) -> List[OptionStrategy]:
        """Find covered call opportunities"""
        strategies = []
        
        try:
            # Get option chain
            contracts = await self.get_option_chain(symbol)
            calls = [c for c in contracts if c.type == 'call']
            
            if not calls:
                return strategies
            
            # Get current stock price
            stock_quote = await self._get_stock_quote(symbol)
            current_price = stock_quote['price']
            
            # Look for OTM calls with good premium
            for call in calls:
                if call.strike > current_price * 1.02:  # At least 2% OTM
                    # Calculate potential returns
                    premium = (call.bid + call.ask) / 2
                    premium_income = premium * shares_owned
                    
                    if call.strike > current_price:
                        max_profit = (call.strike - current_price) * shares_owned + premium_income
                    else:
                        max_profit = premium_income
                    
                    # Return on risk
                    ror = (premium_income / (current_price * shares_owned)) * 100
                    
                    # Annualized return
                    days_to_expiry = self._days_to_expiry(call.expiration)
                    annualized_return = ror * (365 / days_to_expiry) if days_to_expiry > 0 else 0
                    
                    if annualized_return > 10:  # At least 10% annualized
                        strategy = OptionStrategy(
                            name="Covered Call",
                            legs=[{
                                'contract': call,
                                'side': 'sell',
                                'quantity': shares_owned // 100  # Convert to contracts
                            }],
                            max_profit=max_profit,
                            max_loss=float('inf'),  # Unlimited downside on stock
                            breakeven=[current_price - premium],
                            probability_profit=1 - call.delta,  # Probability OTM at expiry
                            net_debit=-premium_income,  # Credit received
                            confidence=0.8 if call.delta < 0.3 else 0.6,
                            metadata={
                                'strike': call.strike,
                                'premium': premium,
                                'ror': ror,
                                'annualized_return': annualized_return,
                                'days_to_expiry': days_to_expiry,
                                'delta': call.delta,
                                'theta': call.theta
                            }
                        )
                        
                        strategies.append(strategy)
            
        except Exception as e:
            logger.error(f"Error finding covered calls for {symbol}: {e}")
        
        return strategies
    
    async def find_cash_secured_puts(self, symbol: str, 
                                    max_contracts: int = 5) -> List[OptionStrategy]:
        """Find cash-secured put opportunities"""
        strategies = []
        
        try:
            # Get option chain
            contracts = await self.get_option_chain(symbol)
            puts = [c for c in contracts if c.type == 'put']
            
            if not puts:
                return strategies
            
            # Get current stock price
            stock_quote = await self._get_stock_quote(symbol)
            current_price = stock_quote['price']
            
            # Look for OTM puts with good premium
            for put in puts:
                if put.strike < current_price * 0.98:  # At least 2% OTM
                    # Calculate potential returns
                    premium = (put.bid + put.ask) / 2
                    contracts_to_sell = min(max_contracts, 
                                          int(self.max_position_size / (put.strike * 100)))
                    
                    if contracts_to_sell < 1:
                        continue
                    
                    premium_income = premium * 100 * contracts_to_sell
                    cash_required = put.strike * 100 * contracts_to_sell
                    
                    # Return on risk
                    ror = (premium_income / cash_required) * 100
                    
                    # Annualized return
                    days_to_expiry = self._days_to_expiry(put.expiration)
                    annualized_return = ror * (365 / days_to_expiry) if days_to_expiry > 0 else 0
                    
                    if annualized_return > 15:  # At least 15% annualized
                        strategy = OptionStrategy(
                            name="Cash-Secured Put",
                            legs=[{
                                'contract': put,
                                'side': 'sell',
                                'quantity': contracts_to_sell
                            }],
                            max_profit=premium_income,
                            max_loss=cash_required - premium_income,
                            breakeven=[put.strike - premium],
                            probability_profit=1 + put.delta,  # Probability OTM at expiry
                            net_debit=-premium_income,  # Credit received
                            confidence=0.8 if abs(put.delta) < 0.3 else 0.6,
                            metadata={
                                'strike': put.strike,
                                'premium': premium,
                                'contracts': contracts_to_sell,
                                'cash_required': cash_required,
                                'ror': ror,
                                'annualized_return': annualized_return,
                                'days_to_expiry': days_to_expiry,
                                'delta': put.delta,
                                'theta': put.theta
                            }
                        )
                        
                        strategies.append(strategy)
            
        except Exception as e:
            logger.error(f"Error finding cash-secured puts for {symbol}: {e}")
        
        return strategies
    
    async def find_vertical_spreads(self, symbol: str) -> List[OptionStrategy]:
        """Find vertical spread opportunities (bull call and bear put spreads)"""
        strategies = []
        
        try:
            # Get option chain
            contracts = await self.get_option_chain(symbol)
            calls = sorted([c for c in contracts if c.type == 'call'], key=lambda x: x.strike)
            puts = sorted([c for c in contracts if c.type == 'put'], key=lambda x: x.strike)
            
            # Get current stock price and momentum
            stock_quote = await self._get_stock_quote(symbol)
            current_price = stock_quote['price']
            
            # Get historical data for trend
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=20)
            )
            bars = self.stock_data_client.get_stock_bars(bars_request)
            
            if symbol in bars.data:
                prices = [bar.close for bar in bars.data[symbol]]
                trend = (prices[-1] - prices[0]) / prices[0] if prices else 0
            else:
                trend = 0
            
            # Bull Call Spreads (if bullish trend)
            if trend > 0.02 and len(calls) > 1:
                for i in range(len(calls) - 1):
                    long_call = calls[i]
                    short_call = calls[i + 1]
                    
                    if (long_call.strike <= current_price * 1.02 and 
                        short_call.strike <= current_price * 1.10):
                        
                        # Calculate spread metrics
                        debit = ((long_call.ask + long_call.bid) / 2 - 
                                (short_call.ask + short_call.bid) / 2)
                        
                        if debit > 0:
                            max_profit = (short_call.strike - long_call.strike - debit) * 100
                            max_loss = debit * 100
                            breakeven = long_call.strike + debit
                            
                            # Probability of profit (price above breakeven)
                            days_to_expiry = self._days_to_expiry(long_call.expiration)
                            T = days_to_expiry / 365
                            vol = long_call.iv
                            
                            d = (np.log(current_price / breakeven) + 
                                (self.risk_free_rate - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
                            prob_profit = stats.norm.cdf(d)
                            
                            if max_profit / max_loss > 1.5:  # Risk/reward > 1.5
                                strategy = OptionStrategy(
                                    name="Bull Call Spread",
                                    legs=[
                                        {'contract': long_call, 'side': 'buy', 'quantity': 1},
                                        {'contract': short_call, 'side': 'sell', 'quantity': 1}
                                    ],
                                    max_profit=max_profit,
                                    max_loss=max_loss,
                                    breakeven=[breakeven],
                                    probability_profit=prob_profit,
                                    net_debit=debit * 100,
                                    confidence=0.7 if prob_profit > 0.5 else 0.5,
                                    metadata={
                                        'long_strike': long_call.strike,
                                        'short_strike': short_call.strike,
                                        'debit': debit,
                                        'days_to_expiry': days_to_expiry,
                                        'trend': trend,
                                        'risk_reward': max_profit / max_loss
                                    }
                                )
                                
                                strategies.append(strategy)
            
            # Bear Put Spreads (if bearish trend)
            if trend < -0.02 and len(puts) > 1:
                for i in range(len(puts) - 1):
                    long_put = puts[i + 1]  # Higher strike
                    short_put = puts[i]     # Lower strike
                    
                    if (long_put.strike >= current_price * 0.98 and 
                        short_put.strike >= current_price * 0.90):
                        
                        # Calculate spread metrics
                        debit = ((long_put.ask + long_put.bid) / 2 - 
                                (short_put.ask + short_put.bid) / 2)
                        
                        if debit > 0:
                            max_profit = (long_put.strike - short_put.strike - debit) * 100
                            max_loss = debit * 100
                            breakeven = long_put.strike - debit
                            
                            # Probability of profit (price below breakeven)
                            days_to_expiry = self._days_to_expiry(long_put.expiration)
                            T = days_to_expiry / 365
                            vol = long_put.iv
                            
                            d = (np.log(current_price / breakeven) + 
                                (self.risk_free_rate - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
                            prob_profit = 1 - stats.norm.cdf(d)
                            
                            if max_profit / max_loss > 1.5:  # Risk/reward > 1.5
                                strategy = OptionStrategy(
                                    name="Bear Put Spread",
                                    legs=[
                                        {'contract': long_put, 'side': 'buy', 'quantity': 1},
                                        {'contract': short_put, 'side': 'sell', 'quantity': 1}
                                    ],
                                    max_profit=max_profit,
                                    max_loss=max_loss,
                                    breakeven=[breakeven],
                                    probability_profit=prob_profit,
                                    net_debit=debit * 100,
                                    confidence=0.7 if prob_profit > 0.5 else 0.5,
                                    metadata={
                                        'long_strike': long_put.strike,
                                        'short_strike': short_put.strike,
                                        'debit': debit,
                                        'days_to_expiry': days_to_expiry,
                                        'trend': trend,
                                        'risk_reward': max_profit / max_loss
                                    }
                                )
                                
                                strategies.append(strategy)
            
        except Exception as e:
            logger.error(f"Error finding vertical spreads for {symbol}: {e}")
        
        return strategies
    
    async def find_iron_condors(self, symbol: str) -> List[OptionStrategy]:
        """Find iron condor opportunities"""
        strategies = []
        
        try:
            # Get option chain
            contracts = await self.get_option_chain(symbol)
            calls = sorted([c for c in contracts if c.type == 'call'], key=lambda x: x.strike)
            puts = sorted([c for c in contracts if c.type == 'put'], key=lambda x: x.strike)
            
            if len(calls) < 2 or len(puts) < 2:
                return strategies
            
            # Get current stock price
            stock_quote = await self._get_stock_quote(symbol)
            current_price = stock_quote['price']
            
            # Find suitable strikes
            # Put spread: sell put 5-10% OTM, buy put 10-15% OTM
            # Call spread: sell call 5-10% OTM, buy call 10-15% OTM
            
            suitable_puts = []
            suitable_calls = []
            
            for put in puts:
                otm_pct = (current_price - put.strike) / current_price
                if 0.05 <= otm_pct <= 0.15:
                    suitable_puts.append(put)
            
            for call in calls:
                otm_pct = (call.strike - current_price) / current_price
                if 0.05 <= otm_pct <= 0.15:
                    suitable_calls.append(call)
            
            # Create iron condor combinations
            if len(suitable_puts) >= 2 and len(suitable_calls) >= 2:
                for i in range(len(suitable_puts) - 1):
                    for j in range(len(suitable_calls) - 1):
                        put_sell = suitable_puts[i + 1]  # Higher strike
                        put_buy = suitable_puts[i]       # Lower strike
                        call_sell = suitable_calls[j]    # Lower strike
                        call_buy = suitable_calls[j + 1] # Higher strike
                        
                        # Calculate net credit
                        put_credit = ((put_sell.bid + put_sell.ask) / 2 - 
                                     (put_buy.bid + put_buy.ask) / 2)
                        call_credit = ((call_sell.bid + call_sell.ask) / 2 - 
                                      (call_buy.bid + call_buy.ask) / 2)
                        net_credit = (put_credit + call_credit) * 100
                        
                        if net_credit > 50:  # Minimum $50 credit
                            # Calculate max loss and profit
                            put_spread_width = put_sell.strike - put_buy.strike
                            call_spread_width = call_buy.strike - call_sell.strike
                            max_spread_width = max(put_spread_width, call_spread_width)
                            
                            max_profit = net_credit
                            max_loss = (max_spread_width * 100) - net_credit
                            
                            # Breakeven points
                            lower_breakeven = put_sell.strike - (net_credit / 100)
                            upper_breakeven = call_sell.strike + (net_credit / 100)
                            
                            # Probability of profit (price between short strikes)
                            days_to_expiry = self._days_to_expiry(put_sell.expiration)
                            T = days_to_expiry / 365
                            vol = (put_sell.iv + call_sell.iv) / 2
                            
                            # Probability price stays between short strikes
                            d1 = (np.log(current_price / put_sell.strike) + 
                                 (self.risk_free_rate - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
                            d2 = (np.log(current_price / call_sell.strike) + 
                                 (self.risk_free_rate - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
                            
                            prob_below_put = stats.norm.cdf(d1)
                            prob_above_call = 1 - stats.norm.cdf(d2)
                            prob_profit = 1 - prob_below_put - prob_above_call
                            
                            if prob_profit > 0.5 and max_profit / max_loss > 0.2:
                                strategy = OptionStrategy(
                                    name="Iron Condor",
                                    legs=[
                                        {'contract': put_buy, 'side': 'buy', 'quantity': 1},
                                        {'contract': put_sell, 'side': 'sell', 'quantity': 1},
                                        {'contract': call_sell, 'side': 'sell', 'quantity': 1},
                                        {'contract': call_buy, 'side': 'buy', 'quantity': 1}
                                    ],
                                    max_profit=max_profit,
                                    max_loss=max_loss,
                                    breakeven=[lower_breakeven, upper_breakeven],
                                    probability_profit=prob_profit,
                                    net_debit=-net_credit,  # Credit strategy
                                    confidence=0.8 if prob_profit > 0.65 else 0.6,
                                    metadata={
                                        'put_buy_strike': put_buy.strike,
                                        'put_sell_strike': put_sell.strike,
                                        'call_sell_strike': call_sell.strike,
                                        'call_buy_strike': call_buy.strike,
                                        'credit': net_credit / 100,
                                        'days_to_expiry': days_to_expiry,
                                        'volatility': vol,
                                        'risk_reward': max_profit / max_loss
                                    }
                                )
                                
                                strategies.append(strategy)
            
        except Exception as e:
            logger.error(f"Error finding iron condors for {symbol}: {e}")
        
        return strategies
    
    # ========== STRATEGY EXECUTION ==========
    
    async def execute_strategy(self, strategy: OptionStrategy) -> bool:
        """Execute an options strategy"""
        try:
            logger.info(f"\n📊 Executing {strategy.name}:")
            
            # Execute each leg
            for leg in strategy.legs:
                contract = leg['contract']
                side = leg['side']
                quantity = leg['quantity']
                
                logger.info(f"   Leg: {side.upper()} {quantity} {contract.symbol}")
                logger.info(f"        Strike: ${contract.strike}, Expiry: {contract.expiration.date()}")
                logger.info(f"        Bid: ${contract.bid}, Ask: ${contract.ask}")
                
                # Place order
                if side == 'buy':
                    order_side = OrderSide.BUY
                    limit_price = contract.ask * 1.01  # Slightly above ask
                else:
                    order_side = OrderSide.SELL
                    limit_price = contract.bid * 0.99  # Slightly below bid
                
                order_request = LimitOrderRequest(
                    symbol=contract.symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    asset_class=AssetClass.OPTION
                )
                
                try:
                    order = self.trading_client.submit_order(order_request)
                    logger.info(f"   ✅ Order placed: {order.id}")
                    
                    # Track the position
                    self.active_positions[order.id] = {
                        'strategy': strategy.name,
                        'contract': contract,
                        'order': order,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"   ❌ Order failed: {e}")
                    return False
                
                await asyncio.sleep(0.5)  # Rate limit
            
            # Record executed trade
            self.executed_trades.append({
                'timestamp': datetime.now(),
                'strategy': strategy,
                'status': 'executed'
            })
            
            logger.info(f"   Strategy executed successfully!")
            logger.info(f"   Max Profit: ${strategy.max_profit:.2f}")
            logger.info(f"   Max Loss: ${strategy.max_loss:.2f}")
            logger.info(f"   Probability of Profit: {strategy.probability_profit:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            return False
    
    # ========== MAIN EXECUTION ==========
    
    async def find_all_opportunities(self) -> List[OptionStrategy]:
        """Find all trading opportunities"""
        all_strategies = []
        
        for symbol in self.symbols:
            logger.info(f"\n🔍 Analyzing {symbol}...")
            
            # Get current positions
            positions = self.trading_client.get_all_positions()
            stock_position = next((p for p in positions if p.symbol == symbol), None)
            
            # 1. Covered Calls (if we own shares)
            if stock_position and int(stock_position.qty) >= 100:
                strategies = await self.find_covered_calls(symbol, int(stock_position.qty))
                all_strategies.extend(strategies)
                logger.info(f"   Found {len(strategies)} covered call opportunities")
            
            # 2. Cash-Secured Puts
            strategies = await self.find_cash_secured_puts(symbol)
            all_strategies.extend(strategies)
            logger.info(f"   Found {len(strategies)} cash-secured put opportunities")
            
            # 3. Vertical Spreads
            strategies = await self.find_vertical_spreads(symbol)
            all_strategies.extend(strategies)
            logger.info(f"   Found {len(strategies)} vertical spread opportunities")
            
            # 4. Iron Condors
            strategies = await self.find_iron_condors(symbol)
            all_strategies.extend(strategies)
            logger.info(f"   Found {len(strategies)} iron condor opportunities")
        
        # Sort by confidence and expected return
        all_strategies.sort(key=lambda x: (x.confidence, -x.net_debit), reverse=True)
        
        return all_strategies
    
    async def run(self):
        """Main execution loop"""
        logger.info("="*80)
        logger.info("ALPACA REAL OPTIONS TRADING BOT")
        logger.info("="*80)
        
        # Check account status
        account = self.trading_client.get_account()
        logger.info(f"\n💰 Account Status:")
        logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        # Check options trading status
        can_trade_options = await self.check_options_trading_status()
        
        # Find all opportunities
        logger.info(f"\n🔍 Searching for options opportunities...")
        strategies = await self.find_all_opportunities()
        logger.info(f"\n📊 Found {len(strategies)} total opportunities")
        
        # Display top opportunities
        logger.info(f"\n🎯 Top Options Opportunities:")
        for i, strategy in enumerate(strategies[:10]):
            logger.info(f"\n{i+1}. {strategy.name}")
            logger.info(f"   Confidence: {strategy.confidence:.2%}")
            logger.info(f"   Max Profit: ${strategy.max_profit:.2f}")
            logger.info(f"   Max Loss: ${strategy.max_loss:.2f}")
            logger.info(f"   Net Debit/Credit: ${strategy.net_debit:.2f}")
            logger.info(f"   Probability of Profit: {strategy.probability_profit:.2%}")
            
            if strategy.metadata:
                for key, value in strategy.metadata.items():
                    if isinstance(value, float):
                        if 'return' in key or 'prob' in key:
                            logger.info(f"   {key}: {value:.2%}")
                        else:
                            logger.info(f"   {key}: {value:.2f}")
                    else:
                        logger.info(f"   {key}: {value}")
        
        # Execute top strategies if options trading is enabled
        if can_trade_options and strategies:
            logger.info(f"\n💰 Executing top strategies...")
            
            executed = 0
            for strategy in strategies[:3]:  # Execute top 3
                # Check buying power
                if strategy.net_debit > 0 and strategy.net_debit > float(account.buying_power):
                    logger.warning(f"⚠️  Insufficient buying power for {strategy.name}")
                    continue
                
                success = await self.execute_strategy(strategy)
                if success:
                    executed += 1
                    
                    # Update buying power estimate
                    if strategy.net_debit > 0:
                        account.buying_power = str(float(account.buying_power) - strategy.net_debit)
                
                await asyncio.sleep(1)  # Rate limit
            
            logger.info(f"\n✅ Executed {executed} strategies")
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'account_value': float(account.portfolio_value),
            'options_enabled': can_trade_options,
            'opportunities_found': len(strategies),
            'trades_executed': len(self.executed_trades),
            'top_strategies': [
                {
                    'name': s.name,
                    'confidence': s.confidence,
                    'max_profit': s.max_profit,
                    'max_loss': s.max_loss,
                    'net_debit': s.net_debit,
                    'probability_profit': s.probability_profit,
                    'metadata': s.metadata
                }
                for s in strategies[:20]
            ]
        }
        
        with open('alpaca_options_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Report saved to: alpaca_options_report.json")
        logger.info("\n" + "="*80)
        logger.info("OPTIONS TRADING COMPLETE")
        logger.info("="*80)


async def main():
    """Entry point"""
    bot = AlpacaRealOptionsBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())