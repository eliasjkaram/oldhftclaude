#!/usr/bin/env python3
"""
Advanced Options & Spreads Trading System
Focuses exclusively on options trading with various spread strategies
Uses Alpaca's real options trading API
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, ContractType
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('options_spreads_trader.log')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedOptionsTrader:
    def __init__(self):
        """Initialize the options trading system"""
        # API credentials
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKEP9PIBDKOSUGHHY44Z')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ')
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.stock_data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.options_data_client = OptionHistoricalDataClient(self.api_key, self.api_secret)
        
        # Trading parameters
        self.max_positions = 20  # More positions for spreads
        self.position_size_pct = 0.03  # 3% per spread
        self.scan_interval = 30  # seconds
        
        # Options parameters
        self.min_dte = 20  # Minimum days to expiration
        self.max_dte = 45  # Maximum days to expiration
        self.min_volume = 100  # Minimum option volume
        self.min_open_interest = 500
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
        # Strategy thresholds
        self.min_credit = 0.30  # Minimum credit for credit spreads
        self.max_spread_width = 10.0  # Maximum spread width
        self.iv_percentile_threshold = 70  # High IV for selling
        self.iv_skew_threshold = 0.05  # 5% IV skew for calendar spreads
        
        # High volume stocks with liquid options
        self.watchlist = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA',
            'NVDA', 'AMD', 'IWM', 'DIA', 'NFLX', 'JPM', 'BAC', 'XLF',
            'GLD', 'TLT', 'VIX', 'EEM', 'FXI', 'EWZ', 'USO', 'SLV'
        ]
        
        # Track active spreads
        self.active_spreads = {}
        self.spread_performance = {
            'iron_condor': {'trades': 0, 'wins': 0, 'pnl': 0},
            'bull_put_spread': {'trades': 0, 'wins': 0, 'pnl': 0},
            'bear_call_spread': {'trades': 0, 'wins': 0, 'pnl': 0},
            'butterfly': {'trades': 0, 'wins': 0, 'pnl': 0},
            'calendar_spread': {'trades': 0, 'wins': 0, 'pnl': 0},
            'diagonal_spread': {'trades': 0, 'wins': 0, 'pnl': 0},
            'straddle': {'trades': 0, 'wins': 0, 'pnl': 0},
            'strangle': {'trades': 0, 'wins': 0, 'pnl': 0}
        }
        
    def round_price(self, price: float) -> float:
        """Round price to 2 decimals"""
        return float(Decimal(str(price)).quantize(Decimal('0.01'), ROUND_HALF_UP))
        
    def calculate_black_scholes(self, S: float, K: float, T: float, r: float, 
                               sigma: float, option_type: str = 'call') -> Dict:
        """Calculate Black-Scholes option price and Greeks"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = -((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                  r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
        
    async def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Advanced Options & Spreads Trading System")
        logger.info(f"Focus: Options spreads on {len(self.watchlist)} liquid underlyings")
        
        await self.display_account_status()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 Options Trading Cycle #{cycle}")
                logger.info(f"{'='*60}")
                
                # Check market status
                if not await self.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                # Scan for various options strategies
                await asyncio.gather(
                    self.scan_iron_condors(),
                    self.scan_credit_spreads(),
                    self.scan_butterfly_spreads(),
                    self.scan_calendar_spreads(),
                    self.scan_diagonal_spreads(),
                    self.scan_straddles_strangles(),
                    self.scan_ratio_spreads(),
                    self.scan_jade_lizard()
                )
                
                # Manage existing positions
                await self.manage_spreads()
                
                # Display status
                await self.display_portfolio_status()
                await self.display_spread_performance()
                
                # Wait for next scan
                logger.info(f"\n⏰ Next scan in {self.scan_interval} seconds...")
                await asyncio.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
                
    async def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'options_buying_power': float(account.options_buying_power) if hasattr(account, 'options_buying_power') else float(account.buying_power),
                'cash': float(account.cash)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    async def display_account_status(self):
        """Display account status"""
        info = await self.get_account_info()
        if info:
            logger.info("\n💼 Account Status:")
            logger.info(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            logger.info(f"Options Buying Power: ${info['options_buying_power']:,.2f}")
            logger.info(f"Cash: ${info['cash']:,.2f}")
            
    async def get_option_chain(self, symbol: str) -> List[Dict]:
        """Get option chain for a symbol"""
        try:
            # Calculate expiration date range
            today = date.today()
            min_expiry = today + timedelta(days=self.min_dte)
            max_expiry = today + timedelta(days=self.max_dte)
            
            # Get current stock price
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
            
            if symbol not in quotes:
                return []
                
            stock_price = float(quotes[symbol].ask_price)
            
            # Get option contracts
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=min_expiry.isoformat(),
                expiration_date_lte=max_expiry.isoformat(),
                strike_price_gte=stock_price * 0.85,
                strike_price_lte=stock_price * 1.15
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            
            # Filter by volume and open interest
            filtered_contracts = []
            for contract in contracts:
                if (contract.open_interest >= self.min_open_interest and
                    contract.volume >= self.min_volume):
                    filtered_contracts.append({
                        'symbol': contract.symbol,
                        'underlying': symbol,
                        'strike': float(contract.strike_price),
                        'expiration': contract.expiration_date,
                        'type': contract.contract_type,
                        'volume': contract.volume,
                        'open_interest': contract.open_interest,
                        'bid': float(contract.bid_price) if contract.bid_price else 0,
                        'ask': float(contract.ask_price) if contract.ask_price else 0,
                        'iv': contract.implied_volatility if hasattr(contract, 'implied_volatility') else 0.3
                    })
                    
            return filtered_contracts
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return []
            
    async def scan_iron_condors(self):
        """Scan for Iron Condor opportunities"""
        logger.info("🦅 Scanning for Iron Condor opportunities...")
        
        for symbol in self.watchlist[:10]:  # Focus on most liquid
            try:
                if f"{symbol}_IC" in self.active_spreads:
                    continue
                    
                # Get option chain
                chain = await self.get_option_chain(symbol)
                if not chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Find suitable strikes for Iron Condor
                # Sell OTM put and call, buy further OTM put and call
                put_strikes = sorted([c['strike'] for c in chain if c['type'] == 'put'])
                call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                
                if len(put_strikes) < 4 or len(call_strikes) < 4:
                    continue
                    
                # Find strikes around 15-20 delta
                put_short_strike = None
                put_long_strike = None
                call_short_strike = None
                call_long_strike = None
                
                # Put side (below current price)
                for i, strike in enumerate(put_strikes):
                    if strike < stock_price * 0.95:  # ~15-20 delta area
                        put_short_strike = strike
                        if i > 0:
                            put_long_strike = put_strikes[i-1]
                        break
                        
                # Call side (above current price)
                for i, strike in enumerate(call_strikes):
                    if strike > stock_price * 1.05:  # ~15-20 delta area
                        call_short_strike = strike
                        if i < len(call_strikes) - 1:
                            call_long_strike = call_strikes[i+1]
                        break
                        
                if all([put_short_strike, put_long_strike, call_short_strike, call_long_strike]):
                    # Calculate potential credit
                    put_short = next((c for c in chain if c['strike'] == put_short_strike and c['type'] == 'put'), None)
                    put_long = next((c for c in chain if c['strike'] == put_long_strike and c['type'] == 'put'), None)
                    call_short = next((c for c in chain if c['strike'] == call_short_strike and c['type'] == 'call'), None)
                    call_long = next((c for c in chain if c['strike'] == call_long_strike and c['type'] == 'call'), None)
                    
                    if all([put_short, put_long, call_short, call_long]):
                        put_credit = put_short['bid'] - put_long['ask']
                        call_credit = call_short['bid'] - call_long['ask']
                        total_credit = put_credit + call_credit
                        
                        if total_credit >= self.min_credit:
                            logger.info(f"✅ Iron Condor opportunity on {symbol}:")
                            logger.info(f"   Put spread: {put_long_strike}/{put_short_strike}")
                            logger.info(f"   Call spread: {call_short_strike}/{call_long_strike}")
                            logger.info(f"   Total credit: ${total_credit:.2f}")
                            
                            await self.execute_iron_condor(
                                symbol, put_long_strike, put_short_strike,
                                call_short_strike, call_long_strike, total_credit
                            )
                            
            except Exception as e:
                logger.error(f"Error scanning Iron Condors for {symbol}: {e}")
                
    async def scan_credit_spreads(self):
        """Scan for Bull Put and Bear Call spreads"""
        logger.info("💳 Scanning for Credit Spread opportunities...")
        
        for symbol in self.watchlist:
            try:
                if f"{symbol}_CS" in self.active_spreads:
                    continue
                    
                # Get historical data for trend analysis
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=20)
                )
                bars = self.stock_data_client.get_stock_bars(request)
                
                if symbol not in bars.data:
                    continue
                    
                df = bars.df.loc[symbol].reset_index()
                sma_20 = df['close'].mean()
                current_price = df['close'].iloc[-1]
                
                # Get option chain
                chain = await self.get_option_chain(symbol)
                if not chain:
                    continue
                    
                # Bull Put Spread if trending up
                if current_price > sma_20 * 1.01:
                    put_strikes = sorted([c['strike'] for c in chain if c['type'] == 'put'])
                    
                    for i in range(len(put_strikes) - 1):
                        short_strike = put_strikes[i+1]
                        long_strike = put_strikes[i]
                        
                        if short_strike < current_price * 0.95:  # OTM puts
                            short_put = next((c for c in chain if c['strike'] == short_strike and c['type'] == 'put'), None)
                            long_put = next((c for c in chain if c['strike'] == long_strike and c['type'] == 'put'), None)
                            
                            if short_put and long_put:
                                credit = short_put['bid'] - long_put['ask']
                                
                                if credit >= self.min_credit:
                                    logger.info(f"✅ Bull Put Spread on {symbol}:")
                                    logger.info(f"   Strikes: {long_strike}/{short_strike}")
                                    logger.info(f"   Credit: ${credit:.2f}")
                                    
                                    await self.execute_bull_put_spread(
                                        symbol, long_strike, short_strike, credit
                                    )
                                    break
                                    
                # Bear Call Spread if trending down
                elif current_price < sma_20 * 0.99:
                    call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                    
                    for i in range(len(call_strikes) - 1):
                        short_strike = call_strikes[i]
                        long_strike = call_strikes[i+1]
                        
                        if short_strike > current_price * 1.05:  # OTM calls
                            short_call = next((c for c in chain if c['strike'] == short_strike and c['type'] == 'call'), None)
                            long_call = next((c for c in chain if c['strike'] == long_strike and c['type'] == 'call'), None)
                            
                            if short_call and long_call:
                                credit = short_call['bid'] - long_call['ask']
                                
                                if credit >= self.min_credit:
                                    logger.info(f"✅ Bear Call Spread on {symbol}:")
                                    logger.info(f"   Strikes: {short_strike}/{long_strike}")
                                    logger.info(f"   Credit: ${credit:.2f}")
                                    
                                    await self.execute_bear_call_spread(
                                        symbol, short_strike, long_strike, credit
                                    )
                                    break
                                    
            except Exception as e:
                logger.error(f"Error scanning credit spreads for {symbol}: {e}")
                
    async def scan_butterfly_spreads(self):
        """Scan for Butterfly spread opportunities"""
        logger.info("🦋 Scanning for Butterfly spread opportunities...")
        
        for symbol in self.watchlist[:5]:  # Focus on most liquid
            try:
                if f"{symbol}_BF" in self.active_spreads:
                    continue
                    
                chain = await self.get_option_chain(symbol)
                if not chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Find ATM and OTM strikes for butterfly
                call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                
                # Find strikes for butterfly (buy 1, sell 2, buy 1)
                atm_strike = min(call_strikes, key=lambda x: abs(x - stock_price))
                atm_index = call_strikes.index(atm_strike)
                
                if atm_index > 0 and atm_index < len(call_strikes) - 1:
                    lower_strike = call_strikes[atm_index - 1]
                    middle_strike = atm_strike
                    upper_strike = call_strikes[atm_index + 1]
                    
                    # Get option prices
                    lower_call = next((c for c in chain if c['strike'] == lower_strike and c['type'] == 'call'), None)
                    middle_call = next((c for c in chain if c['strike'] == middle_strike and c['type'] == 'call'), None)
                    upper_call = next((c for c in chain if c['strike'] == upper_strike and c['type'] == 'call'), None)
                    
                    if all([lower_call, middle_call, upper_call]):
                        # Calculate net debit
                        debit = lower_call['ask'] - 2 * middle_call['bid'] + upper_call['ask']
                        max_profit = (middle_strike - lower_strike) - debit
                        
                        if max_profit > debit * 2:  # Good risk/reward
                            logger.info(f"✅ Butterfly spread on {symbol}:")
                            logger.info(f"   Strikes: {lower_strike}/{middle_strike}/{upper_strike}")
                            logger.info(f"   Net debit: ${debit:.2f}")
                            logger.info(f"   Max profit: ${max_profit:.2f}")
                            
                            await self.execute_butterfly_spread(
                                symbol, lower_strike, middle_strike, upper_strike, debit
                            )
                            
            except Exception as e:
                logger.error(f"Error scanning butterflies for {symbol}: {e}")
                
    async def scan_calendar_spreads(self):
        """Scan for Calendar spread opportunities"""
        logger.info("📅 Scanning for Calendar spread opportunities...")
        
        for symbol in self.watchlist[:10]:
            try:
                if f"{symbol}_CAL" in self.active_spreads:
                    continue
                    
                # Get options for different expirations
                # Near term (20-30 days)
                near_chain = await self.get_option_chain_by_dte(symbol, 20, 30)
                # Far term (40-50 days)
                far_chain = await self.get_option_chain_by_dte(symbol, 40, 50)
                
                if not near_chain or not far_chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Find ATM strike
                strikes = sorted(list(set([c['strike'] for c in near_chain])))
                atm_strike = min(strikes, key=lambda x: abs(x - stock_price))
                
                # Get near and far term options
                near_call = next((c for c in near_chain if c['strike'] == atm_strike and c['type'] == 'call'), None)
                far_call = next((c for c in far_chain if c['strike'] == atm_strike and c['type'] == 'call'), None)
                
                if near_call and far_call:
                    # Check IV skew
                    iv_skew = far_call['iv'] - near_call['iv']
                    
                    if iv_skew > self.iv_skew_threshold:
                        # Calendar spread opportunity (sell near, buy far)
                        debit = far_call['ask'] - near_call['bid']
                        
                        logger.info(f"✅ Calendar spread on {symbol}:")
                        logger.info(f"   Strike: {atm_strike}")
                        logger.info(f"   IV skew: {iv_skew:.2%}")
                        logger.info(f"   Net debit: ${debit:.2f}")
                        
                        await self.execute_calendar_spread(
                            symbol, atm_strike, near_call['expiration'], 
                            far_call['expiration'], debit
                        )
                        
            except Exception as e:
                logger.error(f"Error scanning calendar spreads for {symbol}: {e}")
                
    async def scan_diagonal_spreads(self):
        """Scan for Diagonal spread opportunities"""
        logger.info("↗️ Scanning for Diagonal spread opportunities...")
        
        for symbol in self.watchlist[:5]:
            try:
                if f"{symbol}_DIAG" in self.active_spreads:
                    continue
                    
                # Similar to calendar but different strikes
                near_chain = await self.get_option_chain_by_dte(symbol, 20, 30)
                far_chain = await self.get_option_chain_by_dte(symbol, 40, 50)
                
                if not near_chain or not far_chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Poor man's covered call setup
                # Buy ITM far-dated call, sell OTM near-dated call
                itm_strike = stock_price * 0.95
                otm_strike = stock_price * 1.05
                
                # Find closest strikes
                far_strikes = sorted([c['strike'] for c in far_chain if c['type'] == 'call'])
                near_strikes = sorted([c['strike'] for c in near_chain if c['type'] == 'call'])
                
                if far_strikes and near_strikes:
                    buy_strike = min(far_strikes, key=lambda x: abs(x - itm_strike))
                    sell_strike = min(near_strikes, key=lambda x: abs(x - otm_strike))
                    
                    if buy_strike < stock_price < sell_strike:
                        buy_call = next((c for c in far_chain if c['strike'] == buy_strike and c['type'] == 'call'), None)
                        sell_call = next((c for c in near_chain if c['strike'] == sell_strike and c['type'] == 'call'), None)
                        
                        if buy_call and sell_call:
                            net_debit = buy_call['ask'] - sell_call['bid']
                            
                            logger.info(f"✅ Diagonal spread (PMCC) on {symbol}:")
                            logger.info(f"   Buy: {buy_strike} call (far)")
                            logger.info(f"   Sell: {sell_strike} call (near)")
                            logger.info(f"   Net debit: ${net_debit:.2f}")
                            
                            await self.execute_diagonal_spread(
                                symbol, buy_strike, sell_strike,
                                buy_call['expiration'], sell_call['expiration'], net_debit
                            )
                            
            except Exception as e:
                logger.error(f"Error scanning diagonal spreads for {symbol}: {e}")
                
    async def scan_straddles_strangles(self):
        """Scan for Straddle and Strangle opportunities"""
        logger.info("🎪 Scanning for Straddle/Strangle opportunities...")
        
        for symbol in ['SPY', 'QQQ', 'IWM']:  # High volume ETFs
            try:
                if f"{symbol}_VOL" in self.active_spreads:
                    continue
                    
                # Check for high IV rank
                iv_rank = await self.calculate_iv_rank(symbol)
                
                if iv_rank > self.iv_percentile_threshold:
                    chain = await self.get_option_chain(symbol)
                    if not chain:
                        continue
                        
                    # Get current stock price
                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                    stock_price = float(quotes[symbol].ask_price)
                    
                    # Straddle - ATM
                    strikes = sorted(list(set([c['strike'] for c in chain])))
                    atm_strike = min(strikes, key=lambda x: abs(x - stock_price))
                    
                    atm_call = next((c for c in chain if c['strike'] == atm_strike and c['type'] == 'call'), None)
                    atm_put = next((c for c in chain if c['strike'] == atm_strike and c['type'] == 'put'), None)
                    
                    if atm_call and atm_put:
                        # Sell straddle in high IV
                        credit = atm_call['bid'] + atm_put['bid']
                        
                        logger.info(f"✅ Short Straddle on {symbol}:")
                        logger.info(f"   Strike: {atm_strike}")
                        logger.info(f"   IV Rank: {iv_rank:.0f}%")
                        logger.info(f"   Credit: ${credit:.2f}")
                        
                        await self.execute_straddle(symbol, atm_strike, credit, 'sell')
                        
                elif iv_rank < 30:  # Low IV - buy volatility
                    chain = await self.get_option_chain(symbol)
                    if not chain:
                        continue
                        
                    # Get current stock price
                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                    stock_price = float(quotes[symbol].ask_price)
                    
                    # Strangle - OTM
                    put_strike = stock_price * 0.97
                    call_strike = stock_price * 1.03
                    
                    put_strikes = sorted([c['strike'] for c in chain if c['type'] == 'put'])
                    call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                    
                    if put_strikes and call_strikes:
                        otm_put_strike = min(put_strikes, key=lambda x: abs(x - put_strike))
                        otm_call_strike = min(call_strikes, key=lambda x: abs(x - call_strike))
                        
                        otm_put = next((c for c in chain if c['strike'] == otm_put_strike and c['type'] == 'put'), None)
                        otm_call = next((c for c in chain if c['strike'] == otm_call_strike and c['type'] == 'call'), None)
                        
                        if otm_put and otm_call:
                            # Buy strangle in low IV
                            debit = otm_put['ask'] + otm_call['ask']
                            
                            logger.info(f"✅ Long Strangle on {symbol}:")
                            logger.info(f"   Strikes: {otm_put_strike}/{otm_call_strike}")
                            logger.info(f"   IV Rank: {iv_rank:.0f}%")
                            logger.info(f"   Debit: ${debit:.2f}")
                            
                            await self.execute_strangle(
                                symbol, otm_put_strike, otm_call_strike, debit, 'buy'
                            )
                            
            except Exception as e:
                logger.error(f"Error scanning straddles/strangles for {symbol}: {e}")
                
    async def scan_ratio_spreads(self):
        """Scan for Ratio spread opportunities"""
        logger.info("⚖️ Scanning for Ratio spread opportunities...")
        
        for symbol in self.watchlist[:5]:
            try:
                if f"{symbol}_RATIO" in self.active_spreads:
                    continue
                    
                chain = await self.get_option_chain(symbol)
                if not chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Call ratio spread (1x2) - neutral to bearish
                call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                
                for i in range(len(call_strikes) - 1):
                    buy_strike = call_strikes[i]
                    sell_strike = call_strikes[i+1]
                    
                    if buy_strike > stock_price:  # Both OTM
                        buy_call = next((c for c in chain if c['strike'] == buy_strike and c['type'] == 'call'), None)
                        sell_call = next((c for c in chain if c['strike'] == sell_strike and c['type'] == 'call'), None)
                        
                        if buy_call and sell_call:
                            # Buy 1, sell 2
                            net_credit = 2 * sell_call['bid'] - buy_call['ask']
                            
                            if net_credit > 0:
                                logger.info(f"✅ Call Ratio Spread (1x2) on {symbol}:")
                                logger.info(f"   Buy 1x {buy_strike} call")
                                logger.info(f"   Sell 2x {sell_strike} call")
                                logger.info(f"   Net credit: ${net_credit:.2f}")
                                
                                await self.execute_ratio_spread(
                                    symbol, buy_strike, sell_strike, net_credit, 'call'
                                )
                                break
                                
            except Exception as e:
                logger.error(f"Error scanning ratio spreads for {symbol}: {e}")
                
    async def scan_jade_lizard(self):
        """Scan for Jade Lizard opportunities (short put + short call spread)"""
        logger.info("🦎 Scanning for Jade Lizard opportunities...")
        
        for symbol in self.watchlist[:5]:
            try:
                if f"{symbol}_JL" in self.active_spreads:
                    continue
                    
                chain = await self.get_option_chain(symbol)
                if not chain:
                    continue
                    
                # Get current stock price
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data_client.get_stock_latest_quote(quote_request)
                stock_price = float(quotes[symbol].ask_price)
                
                # Find strikes
                put_strikes = sorted([c['strike'] for c in chain if c['type'] == 'put'])
                call_strikes = sorted([c['strike'] for c in chain if c['type'] == 'call'])
                
                # Short OTM put
                otm_put_strike = None
                for strike in put_strikes:
                    if strike < stock_price * 0.95:
                        otm_put_strike = strike
                        break
                        
                # Short call spread (bear call spread)
                short_call_strike = None
                long_call_strike = None
                for i, strike in enumerate(call_strikes):
                    if strike > stock_price * 1.05:
                        short_call_strike = strike
                        if i < len(call_strikes) - 1:
                            long_call_strike = call_strikes[i+1]
                        break
                        
                if all([otm_put_strike, short_call_strike, long_call_strike]):
                    short_put = next((c for c in chain if c['strike'] == otm_put_strike and c['type'] == 'put'), None)
                    short_call = next((c for c in chain if c['strike'] == short_call_strike and c['type'] == 'call'), None)
                    long_call = next((c for c in chain if c['strike'] == long_call_strike and c['type'] == 'call'), None)
                    
                    if all([short_put, short_call, long_call]):
                        put_credit = short_put['bid']
                        call_spread_credit = short_call['bid'] - long_call['ask']
                        total_credit = put_credit + call_spread_credit
                        
                        # Jade Lizard should collect more than width of call spread
                        call_spread_width = long_call_strike - short_call_strike
                        
                        if total_credit > call_spread_width:
                            logger.info(f"✅ Jade Lizard on {symbol}:")
                            logger.info(f"   Short put: {otm_put_strike}")
                            logger.info(f"   Call spread: {short_call_strike}/{long_call_strike}")
                            logger.info(f"   Total credit: ${total_credit:.2f}")
                            logger.info(f"   No upside risk!")
                            
                            await self.execute_jade_lizard(
                                symbol, otm_put_strike, short_call_strike, 
                                long_call_strike, total_credit
                            )
                            
            except Exception as e:
                logger.error(f"Error scanning Jade Lizards for {symbol}: {e}")
                
    async def get_option_chain_by_dte(self, symbol: str, min_dte: int, max_dte: int) -> List[Dict]:
        """Get option chain for specific DTE range"""
        try:
            today = date.today()
            min_expiry = today + timedelta(days=min_dte)
            max_expiry = today + timedelta(days=max_dte)
            
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=min_expiry.isoformat(),
                expiration_date_lte=max_expiry.isoformat()
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            
            filtered_contracts = []
            for contract in contracts:
                if contract.open_interest >= 100:  # Minimum liquidity
                    filtered_contracts.append({
                        'symbol': contract.symbol,
                        'strike': float(contract.strike_price),
                        'expiration': contract.expiration_date,
                        'type': contract.contract_type,
                        'bid': float(contract.bid_price) if contract.bid_price else 0,
                        'ask': float(contract.ask_price) if contract.ask_price else 0,
                        'iv': 0.3  # Default IV
                    })
                    
            return filtered_contracts
            
        except Exception as e:
            logger.error(f"Error getting option chain by DTE: {e}")
            return []
            
    async def calculate_iv_rank(self, symbol: str) -> float:
        """Calculate IV rank for a symbol"""
        try:
            # This would normally calculate historical IV percentile
            # For demo, return random value
            import random
            return random.uniform(20, 80)
        except:
            return 50
            
    # Execution methods for each strategy
    async def execute_iron_condor(self, symbol: str, put_long: float, put_short: float,
                                  call_short: float, call_long: float, credit: float):
        """Execute Iron Condor spread"""
        try:
            spread_id = f"{symbol}_IC_{int(time.time())}"
            
            # Would place 4 orders here
            logger.info(f"📝 Placing Iron Condor orders for {symbol}")
            
            # Track the spread
            self.active_spreads[f"{symbol}_IC"] = {
                'spread_id': spread_id,
                'type': 'iron_condor',
                'symbol': symbol,
                'strikes': {
                    'put_long': put_long,
                    'put_short': put_short,
                    'call_short': call_short,
                    'call_long': call_long
                },
                'credit': credit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['iron_condor']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Iron Condor: {e}")
            
    async def execute_bull_put_spread(self, symbol: str, long_strike: float, 
                                     short_strike: float, credit: float):
        """Execute Bull Put Spread"""
        try:
            spread_id = f"{symbol}_BPS_{int(time.time())}"
            
            logger.info(f"📝 Placing Bull Put Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_CS"] = {
                'spread_id': spread_id,
                'type': 'bull_put_spread',
                'symbol': symbol,
                'long_strike': long_strike,
                'short_strike': short_strike,
                'credit': credit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['bull_put_spread']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Bull Put Spread: {e}")
            
    async def execute_bear_call_spread(self, symbol: str, short_strike: float,
                                      long_strike: float, credit: float):
        """Execute Bear Call Spread"""
        try:
            spread_id = f"{symbol}_BCS_{int(time.time())}"
            
            logger.info(f"📝 Placing Bear Call Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_CS"] = {
                'spread_id': spread_id,
                'type': 'bear_call_spread',
                'symbol': symbol,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'credit': credit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['bear_call_spread']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Bear Call Spread: {e}")
            
    async def execute_butterfly_spread(self, symbol: str, lower: float, middle: float,
                                      upper: float, debit: float):
        """Execute Butterfly Spread"""
        try:
            spread_id = f"{symbol}_BF_{int(time.time())}"
            
            logger.info(f"📝 Placing Butterfly Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_BF"] = {
                'spread_id': spread_id,
                'type': 'butterfly',
                'symbol': symbol,
                'strikes': [lower, middle, upper],
                'debit': debit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['butterfly']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Butterfly Spread: {e}")
            
    async def execute_calendar_spread(self, symbol: str, strike: float, near_exp: str,
                                     far_exp: str, debit: float):
        """Execute Calendar Spread"""
        try:
            spread_id = f"{symbol}_CAL_{int(time.time())}"
            
            logger.info(f"📝 Placing Calendar Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_CAL"] = {
                'spread_id': spread_id,
                'type': 'calendar_spread',
                'symbol': symbol,
                'strike': strike,
                'near_expiration': near_exp,
                'far_expiration': far_exp,
                'debit': debit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['calendar_spread']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Calendar Spread: {e}")
            
    async def execute_diagonal_spread(self, symbol: str, buy_strike: float, sell_strike: float,
                                     buy_exp: str, sell_exp: str, debit: float):
        """Execute Diagonal Spread"""
        try:
            spread_id = f"{symbol}_DIAG_{int(time.time())}"
            
            logger.info(f"📝 Placing Diagonal Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_DIAG"] = {
                'spread_id': spread_id,
                'type': 'diagonal_spread',
                'symbol': symbol,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike,
                'buy_expiration': buy_exp,
                'sell_expiration': sell_exp,
                'debit': debit,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['diagonal_spread']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Diagonal Spread: {e}")
            
    async def execute_straddle(self, symbol: str, strike: float, premium: float, side: str):
        """Execute Straddle"""
        try:
            spread_id = f"{symbol}_STRDL_{int(time.time())}"
            
            logger.info(f"📝 Placing {side} Straddle orders for {symbol}")
            
            self.active_spreads[f"{symbol}_VOL"] = {
                'spread_id': spread_id,
                'type': 'straddle',
                'symbol': symbol,
                'strike': strike,
                'side': side,
                'premium': premium,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['straddle']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Straddle: {e}")
            
    async def execute_strangle(self, symbol: str, put_strike: float, call_strike: float,
                              premium: float, side: str):
        """Execute Strangle"""
        try:
            spread_id = f"{symbol}_STRGL_{int(time.time())}"
            
            logger.info(f"📝 Placing {side} Strangle orders for {symbol}")
            
            self.active_spreads[f"{symbol}_VOL"] = {
                'spread_id': spread_id,
                'type': 'strangle',
                'symbol': symbol,
                'put_strike': put_strike,
                'call_strike': call_strike,
                'side': side,
                'premium': premium,
                'entry_time': datetime.now()
            }
            
            self.spread_performance['strangle']['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error executing Strangle: {e}")
            
    async def execute_ratio_spread(self, symbol: str, buy_strike: float, sell_strike: float,
                                  credit: float, option_type: str):
        """Execute Ratio Spread"""
        try:
            spread_id = f"{symbol}_RATIO_{int(time.time())}"
            
            logger.info(f"📝 Placing Ratio Spread orders for {symbol}")
            
            self.active_spreads[f"{symbol}_RATIO"] = {
                'spread_id': spread_id,
                'type': 'ratio_spread',
                'symbol': symbol,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike,
                'option_type': option_type,
                'credit': credit,
                'entry_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing Ratio Spread: {e}")
            
    async def execute_jade_lizard(self, symbol: str, put_strike: float, call_short: float,
                                 call_long: float, credit: float):
        """Execute Jade Lizard"""
        try:
            spread_id = f"{symbol}_JL_{int(time.time())}"
            
            logger.info(f"📝 Placing Jade Lizard orders for {symbol}")
            
            self.active_spreads[f"{symbol}_JL"] = {
                'spread_id': spread_id,
                'type': 'jade_lizard',
                'symbol': symbol,
                'put_strike': put_strike,
                'call_strikes': [call_short, call_long],
                'credit': credit,
                'entry_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing Jade Lizard: {e}")
            
    async def manage_spreads(self):
        """Manage existing option spreads"""
        logger.info("🔧 Managing option spreads...")
        
        # In a real implementation, this would check each spread's P&L
        # and close if profit target or stop loss is hit
        
    async def display_portfolio_status(self):
        """Display options portfolio status"""
        try:
            if self.active_spreads:
                logger.info(f"\n📊 Active Option Spreads: {len(self.active_spreads)}")
                
                for key, spread in self.active_spreads.items():
                    logger.info(f"\n{spread['type'].replace('_', ' ').title()} - {spread['symbol']}")
                    if spread['type'] == 'iron_condor':
                        logger.info(f"   Put: {spread['strikes']['put_long']}/{spread['strikes']['put_short']}")
                        logger.info(f"   Call: {spread['strikes']['call_short']}/{spread['strikes']['call_long']}")
                    logger.info(f"   Entry: {spread['entry_time'].strftime('%H:%M:%S')}")
                    
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")
            
    async def display_spread_performance(self):
        """Display spread strategy performance"""
        logger.info("\n📈 Options Strategy Performance:")
        
        total_trades = 0
        for strategy, stats in self.spread_performance.items():
            if stats['trades'] > 0:
                total_trades += stats['trades']
                logger.info(f"\n{strategy.replace('_', ' ').title()}:")
                logger.info(f"  Trades: {stats['trades']}")
                
        logger.info(f"\n🎯 Total Option Spreads: {total_trades}")

async def main():
    """Main entry point"""
    trader = AdvancedOptionsTrader()
    await trader.run()

if __name__ == "__main__":
    # Set environment variables
    os.environ['ALPACA_API_KEY'] = 'PKEP9PIBDKOSUGHHY44Z'
    os.environ['ALPACA_SECRET_KEY'] = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")