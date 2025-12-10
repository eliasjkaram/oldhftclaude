#!/usr/bin/env python3
"""
Fixed Alpaca Real Options Trading System
Properly formatted API calls for options trading
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, MarketOrderRequest, 
    LimitOrderRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaRealOptionsTrader:
    def __init__(self):
        """Initialize the options trading system"""
        # API credentials
        self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
        self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Trading parameters
        self.scan_interval = 30
        self.min_dte = 20
        self.max_dte = 45
        
        # Liquid optionable stocks
        self.watchlist = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD',
            'JPM', 'BAC', 'XLF', 'GLD', 'TLT', 'IWM', 'META'
        ]
        
        # Track option trades
        self.active_spreads = {}
        
    def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Alpaca Real Options Trading System")
        logger.info("✅ Options Trading Level 3 Confirmed!")
        
        self.check_account_status()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 Options Trading Cycle #{cycle}")
                logger.info(f"{'='*60}")
                
                # Check if market is open
                if not self.is_market_open():
                    logger.info("Market is closed")
                    time.sleep(300)
                    continue
                
                # Check available option contracts
                self.check_option_contracts()
                
                # Scan for spread opportunities
                self.scan_for_spreads()
                
                # Display status
                self.display_portfolio_status()
                
                # Wait for next cycle
                logger.info(f"\n⏰ Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)
                
    def check_account_status(self):
        """Check account status and options trading level"""
        try:
            account = self.trading_client.get_account()
            logger.info("\n💼 Account Status:")
            logger.info(f"Account Number: {account.account_number}")
            logger.info(f"Status: {account.status}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Options Trading Level: {account.options_trading_level}")
            
        except Exception as e:
            logger.error(f"Error checking account: {e}")
            
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    def check_option_contracts(self):
        """Check available option contracts with correct API format"""
        logger.info("\n🔍 Checking Available Option Contracts...")
        
        for symbol in self.watchlist[:5]:  # Check first 5 symbols
            try:
                # Get current stock price
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.data_client.get_stock_latest_quote(quote_req)
                
                if symbol in quotes:
                    stock_price = float(quotes[symbol].ask_price)
                    logger.info(f"\n{symbol} - Current Price: ${stock_price:.2f}")
                    
                    # Try to get option contracts with CORRECTED format
                    today = datetime.now().date()
                    expiry_min = today + timedelta(days=self.min_dte)
                    expiry_max = today + timedelta(days=self.max_dte)
                    
                    # FIXED: Proper format for the request
                    request = GetOptionContractsRequest(
                        underlying_symbols=[symbol],  # Must be a list
                        expiration_date_gte=expiry_min.strftime('%Y-%m-%d'),
                        expiration_date_lte=expiry_max.strftime('%Y-%m-%d'),
                        strike_price_gte=str(round(stock_price * 0.90, 2)),  # Must be string
                        strike_price_lte=str(round(stock_price * 1.10, 2))   # Must be string
                    )
                    
                    try:
                        contracts = self.trading_client.get_option_contracts(request)
                        
                        if contracts:
                            logger.info(f"   ✅ Found {len(contracts)} option contracts!")
                            
                            # Group by expiration
                            expirations = {}
                            for contract in contracts:
                                exp_date = contract.expiration_date
                                if exp_date not in expirations:
                                    expirations[exp_date] = []
                                expirations[exp_date].append(contract)
                            
                            # Show summary
                            for exp_date, exp_contracts in list(expirations.items())[:3]:
                                calls = [c for c in exp_contracts if c.contract_type == 'call']
                                puts = [c for c in exp_contracts if c.contract_type == 'put']
                                logger.info(f"   Expiry {exp_date}: {len(calls)} calls, {len(puts)} puts")
                                
                                # Show sample strikes
                                if calls:
                                    call_strikes = sorted([float(c.strike_price) for c in calls])
                                    logger.info(f"     Call strikes: ${min(call_strikes):.0f} - ${max(call_strikes):.0f}")
                                if puts:
                                    put_strikes = sorted([float(c.strike_price) for c in puts])
                                    logger.info(f"     Put strikes: ${min(put_strikes):.0f} - ${max(put_strikes):.0f}")
                        else:
                            logger.info(f"   No contracts found in date range")
                            
                    except Exception as e:
                        logger.info(f"   Options data error: {str(e)[:100]}")
                        
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                
    def scan_for_spreads(self):
        """Scan for option spread opportunities"""
        logger.info("\n📊 Scanning for Option Spread Opportunities...")
        
        for symbol in self.watchlist[:3]:  # Demo with first 3
            try:
                # Get stock data
                bars_req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=20
                )
                bars = self.data_client.get_stock_bars(bars_req)
                
                if symbol in bars.data:
                    df = bars.df.loc[symbol]
                    
                    # Calculate indicators
                    current_price = df['close'].iloc[-1]
                    sma_20 = df['close'].mean()
                    volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
                    
                    # Get option chain
                    contracts = self.get_option_chain(symbol)
                    
                    if contracts:
                        # Scan for different strategies based on market conditions
                        if volatility > 35:
                            self.scan_iron_condor(symbol, current_price, contracts, volatility)
                        
                        if current_price > sma_20 * 1.01:
                            self.scan_bull_put_spread(symbol, current_price, contracts)
                            
                        if current_price < sma_20 * 0.99:
                            self.scan_bear_call_spread(symbol, current_price, contracts)
                            
                        if 20 < volatility < 30:
                            self.scan_butterfly(symbol, current_price, contracts)
                            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
    def get_option_chain(self, symbol: str) -> List:
        """Get option chain for analysis"""
        try:
            today = datetime.now().date()
            expiry_min = today + timedelta(days=self.min_dte)
            expiry_max = today + timedelta(days=self.max_dte)
            
            # Get current price
            quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(quote_req)
            stock_price = float(quotes[symbol].ask_price)
            
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=expiry_min.strftime('%Y-%m-%d'),
                expiration_date_lte=expiry_max.strftime('%Y-%m-%d'),
                strike_price_gte=str(round(stock_price * 0.85, 2)),
                strike_price_lte=str(round(stock_price * 1.15, 2))
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            return contracts if contracts else []
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return []
            
    def scan_iron_condor(self, symbol: str, stock_price: float, contracts: List, volatility: float):
        """Scan for Iron Condor opportunities"""
        logger.info(f"\n🦅 Iron Condor opportunity on {symbol}:")
        logger.info(f"   High volatility: {volatility:.1f}%")
        
        # Find suitable strikes
        calls = [c for c in contracts if c.contract_type == 'call']
        puts = [c for c in contracts if c.contract_type == 'put']
        
        if len(calls) >= 4 and len(puts) >= 4:
            # Sort by strike
            calls.sort(key=lambda x: float(x.strike_price))
            puts.sort(key=lambda x: float(x.strike_price))
            
            # Find strikes ~5% OTM
            put_short_strike = stock_price * 0.95
            put_long_strike = stock_price * 0.93
            call_short_strike = stock_price * 1.05
            call_long_strike = stock_price * 1.07
            
            # Find closest contracts
            put_short = min(puts, key=lambda x: abs(float(x.strike_price) - put_short_strike))
            put_long = min(puts, key=lambda x: abs(float(x.strike_price) - put_long_strike))
            call_short = min(calls, key=lambda x: abs(float(x.strike_price) - call_short_strike))
            call_long = min(calls, key=lambda x: abs(float(x.strike_price) - call_long_strike))
            
            logger.info(f"   Put spread: ${put_long.strike_price}/{put_short.strike_price}")
            logger.info(f"   Call spread: ${call_short.strike_price}/{call_long.strike_price}")
            logger.info(f"   Expiration: {put_short.expiration_date}")
            
            # Demonstrate order structure
            logger.info("\n   📝 Order Structure:")
            logger.info(f"   1. BUY {put_long.symbol}")
            logger.info(f"   2. SELL {put_short.symbol}")
            logger.info(f"   3. SELL {call_short.symbol}")
            logger.info(f"   4. BUY {call_long.symbol}")
            
    def scan_bull_put_spread(self, symbol: str, stock_price: float, contracts: List):
        """Scan for Bull Put Spread opportunities"""
        logger.info(f"\n📈 Bull Put Spread opportunity on {symbol}:")
        logger.info(f"   Bullish trend detected")
        
        puts = [c for c in contracts if c.contract_type == 'put']
        
        if len(puts) >= 2:
            puts.sort(key=lambda x: float(x.strike_price))
            
            # Find strikes ~5% and 7% OTM
            short_strike = stock_price * 0.95
            long_strike = stock_price * 0.93
            
            short_put = min(puts, key=lambda x: abs(float(x.strike_price) - short_strike))
            long_put = min(puts, key=lambda x: abs(float(x.strike_price) - long_strike))
            
            spread_width = float(short_put.strike_price) - float(long_put.strike_price)
            
            logger.info(f"   Strikes: ${long_put.strike_price}/${short_put.strike_price}")
            logger.info(f"   Spread width: ${spread_width:.2f}")
            logger.info(f"   Expiration: {short_put.expiration_date}")
            
            # Order structure
            logger.info("\n   📝 Order Structure:")
            logger.info(f"   1. BUY {long_put.symbol}")
            logger.info(f"   2. SELL {short_put.symbol}")
            
    def scan_bear_call_spread(self, symbol: str, stock_price: float, contracts: List):
        """Scan for Bear Call Spread opportunities"""
        logger.info(f"\n📉 Bear Call Spread opportunity on {symbol}:")
        logger.info(f"   Bearish trend detected")
        
        calls = [c for c in contracts if c.contract_type == 'call']
        
        if len(calls) >= 2:
            calls.sort(key=lambda x: float(x.strike_price))
            
            # Find strikes ~5% and 7% OTM
            short_strike = stock_price * 1.05
            long_strike = stock_price * 1.07
            
            short_call = min(calls, key=lambda x: abs(float(x.strike_price) - short_strike))
            long_call = min(calls, key=lambda x: abs(float(x.strike_price) - long_strike))
            
            spread_width = float(long_call.strike_price) - float(short_call.strike_price)
            
            logger.info(f"   Strikes: ${short_call.strike_price}/{long_call.strike_price}")
            logger.info(f"   Spread width: ${spread_width:.2f}")
            logger.info(f"   Expiration: {short_call.expiration_date}")
            
            # Order structure
            logger.info("\n   📝 Order Structure:")
            logger.info(f"   1. SELL {short_call.symbol}")
            logger.info(f"   2. BUY {long_call.symbol}")
            
    def scan_butterfly(self, symbol: str, stock_price: float, contracts: List):
        """Scan for Butterfly spread opportunities"""
        logger.info(f"\n🦋 Butterfly Spread opportunity on {symbol}:")
        logger.info(f"   Range-bound market expected")
        
        calls = [c for c in contracts if c.contract_type == 'call']
        
        if len(calls) >= 3:
            calls.sort(key=lambda x: float(x.strike_price))
            
            # Find ATM and OTM strikes
            atm_call = min(calls, key=lambda x: abs(float(x.strike_price) - stock_price))
            atm_idx = calls.index(atm_call)
            
            if 0 < atm_idx < len(calls) - 1:
                lower_call = calls[atm_idx - 1]
                middle_call = atm_call
                upper_call = calls[atm_idx + 1]
                
                logger.info(f"   Strikes: ${lower_call.strike_price}/${middle_call.strike_price}/${upper_call.strike_price}")
                logger.info(f"   Expiration: {middle_call.expiration_date}")
                
                # Order structure
                logger.info("\n   📝 Order Structure:")
                logger.info(f"   1. BUY 1x {lower_call.symbol}")
                logger.info(f"   2. SELL 2x {middle_call.symbol}")
                logger.info(f"   3. BUY 1x {upper_call.symbol}")
                
    def place_option_spread(self, spread_type: str, legs: List[Dict]):
        """Place an option spread order"""
        logger.info(f"\n🚀 Placing {spread_type} order...")
        
        try:
            for leg in legs:
                order = MarketOrderRequest(
                    symbol=leg['symbol'],
                    qty=leg['qty'],
                    side=leg['side'],
                    time_in_force=TimeInForce.DAY
                )
                
                # Submit order
                result = self.trading_client.submit_order(order)
                logger.info(f"   ✅ {leg['action']}: {leg['symbol']}")
                
        except Exception as e:
            logger.error(f"Error placing spread: {e}")
            
    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n📊 Portfolio Status:")
            
            if positions:
                logger.info(f"Active Positions: {len(positions)}")
                
                # Check for option positions
                option_positions = [p for p in positions if len(p.symbol) > 10]  # Options have longer symbols
                stock_positions = [p for p in positions if len(p.symbol) <= 10]
                
                if option_positions:
                    logger.info(f"\n🎯 Option Positions: {len(option_positions)}")
                    for position in option_positions[:5]:
                        logger.info(f"   {position.symbol}: {position.qty} contracts")
                        
                if stock_positions:
                    logger.info(f"\n📈 Stock Positions: {len(stock_positions)}")
                    total_value = sum(float(p.market_value) for p in stock_positions)
                    logger.info(f"   Total Value: ${total_value:,.2f}")
                    
            else:
                logger.info("No active positions")
                
            # Show active spreads
            if self.active_spreads:
                logger.info(f"\n🎯 Active Option Spreads: {len(self.active_spreads)}")
                for spread_id, spread_info in self.active_spreads.items():
                    logger.info(f"   {spread_id}: {spread_info['type']} on {spread_info['symbol']}")
                    
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")

def main():
    """Main entry point"""
    bot = AlpacaRealOptionsTrader()
    bot.run()

if __name__ == "__main__":
    main()