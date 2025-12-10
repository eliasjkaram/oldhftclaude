#!/usr/bin/env python3
"""
Alpaca Options Trading System - Working Implementation
Based on Alpaca's actual options API documentation
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import requests
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

class AlpacaOptionsWorkingSystem:
    def __init__(self):
        """Initialize the working options trading system"""
        # API credentials
        self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
        self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
        self.base_url = 'https://paper-api.alpaca.markets'
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Headers for direct API calls
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # Trading parameters
        self.scan_interval = 30
        self.position_size = 1  # 1 contract
        self.max_spreads = 5
        
        # Target stocks
        self.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD']
        
        # Track active spreads
        self.active_spreads = {}
        
    def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Alpaca Options Trading System - Working Implementation")
        logger.info("Trading real option contracts via Alpaca Options API")
        
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
                
                # Scan and trade options
                self.scan_option_opportunities()
                
                # Display portfolio
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
        """Check account status"""
        try:
            account = self.trading_client.get_account()
            logger.info("\n💼 Account Status:")
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
            
    def get_option_contracts(self, underlying: str, expiry_min: str = None, expiry_max: str = None):
        """Get option contracts using Alpaca's options API"""
        try:
            # Use the options contracts endpoint
            url = f"{self.base_url}/v2/options/contracts"
            
            params = {
                'underlying_symbols': underlying,
                'status': 'active'
            }
            
            # Add expiry filters if provided
            if expiry_min:
                params['expiration_date_gte'] = expiry_min
            if expiry_max:
                params['expiration_date_lte'] = expiry_max
                
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                if contracts:
                    logger.info(f"Found {len(contracts)} option contracts for {underlying}")
                    return contracts
                else:
                    logger.info(f"No option contracts found for {underlying}")
                    
            else:
                logger.error(f"API Error: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            logger.error(f"Error getting option contracts: {e}")
            
        return []
        
    def scan_option_opportunities(self):
        """Scan for option trading opportunities"""
        logger.info("\n🔍 Scanning for option opportunities...")
        
        for underlying in self.watchlist[:3]:  # Top 3 for demo
            try:
                # Get current stock price
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=underlying)
                quotes = self.data_client.get_stock_latest_quote(quote_req)
                
                if underlying not in quotes:
                    continue
                    
                stock_price = float(quotes[underlying].ask_price)
                logger.info(f"\n{underlying} - Current Price: ${stock_price:.2f}")
                
                # Set expiry range (20-45 days out)
                today = datetime.now().date()
                expiry_min = (today + timedelta(days=20)).strftime('%Y-%m-%d')
                expiry_max = (today + timedelta(days=45)).strftime('%Y-%m-%d')
                
                # Get option contracts
                contracts = self.get_option_contracts(underlying, expiry_min, expiry_max)
                
                if contracts:
                    # Group by expiration
                    expirations = {}
                    for contract in contracts:
                        exp_date = contract.get('expiration_date')
                        if exp_date not in expirations:
                            expirations[exp_date] = {'calls': [], 'puts': []}
                            
                        if contract.get('type') == 'call':
                            expirations[exp_date]['calls'].append(contract)
                        else:
                            expirations[exp_date]['puts'].append(contract)
                            
                    # Find spread opportunities
                    for exp_date, options in list(expirations.items())[:2]:  # First 2 expirations
                        logger.info(f"\n  Expiration: {exp_date}")
                        
                        # Look for Bull Put Spread
                        if len(options['puts']) >= 2:
                            self.find_bull_put_spread(underlying, options['puts'], stock_price)
                            
                        # Look for Iron Condor
                        if len(options['puts']) >= 2 and len(options['calls']) >= 2:
                            self.find_iron_condor(underlying, options, stock_price)
                            
                        # Look for Calendar Spread
                        if exp_date != list(expirations.keys())[0]:  # Not the nearest expiry
                            self.find_calendar_spread(underlying, expirations, stock_price)
                            
            except Exception as e:
                logger.error(f"Error scanning {underlying}: {e}")
                
    def find_bull_put_spread(self, underlying: str, puts: List[Dict], stock_price: float):
        """Find Bull Put Spread opportunities"""
        try:
            # Sort puts by strike
            puts = sorted(puts, key=lambda x: float(x.get('strike_price', 0)))
            
            # Find OTM puts
            otm_puts = [p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.95]
            
            if len(otm_puts) >= 2:
                # Select strikes
                short_put = otm_puts[-1]  # Higher strike (closer to ATM)
                long_put = otm_puts[-2]   # Lower strike
                
                short_strike = float(short_put.get('strike_price'))
                long_strike = float(long_put.get('strike_price'))
                spread_width = short_strike - long_strike
                
                logger.info(f"\n  💡 Bull Put Spread opportunity:")
                logger.info(f"     Short Put: {short_put.get('symbol')} (Strike: ${short_strike})")
                logger.info(f"     Long Put: {long_put.get('symbol')} (Strike: ${long_strike})")
                logger.info(f"     Spread Width: ${spread_width:.2f}")
                
                # Check if we should trade
                if self.should_trade_bull_put(underlying, stock_price):
                    self.execute_bull_put_spread(short_put, long_put)
                    
        except Exception as e:
            logger.error(f"Error finding bull put spread: {e}")
            
    def find_iron_condor(self, underlying: str, options: Dict, stock_price: float):
        """Find Iron Condor opportunities"""
        try:
            puts = sorted(options['puts'], key=lambda x: float(x.get('strike_price', 0)))
            calls = sorted(options['calls'], key=lambda x: float(x.get('strike_price', 0)))
            
            # Find OTM options
            otm_puts = [p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.95]
            otm_calls = [c for c in calls if float(c.get('strike_price', 0)) > stock_price * 1.05]
            
            if len(otm_puts) >= 2 and len(otm_calls) >= 2:
                # Select strikes
                put_short = otm_puts[-1]
                put_long = otm_puts[-2]
                call_short = otm_calls[0]
                call_long = otm_calls[1]
                
                logger.info(f"\n  🦅 Iron Condor opportunity:")
                logger.info(f"     Put Spread: {put_long.get('symbol')}/{put_short.get('symbol')}")
                logger.info(f"     Call Spread: {call_short.get('symbol')}/{call_long.get('symbol')}")
                
                if underlying in ['SPY', 'QQQ'] and len(self.active_spreads) < self.max_spreads:
                    self.execute_iron_condor(put_long, put_short, call_short, call_long)
                    
        except Exception as e:
            logger.error(f"Error finding iron condor: {e}")
            
    def find_calendar_spread(self, underlying: str, all_expirations: Dict, stock_price: float):
        """Find Calendar Spread opportunities"""
        try:
            exp_dates = sorted(all_expirations.keys())
            if len(exp_dates) >= 2:
                near_exp = exp_dates[0]
                far_exp = exp_dates[1]
                
                # Find ATM options
                near_calls = all_expirations[near_exp]['calls']
                far_calls = all_expirations[far_exp]['calls']
                
                if near_calls and far_calls:
                    # Find closest to ATM
                    near_atm = min(near_calls, key=lambda x: abs(float(x.get('strike_price', 0)) - stock_price))
                    far_atm = next((c for c in far_calls if c.get('strike_price') == near_atm.get('strike_price')), None)
                    
                    if far_atm:
                        logger.info(f"\n  📅 Calendar Spread opportunity:")
                        logger.info(f"     Sell: {near_atm.get('symbol')} (Near: {near_exp})")
                        logger.info(f"     Buy: {far_atm.get('symbol')} (Far: {far_exp})")
                        
        except Exception as e:
            logger.error(f"Error finding calendar spread: {e}")
            
    def should_trade_bull_put(self, underlying: str, stock_price: float) -> bool:
        """Determine if we should trade a bull put spread"""
        try:
            # Get recent price data
            bars_req = StockBarsRequest(
                symbol_or_symbols=underlying,
                timeframe=TimeFrame.Day,
                limit=20
            )
            bars = self.data_client.get_stock_bars(bars_req)
            
            if underlying in bars.data:
                df = bars.df.loc[underlying]
                sma_20 = df['close'].mean()
                
                # Bullish if above SMA
                if stock_price > sma_20:
                    logger.info(f"     ✅ Bullish trend confirmed (Price > SMA20)")
                    return True
                    
        except Exception as e:
            logger.error(f"Error analyzing {underlying}: {e}")
            
        return False
        
    def execute_bull_put_spread(self, short_put: Dict, long_put: Dict):
        """Execute a Bull Put Spread"""
        try:
            logger.info(f"\n  🚀 Executing Bull Put Spread...")
            
            # Sell the higher strike put
            sell_order = MarketOrderRequest(
                symbol=short_put.get('symbol'),
                qty=self.position_size,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Buy the lower strike put
            buy_order = MarketOrderRequest(
                symbol=long_put.get('symbol'),
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit orders
            sell_result = self.trading_client.submit_order(sell_order)
            logger.info(f"     ✅ Sold {short_put.get('symbol')}")
            
            buy_result = self.trading_client.submit_order(buy_order)
            logger.info(f"     ✅ Bought {long_put.get('symbol')}")
            
            # Track the spread
            spread_id = f"BPS_{short_put.get('underlying_symbol')}_{int(time.time())}"
            self.active_spreads[spread_id] = {
                'type': 'bull_put_spread',
                'legs': [short_put.get('symbol'), long_put.get('symbol')],
                'entry_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing bull put spread: {e}")
            
    def execute_iron_condor(self, put_long: Dict, put_short: Dict, 
                           call_short: Dict, call_long: Dict):
        """Execute an Iron Condor"""
        try:
            logger.info(f"\n  🚀 Executing Iron Condor...")
            
            # Place all four legs
            legs = [
                (put_long, OrderSide.BUY, "Long Put"),
                (put_short, OrderSide.SELL, "Short Put"),
                (call_short, OrderSide.SELL, "Short Call"),
                (call_long, OrderSide.BUY, "Long Call")
            ]
            
            for option, side, desc in legs:
                order = MarketOrderRequest(
                    symbol=option.get('symbol'),
                    qty=self.position_size,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                result = self.trading_client.submit_order(order)
                logger.info(f"     ✅ {desc}: {option.get('symbol')}")
                
            # Track the spread
            spread_id = f"IC_{put_short.get('underlying_symbol')}_{int(time.time())}"
            self.active_spreads[spread_id] = {
                'type': 'iron_condor',
                'legs': [leg[0].get('symbol') for leg in legs],
                'entry_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing iron condor: {e}")
            
    def display_portfolio_status(self):
        """Display portfolio status"""
        try:
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n📊 Portfolio Status:")
            logger.info(f"Active Positions: {len(positions)}")
            
            # Separate options and stocks
            option_positions = []
            stock_positions = []
            
            for position in positions:
                # Check if it's an option (has expiry date in symbol)
                if any(char.isdigit() for char in position.symbol[3:]):
                    option_positions.append(position)
                else:
                    stock_positions.append(position)
                    
            if option_positions:
                logger.info(f"\n🎯 Option Positions: {len(option_positions)}")
                for pos in option_positions:
                    logger.info(f"   {pos.symbol}: {pos.qty} contracts")
                    logger.info(f"      Entry: ${float(pos.avg_entry_price):.2f}")
                    logger.info(f"      P&L: ${float(pos.unrealized_pl):,.2f}")
                    
            # Show active spreads
            if self.active_spreads:
                logger.info(f"\n📊 Active Option Spreads: {len(self.active_spreads)}")
                for spread_id, spread_info in self.active_spreads.items():
                    logger.info(f"   {spread_info['type']}: {spread_id}")
                    
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")

def main():
    """Main entry point"""
    system = AlpacaOptionsWorkingSystem()
    system.run()

if __name__ == "__main__":
    main()