#!/usr/bin/env python3
"""
Working Options Trading System for Alpaca
Uses Alpaca's actual options trading capabilities
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

class AlpacaOptionsTrader:
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
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD'
        ]
        
        # Track trades
        self.active_trades = {}
        
    def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Alpaca Options Trading System")
        logger.info("Note: Options trading on Alpaca is currently in development")
        logger.info("This system will demonstrate options strategies using available features")
        
        self.check_account_status()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 Trading Cycle #{cycle}")
                logger.info(f"{'='*60}")
                
                # Check if market is open
                if not self.is_market_open():
                    logger.info("Market is closed")
                    time.sleep(300)
                    continue
                
                # Demonstrate options capabilities
                self.check_options_availability()
                self.demonstrate_spread_strategies()
                self.scan_for_opportunities()
                
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
            logger.info(f"Pattern Day Trader: {account.pattern_day_trader}")
            
            # Check for options trading permissions
            if hasattr(account, 'options_trading_level'):
                logger.info(f"Options Trading Level: {account.options_trading_level}")
            else:
                logger.info("Options Trading Level: Checking with Alpaca support...")
                
        except Exception as e:
            logger.error(f"Error checking account: {e}")
            
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    def check_options_availability(self):
        """Check what options contracts are available"""
        logger.info("\n🔍 Checking Options Availability...")
        
        for symbol in self.watchlist[:3]:  # Check first 3 symbols
            try:
                # Get current stock price
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.data_client.get_stock_latest_quote(quote_req)
                
                if symbol in quotes:
                    stock_price = float(quotes[symbol].ask_price)
                    logger.info(f"\n{symbol} - Current Price: ${stock_price:.2f}")
                    
                    # Try to get option contracts
                    today = datetime.now().date()
                    expiry_min = today + timedelta(days=self.min_dte)
                    expiry_max = today + timedelta(days=self.max_dte)
                    
                    request = GetOptionContractsRequest(
                        underlying_symbols=symbol,
                        expiration_date_gte=expiry_min,
                        expiration_date_lte=expiry_max,
                        strike_price_gte=stock_price * 0.90,
                        strike_price_lte=stock_price * 1.10
                    )
                    
                    try:
                        contracts = self.trading_client.get_option_contracts(request)
                        
                        if contracts:
                            logger.info(f"   Found {len(contracts)} option contracts")
                            
                            # Show sample contracts
                            for contract in contracts[:5]:
                                logger.info(f"   - {contract.symbol}: Strike ${contract.strike_price}, "
                                          f"Exp {contract.expiration_date}")
                        else:
                            logger.info(f"   No option contracts found (may not be available yet)")
                            
                    except Exception as e:
                        logger.info(f"   Options not available: {str(e)[:100]}")
                        
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                
    def demonstrate_spread_strategies(self):
        """Demonstrate various option spread strategies"""
        logger.info("\n📚 Option Spread Strategies Available:")
        
        strategies = {
            "Iron Condor": {
                "description": "Sell OTM put and call, buy further OTM put and call",
                "risk": "Limited",
                "reward": "Limited",
                "market_outlook": "Neutral/Range-bound",
                "example": "SPY: -1 390P, +1 385P, -1 410C, +1 415C"
            },
            "Bull Put Spread": {
                "description": "Sell OTM put, buy lower strike put",
                "risk": "Limited",
                "reward": "Limited", 
                "market_outlook": "Bullish",
                "example": "AAPL: -1 170P, +1 165P for $1.50 credit"
            },
            "Bear Call Spread": {
                "description": "Sell OTM call, buy higher strike call",
                "risk": "Limited",
                "reward": "Limited",
                "market_outlook": "Bearish",
                "example": "TSLA: -1 250C, +1 255C for $2.00 credit"
            },
            "Butterfly Spread": {
                "description": "Buy 1 low, sell 2 middle, buy 1 high strike",
                "risk": "Limited",
                "reward": "Limited",
                "market_outlook": "Neutral with pinning",
                "example": "MSFT: +1 380C, -2 385C, +1 390C"
            },
            "Calendar Spread": {
                "description": "Sell near-term, buy far-term same strike",
                "risk": "Limited",
                "reward": "Limited",
                "market_outlook": "Neutral with IV expansion",
                "example": "QQQ: -1 380C 30DTE, +1 380C 60DTE"
            },
            "Diagonal Spread (PMCC)": {
                "description": "Buy far-term ITM call, sell near-term OTM call",
                "risk": "Limited",
                "reward": "Limited",
                "market_outlook": "Moderately bullish",
                "example": "AMD: +1 120C 90DTE, -1 130C 30DTE"
            },
            "Jade Lizard": {
                "description": "Short OTM put + short call spread",
                "risk": "Downside only",
                "reward": "Limited",
                "market_outlook": "Neutral to bullish",
                "example": "SPY: -1 390P, -1 405C, +1 410C"
            },
            "Ratio Spread": {
                "description": "Buy 1 option, sell 2 further OTM options",
                "risk": "Unlimited on one side",
                "reward": "Limited",
                "market_outlook": "Directional with volatility contraction",
                "example": "NVDA: +1 280C, -2 290C"
            }
        }
        
        for name, details in strategies.items():
            logger.info(f"\n🎯 {name}:")
            logger.info(f"   Description: {details['description']}")
            logger.info(f"   Risk: {details['risk']}, Reward: {details['reward']}")
            logger.info(f"   Market Outlook: {details['market_outlook']}")
            logger.info(f"   Example: {details['example']}")
            
    def scan_for_opportunities(self):
        """Scan for trading opportunities"""
        logger.info("\n🔍 Scanning for Opportunities...")
        
        # Demonstrate what we would look for
        for symbol in self.watchlist:
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
                    
                    # Determine strategy based on conditions
                    if volatility > 40:
                        logger.info(f"\n{symbol}: High volatility ({volatility:.1f}%)")
                        logger.info(f"   Strategy: Iron Condor or Short Straddle")
                        logger.info(f"   Sell volatility when IV is elevated")
                        
                        # Show hypothetical Iron Condor
                        put_short = round(current_price * 0.95, 0)
                        put_long = round(current_price * 0.93, 0)
                        call_short = round(current_price * 1.05, 0)
                        call_long = round(current_price * 1.07, 0)
                        
                        logger.info(f"   Iron Condor: {put_long}P/{put_short}P - {call_short}C/{call_long}C")
                        
                    elif current_price > sma_20 * 1.02:
                        logger.info(f"\n{symbol}: Bullish trend (above SMA)")
                        logger.info(f"   Strategy: Bull Put Spread")
                        logger.info(f"   Collect premium below support")
                        
                        # Show hypothetical Bull Put Spread
                        short_put = round(current_price * 0.95, 0)
                        long_put = round(current_price * 0.93, 0)
                        
                        logger.info(f"   Bull Put Spread: -{short_put}P/+{long_put}P")
                        
                    elif volatility < 20:
                        logger.info(f"\n{symbol}: Low volatility ({volatility:.1f}%)")
                        logger.info(f"   Strategy: Calendar Spread or Diagonal")
                        logger.info(f"   Benefit from volatility expansion")
                        
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n📊 Portfolio Status:")
            
            if positions:
                logger.info(f"Active Positions: {len(positions)}")
                
                for position in positions:
                    logger.info(f"\n{position.symbol}:")
                    logger.info(f"   Quantity: {position.qty}")
                    logger.info(f"   Market Value: ${float(position.market_value):,.2f}")
                    logger.info(f"   Unrealized P&L: ${float(position.unrealized_pl):,.2f}")
                    
                    # Note if this is part of an options strategy
                    if position.symbol in self.active_trades:
                        logger.info(f"   Strategy: {self.active_trades[position.symbol]}")
            else:
                logger.info("No active positions")
                
            # Show what options strategies we would implement
            logger.info("\n🎯 Ready to Execute:")
            logger.info("• Iron Condors on high IV stocks")
            logger.info("• Credit spreads on trending stocks")
            logger.info("• Calendar spreads on low volatility")
            logger.info("• Diagonal spreads for income generation")
            logger.info("• Butterflies for range-bound markets")
            
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")
            
    def demonstrate_options_order(self):
        """Demonstrate how to place options orders when available"""
        logger.info("\n📝 Options Order Example:")
        
        example_code = """
# Bull Put Spread Example
symbol = 'SPY'
expiry = '2024-01-19'  # Options expiration

# Sell put at 395 strike
sell_put = MarketOrderRequest(
    symbol=f'SPY240119P00395000',  # SPY Jan 19 2024 395 Put
    qty=1,
    side=OrderSide.SELL,
    time_in_force=TimeInForce.DAY
)

# Buy put at 390 strike  
buy_put = MarketOrderRequest(
    symbol=f'SPY240119P00390000',  # SPY Jan 19 2024 390 Put
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# Submit orders
trading_client.submit_order(sell_put)
trading_client.submit_order(buy_put)
"""
        
        logger.info(example_code)

def main():
    """Main entry point"""
    bot = AlpacaOptionsTrader()
    bot.run()

if __name__ == "__main__":
    main()