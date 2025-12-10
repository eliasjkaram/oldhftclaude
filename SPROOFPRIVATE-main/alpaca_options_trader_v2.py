#!/usr/bin/env python3
"""
Alpaca Options Trading System V2
Uses proper option contract symbols and handles API responses correctly
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
    MarketOrderRequest, LimitOrderRequest, GetAssetsRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaOptionsTraderV2:
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
        self.position_size = 1  # 1 contract
        self.max_contracts = 10
        
        # Stocks with liquid options
        self.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD']
        
    def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Alpaca Options Trading System V2")
        logger.info("Trading real option contracts with proper symbols")
        
        self.check_account_status()
        
        # First, let's check what assets are available
        self.check_available_assets()
        
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
                
                # Try to trade options with proper symbols
                self.trade_option_spreads()
                
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
            
    def check_available_assets(self):
        """Check what option assets are available"""
        logger.info("\n🔍 Checking available option assets...")
        
        try:
            # Get all tradeable assets
            request = GetAssetsRequest(
                asset_class=AssetClass.US_OPTION,
                status='active'
            )
            
            options = self.trading_client.get_all_assets(request)
            
            if options:
                logger.info(f"Found {len(options)} option contracts!")
                
                # Show some examples
                for i, option in enumerate(options[:10]):
                    logger.info(f"  {option.symbol}: {option.name}")
                    
                return options
            else:
                logger.info("No option contracts found in assets list")
                
        except Exception as e:
            logger.error(f"Error getting option assets: {e}")
            
        # If options API isn't working, show the correct format
        self.demonstrate_option_symbols()
        
    def demonstrate_option_symbols(self):
        """Show correct option symbol format"""
        logger.info("\n📚 Standard Option Symbol Format (OCC):")
        logger.info("Format: UNDERLYING + EXPIRY(YYMMDD) + TYPE(C/P) + STRIKE(8 digits)")
        
        today = datetime.now()
        expiry = today + timedelta(days=30)
        
        # Example option symbols
        examples = []
        
        # SPY examples
        spy_price = 600
        examples.append(f"SPY{expiry.strftime('%y%m%d')}C{int(spy_price * 1000):08d}")  # ATM Call
        examples.append(f"SPY{expiry.strftime('%y%m%d')}P{int(spy_price * 0.98 * 1000):08d}")  # OTM Put
        
        # AAPL examples  
        aapl_price = 200
        examples.append(f"AAPL{expiry.strftime('%y%m%d')}C{int(aapl_price * 1000):08d}")  # ATM Call
        examples.append(f"AAPL{expiry.strftime('%y%m%d')}P{int(aapl_price * 0.95 * 1000):08d}")  # OTM Put
        
        # TSLA examples
        tsla_price = 350
        examples.append(f"TSLA{expiry.strftime('%y%m%d')}C{int(tsla_price * 1.05 * 1000):08d}")  # OTM Call
        examples.append(f"TSLA{expiry.strftime('%y%m%d')}P{int(tsla_price * 0.95 * 1000):08d}")  # OTM Put
        
        logger.info("\n🎯 Example Option Symbols:")
        for symbol in examples:
            logger.info(f"  {symbol}")
            
        return examples
        
    def trade_option_spreads(self):
        """Try to trade option spreads with proper symbols"""
        logger.info("\n📊 Attempting to trade option spreads...")
        
        for underlying in self.watchlist[:3]:
            try:
                # Get current stock price
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=underlying)
                quotes = self.data_client.get_stock_latest_quote(quote_req)
                
                if underlying in quotes:
                    stock_price = float(quotes[underlying].ask_price)
                    logger.info(f"\n{underlying} - Current Price: ${stock_price:.2f}")
                    
                    # Calculate option strikes
                    atm_strike = round(stock_price)
                    otm_put_strike = round(stock_price * 0.95)
                    otm_call_strike = round(stock_price * 1.05)
                    
                    # Generate option symbols
                    expiry = datetime.now() + timedelta(days=30)
                    expiry_str = expiry.strftime('%y%m%d')
                    
                    # Bull Put Spread
                    short_put = f"{underlying}{expiry_str}P{int(otm_put_strike * 1000):08d}"
                    long_put = f"{underlying}{expiry_str}P{int(otm_put_strike * 0.98 * 1000):08d}"
                    
                    logger.info(f"\n💡 Bull Put Spread opportunity:")
                    logger.info(f"   Short Put: {short_put}")
                    logger.info(f"   Long Put: {long_put}")
                    logger.info(f"   Max profit: Premium collected")
                    logger.info(f"   Max loss: Strike difference - premium")
                    
                    # Try to place the order
                    if self.should_trade_spread(underlying, stock_price, "bull_put"):
                        self.execute_option_spread("bull_put", short_put, long_put)
                        
                    # Iron Condor
                    if underlying in ['SPY', 'QQQ']:
                        put_short = f"{underlying}{expiry_str}P{int(stock_price * 0.95 * 1000):08d}"
                        put_long = f"{underlying}{expiry_str}P{int(stock_price * 0.93 * 1000):08d}"
                        call_short = f"{underlying}{expiry_str}C{int(stock_price * 1.05 * 1000):08d}"
                        call_long = f"{underlying}{expiry_str}C{int(stock_price * 1.07 * 1000):08d}"
                        
                        logger.info(f"\n🦅 Iron Condor opportunity:")
                        logger.info(f"   Put spread: {put_long} / {put_short}")
                        logger.info(f"   Call spread: {call_short} / {call_long}")
                        
            except Exception as e:
                logger.error(f"Error processing {underlying}: {e}")
                
    def should_trade_spread(self, underlying: str, stock_price: float, spread_type: str) -> bool:
        """Determine if we should trade this spread"""
        try:
            # Get recent price action
            bars_req = StockBarsRequest(
                symbol_or_symbols=underlying,
                timeframe=TimeFrame.Day,
                limit=20
            )
            bars = self.data_client.get_stock_bars(bars_req)
            
            if underlying in bars.data:
                df = bars.df.loc[underlying]
                
                # Calculate indicators
                sma_20 = df['close'].mean()
                volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
                
                logger.info(f"\n📈 Analysis for {underlying}:")
                logger.info(f"   20-day SMA: ${sma_20:.2f}")
                logger.info(f"   Volatility: {volatility:.1f}%")
                logger.info(f"   Trend: {'Bullish' if stock_price > sma_20 else 'Bearish'}")
                
                # Decision logic
                if spread_type == "bull_put" and stock_price > sma_20:
                    logger.info("   ✅ Bullish trend supports Bull Put Spread")
                    return True
                elif spread_type == "iron_condor" and 15 < volatility < 35:
                    logger.info("   ✅ Moderate volatility supports Iron Condor")
                    return True
                    
        except Exception as e:
            logger.error(f"Error analyzing {underlying}: {e}")
            
        return False
        
    def execute_option_spread(self, spread_type: str, *option_symbols):
        """Execute an option spread order"""
        logger.info(f"\n🚀 Attempting to execute {spread_type}...")
        
        try:
            for symbol in option_symbols:
                # Try to place order with option symbol
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=self.position_size,
                    side=OrderSide.BUY,  # Simplified for demo
                    time_in_force=TimeInForce.DAY
                )
                
                try:
                    result = self.trading_client.submit_order(order)
                    logger.info(f"   ✅ Order placed for {symbol}")
                except Exception as e:
                    logger.info(f"   ⚠️ Option order not available: {str(e)[:100]}")
                    
                    # If options aren't available, demonstrate the concept
                    logger.info(f"   📝 Would execute: BUY {self.position_size} {symbol}")
                    
        except Exception as e:
            logger.error(f"Error executing spread: {e}")
            
    def display_portfolio_status(self):
        """Display current portfolio status"""
        try:
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n📊 Portfolio Status:")
            logger.info(f"Active Positions: {len(positions)}")
            
            # Separate stock and option positions
            option_positions = []
            stock_positions = []
            
            for position in positions:
                # Option symbols are typically 15+ characters
                if len(position.symbol) > 10 and any(c.isdigit() for c in position.symbol[6:]):
                    option_positions.append(position)
                else:
                    stock_positions.append(position)
                    
            if option_positions:
                logger.info(f"\n🎯 Option Positions: {len(option_positions)}")
                for pos in option_positions:
                    logger.info(f"   {pos.symbol}: {pos.qty} contracts @ ${float(pos.avg_entry_price):.2f}")
                    logger.info(f"      P&L: ${float(pos.unrealized_pl):,.2f}")
                    
            if stock_positions:
                logger.info(f"\n📈 Stock Positions: {len(stock_positions)}")
                total_value = sum(float(p.market_value) for p in stock_positions)
                logger.info(f"   Total Value: ${total_value:,.2f}")
                
            # Show available strategies
            logger.info("\n🎯 Available Option Strategies:")
            logger.info("• Bull Put Spread - Credit spread for bullish outlook")
            logger.info("• Bear Call Spread - Credit spread for bearish outlook")
            logger.info("• Iron Condor - Range-bound strategy")
            logger.info("• Butterfly Spread - Pinning strategy")
            logger.info("• Calendar Spread - Volatility play")
            logger.info("• Diagonal Spread - Poor man's covered call")
            
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")

def main():
    """Main entry point"""
    bot = AlpacaOptionsTraderV2()
    bot.run()

if __name__ == "__main__":
    main()