#!/usr/bin/env python3
"""
Continuous Trading Bot with Multiple Strategies
Actively scans and trades using momentum, mean reversion, breakout, and options-like strategies
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopLossRequest,
    TakeProfitRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models import Bar
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('continuous_trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTradingBot:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """Initialize the trading bot with Alpaca credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
        # Trading parameters
        self.max_position_size = 0.05  # 5% of portfolio per trade
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        self.scan_interval = 30  # Scan every 30 seconds
        
        # Strategy parameters
        self.momentum_period = 20
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.breakout_period = 50
        self.volume_threshold = 1.5  # 150% of average volume
        
        # Watchlist of stocks to monitor
        self.watchlist = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            'SPY', 'QQQ', 'IWM', 'DIA', 'NFLX', 'BABA', 'JPM', 'BAC',
            'WMT', 'V', 'MA', 'PG', 'JNJ', 'UNH', 'HD', 'DIS',
            'PYPL', 'ADBE', 'CRM', 'INTC', 'CSCO', 'PFE', 'NKE', 'MRK'
        ]
        
        # Track active positions and performance
        self.active_positions = {}
        self.trade_history = []
        self.total_pnl = 0.0
        
    def run(self):
        """Main loop - continuously scan and trade"""
        logger.info("🚀 Starting Continuous Trading Bot")
        logger.info(f"Paper Trading: {self.paper}")
        logger.info(f"Monitoring {len(self.watchlist)} stocks")
        
        # Display initial account info
        self.display_account_info()
        
        while True:
            try:
                # Check market hours
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Scan for opportunities
                logger.info("\n" + "="*60)
                logger.info("🔍 Scanning for trading opportunities...")
                
                # Run each strategy
                self.scan_momentum_stocks()
                self.scan_mean_reversion()
                self.scan_breakouts()
                self.manage_existing_positions()
                
                # Display current status
                self.display_portfolio_status()
                
                # Wait before next scan
                logger.info(f"💤 Waiting {self.scan_interval} seconds until next scan...")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down trading bot...")
                self.close_all_positions()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
                
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return False
            
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'positions': int(account.position_market_value) if hasattr(account, 'position_market_value') else 0
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    def display_account_info(self):
        """Display current account information"""
        info = self.get_account_info()
        if info:
            logger.info("\n📊 Account Information:")
            logger.info(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            logger.info(f"Buying Power: ${info['buying_power']:,.2f}")
            logger.info(f"Cash: ${info['cash']:,.2f}")
            
    def get_historical_data(self, symbol: str, timeframe: TimeFrame = TimeFrame.Minute, 
                           limit: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                limit=limit
            )
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars.data:
                df = bars.data[symbol].df
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
            
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        
    def scan_momentum_stocks(self):
        """Scan for stocks with strong momentum"""
        logger.info("📈 Scanning for momentum opportunities...")
        
        for symbol in self.watchlist:
            try:
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                    
                # Get historical data
                df = self.get_historical_data(symbol, TimeFrame.Minute, 50)
                if df.empty or len(df) < self.momentum_period:
                    continue
                    
                # Calculate momentum indicators
                returns = df['close'].pct_change(self.momentum_period)
                current_return = returns.iloc[-1]
                
                # Strong momentum criteria
                if current_return > 0.02:  # 2% gain over period
                    volume_ratio = df['volume'].iloc[-1] / df['volume'].mean()
                    
                    if volume_ratio > self.volume_threshold:
                        # Check if price is above moving average
                        ma = df['close'].rolling(20).mean().iloc[-1]
                        current_price = df['close'].iloc[-1]
                        
                        if current_price > ma:
                            logger.info(f"✅ Momentum signal for {symbol}: "
                                      f"{current_return*100:.2f}% gain, "
                                      f"volume ratio: {volume_ratio:.2f}")
                            self.execute_trade(symbol, OrderSide.BUY, 'momentum')
                            
            except Exception as e:
                logger.error(f"Error scanning momentum for {symbol}: {e}")
                
    def scan_mean_reversion(self):
        """Scan for oversold stocks to buy"""
        logger.info("📉 Scanning for mean reversion opportunities...")
        
        for symbol in self.watchlist:
            try:
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                    
                # Get historical data
                df = self.get_historical_data(symbol, TimeFrame.Minute, 50)
                if df.empty or len(df) < 14:
                    continue
                    
                # Calculate RSI
                rsi = self.calculate_rsi(df['close'])
                
                # Check if oversold
                if rsi < self.rsi_oversold:
                    # Additional confirmation: price below lower Bollinger Band
                    ma = df['close'].rolling(20).mean()
                    std = df['close'].rolling(20).std()
                    lower_band = ma - (2 * std)
                    
                    current_price = df['close'].iloc[-1]
                    
                    if current_price < lower_band.iloc[-1]:
                        logger.info(f"✅ Mean reversion signal for {symbol}: "
                                  f"RSI={rsi:.2f}, price below BB")
                        self.execute_trade(symbol, OrderSide.BUY, 'mean_reversion')
                        
            except Exception as e:
                logger.error(f"Error scanning mean reversion for {symbol}: {e}")
                
    def scan_breakouts(self):
        """Scan for breakout opportunities"""
        logger.info("🚀 Scanning for breakout opportunities...")
        
        for symbol in self.watchlist:
            try:
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                    
                # Get historical data
                df = self.get_historical_data(symbol, TimeFrame.Minute, self.breakout_period + 10)
                if df.empty or len(df) < self.breakout_period:
                    continue
                    
                # Calculate resistance level (recent high)
                resistance = df['high'].iloc[-self.breakout_period:-1].max()
                current_price = df['close'].iloc[-1]
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[-self.breakout_period:-1].mean()
                
                # Check for breakout
                if current_price > resistance * 1.001:  # Price above resistance
                    if current_volume > avg_volume * self.volume_threshold:
                        logger.info(f"✅ Breakout signal for {symbol}: "
                                  f"price={current_price:.2f}, "
                                  f"resistance={resistance:.2f}")
                        self.execute_trade(symbol, OrderSide.BUY, 'breakout')
                        
            except Exception as e:
                logger.error(f"Error scanning breakouts for {symbol}: {e}")
                
    def execute_trade(self, symbol: str, side: OrderSide, strategy: str):
        """Execute a trade with proper risk management"""
        try:
            account = self.get_account_info()
            if not account:
                return
                
            # Calculate position size
            portfolio_value = account['portfolio_value']
            position_value = portfolio_value * self.max_position_size
            
            # Get current price
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(quote_request)
            
            if symbol in quote:
                current_price = float(quote[symbol].ask_price)
                quantity = int(position_value / current_price)
                
                if quantity < 1:
                    logger.warning(f"Position too small for {symbol}")
                    return
                    
                # Create bracket order (entry + stop loss + take profit)
                if side == OrderSide.BUY:
                    stop_price = current_price * (1 - self.stop_loss_pct)
                    limit_price = current_price * (1 + self.take_profit_pct)
                else:
                    stop_price = current_price * (1 + self.stop_loss_pct)
                    limit_price = current_price * (1 - self.take_profit_pct)
                    
                # Place market order with bracket
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class='bracket',
                    stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(limit_price, 2))
                )
                
                order = self.trading_client.submit_order(order_request)
                
                # Track position
                self.active_positions[symbol] = {
                    'order_id': order.id,
                    'strategy': strategy,
                    'side': side.value,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'stop_loss': stop_price,
                    'take_profit': limit_price,
                    'entry_time': datetime.now()
                }
                
                logger.info(f"🎯 {side.value} order placed for {symbol}:")
                logger.info(f"   Strategy: {strategy}")
                logger.info(f"   Quantity: {quantity}")
                logger.info(f"   Entry: ${current_price:.2f}")
                logger.info(f"   Stop Loss: ${stop_price:.2f}")
                logger.info(f"   Take Profit: ${limit_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            
    def manage_existing_positions(self):
        """Manage existing positions - check for exits"""
        logger.info("🔧 Managing existing positions...")
        
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                symbol = position.symbol
                
                # Get current market data
                df = self.get_historical_data(symbol, TimeFrame.Minute, 20)
                if df.empty:
                    continue
                    
                current_price = df['close'].iloc[-1]
                entry_price = float(position.avg_entry_price)
                unrealized_pnl = float(position.unrealized_pl)
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check for manual exit conditions
                rsi = self.calculate_rsi(df['close'])
                
                # Exit conditions
                exit_signal = False
                exit_reason = ""
                
                # Momentum exit - RSI overbought
                if rsi > self.rsi_overbought and pnl_pct > 0.01:
                    exit_signal = True
                    exit_reason = "RSI overbought"
                    
                # Trend reversal
                ma_short = df['close'].rolling(5).mean().iloc[-1]
                ma_long = df['close'].rolling(20).mean().iloc[-1]
                
                if position.side == 'long' and ma_short < ma_long:
                    exit_signal = True
                    exit_reason = "Trend reversal"
                    
                if exit_signal:
                    logger.info(f"📤 Exit signal for {symbol}: {exit_reason}")
                    self.close_position(symbol)
                else:
                    logger.info(f"📊 {symbol}: P&L=${unrealized_pnl:.2f} ({pnl_pct*100:.2f}%)")
                    
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            
    def close_position(self, symbol: str):
        """Close a specific position"""
        try:
            # Place market order to close
            position = self.trading_client.get_open_position(symbol)
            
            if position:
                qty = abs(int(position.qty))
                side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
                
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_request)
                
                # Update tracking
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                    
                logger.info(f"✅ Closed position for {symbol}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            
    def close_all_positions(self):
        """Close all open positions"""
        logger.info("🛑 Closing all positions...")
        
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            self.active_positions.clear()
            logger.info("All positions closed")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            
    def display_portfolio_status(self):
        """Display current portfolio status and P&L"""
        try:
            account = self.get_account_info()
            positions = self.trading_client.get_all_positions()
            
            logger.info("\n" + "="*60)
            logger.info("💼 PORTFOLIO STATUS")
            logger.info("="*60)
            
            if account:
                logger.info(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
                logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
                
            if positions:
                logger.info(f"\n📈 Open Positions: {len(positions)}")
                total_unrealized_pnl = 0
                
                for position in positions:
                    unrealized_pnl = float(position.unrealized_pl)
                    total_unrealized_pnl += unrealized_pnl
                    pnl_pct = float(position.unrealized_plpc) * 100
                    
                    emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                    logger.info(f"{emoji} {position.symbol}: "
                              f"{position.qty} shares @ ${float(position.avg_entry_price):.2f} | "
                              f"P&L: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%)")
                              
                logger.info(f"\n💰 Total Unrealized P&L: ${total_unrealized_pnl:.2f}")
            else:
                logger.info("\n📊 No open positions")
                
            # Display recent orders
            orders_request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=5
            )
            recent_orders = self.trading_client.get_orders(orders_request)
            
            if recent_orders:
                logger.info(f"\n📋 Recent Orders:")
                for order in recent_orders[:5]:
                    logger.info(f"   {order.symbol} - {order.side} {order.qty} @ "
                              f"{order.order_type} - {order.status}")
                              
        except Exception as e:
            logger.error(f"Error displaying portfolio status: {e}")

def main():
    """Main entry point"""
    # Trading credentials
    API_KEY = "PKEP9PIBDKOSUGHHY44Z"
    API_SECRET = "VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ"
    
    # Create and run the bot
    bot = ContinuousTradingBot(API_KEY, API_SECRET, paper=True)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()