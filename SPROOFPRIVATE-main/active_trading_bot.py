#!/usr/bin/env python3
"""
Active Trading Bot with Fixed Data Retrieval
Implements momentum, mean reversion, and breakout strategies
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActiveTradingBot:
    def __init__(self):
        """Initialize the trading bot with Alpaca clients"""
        # Get API credentials
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.secret_key = os.getenv('APCA_API_SECRET_KEY')
        self.base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Trading parameters
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMD']
        self.position_size = 1000  # $1000 per position
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.05  # 5%
        self.scan_interval = 30  # seconds
        
        # Strategy parameters
        self.momentum_threshold = 0.01  # 1% gain
        self.mean_reversion_threshold = -0.02  # 2% drop
        self.volume_multiplier = 1.5  # 50% above average
        
        # Track orders and positions
        self.active_orders = {}
        self.positions = {}
        
        logger.info("Active Trading Bot initialized successfully")
        self.display_account_info()
    
    def display_account_info(self):
        """Display account information"""
        try:
            account = self.trading_client.get_account()
            logger.info(f"Account Status: {account.status}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Portfolio Value: ${float(account.equity):,.2f}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
    
    def get_historical_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol - FIXED VERSION"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            # Get bars data - returns a dict with symbol as key
            bars_data = self.data_client.get_stock_bars(request)
            
            # Extract the bars for our symbol
            if symbol in bars_data:
                bars = bars_data[symbol]
                
                # Convert to DataFrame
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    })
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('timestamp', inplace=True)
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Use ask price if available, otherwise bid
                return float(quote.ask_price) if quote.ask_price else float(quote.bid_price)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def check_momentum_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for momentum trading signal"""
        try:
            if len(df) < 2:
                return False, ""
            
            # Calculate price change
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            
            # Calculate volume change
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[:-1].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Momentum signal: price up 1%+ with high volume
            if price_change >= self.momentum_threshold and volume_ratio >= self.volume_multiplier:
                return True, f"Momentum: +{price_change:.2%} on {volume_ratio:.1f}x volume"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking momentum for {symbol}: {e}")
            return False, ""
    
    def check_mean_reversion_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for mean reversion trading signal"""
        try:
            if len(df) < 5:
                return False, ""
            
            # Calculate price drop from 5-day high
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].iloc[-5:].max()
            price_drop = (current_price - recent_high) / recent_high
            
            # Mean reversion signal: price dropped 2%+ from recent high
            if price_drop <= self.mean_reversion_threshold:
                return True, f"Mean Reversion: {price_drop:.2%} from high"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking mean reversion for {symbol}: {e}")
            return False, ""
    
    def check_breakout_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for breakout trading signal"""
        try:
            if len(df) < 20:
                return False, ""
            
            # Get current price and 20-day high
            current_price = df['close'].iloc[-1]
            high_20d = df['high'].iloc[-21:-1].max()  # Exclude today
            
            # Breakout signal: new 20-day high
            if current_price > high_20d:
                return True, f"Breakout: New 20-day high ${current_price:.2f}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking breakout for {symbol}: {e}")
            return False, ""
    
    def place_bracket_order(self, symbol: str, quantity: int, entry_price: float, 
                          strategy: str, signal_msg: str) -> bool:
        """Place a bracket order with stop loss and take profit"""
        try:
            # Calculate stop loss and take profit prices
            stop_price = entry_price * (1 - self.stop_loss_pct)
            limit_price = entry_price * (1 + self.take_profit_pct)
            
            # Create market order
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class='bracket',
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': limit_price}
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_data=market_order_data)
            
            logger.info(f"✅ {strategy} Order Placed: {symbol}")
            logger.info(f"   Signal: {signal_msg}")
            logger.info(f"   Quantity: {quantity} @ ~${entry_price:.2f}")
            logger.info(f"   Stop Loss: ${stop_price:.2f} (-2%)")
            logger.info(f"   Take Profit: ${limit_price:.2f} (+5%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return False
    
    def check_existing_position(self, symbol: str) -> bool:
        """Check if we already have a position in this symbol"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == symbol:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return True  # Assume we have position to be safe
    
    def scan_for_opportunities(self):
        """Scan all symbols for trading opportunities"""
        logger.info("🔍 Scanning for trading opportunities...")
        opportunities_found = 0
        
        for symbol in self.symbols:
            # Skip if we already have a position
            if self.check_existing_position(symbol):
                continue
            
            # Get historical data
            df = self.get_historical_data(symbol, days=30)
            if df is None or df.empty:
                continue
            
            # Get latest price
            latest_price = self.get_latest_price(symbol)
            if latest_price is None:
                continue
            
            # Check all strategies
            strategies = [
                ("Momentum", self.check_momentum_signal),
                ("Mean Reversion", self.check_mean_reversion_signal),
                ("Breakout", self.check_breakout_signal)
            ]
            
            for strategy_name, check_func in strategies:
                signal, msg = check_func(symbol, df)
                if signal:
                    # Calculate quantity
                    quantity = int(self.position_size / latest_price)
                    if quantity > 0:
                        if self.place_bracket_order(symbol, quantity, latest_price, 
                                                   strategy_name, msg):
                            opportunities_found += 1
                            break  # Only one strategy per symbol
        
        if opportunities_found == 0:
            logger.info("No trading opportunities found in this scan")
    
    def update_pnl(self):
        """Update and display current P&L"""
        try:
            # Get current positions
            positions = self.trading_client.get_all_positions()
            
            if not positions:
                logger.info("No open positions")
                return
            
            logger.info("\n📊 Current Positions:")
            total_pnl = 0
            
            for position in positions:
                qty = int(position.qty)
                avg_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                market_value = float(position.market_value)
                unrealized_pnl = float(position.unrealized_pl)
                pnl_pct = (current_price - avg_price) / avg_price * 100
                
                emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                logger.info(f"{emoji} {position.symbol}: {qty} shares @ ${avg_price:.2f}")
                logger.info(f"   Current: ${current_price:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"   P&L: ${unrealized_pnl:+,.2f}")
                
                total_pnl += unrealized_pnl
            
            logger.info(f"\n💰 Total Unrealized P&L: ${total_pnl:+,.2f}")
            
            # Get account info
            account = self.trading_client.get_account()
            logger.info(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"📈 Portfolio Value: ${float(account.equity):,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
    
    def check_orders(self):
        """Check status of recent orders"""
        try:
            # Get recent orders
            request = GetOrdersRequest(
                status=OrderStatus.FILLED,
                limit=10
            )
            orders = self.trading_client.get_orders(filter=request)
            
            if orders:
                logger.info("\n📋 Recent Filled Orders:")
                for order in orders[:5]:  # Show last 5
                    logger.info(f"   {order.symbol}: {order.qty} @ ${float(order.filled_avg_price or 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
    
    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Active Trading Bot")
        logger.info(f"Scanning {len(self.symbols)} symbols every {self.scan_interval} seconds")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check if market is open
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed. Waiting...")
                    time.sleep(60)
                    continue
                
                # Scan for opportunities
                self.scan_for_opportunities()
                
                # Update P&L
                self.update_pnl()
                
                # Check recent orders
                self.check_orders()
                
                # Wait for next scan
                logger.info(f"\n⏰ Next scan in {self.scan_interval} seconds...\n")
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

if __name__ == "__main__":
    # Create and run the bot
    bot = ActiveTradingBot()
    bot.run()