#!/usr/bin/env python3
"""
Aggressive Trading Bot - Actively finds and executes trades with multiple strategies
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveTradingBot:
    def __init__(self):
        """Initialize the aggressive trading bot"""
        # Get API credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        
        # Initialize clients
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Expanded symbol list (30+ stocks)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
            'NFLX', 'CRM', 'XOM', 'CVX', 'PFE', 'TMO', 'CSCO', 'PEP', 'AVGO',
            'ABBV', 'NKE', 'ACN', 'COST', 'TXN', 'LLY', 'ORCL', 'MCD', 'WFC'
        ]
        
        # Strategy thresholds (more aggressive)
        self.momentum_threshold = 0.005  # 0.5% gain
        self.mean_reversion_threshold = 0.01  # 1% drop
        self.breakout_days = 10  # 10-day high
        self.gap_threshold = 0.02  # 2% gap for gap trading
        self.volume_spike_multiplier = 2.0  # 2x average volume
        self.support_resistance_buffer = 0.002  # 0.2% buffer
        
        # Risk management
        self.position_size = 100  # Shares per trade
        self.max_positions = 10
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        
        # Track active positions
        self.active_positions = {}
        
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'positions': len(self.trading_client.get_all_positions())
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=days)
            )
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars.data:
                df = bars.df.loc[symbol]
                df = df.reset_index()
                
                # Calculate technical indicators
                df['returns'] = df['close'].pct_change()
                df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
                df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
                df['volume_avg'] = df['volume'].rolling(window=20, min_periods=1).mean()
                df['high_10d'] = df['high'].rolling(window=self.breakout_days, min_periods=1).max()
                df['low_10d'] = df['low'].rolling(window=self.breakout_days, min_periods=1).min()
                
                # Support and resistance levels
                df['resistance'] = df['high'].rolling(window=20, min_periods=1).max()
                df['support'] = df['low'].rolling(window=20, min_periods=1).min()
                
                return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def check_momentum_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for momentum trading opportunity"""
        if len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish momentum
        if latest['returns'] > self.momentum_threshold:
            if latest['close'] > latest['sma_5'] > latest['sma_20']:
                return 'BUY'
        
        return None
    
    def check_mean_reversion_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for mean reversion opportunity"""
        if len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        
        # Oversold condition
        if latest['returns'] < -self.mean_reversion_threshold:
            if latest['close'] < latest['sma_20'] * 0.98:  # 2% below SMA
                return 'BUY'
        
        return None
    
    def check_breakout_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for breakout trading opportunity"""
        if len(df) < self.breakout_days:
            return None
        
        latest = df.iloc[-1]
        
        # Breakout above 10-day high
        if latest['close'] > latest['high_10d'] * 0.998:  # Near or above 10-day high
            return 'BUY'
        
        return None
    
    def check_gap_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for gap up/down trading opportunity"""
        if len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        gap = (latest['open'] - prev['close']) / prev['close']
        
        # Gap up - potential continuation
        if gap > self.gap_threshold:
            return 'BUY'
        
        # Gap down - potential reversal
        if gap < -self.gap_threshold and latest['close'] > latest['open']:
            return 'BUY'
        
        return None
    
    def check_volume_spike_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for volume spike trading opportunity"""
        if len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        
        if latest['volume'] > latest['volume_avg'] * self.volume_spike_multiplier:
            # Volume spike with positive price action
            if latest['returns'] > 0:
                return 'BUY'
        
        return None
    
    def check_support_resistance_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Check for support/resistance break trading opportunity"""
        if len(df) < 20:
            return None
        
        latest = df.iloc[-1]
        
        # Resistance break
        if latest['close'] > latest['resistance'] * (1 - self.support_resistance_buffer):
            return 'BUY'
        
        # Bounce off support
        if latest['close'] > latest['support'] * (1 + self.support_resistance_buffer):
            if latest['low'] <= latest['support'] * (1 + self.support_resistance_buffer):
                return 'BUY'
        
        return None
    
    def calculate_opportunity_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate opportunity score for forced trading"""
        score = 0.0
        
        if len(df) < 2:
            return score
        
        latest = df.iloc[-1]
        
        # Momentum score
        if latest['returns'] > 0:
            score += latest['returns'] * 100
        
        # Trend score
        if latest['close'] > latest['sma_5']:
            score += 1
        if latest['close'] > latest['sma_20']:
            score += 1
        
        # Volume score
        if latest['volume'] > latest['volume_avg']:
            score += (latest['volume'] / latest['volume_avg'])
        
        # Volatility score (prefer some volatility)
        volatility = df['returns'].std()
        if 0.01 < volatility < 0.05:
            score += 2
        
        return score
    
    def scan_for_opportunities(self, force_trade: bool = False) -> List[Dict]:
        """Scan all symbols for trading opportunities"""
        opportunities = []
        
        logger.info(f"Scanning {len(self.symbols)} symbols for opportunities...")
        
        for symbol in self.symbols:
            try:
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                
                # Get historical data
                df = self.get_historical_data(symbol)
                if df.empty:
                    continue
                
                # Check all strategies
                signals = {
                    'momentum': self.check_momentum_strategy(df),
                    'mean_reversion': self.check_mean_reversion_strategy(df),
                    'breakout': self.check_breakout_strategy(df),
                    'gap': self.check_gap_strategy(df),
                    'volume_spike': self.check_volume_spike_strategy(df),
                    'support_resistance': self.check_support_resistance_strategy(df)
                }
                
                # Count buy signals
                buy_signals = [s for s in signals.values() if s == 'BUY']
                
                if buy_signals or force_trade:
                    opportunity = {
                        'symbol': symbol,
                        'price': df.iloc[-1]['close'],
                        'signals': signals,
                        'signal_count': len(buy_signals),
                        'score': self.calculate_opportunity_score(symbol, df),
                        'volume': df.iloc[-1]['volume'],
                        'returns': df.iloc[-1]['returns']
                    }
                    opportunities.append(opportunity)
                    
                    logger.info(f"Opportunity found: {symbol} - Signals: {len(buy_signals)}, Score: {opportunity['score']:.2f}")
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by score (higher is better)
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def place_order(self, symbol: str, side: OrderSide, qty: int) -> Optional[str]:
        """Place a market order"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            logger.info(f"Order placed: {side} {qty} shares of {symbol} at market")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def check_positions_for_exit(self):
        """Check existing positions for stop loss or take profit"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                avg_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                pnl_pct = (current_price - avg_price) / avg_price
                
                # Check stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.2%} loss")
                    self.place_order(symbol, OrderSide.SELL, qty)
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                
                # Check take profit
                elif pnl_pct >= self.take_profit_pct:
                    logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%} gain")
                    self.place_order(symbol, OrderSide.SELL, qty)
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def execute_trades(self, opportunities: List[Dict], force_trade: bool = False):
        """Execute trades based on opportunities"""
        account = self.get_account_info()
        buying_power = account.get('buying_power', 0)
        current_positions = account.get('positions', 0)
        
        if current_positions >= self.max_positions:
            logger.info(f"Max positions reached ({self.max_positions})")
            return
        
        # Execute trades for best opportunities
        trades_to_place = min(len(opportunities), self.max_positions - current_positions)
        
        if force_trade and trades_to_place == 0 and opportunities:
            # Force at least one trade if requested
            trades_to_place = 1
            logger.info("Force trade activated - will execute best opportunity")
        
        for i in range(trades_to_place):
            if i >= len(opportunities):
                break
                
            opp = opportunities[i]
            symbol = opp['symbol']
            price = opp['price']
            
            # Calculate position size
            position_value = self.position_size * price
            
            if position_value > buying_power * 0.95:  # Leave some buffer
                logger.warning(f"Insufficient buying power for {symbol}")
                continue
            
            # Place buy order
            order_id = self.place_order(symbol, OrderSide.BUY, self.position_size)
            
            if order_id:
                self.active_positions[symbol] = {
                    'order_id': order_id,
                    'qty': self.position_size,
                    'entry_price': price,
                    'timestamp': datetime.now()
                }
                buying_power -= position_value
    
    def get_recent_orders(self):
        """Get recent orders with fixed status check"""
        try:
            # Use 'all' instead of OrderStatus.FILLED
            request = GetOrdersRequest(
                status='all',  # Fixed: use string 'all' instead of enum
                limit=10
            )
            orders = self.trading_client.get_orders(filter=request)
            
            logger.info(f"Recent orders: {len(orders)} found")
            for order in orders[:5]:  # Show last 5 orders
                logger.info(f"  {order.symbol}: {order.side} {order.qty} @ {order.filled_avg_price or 'pending'} - Status: {order.status}")
                
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
    
    def run(self, force_trade: bool = False):
        """Run a single scan and trade execution cycle"""
        logger.info("="*50)
        logger.info("Starting aggressive trading scan...")
        
        # Get account info
        account = self.get_account_info()
        logger.info(f"Account - Buying Power: ${account.get('buying_power', 0):,.2f}, Positions: {account.get('positions', 0)}")
        
        # Check existing positions for exit
        self.check_positions_for_exit()
        
        # Scan for opportunities
        opportunities = self.scan_for_opportunities(force_trade)
        logger.info(f"Found {len(opportunities)} trading opportunities")
        
        if opportunities:
            # Show top opportunities
            logger.info("\nTop opportunities:")
            for i, opp in enumerate(opportunities[:5]):
                logger.info(f"{i+1}. {opp['symbol']} - Score: {opp['score']:.2f}, Signals: {opp['signal_count']}, Returns: {opp['returns']:.2%}")
            
            # Execute trades
            self.execute_trades(opportunities, force_trade)
        
        # Show recent orders
        self.get_recent_orders()
        
        logger.info("Scan complete")
        logger.info("="*50)
    
    def run_continuous(self, interval: int = 60, force_trade_interval: int = 300):
        """Run continuously with periodic scans"""
        logger.info(f"Starting continuous trading - Scan interval: {interval}s, Force trade every: {force_trade_interval}s")
        
        last_force_trade = time.time()
        
        while True:
            try:
                # Check if it's time for a forced trade
                current_time = time.time()
                force_trade = (current_time - last_force_trade) >= force_trade_interval
                
                if force_trade:
                    logger.info(">>> FORCE TRADE ACTIVATED <<<")
                    last_force_trade = current_time
                
                # Run scan and trade
                self.run(force_trade=force_trade)
                
                # Wait for next scan
                logger.info(f"Waiting {interval} seconds until next scan...\n")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping trading bot...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggressive Trading Bot')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval in seconds')
    parser.add_argument('--force-trade', action='store_true', help='Force at least one trade per scan')
    parser.add_argument('--force-interval', type=int, default=300, help='Force trade interval in seconds')
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = AggressiveTradingBot()
    
    if args.continuous:
        bot.run_continuous(
            interval=args.interval,
            force_trade_interval=args.force_interval
        )
    else:
        bot.run(force_trade=args.force_trade)


if __name__ == "__main__":
    main()