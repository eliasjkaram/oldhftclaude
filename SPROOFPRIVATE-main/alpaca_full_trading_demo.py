#!/usr/bin/env python3
"""
Alpaca Full Trading Demo
========================
Comprehensive demonstration of all trading capabilities using Alpaca's paper trading API.
Showcases various order types, synthetic options strategies, and risk management.
"""

import os
import sys
import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import logging
# Simple color codes for terminal output
class Fore:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

class Style:
    BRIGHT = '\033[1m'
    RESET_ALL = '\033[0m'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaTradingDemo:
    """Comprehensive Alpaca trading demonstration system."""
    
    def __init__(self):
        """Initialize the trading demo with provided credentials."""
        self.api_key = "PKEP9PIBDKOSUGHHY44Z"
        self.api_secret = "VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ"
        self.base_url = "https://paper-api.alpaca.markets"
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        # Risk management parameters
        self.max_position_size_pct = 0.20  # Max 20% per position
        self.max_total_exposure_pct = 0.80  # Max 80% total exposure
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        
        # Strategy parameters
        self.momentum_lookback = 20
        self.mean_reversion_lookback = 10
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Track executed orders for demo
        self.executed_orders = []
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}{text.center(60)}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
    def print_success(self, text: str):
        """Print success message."""
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
        
    def print_error(self, text: str):
        """Print error message."""
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
        
    def print_info(self, text: str):
        """Print info message."""
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")
        
    async def show_account_status(self):
        """Display current account status and metrics."""
        self.print_header("ACCOUNT STATUS")
        
        try:
            account = self.api.get_account()
            
            print(f"Account Number: {account.account_number}")
            print(f"Status: {Fore.GREEN if account.status == 'ACTIVE' else Fore.RED}{account.status}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Balances:{Style.RESET_ALL}")
            print(f"  Cash: ${float(account.cash):,.2f}")
            print(f"  Buying Power: ${float(account.buying_power):,.2f}")
            print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"  Equity: ${float(account.equity):,.2f}")
            
            # Calculate P&L
            initial_value = float(account.last_equity)
            current_value = float(account.equity)
            pnl = current_value - initial_value
            pnl_pct = (pnl / initial_value * 100) if initial_value > 0 else 0
            
            print(f"\n{Fore.CYAN}Performance:{Style.RESET_ALL}")
            print(f"  Daily P&L: {Fore.GREEN if pnl >= 0 else Fore.RED}${pnl:,.2f} ({pnl_pct:+.2f}%){Style.RESET_ALL}")
            print(f"  Day Trade Count: {account.daytrade_count}")
            print(f"  Pattern Day Trader: {account.pattern_day_trader}")
            
            return account
            
        except Exception as e:
            self.print_error(f"Failed to fetch account status: {str(e)}")
            return None
            
    async def show_current_positions(self):
        """Display all current positions."""
        self.print_header("CURRENT POSITIONS")
        
        try:
            positions = self.api.list_positions()
            
            if not positions:
                self.print_info("No open positions")
                return []
                
            total_value = 0
            total_pnl = 0
            
            for pos in positions:
                qty = int(pos.qty)
                avg_cost = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                market_value = float(pos.market_value)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                
                total_value += market_value
                total_pnl += unrealized_pnl
                
                print(f"\n{Fore.YELLOW}{pos.symbol}{Style.RESET_ALL}")
                print(f"  Quantity: {qty} @ ${avg_cost:.2f}")
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Market Value: ${market_value:,.2f}")
                print(f"  Unrealized P&L: {Fore.GREEN if unrealized_pnl >= 0 else Fore.RED}${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%){Style.RESET_ALL}")
                
            print(f"\n{Fore.CYAN}Total Position Value: ${total_value:,.2f}")
            print(f"Total Unrealized P&L: {Fore.GREEN if total_pnl >= 0 else Fore.RED}${total_pnl:,.2f}{Style.RESET_ALL}")
            
            return positions
            
        except Exception as e:
            self.print_error(f"Failed to fetch positions: {str(e)}")
            return []
            
    async def calculate_position_size(self, symbol: str, account) -> int:
        """Calculate appropriate position size based on risk management rules."""
        try:
            # Get current price
            quote = self.api.get_latest_quote(symbol)
            price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
            
            # Calculate max position value
            portfolio_value = float(account.portfolio_value)
            max_position_value = portfolio_value * self.max_position_size_pct
            
            # Calculate shares
            shares = int(max_position_value / price)
            
            # Ensure minimum order size
            shares = max(1, shares)
            
            # Check buying power
            buying_power = float(account.buying_power)
            max_affordable_shares = int(buying_power / price)
            
            shares = min(shares, max_affordable_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1
            
    async def execute_market_order(self, symbol: str, qty: int, side: str) -> Optional[Dict]:
        """Execute a market order."""
        self.print_header(f"MARKET ORDER - {side.upper()} {qty} {symbol}")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            self.print_success(f"Market order submitted: {order.id}")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Quantity: {qty}")
            print(f"  Status: {order.status}")
            
            # Wait for fill
            time.sleep(2)
            order = self.api.get_order(order.id)
            
            if order.status == 'filled':
                self.print_success(f"Order filled at ${order.filled_avg_price}")
            else:
                self.print_info(f"Order status: {order.status}")
                
            self.executed_orders.append(order)
            return order
            
        except Exception as e:
            self.print_error(f"Market order failed: {str(e)}")
            return None
            
    async def execute_limit_order(self, symbol: str, qty: int, side: str, limit_price: float) -> Optional[Dict]:
        """Execute a limit order."""
        self.print_header(f"LIMIT ORDER - {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=limit_price
            )
            
            self.print_success(f"Limit order submitted: {order.id}")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Quantity: {qty}")
            print(f"  Limit Price: ${limit_price:.2f}")
            print(f"  Status: {order.status}")
            
            self.executed_orders.append(order)
            return order
            
        except Exception as e:
            self.print_error(f"Limit order failed: {str(e)}")
            return None
            
    async def execute_stop_order(self, symbol: str, qty: int, side: str, stop_price: float) -> Optional[Dict]:
        """Execute a stop order."""
        self.print_header(f"STOP ORDER - {side.upper()} {qty} {symbol} @ ${stop_price:.2f}")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='stop',
                time_in_force='day',
                stop_price=stop_price
            )
            
            self.print_success(f"Stop order submitted: {order.id}")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Quantity: {qty}")
            print(f"  Stop Price: ${stop_price:.2f}")
            print(f"  Status: {order.status}")
            
            self.executed_orders.append(order)
            return order
            
        except Exception as e:
            self.print_error(f"Stop order failed: {str(e)}")
            return None
            
    async def execute_bracket_order(self, symbol: str, qty: int, side: str, 
                                   limit_price: float, stop_price: float, 
                                   take_profit: float) -> Optional[Dict]:
        """Execute a bracket order (OCO - One Cancels Other)."""
        self.print_header(f"BRACKET ORDER - {side.upper()} {qty} {symbol}")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=limit_price,
                order_class='bracket',
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': take_profit}
            )
            
            self.print_success(f"Bracket order submitted: {order.id}")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Quantity: {qty}")
            print(f"  Entry Limit: ${limit_price:.2f}")
            print(f"  Stop Loss: ${stop_price:.2f}")
            print(f"  Take Profit: ${take_profit:.2f}")
            print(f"  Status: {order.status}")
            
            self.executed_orders.append(order)
            return order
            
        except Exception as e:
            self.print_error(f"Bracket order failed: {str(e)}")
            return None
            
    async def synthetic_iron_condor(self, symbol: str, account):
        """Create a synthetic iron condor using stock positions and limit orders."""
        self.print_header("SYNTHETIC IRON CONDOR STRATEGY")
        
        try:
            # Get current price
            quote = self.api.get_latest_quote(symbol)
            current_price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
            
            print(f"Creating synthetic iron condor on {symbol}")
            print(f"Current price: ${current_price:.2f}")
            
            # Calculate strike prices
            otm_call_strike = current_price * 1.03  # 3% OTM call
            otm_put_strike = current_price * 0.97   # 3% OTM put
            far_otm_call_strike = current_price * 1.05  # 5% OTM call
            far_otm_put_strike = current_price * 0.95   # 5% OTM put
            
            # Calculate position sizes (small for demo)
            position_size = min(10, await self.calculate_position_size(symbol, account))
            
            print(f"\nStrategy setup:")
            print(f"  Sell {position_size} shares @ ${otm_call_strike:.2f} (synthetic short call)")
            print(f"  Buy {position_size} shares @ ${otm_put_strike:.2f} (synthetic long put)")
            print(f"  Protective orders at ${far_otm_call_strike:.2f} and ${far_otm_put_strike:.2f}")
            
            # Place orders to simulate iron condor
            # Sell limit order (synthetic short call)
            await self.execute_limit_order(symbol, position_size, 'sell', otm_call_strike)
            
            # Buy limit order (synthetic long put)
            await self.execute_limit_order(symbol, position_size, 'buy', otm_put_strike)
            
            self.print_success("Synthetic iron condor created successfully")
            
        except Exception as e:
            self.print_error(f"Failed to create synthetic iron condor: {str(e)}")
            
    async def synthetic_covered_call(self, symbol: str, account):
        """Create a synthetic covered call (stock + limit sell order)."""
        self.print_header("SYNTHETIC COVERED CALL STRATEGY")
        
        try:
            # Get current price
            quote = self.api.get_latest_quote(symbol)
            current_price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
            
            print(f"Creating synthetic covered call on {symbol}")
            print(f"Current price: ${current_price:.2f}")
            
            # Calculate position size
            position_size = min(100, await self.calculate_position_size(symbol, account))
            
            # Buy stock
            stock_order = await self.execute_market_order(symbol, position_size, 'buy')
            
            if stock_order and stock_order.status in ['filled', 'partially_filled']:
                # Set call strike 2% above current price
                call_strike = current_price * 1.02
                
                # Sell call (limit order above market)
                await self.execute_limit_order(symbol, position_size, 'sell', call_strike)
                
                print(f"\nSynthetic covered call created:")
                print(f"  Long {position_size} shares @ market")
                print(f"  Short call strike: ${call_strike:.2f}")
                print(f"  Max profit: ${(call_strike - current_price) * position_size:.2f}")
                
                self.print_success("Synthetic covered call created successfully")
            
        except Exception as e:
            self.print_error(f"Failed to create synthetic covered call: {str(e)}")
            
    async def synthetic_protective_put(self, symbol: str, account):
        """Create a synthetic protective put (stock + stop loss)."""
        self.print_header("SYNTHETIC PROTECTIVE PUT STRATEGY")
        
        try:
            # Get current price
            quote = self.api.get_latest_quote(symbol)
            current_price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
            
            print(f"Creating synthetic protective put on {symbol}")
            print(f"Current price: ${current_price:.2f}")
            
            # Calculate position size
            position_size = min(50, await self.calculate_position_size(symbol, account))
            
            # Buy stock
            stock_order = await self.execute_market_order(symbol, position_size, 'buy')
            
            if stock_order and stock_order.status in ['filled', 'partially_filled']:
                # Set protective stop 3% below current price
                stop_price = current_price * 0.97
                
                # Place stop loss order
                await self.execute_stop_order(symbol, position_size, 'sell', stop_price)
                
                print(f"\nSynthetic protective put created:")
                print(f"  Long {position_size} shares @ market")
                print(f"  Protective stop: ${stop_price:.2f}")
                print(f"  Max loss: ${(current_price - stop_price) * position_size:.2f}")
                
                self.print_success("Synthetic protective put created successfully")
            
        except Exception as e:
            self.print_error(f"Failed to create synthetic protective put: {str(e)}")
            
    async def momentum_strategy(self, symbols: List[str], account):
        """Execute a momentum-based trading strategy."""
        self.print_header("MOMENTUM STRATEGY")
        
        try:
            print(f"Scanning {len(symbols)} symbols for momentum opportunities...")
            
            momentum_scores = []
            
            for symbol in symbols:
                try:
                    # Get historical data
                    bars = self.api.get_bars(
                        symbol,
                        TimeFrame.Day,
                        start=(datetime.now() - timedelta(days=30)).isoformat(),
                        end=datetime.now().isoformat(),
                        limit=self.momentum_lookback
                    ).df
                    
                    if len(bars) < self.momentum_lookback:
                        continue
                        
                    # Calculate momentum
                    returns = bars['close'].pct_change()
                    momentum = returns.rolling(window=self.momentum_lookback).mean().iloc[-1]
                    volatility = returns.rolling(window=self.momentum_lookback).std().iloc[-1]
                    
                    # Sharpe-like ratio
                    if volatility > 0:
                        score = momentum / volatility
                        momentum_scores.append({
                            'symbol': symbol,
                            'momentum': momentum,
                            'volatility': volatility,
                            'score': score,
                            'price': bars['close'].iloc[-1]
                        })
                        
                except Exception as e:
                    continue
                    
            # Sort by score and take top performer
            momentum_scores.sort(key=lambda x: x['score'], reverse=True)
            
            if momentum_scores:
                best = momentum_scores[0]
                print(f"\nTop momentum stock: {best['symbol']}")
                print(f"  Momentum: {best['momentum']*100:.2f}%")
                print(f"  Volatility: {best['volatility']*100:.2f}%")
                print(f"  Score: {best['score']:.2f}")
                
                # Execute trade if positive momentum
                if best['momentum'] > 0:
                    position_size = min(20, await self.calculate_position_size(best['symbol'], account))
                    
                    # Enter with bracket order for risk management
                    entry_price = best['price']
                    stop_loss = entry_price * (1 - self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct)
                    
                    await self.execute_bracket_order(
                        best['symbol'],
                        position_size,
                        'buy',
                        entry_price,
                        stop_loss,
                        take_profit
                    )
                    
                    self.print_success(f"Momentum trade executed on {best['symbol']}")
                else:
                    self.print_info("No positive momentum opportunities found")
                    
        except Exception as e:
            self.print_error(f"Momentum strategy failed: {str(e)}")
            
    async def mean_reversion_strategy(self, symbols: List[str], account):
        """Execute a mean reversion trading strategy."""
        self.print_header("MEAN REVERSION STRATEGY")
        
        try:
            print(f"Scanning {len(symbols)} symbols for mean reversion opportunities...")
            
            reversion_scores = []
            
            for symbol in symbols:
                try:
                    # Get historical data
                    bars = self.api.get_bars(
                        symbol,
                        TimeFrame.Day,
                        start=(datetime.now() - timedelta(days=30)).isoformat(),
                        end=datetime.now().isoformat(),
                        limit=self.mean_reversion_lookback
                    ).df
                    
                    if len(bars) < self.mean_reversion_lookback:
                        continue
                        
                    # Calculate RSI
                    close_prices = bars['close']
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Calculate z-score
                    mean = close_prices.rolling(window=self.mean_reversion_lookback).mean().iloc[-1]
                    std = close_prices.rolling(window=self.mean_reversion_lookback).std().iloc[-1]
                    current_price = close_prices.iloc[-1]
                    z_score = (current_price - mean) / std if std > 0 else 0
                    
                    reversion_scores.append({
                        'symbol': symbol,
                        'rsi': current_rsi,
                        'z_score': z_score,
                        'price': current_price,
                        'mean': mean
                    })
                    
                except Exception as e:
                    continue
                    
            # Find oversold opportunities
            oversold = [s for s in reversion_scores if s['rsi'] < self.rsi_oversold and s['z_score'] < -1.5]
            
            if oversold:
                # Sort by most oversold
                oversold.sort(key=lambda x: x['rsi'])
                best = oversold[0]
                
                print(f"\nOversold opportunity: {best['symbol']}")
                print(f"  RSI: {best['rsi']:.2f}")
                print(f"  Z-Score: {best['z_score']:.2f}")
                print(f"  Current Price: ${best['price']:.2f}")
                print(f"  Mean Price: ${best['mean']:.2f}")
                
                # Execute mean reversion trade
                position_size = min(30, await self.calculate_position_size(best['symbol'], account))
                
                # Place limit order below market for better entry
                entry_price = best['price'] * 0.995
                stop_loss = entry_price * 0.98
                take_profit = best['mean']  # Target mean price
                
                await self.execute_bracket_order(
                    best['symbol'],
                    position_size,
                    'buy',
                    entry_price,
                    stop_loss,
                    take_profit
                )
                
                self.print_success(f"Mean reversion trade executed on {best['symbol']}")
            else:
                self.print_info("No oversold opportunities found")
                
        except Exception as e:
            self.print_error(f"Mean reversion strategy failed: {str(e)}")
            
    async def display_real_time_pnl(self):
        """Display real-time P&L for all positions."""
        self.print_header("REAL-TIME P&L MONITOR")
        
        try:
            positions = self.api.list_positions()
            
            if not positions:
                self.print_info("No positions to monitor")
                return
                
            print("Monitoring P&L for 30 seconds...")
            print("Press Ctrl+C to stop\n")
            
            start_time = time.time()
            
            while time.time() - start_time < 30:
                # Clear previous output
                print("\033[F" * (len(positions) + 3))
                
                total_pnl = 0
                
                for pos in positions:
                    # Get latest quote
                    quote = self.api.get_latest_quote(pos.symbol)
                    current_price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
                    
                    # Calculate P&L
                    qty = int(pos.qty)
                    avg_cost = float(pos.avg_entry_price)
                    unrealized_pnl = (current_price - avg_cost) * qty
                    unrealized_pnl_pct = (current_price / avg_cost - 1) * 100
                    
                    total_pnl += unrealized_pnl
                    
                    # Display
                    color = Fore.GREEN if unrealized_pnl >= 0 else Fore.RED
                    print(f"{pos.symbol}: {color}${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%){Style.RESET_ALL}")
                    
                # Total P&L
                color = Fore.GREEN if total_pnl >= 0 else Fore.RED
                print(f"\nTotal P&L: {color}${total_pnl:,.2f}{Style.RESET_ALL}")
                print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nP&L monitoring stopped")
        except Exception as e:
            self.print_error(f"Failed to display P&L: {str(e)}")
            
    async def cleanup_orders(self):
        """Cancel all open orders for cleanup."""
        self.print_header("CLEANUP - CANCELING OPEN ORDERS")
        
        try:
            orders = self.api.list_orders(status='open')
            
            if not orders:
                self.print_info("No open orders to cancel")
                return
                
            for order in orders:
                try:
                    self.api.cancel_order(order.id)
                    self.print_success(f"Cancelled order {order.id} for {order.symbol}")
                except Exception as e:
                    self.print_error(f"Failed to cancel order {order.id}: {str(e)}")
                    
        except Exception as e:
            self.print_error(f"Failed to cleanup orders: {str(e)}")
            
    async def run_full_demo(self):
        """Run the complete trading demonstration."""
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}ALPACA FULL TRADING DEMONSTRATION")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"\nStarting comprehensive trading demo at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test symbols for demo
        test_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA']
        
        try:
            # 1. Show account status
            account = await self.show_account_status()
            if not account:
                return
                
            # 2. Show current positions
            await self.show_current_positions()
            
            # 3. Execute different order types
            self.print_header("ORDER TYPE DEMONSTRATIONS")
            
            # Market order
            await self.execute_market_order('SPY', 1, 'buy')
            
            # Limit order
            quote = self.api.get_latest_quote('AAPL')
            limit_price = quote.bid_price * 0.99  # 1% below bid
            await self.execute_limit_order('AAPL', 1, 'buy', limit_price)
            
            # Stop order
            quote = self.api.get_latest_quote('MSFT')
            stop_price = quote.ask_price * 1.01  # 1% above ask
            await self.execute_stop_order('MSFT', 1, 'sell', stop_price)
            
            # Bracket order
            quote = self.api.get_latest_quote('QQQ')
            current_price = quote.ask_price
            await self.execute_bracket_order(
                'QQQ', 1, 'buy',
                current_price * 0.995,  # Entry limit
                current_price * 0.98,   # Stop loss
                current_price * 1.02    # Take profit
            )
            
            # 4. Synthetic options strategies
            await self.synthetic_covered_call('TSLA', account)
            await self.synthetic_protective_put('SPY', account)
            await self.synthetic_iron_condor('QQQ', account)
            
            # 5. Execute trading strategies
            await self.momentum_strategy(test_symbols, account)
            await self.mean_reversion_strategy(test_symbols, account)
            
            # 6. Display real-time P&L
            await self.display_real_time_pnl()
            
            # 7. Show final positions
            await self.show_current_positions()
            
            # 8. Summary
            self.print_header("DEMO SUMMARY")
            print(f"Total orders executed: {len(self.executed_orders)}")
            print(f"Order types demonstrated: Market, Limit, Stop, Bracket")
            print(f"Strategies demonstrated: Momentum, Mean Reversion")
            print(f"Synthetic options: Iron Condor, Covered Call, Protective Put")
            
            # 9. Cleanup
            await self.cleanup_orders()
            
            self.print_success("\nTrading demonstration completed successfully!")
            
        except Exception as e:
            self.print_error(f"Demo failed: {str(e)}")
            logger.exception("Demo error")
            
async def main():
    """Main entry point."""
    demo = AlpacaTradingDemo()
    await demo.run_full_demo()
    
if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())