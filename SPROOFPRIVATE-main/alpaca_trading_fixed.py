#!/usr/bin/env python3
"""
Fixed Alpaca Trading Demo - Properly handles price rounding
"""

import os
import time
from decimal import Decimal, ROUND_HALF_UP
from alpaca_trade_api import REST
from datetime import datetime, timedelta
import pandas as pd

# Alpaca API credentials
API_KEY = "PKEP9PIBDKOSUGHHY44Z"
API_SECRET = "VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca client
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def round_price(price, decimals=2):
    """Round price to specified decimal places using proper rounding"""
    return float(Decimal(str(price)).quantize(Decimal(f'0.{"0" * decimals}'), rounding=ROUND_HALF_UP))

def show_account_status():
    """Display account status"""
    print("\n" + "="*60)
    print("ACCOUNT STATUS")
    print("="*60)
    
    account = api.get_account()
    print(f"Account Number: {account.account_number}")
    print(f"Status: {account.status}")
    print(f"Currency: {account.currency}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Day Trade Count: {account.daytrade_count}")
    print(f"Pattern Day Trader: {account.pattern_day_trader}")
    
    return account

def get_latest_price(symbol):
    """Get latest price for a symbol"""
    try:
        # Get latest trade
        trades = api.get_latest_trade(symbol)
        return round_price(trades.price)
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        # Fallback to quote
        try:
            quote = api.get_latest_quote(symbol)
            mid_price = (quote.ask_price + quote.bid_price) / 2
            return round_price(mid_price)
        except:
            return None

def place_market_order(symbol, qty, side):
    """Place a simple market order"""
    print(f"\nPlacing {side} market order for {qty} shares of {symbol}")
    
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"✓ Market order placed: {order.id}")
        return order
    except Exception as e:
        print(f"✗ Error placing market order: {e}")
        return None

def place_limit_order(symbol, qty, side, limit_price):
    """Place a limit order with properly rounded price"""
    # Round the limit price
    limit_price = round_price(limit_price)
    
    print(f"\nPlacing {side} limit order for {qty} shares of {symbol} at ${limit_price:.2f}")
    
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='day',
            limit_price=limit_price
        )
        print(f"✓ Limit order placed: {order.id}")
        return order
    except Exception as e:
        print(f"✗ Error placing limit order: {e}")
        return None

def place_stop_order(symbol, qty, side, stop_price):
    """Place a stop order with properly rounded price"""
    # Round the stop price
    stop_price = round_price(stop_price)
    
    print(f"\nPlacing {side} stop order for {qty} shares of {symbol} at ${stop_price:.2f}")
    
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='stop',
            time_in_force='day',
            stop_price=stop_price
        )
        print(f"✓ Stop order placed: {order.id}")
        return order
    except Exception as e:
        print(f"✗ Error placing stop order: {e}")
        return None

def place_bracket_order(symbol, qty, side, limit_price, take_profit_price, stop_loss_price):
    """Place a bracket order with properly rounded prices"""
    # Round all prices
    limit_price = round_price(limit_price)
    take_profit_price = round_price(take_profit_price)
    stop_loss_price = round_price(stop_loss_price)
    
    print(f"\nPlacing bracket order for {qty} shares of {symbol}")
    print(f"  Entry: ${limit_price:.2f}")
    print(f"  Take Profit: ${take_profit_price:.2f}")
    print(f"  Stop Loss: ${stop_loss_price:.2f}")
    
    try:
        # Create the bracket order
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='day',
            limit_price=limit_price,
            order_class='bracket',
            take_profit=dict(
                limit_price=take_profit_price
            ),
            stop_loss=dict(
                stop_price=stop_loss_price
            )
        )
        print(f"✓ Bracket order placed: {order.id}")
        return order
    except Exception as e:
        print(f"✗ Error placing bracket order: {e}")
        return None

def show_positions():
    """Display current positions"""
    print("\n" + "="*60)
    print("CURRENT POSITIONS")
    print("="*60)
    
    positions = api.list_positions()
    
    if not positions:
        print("No open positions")
        return
    
    total_value = 0
    total_pl = 0
    
    for position in positions:
        current_price = float(position.current_price)
        market_value = float(position.market_value)
        unrealized_pl = float(position.unrealized_pl)
        unrealized_plpc = float(position.unrealized_plpc) * 100
        
        total_value += market_value
        total_pl += unrealized_pl
        
        print(f"\n{position.symbol}:")
        print(f"  Quantity: {position.qty}")
        print(f"  Avg Entry: ${float(position.avg_entry_price):.2f}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Market Value: ${market_value:,.2f}")
        print(f"  Unrealized P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")
    
    print(f"\nTotal Position Value: ${total_value:,.2f}")
    print(f"Total Unrealized P&L: ${total_pl:,.2f}")

def show_recent_orders(limit=10):
    """Display recent orders"""
    print("\n" + "="*60)
    print(f"RECENT ORDERS (Last {limit})")
    print("="*60)
    
    orders = api.list_orders(status='all', limit=limit)
    
    for order in orders:
        print(f"\n{order.symbol} - {order.side.upper()} {order.qty} @ {order.order_type.upper()}")
        print(f"  Status: {order.status}")
        print(f"  Submitted: {order.submitted_at}")
        
        if order.filled_qty and int(order.filled_qty) > 0:
            print(f"  Filled: {order.filled_qty} @ ${float(order.filled_avg_price or 0):.2f}")
        
        if order.limit_price:
            print(f"  Limit Price: ${float(order.limit_price):.2f}")
        
        if order.stop_price:
            print(f"  Stop Price: ${float(order.stop_price):.2f}")

def cancel_all_orders():
    """Cancel all open orders"""
    print("\nCancelling all open orders...")
    try:
        api.cancel_all_orders()
        print("✓ All orders cancelled")
    except Exception as e:
        print(f"✗ Error cancelling orders: {e}")

def close_all_positions():
    """Close all open positions"""
    print("\nClosing all open positions...")
    try:
        api.close_all_positions()
        print("✓ All positions closed")
    except Exception as e:
        print(f"✗ Error closing positions: {e}")

def main():
    """Run the trading demo"""
    print("\n" + "="*60)
    print("ALPACA TRADING DEMO - FIXED PRICING")
    print("="*60)
    
    # Show initial account status
    account = show_account_status()
    
    # Cancel any existing orders
    cancel_all_orders()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'SPY']
    
    print("\n" + "="*60)
    print("EXECUTING DEMO TRADES")
    print("="*60)
    
    # 1. Market Orders
    print("\n--- Market Orders ---")
    for symbol in symbols[:2]:
        place_market_order(symbol, 1, 'buy')
        time.sleep(1)
    
    # Wait for fills
    time.sleep(3)
    
    # 2. Limit Orders with proper rounding
    print("\n--- Limit Orders ---")
    for symbol in symbols[:2]:
        current_price = get_latest_price(symbol)
        if current_price:
            # Buy limit below market
            buy_limit = round_price(current_price * 0.99)
            place_limit_order(symbol, 1, 'buy', buy_limit)
            
            # Sell limit above market
            sell_limit = round_price(current_price * 1.01)
            place_limit_order(symbol, 1, 'sell', sell_limit)
            
            time.sleep(1)
    
    # 3. Stop Orders
    print("\n--- Stop Orders ---")
    spy_price = get_latest_price('SPY')
    if spy_price:
        # Stop loss below market
        stop_loss = round_price(spy_price * 0.98)
        place_stop_order('SPY', 1, 'sell', stop_loss)
    
    # 4. Bracket Order
    print("\n--- Bracket Order ---")
    aapl_price = get_latest_price('AAPL')
    if aapl_price:
        # Calculate bracket prices
        entry_price = round_price(aapl_price * 0.995)  # Buy slightly below market
        take_profit = round_price(aapl_price * 1.02)   # 2% profit target
        stop_loss = round_price(aapl_price * 0.98)     # 2% stop loss
        
        place_bracket_order('AAPL', 2, 'buy', entry_price, take_profit, stop_loss)
    
    # Wait for order processing
    print("\nWaiting for order processing...")
    time.sleep(5)
    
    # Show results
    show_positions()
    show_recent_orders()
    
    # Final account status
    print("\n" + "="*60)
    print("FINAL ACCOUNT STATUS")
    print("="*60)
    
    final_account = api.get_account()
    initial_value = float(account.portfolio_value)
    final_value = float(final_account.portfolio_value)
    change = final_value - initial_value
    change_pct = (change / initial_value * 100) if initial_value > 0 else 0
    
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Change: ${change:,.2f} ({change_pct:+.2f}%)")
    
    # Cleanup option
    print("\n" + "="*60)
    response = input("\nDo you want to close all positions and cancel all orders? (y/n): ")
    if response.lower() == 'y':
        cancel_all_orders()
        close_all_positions()
        print("\n✓ Cleanup complete")
    else:
        print("\n✓ Positions and orders left open")
    
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()