#!/usr/bin/env python3
"""
Simple Alpaca Paper Trading Account Status Checker
Displays current account information including portfolio value, positions, and recent orders.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def check_account_status():
    """Check and display Alpaca paper trading account status."""
    
    # Get API credentials from environment
    api_key = os.getenv('ALPACA_PAPER_API_KEY')
    secret_key = os.getenv('ALPACA_PAPER_API_SECRET')
    
    if not api_key or not secret_key:
        print("Error: Alpaca API credentials not found in environment variables.")
        print("Please set ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET")
        return
    
    try:
        # Create trading client for paper trading
        client = TradingClient(api_key, secret_key, paper=True)
        
        print("=" * 60)
        print("ALPACA PAPER TRADING ACCOUNT STATUS")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        # Get account information
        account = client.get_account()
        
        # Display account details
        print("\nACCOUNT SUMMARY:")
        print(f"  Portfolio Value: ${float(account.equity):,.2f}")
        print(f"  Cash Balance: ${float(account.cash):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        print(f"  Pattern Day Trader: {account.pattern_day_trader}")
        print(f"  Account Blocked: {account.account_blocked}")
        print(f"  Trading Blocked: {account.trading_blocked}")
        
        # Get positions
        positions = client.get_all_positions()
        print(f"\nPOSITIONS ({len(positions)} total):")
        
        if positions:
            total_market_value = 0
            total_unrealized_pl = 0
            
            for pos in positions:
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100
                
                total_market_value += market_value
                total_unrealized_pl += unrealized_pl
                
                print(f"  {pos.symbol}:")
                print(f"    Quantity: {pos.qty} shares")
                print(f"    Avg Entry Price: ${float(pos.avg_entry_price):.2f}")
                print(f"    Current Price: ${float(pos.current_price):.2f}")
                print(f"    Market Value: ${market_value:,.2f}")
                print(f"    Unrealized P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")
                print()
            
            print(f"  Total Market Value: ${total_market_value:,.2f}")
            print(f"  Total Unrealized P&L: ${total_unrealized_pl:,.2f}")
        else:
            print("  No open positions")
        
        # Get recent orders
        print("\nRECENT ORDERS (last 10):")
        
        # Request all orders, limited to 10
        orders_request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=10
        )
        orders = client.get_orders(orders_request)
        
        if orders:
            for order in orders:
                created_at = order.created_at.strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {order.symbol} - {order.side} {order.qty} shares")
                print(f"    Type: {order.order_type}")
                print(f"    Status: {order.status}")
                if order.filled_avg_price:
                    print(f"    Filled Price: ${float(order.filled_avg_price):.2f}")
                print(f"    Created: {created_at}")
                print()
        else:
            print("  No recent orders")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return

if __name__ == "__main__":
    check_account_status()