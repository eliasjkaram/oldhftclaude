#!/usr/bin/env python3
"""Check recent option trades"""

import logging
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paper trading credentials
api_key = 'PKEP9PIBDKOSUGHHY44Z'
api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'

client = TradingClient(api_key, api_secret, paper=True)

try:
    # Get orders from last 7 days
    request = GetOrdersRequest(
        after=datetime.now() - timedelta(days=7),
        limit=100
    )
    
    orders = client.get_orders(request)
    
    logger.info('\n📜 RECENT ORDERS (Last 7 Days)')
    logger.info('═' * 80)
    
    if not orders:
        logger.info('No recent orders found')
    else:
        # Count by status
        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        logger.info(f'\nTotal Orders: {len(orders)} ({len(filled)} filled)')
        
        # Group by underlying
        option_orders = {}
        for order in filled:
            if len(order.symbol) > 10:  # Options
                # Extract underlying
                for i, char in enumerate(order.symbol):
                    if char.isdigit():
                        underlying = order.symbol[:i]
                        break
                
                if underlying not in option_orders:
                    option_orders[underlying] = []
                option_orders[underlying].append(order)
        
        # Show by underlying
        for underlying, orders in option_orders.items():
            logger.info(f'\n{underlying}:')
            for order in orders[-10:]:  # Last 10 per symbol
                created = order.created_at.strftime('%m-%d %H:%M')
                logger.info(f'  {created} | {order.symbol} | {order.side} {order.qty} @ ${float(order.filled_avg_price):.2f}')
                    
except Exception as e:
    logger.error(f'Error: {e}')