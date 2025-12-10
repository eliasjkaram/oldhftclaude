#!/usr/bin/env python3
"""
Continuous Multi-Leg Options Trader
Runs during market hours and continuously scans for opportunities
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
import threading
import queue
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousMultiLegTrader:
    def __init__(self, paper=True):
        """Initialize continuous trader"""
        # API credentials
        if paper:
            self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
            self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.api_key = 'AK7LZKPVTPZTOTO9VVPM'
            self.api_secret = '2TjtRymW9aWXkWWQFwTThQGAKQrTbSWwLdz1LGKI'
            self.base_url = 'https://api.alpaca.markets'
            
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # API headers
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
        
        # Watchlist
        self.primary_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
        self.secondary_symbols = ['META', 'MSFT', 'AMD', 'AMZN', 'GOOGL']
        self.all_symbols = self.primary_symbols + self.secondary_symbols
        
        # Trading parameters
        self.scan_interval = 300  # 5 minutes
        self.max_positions = 20
        self.max_trades_per_hour = 5
        self.position_size = 2
        
        # Tracking
        self.active_strategies = {}
        self.hourly_trades = []
        self.total_pnl = 0
        self.opportunities_found = 0
        self.trades_executed = 0
        
        # Control flags
        self.running = False
        self.market_open = False
        
    def run(self):
        """Main continuous trading loop"""
        logger.info("🤖 CONTINUOUS MULTI-LEG OPTIONS TRADER")
        logger.info("=" * 70)
        
        # Verify account
        if not self.verify_account():
            return
            
        # Start continuous monitoring
        self.running = True
        
        try:
            while self.running:
                # Check market status
                self.market_open = self.is_market_open()
                
                if not self.market_open:
                    self.wait_for_market_open()
                    continue
                    
                # Run trading cycle
                self.trading_cycle()
                
                # Wait for next scan
                logger.info(f"\n⏳ Waiting {self.scan_interval} seconds until next scan...")
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down gracefully...")
            self.shutdown()
            
    def verify_account(self) -> bool:
        """Verify account status"""
        try:
            account = self.trading_client.get_account()
            
            logger.info(f"\n💼 Account Status:")
            logger.info(f"   Portfolio: ${float(account.portfolio_value):,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"   Options Level: {account.options_trading_level}")
            
            return account.options_trading_level >= 2
            
        except Exception as e:
            logger.error(f"Account error: {e}")
            return False
            
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            # Fallback to time check
            now = datetime.now(timezone(timedelta(hours=-5)))  # ET
            if now.weekday() >= 5:  # Weekend
                return False
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)
            return market_open <= now <= market_close
            
    def wait_for_market_open(self):
        """Wait for market to open"""
        try:
            clock = self.trading_client.get_clock()
            if not clock.is_open and clock.next_open:
                time_to_open = (clock.next_open - datetime.now(timezone.utc)).total_seconds()
                if time_to_open > 0 and time_to_open < 3600:  # Less than 1 hour
                    logger.info(f"\n⏰ Market opens in {time_to_open/60:.1f} minutes")
                    logger.info("   Waiting for market open...")
                    time.sleep(min(time_to_open, 300))  # Wait up to 5 minutes
                else:
                    logger.info("\n🌙 Market is closed. Exiting...")
                    self.running = False
        except Exception as e:
            logger.error(f"Clock error: {e}")
            self.running = False
            
    def trading_cycle(self):
        """Execute one trading cycle"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*70}")
        
        # Update positions
        self.update_positions()
        
        # Check rate limits
        if not self.check_rate_limits():
            logger.warning("⚠️  Rate limit reached - skipping cycle")
            return
            
        # Scan for opportunities
        opportunities = self.scan_for_opportunities()
        
        if opportunities:
            logger.info(f"\n🎯 Found {len(opportunities)} opportunities")
            self.opportunities_found += len(opportunities)
            
            # Execute best opportunities
            for opp in opportunities[:3]:  # Top 3
                if self.execute_opportunity(opp):
                    self.trades_executed += 1
                    self.hourly_trades.append(datetime.now())
                    
        else:
            logger.info("\n💤 No opportunities found this cycle")
            
        # Show status
        self.show_status()
        
    def update_positions(self):
        """Update current positions and P&L"""
        try:
            positions = self.trading_client.get_all_positions()
            option_positions = [p for p in positions if len(p.symbol) > 10]
            
            if option_positions:
                logger.info(f"\n📊 Active Positions: {len(option_positions)}")
                
                total_pnl = 0
                for pos in option_positions:
                    pnl = float(pos.unrealized_pl)
                    total_pnl += pnl
                    
                    # Check if position needs management
                    if pnl < -100:  # Loss threshold
                        logger.warning(f"   ⚠️  {pos.symbol}: ${pnl:+.2f} - Consider closing")
                    elif pnl > 200:  # Profit threshold
                        logger.info(f"   ✅ {pos.symbol}: ${pnl:+.2f} - Consider taking profit")
                        
                self.total_pnl = total_pnl
                logger.info(f"   Total P&L: ${total_pnl:+,.2f}")
                
        except Exception as e:
            logger.error(f"Position update error: {e}")
            
    def check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        # Clean old trades from hourly list
        one_hour_ago = datetime.now() - timedelta(hours=1)
        self.hourly_trades = [t for t in self.hourly_trades if t > one_hour_ago]
        
        # Check limits
        if len(self.hourly_trades) >= self.max_trades_per_hour:
            return False
            
        # Check position limits
        try:
            positions = self.trading_client.get_all_positions()
            if len(positions) >= self.max_positions:
                logger.warning(f"Position limit reached ({len(positions)}/{self.max_positions})")
                return False
        except:
            pass
            
        return True
        
    def scan_for_opportunities(self) -> List[Dict]:
        """Scan all symbols for trading opportunities"""
        opportunities = []
        
        # Prioritize symbols with recent activity
        for symbol in self.primary_symbols:
            opp = self.analyze_symbol(symbol, priority=True)
            if opp:
                opportunities.append(opp)
                
        # Check secondary symbols if needed
        if len(opportunities) < 3:
            for symbol in self.secondary_symbols:
                opp = self.analyze_symbol(symbol, priority=False)
                if opp:
                    opportunities.append(opp)
                    
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
        
    def analyze_symbol(self, symbol: str, priority: bool) -> Optional[Dict]:
        """Analyze symbol for opportunities"""
        try:
            # Get market data
            quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            
            if symbol not in quote:
                return None
                
            price = float(quote[symbol].ask_price)
            
            # Get recent bars
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=30
                )
            )
            
            if symbol not in bars.df.index:
                return None
                
            df = bars.df.loc[symbol]
            
            # Quick technical analysis
            analysis = self.quick_analysis(df, price)
            
            # Check for opportunity
            if analysis['signal'] != 'none':
                # Get option chain
                chain = self.get_quick_option_chain(symbol, price)
                if not chain:
                    return None
                    
                # Find best strategy
                strategy = self.find_quick_strategy(symbol, price, analysis, chain)
                if strategy:
                    return {
                        'symbol': symbol,
                        'price': price,
                        'signal': analysis['signal'],
                        'confidence': analysis['confidence'],
                        'strategy': strategy,
                        'score': analysis['confidence'] * (1.2 if priority else 1.0)
                    }
                    
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
        return None
        
    def quick_analysis(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Quick technical analysis for signals"""
        try:
            # Recent price action
            close_prices = df['close'].values
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            
            # Simple indicators
            sma_10 = np.mean(close_prices[-10:])
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            
            # Volume surge
            volume_avg = df['volume'].mean()
            recent_volume = df['volume'].iloc[-1]
            volume_surge = recent_volume > volume_avg * 1.5
            
            # Detect signals
            signal = 'none'
            confidence = 0
            
            # Breakout
            if current_price > recent_high * 0.995 and volume_surge:
                signal = 'bullish_breakout'
                confidence = 0.75
                
            # Bounce from support
            elif price_position < 0.2 and current_price > sma_10:
                signal = 'bullish_bounce'
                confidence = 0.70
                
            # Rejection from resistance
            elif price_position > 0.8 and current_price < sma_10:
                signal = 'bearish_rejection'
                confidence = 0.70
                
            # Range bound
            elif 0.3 < price_position < 0.7:
                signal = 'range_bound'
                confidence = 0.65
                
            return {
                'signal': signal,
                'confidence': confidence,
                'price_position': price_position,
                'volume_surge': volume_surge
            }
            
        except Exception as e:
            logger.error(f"Quick analysis error: {e}")
            return {'signal': 'none', 'confidence': 0}
            
    def get_quick_option_chain(self, symbol: str, price: float) -> Optional[Dict]:
        """Get simplified option chain for quick decisions"""
        try:
            today = datetime.now().date()
            
            # Target 2-4 week expiration
            exp_min = (today + timedelta(days=14)).strftime('%Y-%m-%d')
            exp_max = (today + timedelta(days=28)).strftime('%Y-%m-%d')
            
            # Narrow strike range
            strike_min = price * 0.95
            strike_max = price * 1.05
            
            url = f"{self.base_url}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'expiration_date_gte': exp_min,
                'expiration_date_lte': exp_max,
                'strike_price_gte': strike_min,
                'strike_price_lte': strike_max,
                'limit': 100
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                if contracts:
                    return self.organize_quick_chain(contracts, price)
                    
        except Exception as e:
            logger.error(f"Chain error: {e}")
            
        return None
        
    def organize_quick_chain(self, contracts: List[Dict], price: float) -> Dict:
        """Organize chain for quick access"""
        chain = {
            'puts': [],
            'calls': [],
            'atm_strike': round(price / 5) * 5
        }
        
        for contract in contracts:
            if contract.get('type') == 'put':
                chain['puts'].append(contract)
            else:
                chain['calls'].append(contract)
                
        # Sort by strike
        chain['puts'].sort(key=lambda x: float(x.get('strike_price', 0)))
        chain['calls'].sort(key=lambda x: float(x.get('strike_price', 0)))
        
        return chain
        
    def find_quick_strategy(self, symbol: str, price: float, 
                          analysis: Dict, chain: Dict) -> Optional[Dict]:
        """Find appropriate strategy based on signal"""
        signal = analysis['signal']
        
        if signal == 'bullish_breakout':
            # Bull call spread
            calls = chain['calls']
            if len(calls) >= 2:
                long_call = calls[0]  # ATM
                short_call = calls[2] if len(calls) > 2 else calls[1]  # OTM
                
                return {
                    'type': 'bull_call_spread',
                    'legs': [
                        {'option': long_call, 'action': 'buy', 'quantity': self.position_size},
                        {'option': short_call, 'action': 'sell', 'quantity': self.position_size}
                    ]
                }
                
        elif signal == 'bearish_rejection':
            # Bear put spread
            puts = chain['puts']
            if len(puts) >= 2:
                long_put = puts[-1]  # ATM
                short_put = puts[-3] if len(puts) > 2 else puts[-2]  # OTM
                
                return {
                    'type': 'bear_put_spread',
                    'legs': [
                        {'option': long_put, 'action': 'buy', 'quantity': self.position_size},
                        {'option': short_put, 'action': 'sell', 'quantity': self.position_size}
                    ]
                }
                
        elif signal == 'range_bound':
            # Iron condor
            puts = chain['puts']
            calls = chain['calls']
            
            if len(puts) >= 2 and len(calls) >= 2:
                # Find OTM options
                otm_puts = [p for p in puts if float(p['strike_price']) < price * 0.97]
                otm_calls = [c for c in calls if float(c['strike_price']) > price * 1.03]
                
                if len(otm_puts) >= 2 and len(otm_calls) >= 2:
                    return {
                        'type': 'iron_condor',
                        'legs': [
                            {'option': otm_puts[0], 'action': 'buy', 'quantity': self.position_size},
                            {'option': otm_puts[-1], 'action': 'sell', 'quantity': self.position_size},
                            {'option': otm_calls[0], 'action': 'sell', 'quantity': self.position_size},
                            {'option': otm_calls[-1], 'action': 'buy', 'quantity': self.position_size}
                        ]
                    }
                    
        elif signal == 'bullish_bounce':
            # Bull put spread (credit)
            puts = chain['puts']
            if len(puts) >= 2:
                # OTM puts
                otm_puts = [p for p in puts if float(p['strike_price']) < price * 0.98]
                if len(otm_puts) >= 2:
                    return {
                        'type': 'bull_put_spread',
                        'legs': [
                            {'option': otm_puts[-1], 'action': 'sell', 'quantity': self.position_size},
                            {'option': otm_puts[0], 'action': 'buy', 'quantity': self.position_size}
                        ]
                    }
                    
        return None
        
    def execute_opportunity(self, opportunity: Dict) -> bool:
        """Execute trading opportunity"""
        try:
            symbol = opportunity['symbol']
            strategy = opportunity['strategy']
            
            logger.info(f"\n🎯 Executing {strategy['type']} on {symbol}")
            logger.info(f"   Signal: {opportunity['signal']} ({opportunity['confidence']:.1%})")
            
            # Execute each leg
            success = True
            for leg in strategy['legs']:
                option = leg['option']
                action = leg['action']
                quantity = leg['quantity']
                
                logger.info(f"   {action.upper()} {quantity}x {option['symbol']}")
                
                # Submit order
                order = MarketOrderRequest(
                    symbol=option['symbol'],
                    qty=quantity,
                    side=OrderSide.BUY if action == 'buy' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                try:
                    result = self.trading_client.submit_order(order)
                    logger.info(f"   ✅ Order submitted: {result.id}")
                except Exception as e:
                    logger.error(f"   ❌ Order failed: {e}")
                    success = False
                    break
                    
            if success:
                # Track strategy
                self.active_strategies[symbol] = {
                    'type': strategy['type'],
                    'entry_time': datetime.now(),
                    'entry_price': opportunity['price']
                }
                
            return success
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
            
    def show_status(self):
        """Show current trading status"""
        logger.info(f"\n📈 Session Status:")
        logger.info(f"   Opportunities Found: {self.opportunities_found}")
        logger.info(f"   Trades Executed: {self.trades_executed}")
        logger.info(f"   Active Strategies: {len(self.active_strategies)}")
        logger.info(f"   Session P&L: ${self.total_pnl:+,.2f}")
        
        # Show active strategies
        if self.active_strategies:
            logger.info(f"\n   Active Strategies:")
            for symbol, strategy in self.active_strategies.items():
                elapsed = (datetime.now() - strategy['entry_time']).seconds / 60
                logger.info(f"      {symbol}: {strategy['type']} ({elapsed:.0f} min)")
                
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        logger.info("\n📊 Final Summary:")
        logger.info(f"   Total Opportunities: {self.opportunities_found}")
        logger.info(f"   Total Trades: {self.trades_executed}")
        logger.info(f"   Final P&L: ${self.total_pnl:+,.2f}")
        
        # Show all positions
        try:
            positions = self.trading_client.get_all_positions()
            if positions:
                logger.info(f"\n   Open Positions: {len(positions)}")
                for pos in positions:
                    if len(pos.symbol) > 10:
                        logger.info(f"      {pos.symbol}: {float(pos.qty):+.0f} @ ${float(pos.avg_entry_price):.2f}")
        except:
            pass

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Multi-Leg Options Trader')
    parser.add_argument('--live', action='store_true', help='Use live trading')
    parser.add_argument('--interval', type=int, default=300, help='Scan interval in seconds')
    args = parser.parse_args()
    
    trader = ContinuousMultiLegTrader(paper=not args.live)
    trader.scan_interval = args.interval
    
    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        trader.shutdown()

if __name__ == "__main__":
    main()