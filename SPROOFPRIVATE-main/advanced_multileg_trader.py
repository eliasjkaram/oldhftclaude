#!/usr/bin/env python3
"""
Advanced Multi-Leg Options Trader
Executes complex 3-4 leg option strategies based on market conditions
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedMultiLegTrader:
    def __init__(self):
        """Initialize advanced multi-leg trader"""
        # Paper trading credentials
        self.api_key = 'PKEP9PIBDKOSUGHHY44Z'
        self.api_secret = 'VtNWykIafQe7VfjPWKUVRu8RXOnpgBYgndyFCwTZ'
        self.base_url = 'https://paper-api.alpaca.markets'
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # API headers
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
        
        # Trading universe
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMD', 'GOOGL', 'AMZN']
        
        # Strategy parameters
        self.max_trades_per_run = 5
        self.position_size = 1
        
        # Track executions
        self.executed_strategies = []
        
    def run(self):
        """Main execution loop"""
        logger.info("🚀 ADVANCED MULTI-LEG OPTIONS TRADER")
        logger.info("Executing complex 3-4 leg strategies")
        logger.info("=" * 70)
        
        # Check account
        if not self.check_account():
            return
            
        # Execute strategies
        trades_executed = 0
        
        for symbol in self.symbols:
            if trades_executed >= self.max_trades_per_run:
                break
                
            try:
                logger.info(f"\n🔍 Analyzing {symbol}...")
                
                # Get market data and analysis
                market_data = self.get_market_data(symbol)
                if not market_data:
                    continue
                    
                # Get option chain
                options = self.get_option_chain(symbol, market_data['price'])
                if not options:
                    logger.info(f"   No options available for {symbol}")
                    continue
                    
                # Execute advanced strategy
                if self.execute_advanced_strategy(symbol, market_data, options):
                    trades_executed += 1
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                
        # Show results
        self.show_results()
        
    def check_account(self) -> bool:
        """Check account readiness"""
        try:
            account = self.trading_client.get_account()
            logger.info(f"\n💼 Account Status:")
            logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"   Options Level: {account.options_trading_level}")
            
            return account.options_trading_level >= 2
            
        except Exception as e:
            logger.error(f"Account error: {e}")
            return False
            
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data and technical analysis"""
        try:
            # Get current quote
            quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            
            if symbol not in quote:
                return None
                
            price = float(quote[symbol].ask_price)
            
            # Get historical data for analysis
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=30
                )
            )
            
            # Calculate indicators
            df = bars.df.loc[symbol] if symbol in bars.df.index else None
            
            if df is not None and len(df) > 20:
                # Technical indicators
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                sma_5 = df['close'].rolling(5).mean().iloc[-1]
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # Volatility
                volatility = df['close'].pct_change().std() * np.sqrt(252)
                
                # Volume analysis
                vol_avg = df['volume'].mean()
                vol_recent = df['volume'].iloc[-5:].mean()
                volume_surge = vol_recent / vol_avg if vol_avg > 0 else 1
                
                # Market regime
                if price > sma_20 * 1.02 and sma_5 > sma_20:
                    trend = 'strong_bullish'
                elif price > sma_20:
                    trend = 'bullish'
                elif price < sma_20 * 0.98 and sma_5 < sma_20:
                    trend = 'strong_bearish'
                elif price < sma_20:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
                    
                # Volatility regime
                if volatility < 0.15:
                    vol_regime = 'low'
                elif volatility < 0.30:
                    vol_regime = 'normal'
                else:
                    vol_regime = 'high'
                    
                return {
                    'symbol': symbol,
                    'price': price,
                    'trend': trend,
                    'volatility': volatility,
                    'vol_regime': vol_regime,
                    'rsi': rsi,
                    'volume_surge': volume_surge,
                    'sma_20': sma_20
                }
            else:
                # Simplified analysis
                return {
                    'symbol': symbol,
                    'price': price,
                    'trend': 'neutral',
                    'volatility': 0.25,
                    'vol_regime': 'normal',
                    'rsi': 50,
                    'volume_surge': 1.0,
                    'sma_20': price
                }
                
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None
            
    def get_option_chain(self, symbol: str, stock_price: float) -> Dict:
        """Get comprehensive option chain"""
        try:
            # Target multiple expiration dates
            today = datetime.now().date()
            
            # Near-term (25-35 days)
            near_exp_min = (today + timedelta(days=20)).strftime('%Y-%m-%d')
            near_exp_max = (today + timedelta(days=40)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'expiration_date_gte': near_exp_min,
                'expiration_date_lte': near_exp_max,
                'limit': 300
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                if contracts:
                    # Organize contracts
                    puts = sorted([c for c in contracts if c.get('type') == 'put'], 
                                 key=lambda x: float(x.get('strike_price', 0)))
                    calls = sorted([c for c in contracts if c.get('type') == 'call'], 
                                  key=lambda x: float(x.get('strike_price', 0)))
                    
                    # Find key strikes
                    atm_strike = round(stock_price / 5) * 5
                    
                    # Strike ranges
                    deep_otm_put_strikes = [p for p in puts if float(p.get('strike_price', 0)) < stock_price * 0.85]
                    otm_put_strikes = [p for p in puts if stock_price * 0.85 <= float(p.get('strike_price', 0)) < stock_price * 0.95]
                    near_put_strikes = [p for p in puts if stock_price * 0.95 <= float(p.get('strike_price', 0)) < stock_price]
                    
                    near_call_strikes = [c for c in calls if stock_price < float(c.get('strike_price', 0)) <= stock_price * 1.05]
                    otm_call_strikes = [c for c in calls if stock_price * 1.05 < float(c.get('strike_price', 0)) <= stock_price * 1.15]
                    deep_otm_call_strikes = [c for c in calls if float(c.get('strike_price', 0)) > stock_price * 1.15]
                    
                    logger.info(f"   Options available: {len(puts)} puts, {len(calls)} calls")
                    logger.info(f"   ATM strike: ${atm_strike}")
                    
                    return {
                        'symbol': symbol,
                        'stock_price': stock_price,
                        'atm_strike': atm_strike,
                        'puts': puts,
                        'calls': calls,
                        'deep_otm_puts': deep_otm_put_strikes,
                        'otm_puts': otm_put_strikes,
                        'near_puts': near_put_strikes,
                        'near_calls': near_call_strikes,
                        'otm_calls': otm_call_strikes,
                        'deep_otm_calls': deep_otm_call_strikes
                    }
                    
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            
        return None
        
    def execute_advanced_strategy(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute advanced multi-leg strategies"""
        try:
            trend = market_data['trend']
            vol_regime = market_data['vol_regime']
            rsi = market_data['rsi']
            volume_surge = market_data['volume_surge']
            
            logger.info(f"\n📊 Market Analysis:")
            logger.info(f"   Trend: {trend.upper()}")
            logger.info(f"   Volatility: {vol_regime} ({market_data['volatility']:.1%})")
            logger.info(f"   RSI: {rsi:.1f}")
            logger.info(f"   Volume: {volume_surge:.1f}x average")
            
            # STRONG TRENDING MARKETS
            if trend == 'strong_bullish' and volume_surge > 1.2:
                # Aggressive: Call Ratio Spread
                if self.execute_call_ratio_spread(symbol, market_data, options):
                    return True
                # Alternative: Call Diagonal
                if self.execute_call_diagonal(symbol, market_data, options):
                    return True
                    
            elif trend == 'strong_bearish' and volume_surge > 1.2:
                # Aggressive: Put Ratio Spread
                if self.execute_put_ratio_spread(symbol, market_data, options):
                    return True
                    
            # MODERATE TRENDS WITH HIGH VOLATILITY
            elif trend in ['bullish', 'bearish'] and vol_regime == 'high':
                # Broken Wing Butterfly
                if self.execute_broken_wing_butterfly(symbol, market_data, options, trend):
                    return True
                    
            # NEUTRAL MARKETS
            elif trend == 'neutral':
                if vol_regime == 'low':
                    # Low vol: Short Iron Butterfly
                    if self.execute_short_iron_butterfly(symbol, market_data, options):
                        return True
                elif vol_regime == 'high':
                    # High vol: Double Diagonal
                    if self.execute_double_diagonal(symbol, market_data, options):
                        return True
                else:
                    # Normal vol: Iron Condor
                    if self.execute_advanced_iron_condor(symbol, market_data, options):
                        return True
                        
            # OVERSOLD/OVERBOUGHT CONDITIONS
            if rsi < 30:
                # Oversold: Jade Lizard (bullish credit)
                if self.execute_jade_lizard(symbol, market_data, options):
                    return True
            elif rsi > 70:
                # Overbought: Twisted Sister (bearish credit)
                if self.execute_twisted_sister(symbol, market_data, options):
                    return True
                    
            # DEFAULT: Standard multi-leg
            logger.info("   Attempting standard multi-leg strategy...")
            if self.execute_standard_multileg(symbol, market_data, options):
                return True
                
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            
        return False
        
    def execute_call_ratio_spread(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute 1x2 call ratio spread for aggressive bullish"""
        try:
            logger.info("   🚀 Executing CALL RATIO SPREAD (1x2)")
            
            price = market_data['price']
            near_calls = options.get('near_calls', [])
            otm_calls = options.get('otm_calls', [])
            
            if near_calls and len(otm_calls) >= 2:
                # Buy 1 near call
                buy_call = near_calls[0]
                # Sell 2 OTM calls
                sell_call = otm_calls[0]
                
                buy_strike = float(buy_call.get('strike_price'))
                sell_strike = float(sell_call.get('strike_price'))
                
                if sell_strike > buy_strike + 5:
                    logger.info(f"   Buy 1x {buy_call.get('symbol')} @ ${buy_strike}")
                    logger.info(f"   Sell 2x {sell_call.get('symbol')} @ ${sell_strike}")
                    logger.info(f"   Max profit at ${sell_strike}")
                    
                    # Execute
                    logger.info("\n   🚀 EXECUTING 3-LEG RATIO:")
                    
                    # Buy 1 call
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=buy_call.get('symbol'),
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ BOUGHT 1x {buy_call.get('symbol')}")
                    
                    # Sell 2 calls
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=sell_call.get('symbol'),
                        qty=2,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ SOLD 2x {sell_call.get('symbol')}")
                    
                    self.executed_strategies.append({
                        'strategy': 'Call Ratio Spread (1x2)',
                        'symbol': symbol,
                        'legs': 3,
                        'details': [
                            f"BUY 1x {buy_call.get('symbol')} @ ${buy_strike}",
                            f"SELL 2x {sell_call.get('symbol')} @ ${sell_strike}"
                        ],
                        'market': market_data['trend']
                    })
                    
                    return True
                    
        except Exception as e:
            logger.error(f"   Ratio spread error: {e}")
            
        return False
        
    def execute_broken_wing_butterfly(self, symbol: str, market_data: Dict, 
                                     options: Dict, direction: str) -> bool:
        """Execute broken wing butterfly (3-leg asymmetric)"""
        try:
            logger.info(f"   🦋 Executing BROKEN WING BUTTERFLY ({direction})")
            
            atm_strike = options['atm_strike']
            
            if direction == 'bullish':
                calls = options['calls']
                
                # Find strikes (asymmetric wings)
                lower = next((c for c in calls if float(c.get('strike_price')) == atm_strike - 5), None)
                middle = next((c for c in calls if float(c.get('strike_price')) == atm_strike), None)
                upper = next((c for c in calls if float(c.get('strike_price')) == atm_strike + 10), None)  # Wider upper wing
                
                if all([lower, middle, upper]):
                    logger.info(f"   Lower wing: {lower.get('symbol')} @ ${float(lower.get('strike_price'))}")
                    logger.info(f"   Body: {middle.get('symbol')} @ ${float(middle.get('strike_price'))} (x2)")
                    logger.info(f"   Upper wing: {upper.get('symbol')} @ ${float(upper.get('strike_price'))}")
                    logger.info("   Profit skewed to upside")
                    
                    # Execute
                    logger.info("\n   🚀 EXECUTING BROKEN WING:")
                    
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=lower.get('symbol'),
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ BOUGHT {lower.get('symbol')}")
                    
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=upper.get('symbol'),
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ BOUGHT {upper.get('symbol')}")
                    
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=middle.get('symbol'),
                        qty=2,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ SOLD 2x {middle.get('symbol')}")
                    
                    self.executed_strategies.append({
                        'strategy': 'Broken Wing Butterfly',
                        'symbol': symbol,
                        'legs': 3,
                        'details': [
                            f"BUY {lower.get('symbol')}",
                            f"SELL 2x {middle.get('symbol')}",
                            f"BUY {upper.get('symbol')} (wider wing)"
                        ],
                        'market': market_data['trend']
                    })
                    
                    return True
                    
        except Exception as e:
            logger.error(f"   Broken wing error: {e}")
            
        return False
        
    def execute_jade_lizard(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute jade lizard (3-leg bullish credit)"""
        try:
            logger.info("   🦎 Executing JADE LIZARD (Bullish Credit)")
            
            price = market_data['price']
            otm_puts = options.get('otm_puts', [])
            otm_calls = options.get('otm_calls', [])
            
            if len(otm_puts) >= 2 and otm_calls:
                # Sell put spread + sell call
                sell_put = otm_puts[-1]  # Higher strike put
                buy_put = otm_puts[0]    # Lower strike put
                sell_call = otm_calls[0] # OTM call
                
                sell_put_strike = float(sell_put.get('strike_price'))
                buy_put_strike = float(buy_put.get('strike_price'))
                sell_call_strike = float(sell_call.get('strike_price'))
                
                logger.info(f"   Put Spread: SELL ${sell_put_strike} / BUY ${buy_put_strike}")
                logger.info(f"   Naked Call: SELL ${sell_call_strike}")
                logger.info("   No upside risk if credit > spread width")
                
                # Execute
                logger.info("\n   🚀 EXECUTING JADE LIZARD:")
                
                # Buy protective put
                self.trading_client.submit_order(MarketOrderRequest(
                    symbol=buy_put.get('symbol'),
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                ))
                logger.info(f"   ✅ BOUGHT {buy_put.get('symbol')}")
                
                # Sell put
                self.trading_client.submit_order(MarketOrderRequest(
                    symbol=sell_put.get('symbol'),
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                ))
                logger.info(f"   ✅ SOLD {sell_put.get('symbol')}")
                
                # Sell call
                self.trading_client.submit_order(MarketOrderRequest(
                    symbol=sell_call.get('symbol'),
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                ))
                logger.info(f"   ✅ SOLD {sell_call.get('symbol')}")
                
                self.executed_strategies.append({
                    'strategy': 'Jade Lizard',
                    'symbol': symbol,
                    'legs': 3,
                    'details': [
                        f"SELL {sell_put.get('symbol')} @ ${sell_put_strike}",
                        f"BUY {buy_put.get('symbol')} @ ${buy_put_strike}",
                        f"SELL {sell_call.get('symbol')} @ ${sell_call_strike}"
                    ],
                    'market': f"Oversold (RSI: {market_data['rsi']:.1f})"
                })
                
                return True
                
        except Exception as e:
            logger.error(f"   Jade lizard error: {e}")
            
        return False
        
    def execute_advanced_iron_condor(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute advanced iron condor with optimal strikes"""
        try:
            logger.info("   🦅 Executing ADVANCED IRON CONDOR")
            
            price = market_data['price']
            volatility = market_data['volatility']
            
            # Calculate expected range
            expected_move = price * volatility * np.sqrt(30/365)
            
            # Find strikes outside expected move
            put_short_target = price - expected_move
            put_long_target = put_short_target - 10
            call_short_target = price + expected_move
            call_long_target = call_short_target + 10
            
            puts = options['puts']
            calls = options['calls']
            
            # Find closest strikes
            put_short = min(puts, key=lambda x: abs(float(x.get('strike_price', 0)) - put_short_target))
            put_long = min(puts, key=lambda x: abs(float(x.get('strike_price', 0)) - put_long_target))
            call_short = min(calls, key=lambda x: abs(float(x.get('strike_price', 0)) - call_short_target))
            call_long = min(calls, key=lambda x: abs(float(x.get('strike_price', 0)) - call_long_target))
            
            # Verify 4 different strikes
            strikes = [
                float(put_long.get('strike_price')),
                float(put_short.get('strike_price')),
                float(call_short.get('strike_price')),
                float(call_long.get('strike_price'))
            ]
            
            if len(set(strikes)) == 4 and strikes[0] < strikes[1] < strikes[2] < strikes[3]:
                logger.info(f"   Expected move: ±${expected_move:.2f}")
                logger.info(f"   Put side: ${strikes[0]} / ${strikes[1]}")
                logger.info(f"   Call side: ${strikes[2]} / ${strikes[3]}")
                logger.info(f"   Profit zone: ${strikes[1]:.0f} - ${strikes[2]:.0f}")
                
                # Execute all 4 legs
                logger.info("\n   🚀 EXECUTING 4-LEG CONDOR:")
                
                legs = [
                    (put_long, OrderSide.BUY, "BUY PUT wing"),
                    (call_long, OrderSide.BUY, "BUY CALL wing"),
                    (put_short, OrderSide.SELL, "SELL PUT body"),
                    (call_short, OrderSide.SELL, "SELL CALL body")
                ]
                
                details = []
                for option, side, desc in legs:
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=option.get('symbol'),
                        qty=1,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ {desc}: {option.get('symbol')}")
                    
                    action = "BUY" if side == OrderSide.BUY else "SELL"
                    details.append(f"{action} {option.get('symbol')} @ ${float(option.get('strike_price'))}")
                    
                self.executed_strategies.append({
                    'strategy': 'Advanced Iron Condor',
                    'symbol': symbol,
                    'legs': 4,
                    'details': details,
                    'market': f"{market_data['trend']} (vol: {volatility:.1%})"
                })
                
                return True
                
        except Exception as e:
            logger.error(f"   Iron condor error: {e}")
            
        return False
        
    def execute_short_iron_butterfly(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute short iron butterfly for low volatility"""
        try:
            logger.info("   🦋 Executing SHORT IRON BUTTERFLY")
            
            atm_strike = options['atm_strike']
            puts = options['puts']
            calls = options['calls']
            
            # Find ATM and wings
            atm_put = next((p for p in puts if float(p.get('strike_price')) == atm_strike), None)
            atm_call = next((c for c in calls if float(c.get('strike_price')) == atm_strike), None)
            
            wing_width = 10
            lower_put = next((p for p in puts if float(p.get('strike_price')) == atm_strike - wing_width), None)
            upper_call = next((c for c in calls if float(c.get('strike_price')) == atm_strike + wing_width), None)
            
            if all([atm_put, atm_call, lower_put, upper_call]):
                logger.info(f"   ATM: SELL both ${atm_strike} put/call")
                logger.info(f"   Wings: BUY ${atm_strike - wing_width} put / ${atm_strike + wing_width} call")
                logger.info("   Max profit at pin to ATM strike")
                
                # Execute
                logger.info("\n   🚀 EXECUTING IRON BUTTERFLY:")
                
                # Buy wings first
                for option, desc in [(lower_put, "lower put"), (upper_call, "upper call")]:
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=option.get('symbol'),
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ BOUGHT {desc}: {option.get('symbol')}")
                    
                # Sell ATM straddle
                for option, desc in [(atm_put, "ATM put"), (atm_call, "ATM call")]:
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=option.get('symbol'),
                        qty=1,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ SOLD {desc}: {option.get('symbol')}")
                    
                self.executed_strategies.append({
                    'strategy': 'Short Iron Butterfly',
                    'symbol': symbol,
                    'legs': 4,
                    'details': [
                        f"BUY {lower_put.get('symbol')} @ ${atm_strike - wing_width}",
                        f"SELL {atm_put.get('symbol')} @ ${atm_strike}",
                        f"SELL {atm_call.get('symbol')} @ ${atm_strike}",
                        f"BUY {upper_call.get('symbol')} @ ${atm_strike + wing_width}"
                    ],
                    'market': f"Low volatility ({market_data['volatility']:.1%})"
                })
                
                return True
                
        except Exception as e:
            logger.error(f"   Iron butterfly error: {e}")
            
        return False
        
    def execute_standard_multileg(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute standard multi-leg as fallback"""
        try:
            trend = market_data['trend']
            
            if 'bullish' in trend:
                # Bull put spread
                otm_puts = options.get('otm_puts', [])
                if len(otm_puts) >= 2:
                    logger.info("   📈 Executing standard BULL PUT SPREAD")
                    
                    sell_put = otm_puts[-1]
                    buy_put = otm_puts[0]
                    
                    # Buy protective first
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=buy_put.get('symbol'),
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ BOUGHT {buy_put.get('symbol')}")
                    
                    self.trading_client.submit_order(MarketOrderRequest(
                        symbol=sell_put.get('symbol'),
                        qty=1,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    ))
                    logger.info(f"   ✅ SOLD {sell_put.get('symbol')}")
                    
                    self.executed_strategies.append({
                        'strategy': 'Bull Put Spread',
                        'symbol': symbol,
                        'legs': 2,
                        'details': [
                            f"SELL {sell_put.get('symbol')}",
                            f"BUY {buy_put.get('symbol')}"
                        ],
                        'market': trend
                    })
                    
                    return True
                    
        except Exception as e:
            logger.error(f"   Standard strategy error: {e}")
            
        return False
        
    def execute_put_ratio_spread(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute put ratio spread for bearish markets"""
        # Implementation similar to call ratio but with puts
        return False
        
    def execute_call_diagonal(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute call diagonal spread"""
        # Would need different expirations
        return False
        
    def execute_double_diagonal(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute double diagonal for high volatility neutral"""
        # Would need different expirations
        return False
        
    def execute_twisted_sister(self, symbol: str, market_data: Dict, options: Dict) -> bool:
        """Execute twisted sister (reverse jade lizard) for overbought"""
        # Implementation similar to jade lizard but bearish
        return False
        
    def show_results(self):
        """Show comprehensive results"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 ADVANCED MULTI-LEG TRADING SUMMARY")
        logger.info("=" * 70)
        
        if self.executed_strategies:
            logger.info(f"\n✅ Executed {len(self.executed_strategies)} advanced strategies:\n")
            
            total_legs = 0
            strategy_types = defaultdict(int)
            
            for i, strategy in enumerate(self.executed_strategies, 1):
                logger.info(f"{i}. {strategy['strategy']} on {strategy['symbol']}")
                logger.info(f"   Market condition: {strategy['market']}")
                logger.info(f"   Legs: {strategy['legs']}")
                for detail in strategy['details']:
                    logger.info(f"   • {detail}")
                logger.info("")
                
                total_legs += strategy['legs']
                strategy_types[strategy['strategy']] += 1
                
            logger.info(f"📈 Strategy Distribution:")
            for stype, count in strategy_types.items():
                logger.info(f"   • {stype}: {count}")
                
            logger.info(f"\n🎯 Total option contracts: {total_legs}")
            logger.info("📱 Monitor positions in your Alpaca account")
            
            # Get current P&L
            try:
                positions = self.trading_client.get_all_positions()
                option_positions = [p for p in positions if len(p.symbol) > 10]
                
                if option_positions:
                    total_pnl = sum(float(p.unrealized_pl) for p in option_positions)
                    logger.info(f"\n💰 Current Options P&L: ${total_pnl:+,.2f}")
                    
            except Exception as e:
                logger.error(f"P&L error: {e}")
                
        else:
            logger.info("\n❌ No strategies executed")

def main():
    """Main entry point"""
    trader = AdvancedMultiLegTrader()
    trader.run()

if __name__ == "__main__":
    main()