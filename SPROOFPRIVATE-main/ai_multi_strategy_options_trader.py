#!/usr/bin/env python3
"""
AI Multi-Strategy Options Trader
Uses predictions to execute various multi-leg option strategies
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

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AIMultiStrategyOptionsTrader:
    def __init__(self):
        """Initialize the AI options trader"""
        # API credentials
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
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMD']
        
        # Strategy parameters
        self.max_trades_per_run = 5
        self.position_size = 1  # contracts
        
        # Track trades
        self.executed_trades = []
        
    def run(self):
        """Main execution"""
        logger.info("🤖 AI MULTI-STRATEGY OPTIONS TRADER")
        logger.info("Using predictions to select and execute multi-leg strategies")
        logger.info("="*70)
        
        # Check account
        if not self.check_account():
            return
            
        # Process each symbol
        trades_count = 0
        
        for symbol in self.symbols:
            if trades_count >= self.max_trades_per_run:
                break
                
            try:
                logger.info(f"\n🔍 Analyzing {symbol}...")
                
                # Generate AI prediction
                prediction = self.generate_ai_prediction(symbol)
                if not prediction:
                    continue
                    
                # Display prediction
                self.display_prediction(prediction)
                
                # Get option contracts
                options = self.get_option_contracts(symbol)
                if not options:
                    continue
                    
                # Select and execute strategy based on prediction
                if self.execute_ai_strategy(prediction, options):
                    trades_count += 1
                    time.sleep(2)  # Brief pause between trades
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                
        # Show results
        self.show_results()
        
    def check_account(self) -> bool:
        """Check account status"""
        try:
            account = self.trading_client.get_account()
            logger.info(f"\n💼 Account Status:")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"   Options Level: {account.options_trading_level}")
            
            return account.options_trading_level >= 2
            
        except Exception as e:
            logger.error(f"Account error: {e}")
            return False
            
    def generate_ai_prediction(self, symbol: str) -> Dict:
        """Generate AI prediction for stock"""
        try:
            # Get current price and historical data
            quote = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            
            if symbol not in quote:
                return None
                
            current_price = float(quote[symbol].ask_price)
            
            # Get historical bars
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=30
                )
            )
            
            if symbol not in bars.data:
                return None
                
            df = bars.df.loc[symbol]
            
            # Calculate technical indicators
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
            
            # Momentum
            momentum_5d = (current_price / df['close'].iloc[-5] - 1) * 100
            momentum_20d = (current_price / df['close'].iloc[-20] - 1) * 100
            
            # Volume trend
            vol_avg = df['volume'].mean()
            vol_recent = df['volume'].iloc[-5:].mean()
            volume_trend = vol_recent / vol_avg
            
            # Generate prediction
            bull_score = 0
            bear_score = 0
            
            # Price vs SMA
            if current_price > sma_20 * 1.01:
                bull_score += 3
            elif current_price < sma_20 * 0.99:
                bear_score += 3
                
            # Short-term trend
            if sma_5 > sma_20:
                bull_score += 2
            else:
                bear_score += 2
                
            # RSI
            if rsi < 30:
                bull_score += 3  # Oversold
            elif rsi > 70:
                bear_score += 3  # Overbought
            elif rsi < 45:
                bull_score += 1
            elif rsi > 55:
                bear_score += 1
                
            # Momentum
            if momentum_5d > 3:
                bull_score += 2
            elif momentum_5d < -3:
                bear_score += 2
                
            if momentum_20d > 5:
                bull_score += 1
            elif momentum_20d < -5:
                bear_score += 1
                
            # Volume
            if volume_trend > 1.2:
                if momentum_5d > 0:
                    bull_score += 1
                else:
                    bear_score += 1
                    
            # Determine prediction
            net_score = bull_score - bear_score
            
            if net_score >= 4:
                direction = 'strong_bullish'
                confidence = min(0.85, 0.65 + net_score * 0.03)
                expected_return = 0.04 + (net_score - 4) * 0.01
            elif net_score >= 2:
                direction = 'bullish'
                confidence = 0.70 + net_score * 0.02
                expected_return = 0.02 + net_score * 0.005
            elif net_score <= -4:
                direction = 'strong_bearish'
                confidence = min(0.85, 0.65 + abs(net_score) * 0.03)
                expected_return = -0.04 - (abs(net_score) - 4) * 0.01
            elif net_score <= -2:
                direction = 'bearish'
                confidence = 0.70 + abs(net_score) * 0.02
                expected_return = -0.02 - abs(net_score) * 0.005
            else:
                direction = 'neutral'
                confidence = 0.60
                expected_return = net_score * 0.002
                
            return {
                'symbol': symbol,
                'current_price': current_price,
                'direction': direction,
                'confidence': confidence,
                'expected_return': expected_return,
                'volatility': volatility,
                'target_price': current_price * (1 + expected_return),
                'indicators': {
                    'sma_20': sma_20,
                    'rsi': rsi,
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    'volume_trend': volume_trend
                },
                'score': net_score
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None
            
    def display_prediction(self, prediction: Dict):
        """Display AI prediction"""
        logger.info(f"   📊 AI Prediction:")
        logger.info(f"      Direction: {prediction['direction'].upper()}")
        logger.info(f"      Confidence: {prediction['confidence']:.1%}")
        logger.info(f"      Expected Return: {prediction['expected_return']:+.2%}")
        logger.info(f"      Volatility: {prediction['volatility']:.1%}")
        logger.info(f"      Score: {prediction['score']:+d}")
        
    def get_option_contracts(self, symbol: str) -> Dict:
        """Get available option contracts"""
        try:
            # Target 25-40 days out
            today = datetime.now().date()
            exp_min = (today + timedelta(days=20)).strftime('%Y-%m-%d')
            exp_max = (today + timedelta(days=45)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'expiration_date_gte': exp_min,
                'expiration_date_lte': exp_max,
                'limit': 100
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])
                
                if contracts:
                    puts = sorted([c for c in contracts if c.get('type') == 'put'], 
                                 key=lambda x: float(x.get('strike_price', 0)))
                    calls = sorted([c for c in contracts if c.get('type') == 'call'], 
                                  key=lambda x: float(x.get('strike_price', 0)))
                    
                    logger.info(f"   📋 Found {len(puts)} puts and {len(calls)} calls")
                    
                    return {
                        'symbol': symbol,
                        'puts': puts,
                        'calls': calls,
                        'all': contracts
                    }
                    
        except Exception as e:
            logger.error(f"Options error: {e}")
            
        return None
        
    def execute_ai_strategy(self, prediction: Dict, options: Dict) -> bool:
        """Execute strategy based on AI prediction"""
        try:
            direction = prediction['direction']
            confidence = prediction['confidence']
            volatility = prediction['volatility']
            price = prediction['current_price']
            
            logger.info(f"\n   🎲 Selecting strategy based on prediction...")
            
            # STRONG BULLISH - Aggressive strategies
            if direction == 'strong_bullish' and confidence > 0.75:
                # Vertical Call Spread (Debit)
                if self.execute_vertical_call_spread(prediction, options):
                    return True
                # Bull Call Spread
                if self.execute_bull_call_spread(prediction, options):
                    return True
                    
            # BULLISH - Moderate strategies
            elif direction == 'bullish':
                # Bull Put Spread (Credit)
                if self.execute_bull_put_spread(prediction, options):
                    return True
                # Call Calendar Spread
                if self.execute_calendar_spread(prediction, options, 'call'):
                    return True
                    
            # STRONG BEARISH - Aggressive strategies
            elif direction == 'strong_bearish' and confidence > 0.75:
                # Vertical Put Spread (Debit)
                if self.execute_vertical_put_spread(prediction, options):
                    return True
                # Bear Put Spread
                if self.execute_bear_put_spread(prediction, options):
                    return True
                    
            # BEARISH - Moderate strategies
            elif direction == 'bearish':
                # Bear Call Spread (Credit)
                if self.execute_bear_call_spread(prediction, options):
                    return True
                # Put Calendar Spread
                if self.execute_calendar_spread(prediction, options, 'put'):
                    return True
                    
            # NEUTRAL with HIGH VOLATILITY
            elif direction == 'neutral' and volatility > 0.30:
                # Long Straddle or Strangle
                if self.execute_straddle(prediction, options):
                    return True
                if self.execute_strangle(prediction, options):
                    return True
                    
            # NEUTRAL with LOW VOLATILITY
            elif direction == 'neutral':
                # Iron Condor
                if self.execute_iron_condor(prediction, options):
                    return True
                # Butterfly Spread
                if self.execute_butterfly(prediction, options):
                    return True
                    
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            
        return False
        
    def execute_vertical_call_spread(self, prediction: Dict, options: Dict) -> bool:
        """Execute vertical call spread (bull call spread)"""
        try:
            logger.info("   📈 Executing Vertical Call Spread (Bullish Debit)")
            
            price = prediction['current_price']
            target = prediction['target_price']
            calls = options['calls']
            
            # Find suitable strikes
            buy_strike = price * 0.99  # Slightly ITM
            sell_strike = target  # At target
            
            buy_call = min(calls, key=lambda x: abs(float(x.get('strike_price', 0)) - buy_strike))
            sell_call = min(calls, key=lambda x: abs(float(x.get('strike_price', 0)) - sell_strike))
            
            if buy_call == sell_call:
                return False
                
            buy_strike_actual = float(buy_call.get('strike_price'))
            sell_strike_actual = float(sell_call.get('strike_price'))
            
            if sell_strike_actual <= buy_strike_actual:
                return False
                
            logger.info(f"      Buy: {buy_call.get('symbol')} @ ${buy_strike_actual}")
            logger.info(f"      Sell: {sell_call.get('symbol')} @ ${sell_strike_actual}")
            logger.info(f"      Max Profit: ${sell_strike_actual - buy_strike_actual:.0f}")
            
            # Execute
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=buy_call.get('symbol'),
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {buy_call.get('symbol')}")
            
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=sell_call.get('symbol'),
                qty=self.position_size,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ SOLD {sell_call.get('symbol')}")
            
            self.executed_trades.append({
                'strategy': 'Vertical Call Spread',
                'symbol': prediction['symbol'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
            
    def execute_bull_put_spread(self, prediction: Dict, options: Dict) -> bool:
        """Execute bull put spread (credit spread)"""
        try:
            logger.info("   📈 Executing Bull Put Spread (Credit)")
            
            price = prediction['current_price']
            puts = options['puts']
            
            # Find OTM puts
            otm_puts = [p for p in puts if float(p.get('strike_price', 0)) < price * 0.95]
            
            if len(otm_puts) < 2:
                return False
                
            # Select strikes with $5 width
            sell_put = otm_puts[-1]
            sell_strike = float(sell_put.get('strike_price'))
            
            buy_put = None
            for put in otm_puts:
                if float(put.get('strike_price')) <= sell_strike - 5:
                    buy_put = put
                    
            if not buy_put:
                buy_put = otm_puts[0]
                
            buy_strike = float(buy_put.get('strike_price'))
            
            logger.info(f"      Sell: {sell_put.get('symbol')} @ ${sell_strike}")
            logger.info(f"      Buy: {buy_put.get('symbol')} @ ${buy_strike}")
            logger.info(f"      Max Risk: ${sell_strike - buy_strike:.0f}")
            
            # Execute (buy protective first)
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=buy_put.get('symbol'),
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {buy_put.get('symbol')} (protective)")
            
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=sell_put.get('symbol'),
                qty=self.position_size,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ SOLD {sell_put.get('symbol')} (credit)")
            
            self.executed_trades.append({
                'strategy': 'Bull Put Spread',
                'symbol': prediction['symbol'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
            
    def execute_iron_condor(self, prediction: Dict, options: Dict) -> bool:
        """Execute iron condor for neutral outlook"""
        try:
            logger.info("   🦅 Executing Iron Condor (Neutral)")
            
            price = prediction['current_price']
            puts = options['puts']
            calls = options['calls']
            
            # Find strikes ~10% OTM
            put_short = next((p for p in reversed(puts) if float(p.get('strike_price')) < price * 0.90), None)
            call_short = next((c for c in calls if float(c.get('strike_price')) > price * 1.10), None)
            
            if not put_short or not call_short:
                return False
                
            # Find protective strikes
            put_long = next((p for p in puts if float(p.get('strike_price')) < float(put_short.get('strike_price')) - 5), None)
            call_long = next((c for c in calls if float(c.get('strike_price')) > float(call_short.get('strike_price')) + 5), None)
            
            if not put_long or not call_long:
                return False
                
            logger.info(f"      Put Spread: {put_long.get('symbol')} / {put_short.get('symbol')}")
            logger.info(f"      Call Spread: {call_short.get('symbol')} / {call_long.get('symbol')}")
            
            # Execute (buy protective first)
            for option, side in [(put_long, OrderSide.BUY), (call_long, OrderSide.BUY),
                                (put_short, OrderSide.SELL), (call_short, OrderSide.SELL)]:
                self.trading_client.submit_order(MarketOrderRequest(
                    symbol=option.get('symbol'),
                    qty=self.position_size,
                    side=side,
                    time_in_force=TimeInForce.DAY
                ))
                action = "BOUGHT" if side == OrderSide.BUY else "SOLD"
                logger.info(f"      ✅ {action} {option.get('symbol')}")
                
            self.executed_trades.append({
                'strategy': 'Iron Condor',
                'symbol': prediction['symbol'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
            
    def execute_butterfly(self, prediction: Dict, options: Dict) -> bool:
        """Execute butterfly spread"""
        try:
            logger.info("   🦋 Executing Butterfly Spread")
            
            price = prediction['current_price']
            calls = options['calls']
            
            # Find ATM and wings
            atm_strike = round(price / 5) * 5
            
            lower = next((c for c in calls if float(c.get('strike_price')) == atm_strike - 5), None)
            middle = next((c for c in calls if float(c.get('strike_price')) == atm_strike), None)
            upper = next((c for c in calls if float(c.get('strike_price')) == atm_strike + 5), None)
            
            if not (lower and middle and upper):
                return False
                
            logger.info(f"      Lower: {lower.get('symbol')} @ ${float(lower.get('strike_price'))}")
            logger.info(f"      Middle: {middle.get('symbol')} @ ${float(middle.get('strike_price'))}")
            logger.info(f"      Upper: {upper.get('symbol')} @ ${float(upper.get('strike_price'))}")
            
            # Execute 1-2-1
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=lower.get('symbol'),
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {lower.get('symbol')}")
            
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=upper.get('symbol'),
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {upper.get('symbol')}")
            
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=middle.get('symbol'),
                qty=2,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ SOLD 2x {middle.get('symbol')}")
            
            self.executed_trades.append({
                'strategy': 'Butterfly Spread',
                'symbol': prediction['symbol'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
            
    def execute_straddle(self, prediction: Dict, options: Dict) -> bool:
        """Execute long straddle for high volatility"""
        try:
            logger.info("   🌊 Executing Long Straddle (High Volatility)")
            
            price = prediction['current_price']
            puts = options['puts']
            calls = options['calls']
            
            # Find ATM options
            atm_strike = round(price / 5) * 5
            
            atm_put = next((p for p in puts if float(p.get('strike_price')) == atm_strike), None)
            atm_call = next((c for c in calls if float(c.get('strike_price')) == atm_strike), None)
            
            if not (atm_put and atm_call):
                return False
                
            logger.info(f"      Buy Put: {atm_put.get('symbol')} @ ${atm_strike}")
            logger.info(f"      Buy Call: {atm_call.get('symbol')} @ ${atm_strike}")
            logger.info(f"      Need {prediction['volatility']/2:.1%} move to profit")
            
            # Execute
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=atm_put.get('symbol'),
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {atm_put.get('symbol')}")
            
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=atm_call.get('symbol'),
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            logger.info(f"      ✅ BOUGHT {atm_call.get('symbol')}")
            
            self.executed_trades.append({
                'strategy': 'Long Straddle',
                'symbol': prediction['symbol'],
                'direction': 'volatility',
                'confidence': prediction['confidence']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
            
    # Add more strategy methods as needed...
    
    def show_results(self):
        """Show trading results"""
        logger.info("\n" + "="*70)
        logger.info("📊 AI TRADING RESULTS")
        logger.info("="*70)
        
        if self.executed_trades:
            logger.info(f"\n✅ Executed {len(self.executed_trades)} AI-driven strategies:\n")
            
            for i, trade in enumerate(self.executed_trades, 1):
                logger.info(f"{i}. {trade['strategy']} on {trade['symbol']}")
                logger.info(f"   Direction: {trade['direction']}")
                logger.info(f"   Confidence: {trade['confidence']:.1%}\n")
                
            logger.info("🎯 All strategies are now active!")
            logger.info("📱 Check your Alpaca account for positions")
            
            # Show positions
            try:
                positions = self.trading_client.get_all_positions()
                option_positions = [p for p in positions if len(p.symbol) > 10]
                
                if option_positions:
                    logger.info(f"\n📋 Active Option Positions: {len(option_positions)}")
                    
                    total_pnl = 0
                    for pos in option_positions[:20]:
                        pnl = float(pos.unrealized_pl)
                        total_pnl += pnl
                        logger.info(f"   {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} "
                                  f"P&L: ${pnl:+.2f}")
                        
                    logger.info(f"\n💰 Total Options P&L: ${total_pnl:+.2f}")
                    
            except Exception as e:
                logger.error(f"Error showing positions: {e}")
                
        else:
            logger.info("\n❌ No trades executed")

def main():
    """Main entry point"""
    trader = AIMultiStrategyOptionsTrader()
    trader.run()

if __name__ == "__main__":
    main()