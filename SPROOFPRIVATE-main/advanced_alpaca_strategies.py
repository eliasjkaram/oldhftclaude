#!/usr/bin/env python3
"""
Advanced Alpaca Trading Strategies
==================================
Multiple algorithms for paper trading with real data
Includes options-like strategies, pairs trading, and arbitrage
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from scipy import stats
from dataclasses import dataclass

# Set Alpaca credentials
os.environ['ALPACA_API_KEY'] = 'PKCX98VZSJBQF79C1SD8'
os.environ['ALPACA_SECRET_KEY'] = 'KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

@dataclass
class Signal:
    symbol: str
    strategy: str
    action: str
    confidence: float
    expected_return: float
    risk_score: float
    metadata: Dict

class AdvancedAlpacaStrategies:
    """Advanced trading strategies using Alpaca API"""
    
    def __init__(self):
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY'],
            paper=True
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=os.environ['ALPACA_API_KEY'],
            secret_key=os.environ['ALPACA_SECRET_KEY']
        )
        
        # Trading universe
        self.stocks = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD']
        self.etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'VXX']
        self.pairs = [('GOOGL', 'GOOG'), ('XOM', 'CVX'), ('GM', 'F')]
        
        logger.info("✅ Advanced Alpaca Strategies initialized")
    
    async def get_market_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Get market data from Alpaca"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            if symbol in bars.data:
                df = bars.data[symbol]
                data = pd.DataFrame({
                    'Open': [bar.open for bar in df],
                    'High': [bar.high for bar in df],
                    'Low': [bar.low for bar in df],
                    'Close': [bar.close for bar in df],
                    'Volume': [bar.volume for bar in df]
                }, index=[bar.timestamp for bar in df])
                
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    # ========== STRATEGY 1: ML-inspired Predictions ==========
    async def ml_inspired_strategy(self) -> List[Signal]:
        """ML-inspired strategy using technical features"""
        signals = []
        
        for symbol in self.stocks:
            try:
                data = await self.get_market_data(symbol)
                if len(data) < 50:
                    continue
                
                # Calculate features
                features = self._calculate_features(data)
                
                # Simple scoring system (simulating ML prediction)
                score = 0
                if features['rsi'] < 30:
                    score += 2
                elif features['rsi'] > 70:
                    score -= 2
                
                if features['macd_signal'] > 0:
                    score += 1
                else:
                    score -= 1
                
                if features['bb_position'] < 0.2:
                    score += 1
                elif features['bb_position'] > 0.8:
                    score -= 1
                
                # Generate signal
                if abs(score) >= 2:
                    signal = Signal(
                        symbol=symbol,
                        strategy="ML_Inspired",
                        action="BUY" if score > 0 else "SELL",
                        confidence=min(0.9, abs(score) / 4),
                        expected_return=0.02 * (abs(score) / 4),
                        risk_score=1 - min(0.9, abs(score) / 4),
                        metadata=features
                    )
                    signals.append(signal)
                    logger.info(f"📊 ML Signal: {symbol} {signal.action} (score: {score})")
                    
            except Exception as e:
                logger.error(f"ML strategy error for {symbol}: {e}")
        
        return signals
    
    # ========== STRATEGY 2: Pairs Trading ==========
    async def pairs_trading_strategy(self) -> List[Signal]:
        """Statistical arbitrage pairs trading"""
        signals = []
        
        for pair in self.pairs:
            try:
                data1 = await self.get_market_data(pair[0])
                data2 = await self.get_market_data(pair[1])
                
                if len(data1) < 30 or len(data2) < 30:
                    continue
                
                # Align data
                common_dates = data1.index.intersection(data2.index)
                if len(common_dates) < 20:
                    continue
                
                prices1 = data1.loc[common_dates, 'Close']
                prices2 = data2.loc[common_dates, 'Close']
                
                # Calculate spread
                ratio = prices1 / prices2
                mean_ratio = ratio.rolling(20).mean()
                std_ratio = ratio.rolling(20).std()
                
                if pd.isna(mean_ratio.iloc[-1]) or pd.isna(std_ratio.iloc[-1]):
                    continue
                
                z_score = (ratio.iloc[-1] - mean_ratio.iloc[-1]) / std_ratio.iloc[-1]
                
                if abs(z_score) > 2:
                    signal = Signal(
                        symbol=f"{pair[0]}/{pair[1]}",
                        strategy="Pairs_Trading",
                        action="SELL_SPREAD" if z_score > 2 else "BUY_SPREAD",
                        confidence=min(0.85, abs(z_score) / 3),
                        expected_return=0.015 * min(1, abs(z_score) / 2),
                        risk_score=0.3,
                        metadata={
                            'z_score': z_score,
                            'ratio': ratio.iloc[-1],
                            'mean_ratio': mean_ratio.iloc[-1],
                            'pair': pair
                        }
                    )
                    signals.append(signal)
                    logger.info(f"📊 Pairs: {pair[0]}/{pair[1]} z-score: {z_score:.2f}")
                    
            except Exception as e:
                logger.error(f"Pairs trading error for {pair}: {e}")
        
        return signals
    
    # ========== STRATEGY 3: Volatility Trading ==========
    async def volatility_strategy(self) -> List[Signal]:
        """Trade based on volatility patterns"""
        signals = []
        
        for symbol in self.stocks:
            try:
                data = await self.get_market_data(symbol)
                if len(data) < 30:
                    continue
                
                # Calculate volatility metrics
                returns = data['Close'].pct_change().dropna()
                current_vol = returns.rolling(10).std().iloc[-1] * np.sqrt(252)
                avg_vol = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
                
                # High-Low volatility
                hl_vol = ((data['High'] - data['Low']) / data['Close']).rolling(10).mean().iloc[-1]
                
                # Volatility regime
                if current_vol > avg_vol * 1.5 and hl_vol > 0.03:
                    # High volatility - consider straddle-like position
                    signal = Signal(
                        symbol=symbol,
                        strategy="Volatility_High",
                        action="STRADDLE",
                        confidence=0.7,
                        expected_return=0.025,
                        risk_score=0.4,
                        metadata={
                            'current_vol': current_vol,
                            'avg_vol': avg_vol,
                            'hl_vol': hl_vol
                        }
                    )
                    signals.append(signal)
                    logger.info(f"📊 Vol High: {symbol} vol: {current_vol:.2%}")
                    
                elif current_vol < avg_vol * 0.7:
                    # Low volatility - trend following
                    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                    if data['Close'].iloc[-1] > sma_20:
                        signal = Signal(
                            symbol=symbol,
                            strategy="Volatility_Low",
                            action="BUY",
                            confidence=0.65,
                            expected_return=0.015,
                            risk_score=0.25,
                            metadata={
                                'current_vol': current_vol,
                                'avg_vol': avg_vol,
                                'trend': 'up'
                            }
                        )
                        signals.append(signal)
                        logger.info(f"📊 Vol Low: {symbol} trend up")
                        
            except Exception as e:
                logger.error(f"Volatility strategy error for {symbol}: {e}")
        
        return signals
    
    # ========== STRATEGY 4: Mean Reversion ==========
    async def mean_reversion_strategy(self) -> List[Signal]:
        """Trade oversold/overbought conditions"""
        signals = []
        
        for symbol in self.stocks:
            try:
                data = await self.get_market_data(symbol, days=30)
                if len(data) < 20:
                    continue
                
                # Bollinger Bands
                sma = data['Close'].rolling(20).mean()
                std = data['Close'].rolling(20).std()
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                
                current_price = data['Close'].iloc[-1]
                
                # Calculate position in bands
                band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
                position_in_band = (current_price - lower_band.iloc[-1]) / band_width
                
                # RSI
                rsi = self._calculate_rsi(data['Close'])
                
                if position_in_band < 0.1 and rsi < 35:
                    signal = Signal(
                        symbol=symbol,
                        strategy="Mean_Reversion",
                        action="BUY",
                        confidence=0.75,
                        expected_return=0.02,
                        risk_score=0.3,
                        metadata={
                            'position_in_band': position_in_band,
                            'rsi': rsi,
                            'price': current_price
                        }
                    )
                    signals.append(signal)
                    logger.info(f"📊 Mean Rev: {symbol} oversold")
                    
                elif position_in_band > 0.9 and rsi > 65:
                    signal = Signal(
                        symbol=symbol,
                        strategy="Mean_Reversion",
                        action="SELL",
                        confidence=0.75,
                        expected_return=0.02,
                        risk_score=0.3,
                        metadata={
                            'position_in_band': position_in_band,
                            'rsi': rsi,
                            'price': current_price
                        }
                    )
                    signals.append(signal)
                    logger.info(f"📊 Mean Rev: {symbol} overbought")
                    
            except Exception as e:
                logger.error(f"Mean reversion error for {symbol}: {e}")
        
        return signals
    
    # ========== STRATEGY 5: Sector Rotation ==========
    async def sector_rotation_strategy(self) -> List[Signal]:
        """Rotate between sectors based on momentum"""
        signals = []
        
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials'
        }
        
        momentum_scores = {}
        
        for etf, sector in sector_etfs.items():
            try:
                data = await self.get_market_data(etf, days=60)
                if len(data) < 20:
                    continue
                
                # Calculate momentum
                returns_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                returns_20d = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
                
                momentum_score = returns_5d * 0.7 + returns_20d * 0.3
                momentum_scores[etf] = momentum_score
                
            except Exception as e:
                logger.error(f"Sector rotation error for {etf}: {e}")
        
        if momentum_scores:
            # Find best and worst sectors
            sorted_sectors = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_sectors) >= 2:
                # Long best sector
                best_etf = sorted_sectors[0][0]
                signal = Signal(
                    symbol=best_etf,
                    strategy="Sector_Rotation",
                    action="BUY",
                    confidence=0.7,
                    expected_return=0.025,
                    risk_score=0.35,
                    metadata={
                        'sector': sector_etfs[best_etf],
                        'momentum_score': sorted_sectors[0][1]
                    }
                )
                signals.append(signal)
                logger.info(f"📊 Sector: BUY {best_etf} ({sector_etfs[best_etf]})")
                
                # Short worst sector if momentum is negative
                worst_etf = sorted_sectors[-1][0]
                if sorted_sectors[-1][1] < -0.02:
                    signal = Signal(
                        symbol=worst_etf,
                        strategy="Sector_Rotation",
                        action="SELL",
                        confidence=0.65,
                        expected_return=0.02,
                        risk_score=0.4,
                        metadata={
                            'sector': sector_etfs[worst_etf],
                            'momentum_score': sorted_sectors[-1][1]
                        }
                    )
                    signals.append(signal)
                    logger.info(f"📊 Sector: SELL {worst_etf} ({sector_etfs[worst_etf]})")
        
        return signals
    
    # ========== Helper Methods ==========
    def _calculate_features(self, data: pd.DataFrame) -> Dict:
        """Calculate technical features"""
        close = data['Close']
        
        # RSI
        rsi = self._calculate_rsi(close)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        macd_signal = 1 if macd.iloc[-1] > signal_line.iloc[-1] else -1
        
        # Bollinger Bands position
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        return {
            'rsi': rsi,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'price': close.iloc[-1],
            'volume': data['Volume'].iloc[-1]
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def execute_signals(self, signals: List[Signal]):
        """Execute trading signals"""
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        
        logger.info(f"\n💰 Buying Power: ${buying_power:,.2f}")
        
        # Sort by expected return / risk
        sorted_signals = sorted(
            signals,
            key=lambda s: s.expected_return / (s.risk_score + 0.1),
            reverse=True
        )
        
        executed = 0
        for signal in sorted_signals[:5]:  # Top 5 signals
            try:
                if signal.action in ['BUY', 'SELL']:
                    # Get current price
                    request = StockLatestQuoteRequest(symbol_or_symbols=signal.symbol)
                    quote = self.data_client.get_stock_latest_quote(request)
                    
                    if signal.symbol not in quote:
                        continue
                    
                    current_price = float(quote[signal.symbol].ask_price)
                    
                    # Calculate position size
                    position_size = min(5000, buying_power * 0.05)
                    shares = int(position_size / current_price)
                    
                    if shares < 1:
                        continue
                    
                    logger.info(f"\n📈 Executing: {signal.symbol} {signal.action}")
                    logger.info(f"   Strategy: {signal.strategy}")
                    logger.info(f"   Shares: {shares} @ ${current_price:.2f}")
                    logger.info(f"   Confidence: {signal.confidence:.2%}")
                    
                    order_data = MarketOrderRequest(
                        symbol=signal.symbol,
                        qty=shares,
                        side=OrderSide.BUY if signal.action == 'BUY' else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    order = self.trading_client.submit_order(order_data)
                    logger.info(f"✅ Order placed: {order.id}")
                    
                    executed += 1
                    buying_power -= position_size
                    
                    await asyncio.sleep(0.5)
                
                else:
                    # Log complex strategies
                    logger.info(f"\n📊 {signal.strategy}: {signal.symbol} - {signal.action}")
                    logger.info(f"   Expected Return: {signal.expected_return:.2%}")
                    
            except Exception as e:
                logger.error(f"Execution error for {signal.symbol}: {e}")
        
        logger.info(f"\n📊 Executed {executed} trades")
    
    async def run(self):
        """Run all strategies"""
        logger.info("="*80)
        logger.info("ADVANCED ALPACA TRADING STRATEGIES")
        logger.info("="*80)
        
        # Check account
        account = self.trading_client.get_account()
        logger.info(f"\n💰 Account Status:")
        logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        # Run all strategies
        all_signals = []
        
        logger.info(f"\n🔍 Running Trading Strategies...")
        
        # 1. ML-inspired
        ml_signals = await self.ml_inspired_strategy()
        all_signals.extend(ml_signals)
        
        # 2. Pairs trading
        pairs_signals = await self.pairs_trading_strategy()
        all_signals.extend(pairs_signals)
        
        # 3. Volatility
        vol_signals = await self.volatility_strategy()
        all_signals.extend(vol_signals)
        
        # 4. Mean reversion
        mean_signals = await self.mean_reversion_strategy()
        all_signals.extend(mean_signals)
        
        # 5. Sector rotation
        sector_signals = await self.sector_rotation_strategy()
        all_signals.extend(sector_signals)
        
        logger.info(f"\n📊 Total signals generated: {len(all_signals)}")
        
        # Execute signals
        if all_signals:
            await self.execute_signals(all_signals)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'account_value': float(account.portfolio_value),
            'total_signals': len(all_signals),
            'signals_by_strategy': {},
            'signals': []
        }
        
        for signal in all_signals:
            if signal.strategy not in report['signals_by_strategy']:
                report['signals_by_strategy'][signal.strategy] = 0
            report['signals_by_strategy'][signal.strategy] += 1
            
            report['signals'].append({
                'symbol': signal.symbol,
                'strategy': signal.strategy,
                'action': signal.action,
                'confidence': signal.confidence,
                'expected_return': signal.expected_return,
                'risk_score': signal.risk_score,
                'metadata': signal.metadata
            })
        
        with open('advanced_strategies_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Report saved to: advanced_strategies_report.json")
        logger.info("\n" + "="*80)
        logger.info("ADVANCED STRATEGIES COMPLETE")
        logger.info("="*80)


async def main():
    """Entry point"""
    trader = AdvancedAlpacaStrategies()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())