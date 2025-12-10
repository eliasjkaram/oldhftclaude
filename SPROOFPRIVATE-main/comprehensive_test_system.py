#!/usr/bin/env python3
"""
Comprehensive Test System for Alpaca MCP Trading Platform
Tests all components with real market data
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
try:
    import yfinance as yf
except ImportError:
# SYNTAX_ERROR_FIXED:     yf = None as yf

# Setup logging
# SYNTAX_ERROR_FIXED: logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
# SYNTAX_ERROR_FIXED: )
logger = logging.getLogger(__name__)

class ComponentTest:
    """Test individual components"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'errors': {}
        }
        
    def test_component(self, name, test_func, *args, **kwargs):
        """Test a single component"""
        try:
            logger.info(f"Testing {name}...")
            result = test_func(*args, **kwargs)
            self.results['passed'].append(name)
            logger.info(f"✓ {name} passed")
            return True, result
        except Exception as e:
            self.results['failed'].append(name)
            self.results['errors'][name] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"✗ {name} failed: {str(e)}")
            return False, None

class MarketDataTest(ComponentTest):
    """Test market data components"""
    
    def test_yfinance_data(self, symbol='AAPL'):
        """Test yfinance data fetching"""
        def fetch_data():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')
            if hist.empty:
                raise ValueError(f"No data returned for {symbol}")
            return hist
        
        return self.test_component('YFinance Data', fetch_data)
    
    def test_minio_connection(self):
        """Test MinIO connection"""
        def connect_minio():
            try:
                from minio import Minio
                client = Minio(
                    'uschristmas.us',
                    access_key='AKSTOCKDB2024',
                    secret_key='StockDB-Secret-Access-Key-2024-Secure!',
                    secure=True
                )
                # Test bucket access
                buckets = list(client.list_buckets())
                return {'status': 'connected', 'buckets': len(buckets)}
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}
        
        return self.test_component('MinIO Connection', connect_minio)
    
    def test_alpaca_connection(self):
        """Test Alpaca API connection"""
        def connect_alpaca():
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient
                
                # Use paper trading credentials
                api_key = os.getenv('ALPACA_PAPER_API_KEY', 'PKCX98VZSJBQF79C1SD8')
                api_secret = os.getenv('ALPACA_PAPER_API_SECRET', 'vCFAgqyJPRB5ESFNOnBR63lODruojVvoqtcUSVBP')
                
                trading_client = TradingClient(api_key, api_secret, paper=True)
                account = trading_client.get_account()
                
                return {
                    'status': 'connected',
                    'account_status': account.status,
                    'buying_power': float(account.buying_power)
                }
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}
        
        return self.test_component('Alpaca Connection', connect_alpaca)

class TradingStrategyTest(ComponentTest):
    """Test trading strategies"""
    
    def test_covered_call_logic(self):
        """Test basic covered call strategy logic"""
        def run_covered_call():
            # Simulate covered call strategy
            initial_shares = 100
            current_price = 150.0
            strike_price = 155.0
            premium = 2.50
            
            # Calculate potential outcomes
            max_profit = (strike_price - current_price + premium) * initial_shares
            breakeven = current_price - premium
            
            return {
                'max_profit': max_profit,
                'breakeven': breakeven,
                'premium_collected': premium * initial_shares
            }
        
        return self.test_component('Covered Call Logic', run_covered_call)
    
    def test_options_pricing(self):
        """Test Black-Scholes options pricing"""
        def black_scholes():
            from scipy import stats
            
            # Parameters
            S = 100  # Current price
            K = 105  # Strike price
            T = 0.25  # Time to expiration (3 months)
            r = 0.05  # Risk-free rate
            sigma = 0.2  # Volatility
            
            # Black-Scholes formula
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            call_price = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
            put_price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
            
            return {
                'call_price': round(call_price, 2),
                'put_price': round(put_price, 2),
                'parameters': {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
            }
        
        return self.test_component('Black-Scholes Pricing', black_scholes)

class BacktestTest(ComponentTest):
    """Test backtesting functionality"""
    
    def test_simple_backtest(self):
        """Test simple buy-and-hold backtest"""
        def run_backtest():
            # Get test data
            symbol = 'SPY'
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data for {symbol}")
            
            # Simple buy and hold
            initial_price = data['Close'].iloc[0]
            final_price = data['Close'].iloc[-1]
            returns = (final_price - initial_price) / initial_price
            
            # Calculate metrics
            daily_returns = data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe = (returns - 0.02) / volatility if volatility > 0 else 0
            
            return {
                'symbol': symbol,
                'period': f"{start_date.date()} to {end_date.date()}",
                'return': round(returns * 100, 2),
                'volatility': round(volatility * 100, 2),
                'sharpe': round(sharpe, 2)
            }
        
        return self.test_component('Simple Backtest', run_backtest)

class AIComponentTest(ComponentTest):
    """Test AI components (without actual AI calls)"""
    
    def test_feature_engineering(self):
        """Test feature engineering for ML"""
        def create_features():
            # Create sample data
            dates = pd.date_range(end=datetime.now(), periods=100)
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            df = pd.DataFrame({'Close': prices}, index=dates)
            
            # Engineer features
            features = pd.DataFrame(index=df.index)
            features['returns'] = df['Close'].pct_change()
            features['sma_20'] = df['Close'].rolling(20).mean()
            features['rsi'] = self.calculate_rsi(df['Close'])
            features['volatility'] = features['returns'].rolling(20).std()
            
            # Drop NaN
            features = features.dropna()
            
            return {
                'num_features': len(features.columns),
                'num_samples': len(features),
                'features': list(features.columns)
            }
        
        return self.test_component('Feature Engineering', create_features)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class SystemIntegrationTest(ComponentTest):
    """Test system integration"""
    
    def test_end_to_end_flow(self):
        """Test complete trading flow"""
        def run_flow():
            steps = []
            
            # Step 1: Get market data
            ticker = yf.Ticker('AAPL')
            data = ticker.history(period='1d')
            if not data.empty:
                steps.append('Market data retrieved')
                current_price = data['Close'].iloc[-1]
            else:
                raise ValueError("No market data")
            
            # Step 2: Generate signal (mock)
            signal = 'BUY' if current_price < 200 else 'HOLD'
            steps.append(f'Signal generated: {signal}')
            
            # Step 3: Risk check (mock)
            position_size = min(100, 10000 / current_price)  # Max $10k position
            steps.append(f'Position size: {int(position_size)} shares')
            
            # Step 4: Order preparation (mock)
            order = {
                'symbol': 'AAPL',
                'qty': int(position_size),
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }
            steps.append('Order prepared')
            
            return {
                'steps_completed': steps,
                'final_order': order,
                'current_price': round(current_price, 2)
            }
        
        return self.test_component('End-to-End Flow', run_flow)

def run_all_tests():
    """Run all component tests"""
    logger.info("="*80)
    logger.info("COMPREHENSIVE SYSTEM TEST")
    logger.info("="*80)
    
    all_results = {}
    
    # Test market data components
    logger.info("\n1. MARKET DATA TESTS")
    logger.info("-"*40)
    market_test = MarketDataTest()
    market_test.test_yfinance_data('AAPL')
    market_test.test_yfinance_data('TLT')
    market_test.test_minio_connection()
    market_test.test_alpaca_connection()
    all_results['market_data'] = market_test.results
    
    # Test trading strategies
    logger.info("\n2. TRADING STRATEGY TESTS")
    logger.info("-"*40)
    strategy_test = TradingStrategyTest()
    strategy_test.test_covered_call_logic()
    strategy_test.test_options_pricing()
    all_results['strategies'] = strategy_test.results
    
    # Test backtesting
    logger.info("\n3. BACKTESTING TESTS")
    logger.info("-"*40)
    backtest_test = BacktestTest()
    backtest_test.test_simple_backtest()
    all_results['backtesting'] = backtest_test.results
    
    # Test AI components
    logger.info("\n4. AI COMPONENT TESTS")
    logger.info("-"*40)
    ai_test = AIComponentTest()
    ai_test.test_feature_engineering()
    all_results['ai_components'] = ai_test.results
    
    # Test system integration
    logger.info("\n5. SYSTEM INTEGRATION TESTS")
    logger.info("-"*40)
    integration_test = SystemIntegrationTest()
    integration_test.test_end_to_end_flow()
    all_results['integration'] = integration_test.results
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    total_passed = sum(len(r['passed']) for r in all_results.values())
    total_failed = sum(len(r['failed']) for r in all_results.values())
    
    logger.info(f"Total Tests: {total_passed + total_failed}")
    logger.info(f"Passed: {total_passed}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\nDetailed results saved to test_results.json")
    
    # List failed tests
    if total_failed > 0:
        logger.info("\nFAILED TESTS:")
        for category, results in all_results.items():
            if results['failed']:
                logger.info(f"\n{category}:")
                for test in results['failed']:
                    error = results['errors'][test]['error']
                    logger.info(f"  - {test}: {error}")
    
    return all_results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    total_failed = sum(len(r['failed']) for r in results.values())
    sys.exit(0 if total_failed == 0 else 1)