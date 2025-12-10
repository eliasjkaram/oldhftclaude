#!/usr/bin/env python3
"""
Complete Options Trading Demo with Proper Dict Handling
Demonstrates all major options strategies with correct data structure access
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptionsChainAnalyzer:
    """Analyzes options chains and finds appropriate strikes for strategies"""
    
    def __init__(self, chain_data: Dict):
        self.chain = chain_data
        
    def get_sorted_strikes(self, expiry: str) -> List[float]:
        """Get sorted list of strikes for a given expiry"""
        if expiry not in self.chain:
            return []
        return sorted([float(strike) for strike in self.chain[expiry].keys()])
    
    def find_atm_strike(self, expiry: str, underlying_price: float) -> Optional[float]:
        """Find the at-the-money strike"""
        strikes = self.get_sorted_strikes(expiry)
        if not strikes:
            return None
            
        # Find closest strike to underlying price
        closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        return closest_strike
    
    def find_otm_strikes(self, expiry: str, underlying_price: float, 
                        num_strikes: int = 2, call_side: bool = True) -> List[float]:
        """Find out-of-the-money strikes"""
        strikes = self.get_sorted_strikes(expiry)
        if not strikes:
            return []
            
        if call_side:
            # OTM calls are above underlying price
            otm_strikes = [s for s in strikes if s > underlying_price]
            return sorted(otm_strikes)[:num_strikes]
        else:
            # OTM puts are below underlying price
            otm_strikes = [s for s in strikes if s < underlying_price]
            return sorted(otm_strikes, reverse=True)[:num_strikes]
    
    def get_option_data(self, expiry: str, strike: float, option_type: str) -> Optional[Dict]:
        """Get option data for specific contract"""
        try:
            if expiry in self.chain and str(strike) in self.chain[expiry]:
                return self.chain[expiry][str(strike)].get(option_type.lower())
        except Exception as e:
            logger.error(f"Error getting option data: {e}")
        return None

class OptionsStrategy:
    """Base class for options strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.legs = []
        self.underlying_price = None
        
    def add_leg(self, option_type: str, strike: float, quantity: int, 
                premium: float, action: str = "BUY"):
        """Add a leg to the strategy"""
        self.legs.append({
            "type": option_type,
            "strike": strike,
            "quantity": quantity,
            "premium": premium,
            "action": action,
            "cost": premium * quantity * 100 * (-1 if action == "BUY" else 1)
        })
    
    def calculate_pnl(self, underlying_at_expiry: float) -> float:
        """Calculate P&L at expiration for given underlying price"""
        total_pnl = 0
        
        for leg in self.legs:
            # Calculate intrinsic value at expiry
            if leg["type"] == "CALL":
                intrinsic = max(0, underlying_at_expiry - leg["strike"])
            else:  # PUT
                intrinsic = max(0, leg["strike"] - underlying_at_expiry)
            
            # Calculate P&L for this leg
            if leg["action"] == "BUY":
                leg_pnl = (intrinsic - leg["premium"]) * leg["quantity"] * 100
            else:  # SELL
                leg_pnl = (leg["premium"] - intrinsic) * leg["quantity"] * 100
            
            total_pnl += leg_pnl
        
        return total_pnl
    
    def get_breakeven_points(self) -> List[float]:
        """Calculate breakeven points"""
        # This is a simplified calculation - real implementation would be more complex
        net_premium = sum(leg["cost"] for leg in self.legs)
        strikes = [leg["strike"] for leg in self.legs]
        
        if len(strikes) == 1:
            # Single leg strategy
            if self.legs[0]["type"] == "CALL":
                return [strikes[0] + abs(net_premium) / 100]
            else:
                return [strikes[0] - abs(net_premium) / 100]
        else:
            # Multi-leg strategy - return approximate breakevens
            return sorted(set(strikes))
    
    def display_summary(self):
        """Display strategy summary"""
        print(f"\n{'='*60}")
        print(f"Strategy: {self.name}")
        print(f"{'='*60}")
        
        total_cost = 0
        for i, leg in enumerate(self.legs, 1):
            print(f"Leg {i}: {leg['action']} {leg['quantity']} {leg['type']} @ ${leg['strike']}")
            print(f"  Premium: ${leg['premium']:.2f} per contract")
            print(f"  Cost: ${leg['cost']:.2f}")
            total_cost += leg['cost']
        
        print(f"\nTotal Cost/Credit: ${total_cost:.2f}")
        print(f"Max Profit: ${self.calculate_max_profit():.2f}")
        print(f"Max Loss: ${self.calculate_max_loss():.2f}")
        
        breakevens = self.get_breakeven_points()
        print(f"Breakeven Points: {', '.join([f'${b:.2f}' for b in breakevens])}")
    
    def calculate_max_profit(self) -> float:
        """Calculate maximum profit"""
        # Test various underlying prices to find max profit
        strikes = [leg["strike"] for leg in self.legs]
        test_prices = strikes + [0, max(strikes) * 2]
        
        max_profit = max(self.calculate_pnl(price) for price in test_prices)
        return max_profit
    
    def calculate_max_loss(self) -> float:
        """Calculate maximum loss"""
        # Test various underlying prices to find max loss
        strikes = [leg["strike"] for leg in self.legs]
        test_prices = strikes + [0, max(strikes) * 2]
        
        max_loss = min(self.calculate_pnl(price) for price in test_prices)
        return max_loss

class OptionsTradingDemo:
    """Main demo class for options trading strategies"""
    
    def __init__(self):
        self.analyzer = None
        self.strategies = []
        self.underlying_price = 170.0  # TLT example price
        
    def generate_sample_chain(self) -> Dict:
        """Generate a sample options chain for demonstration"""
        # Create realistic options chain data
        chain = {}
        
        # Generate for 3 expiries
        base_date = datetime.now()
        expiries = [
            (base_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            (base_date + timedelta(days=60)).strftime("%Y-%m-%d"),
            (base_date + timedelta(days=90)).strftime("%Y-%m-%d")
        ]
        
        for expiry in expiries:
            chain[expiry] = {}
            
            # Generate strikes around current price
            for strike_offset in range(-10, 11):
                strike = self.underlying_price + (strike_offset * 2.5)
                if strike > 0:
                    chain[expiry][str(strike)] = {
                        "call": {
                            "bid": max(0.10, (self.underlying_price - strike) * 0.4 + 2.0) if strike < self.underlying_price else max(0.10, 2.0 - (strike - self.underlying_price) * 0.3),
                            "ask": max(0.15, (self.underlying_price - strike) * 0.4 + 2.2) if strike < self.underlying_price else max(0.15, 2.2 - (strike - self.underlying_price) * 0.3),
                            "iv": 0.25 + abs(strike - self.underlying_price) * 0.001,
                            "volume": 100 + int(50 / (1 + abs(strike - self.underlying_price))),
                            "open_interest": 500 + int(200 / (1 + abs(strike - self.underlying_price)))
                        },
                        "put": {
                            "bid": max(0.10, (strike - self.underlying_price) * 0.4 + 2.0) if strike > self.underlying_price else max(0.10, 2.0 - (self.underlying_price - strike) * 0.3),
                            "ask": max(0.15, (strike - self.underlying_price) * 0.4 + 2.2) if strike > self.underlying_price else max(0.15, 2.2 - (self.underlying_price - strike) * 0.3),
                            "iv": 0.25 + abs(strike - self.underlying_price) * 0.001,
                            "volume": 100 + int(50 / (1 + abs(strike - self.underlying_price))),
                            "open_interest": 500 + int(200 / (1 + abs(strike - self.underlying_price)))
                        }
                    }
        
        return chain
    
    def demo_iron_condor(self, expiry: str):
        """Demonstrate Iron Condor strategy"""
        print("\n" + "="*80)
        print("IRON CONDOR STRATEGY")
        print("="*80)
        
        strikes = self.analyzer.get_sorted_strikes(expiry)
        atm_strike = self.analyzer.find_atm_strike(expiry, self.underlying_price)
        
        if not atm_strike or len(strikes) < 4:
            print("Not enough strikes available for Iron Condor")
            return
        
        # Find strikes for Iron Condor
        # Sell OTM put and call, buy further OTM put and call
        put_strikes = [s for s in strikes if s < atm_strike]
        call_strikes = [s for s in strikes if s > atm_strike]
        
        if len(put_strikes) < 2 or len(call_strikes) < 2:
            print("Not enough OTM strikes for Iron Condor")
            return
        
        # Select strikes
        short_put = put_strikes[-1]  # Closest OTM put
        long_put = put_strikes[-2]   # Further OTM put
        short_call = call_strikes[0]  # Closest OTM call
        long_call = call_strikes[1]   # Further OTM call
        
        # Create strategy
        strategy = OptionsStrategy("Iron Condor")
        
        # Add legs with proper data access
        for strike, option_type, action in [
            (short_put, "PUT", "SELL"),
            (long_put, "PUT", "BUY"),
            (short_call, "CALL", "SELL"),
            (long_call, "CALL", "BUY")
        ]:
            option_data = self.analyzer.get_option_data(expiry, strike, option_type)
            if option_data:
                premium = (option_data["bid"] + option_data["ask"]) / 2
                strategy.add_leg(option_type, strike, 1, premium, action)
        
        strategy.display_summary()
        self.strategies.append(strategy)
        
        # Show P&L profile
        self.display_pnl_profile(strategy, strikes)
    
    def demo_bull_put_spread(self, expiry: str):
        """Demonstrate Bull Put Spread strategy"""
        print("\n" + "="*80)
        print("BULL PUT SPREAD STRATEGY")
        print("="*80)
        
        strikes = self.analyzer.get_sorted_strikes(expiry)
        atm_strike = self.analyzer.find_atm_strike(expiry, self.underlying_price)
        
        if not atm_strike:
            print("Could not find ATM strike")
            return
        
        # Find OTM put strikes
        put_strikes = sorted([s for s in strikes if s < atm_strike], reverse=True)
        
        if len(put_strikes) < 2:
            print("Not enough put strikes for spread")
            return
        
        # Select strikes
        short_put = put_strikes[0]  # Higher strike (less OTM)
        long_put = put_strikes[1]   # Lower strike (more OTM)
        
        # Create strategy
        strategy = OptionsStrategy("Bull Put Spread")
        
        # Sell higher strike put
        short_data = self.analyzer.get_option_data(expiry, short_put, "PUT")
        if short_data:
            premium = (short_data["bid"] + short_data["ask"]) / 2
            strategy.add_leg("PUT", short_put, 1, premium, "SELL")
        
        # Buy lower strike put
        long_data = self.analyzer.get_option_data(expiry, long_put, "PUT")
        if long_data:
            premium = (long_data["bid"] + long_data["ask"]) / 2
            strategy.add_leg("PUT", long_put, 1, premium, "BUY")
        
        strategy.display_summary()
        self.strategies.append(strategy)
        
        # Show P&L profile
        self.display_pnl_profile(strategy, [long_put, short_put])
    
    def demo_covered_call(self, expiry: str):
        """Demonstrate Covered Call strategy"""
        print("\n" + "="*80)
        print("COVERED CALL STRATEGY")
        print("="*80)
        
        # Find OTM call strikes
        call_strikes = self.analyzer.find_otm_strikes(expiry, self.underlying_price, 3, call_side=True)
        
        if not call_strikes:
            print("No OTM call strikes available")
            return
        
        # Select first OTM call
        call_strike = call_strikes[0]
        
        # Create strategy
        strategy = OptionsStrategy("Covered Call")
        strategy.underlying_price = self.underlying_price
        
        # Add stock position (simplified - not included in P&L calculation here)
        print(f"Long 100 shares at ${self.underlying_price:.2f}")
        
        # Sell call
        call_data = self.analyzer.get_option_data(expiry, call_strike, "CALL")
        if call_data:
            premium = (call_data["bid"] + call_data["ask"]) / 2
            strategy.add_leg("CALL", call_strike, 1, premium, "SELL")
        
        strategy.display_summary()
        self.strategies.append(strategy)
        
        # Show P&L profile (excluding stock position for simplicity)
        self.display_pnl_profile(strategy, [call_strike])
    
    def demo_butterfly_spread(self, expiry: str):
        """Demonstrate Butterfly Spread strategy"""
        print("\n" + "="*80)
        print("BUTTERFLY SPREAD STRATEGY")
        print("="*80)
        
        strikes = self.analyzer.get_sorted_strikes(expiry)
        atm_strike = self.analyzer.find_atm_strike(expiry, self.underlying_price)
        
        if not atm_strike:
            print("Could not find ATM strike")
            return
        
        # Find strikes for butterfly
        atm_index = strikes.index(atm_strike)
        
        if atm_index == 0 or atm_index == len(strikes) - 1:
            print("ATM strike at edge of chain, cannot create butterfly")
            return
        
        # Select strikes
        lower_strike = strikes[max(0, atm_index - 1)]
        middle_strike = atm_strike
        upper_strike = strikes[min(len(strikes) - 1, atm_index + 1)]
        
        # Create strategy (using calls)
        strategy = OptionsStrategy("Butterfly Spread")
        
        # Buy lower strike call
        lower_data = self.analyzer.get_option_data(expiry, lower_strike, "CALL")
        if lower_data:
            premium = (lower_data["bid"] + lower_data["ask"]) / 2
            strategy.add_leg("CALL", lower_strike, 1, premium, "BUY")
        
        # Sell 2 middle strike calls
        middle_data = self.analyzer.get_option_data(expiry, middle_strike, "CALL")
        if middle_data:
            premium = (middle_data["bid"] + middle_data["ask"]) / 2
            strategy.add_leg("CALL", middle_strike, 2, premium, "SELL")
        
        # Buy upper strike call
        upper_data = self.analyzer.get_option_data(expiry, upper_strike, "CALL")
        if upper_data:
            premium = (upper_data["bid"] + upper_data["ask"]) / 2
            strategy.add_leg("CALL", upper_strike, 1, premium, "BUY")
        
        strategy.display_summary()
        self.strategies.append(strategy)
        
        # Show P&L profile
        self.display_pnl_profile(strategy, [lower_strike, middle_strike, upper_strike])
    
    def demo_long_straddle(self, expiry: str):
        """Demonstrate Long Straddle strategy"""
        print("\n" + "="*80)
        print("LONG STRADDLE STRATEGY")
        print("="*80)
        
        atm_strike = self.analyzer.find_atm_strike(expiry, self.underlying_price)
        
        if not atm_strike:
            print("Could not find ATM strike")
            return
        
        # Create strategy
        strategy = OptionsStrategy("Long Straddle")
        
        # Buy ATM call
        call_data = self.analyzer.get_option_data(expiry, atm_strike, "CALL")
        if call_data:
            premium = (call_data["bid"] + call_data["ask"]) / 2
            strategy.add_leg("CALL", atm_strike, 1, premium, "BUY")
        
        # Buy ATM put
        put_data = self.analyzer.get_option_data(expiry, atm_strike, "PUT")
        if put_data:
            premium = (put_data["bid"] + put_data["ask"]) / 2
            strategy.add_leg("PUT", atm_strike, 1, premium, "BUY")
        
        strategy.display_summary()
        self.strategies.append(strategy)
        
        # Show P&L profile
        self.display_pnl_profile(strategy, [atm_strike])
    
    def display_pnl_profile(self, strategy: OptionsStrategy, key_strikes: List[float]):
        """Display P&L profile for a strategy"""
        print(f"\nP&L Profile for {strategy.name}:")
        print("-" * 50)
        
        # Calculate P&L at various price points
        min_strike = min(key_strikes) - 10
        max_strike = max(key_strikes) + 10
        
        test_prices = []
        test_prices.extend(key_strikes)  # Include all key strikes
        test_prices.extend([min_strike, max_strike])  # Include extremes
        test_prices.append(self.underlying_price)  # Include current price
        
        # Add some intermediate points
        for i in range(5):
            price = min_strike + (max_strike - min_strike) * i / 4
            test_prices.append(price)
        
        test_prices = sorted(set(test_prices))
        
        print(f"{'Underlying Price':>20} | {'P&L':>15} | {'Status':>10}")
        print("-" * 50)
        
        for price in test_prices:
            pnl = strategy.calculate_pnl(price)
            status = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
            
            # Highlight current price
            marker = " <-- Current" if abs(price - self.underlying_price) < 0.01 else ""
            
            print(f"${price:>18.2f} | ${pnl:>14.2f} | {status:>10}{marker}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("TRADING SESSION SUMMARY")
        print("="*80)
        
        print(f"\nUnderlying: TLT")
        print(f"Current Price: ${self.underlying_price:.2f}")
        print(f"Strategies Analyzed: {len(self.strategies)}")
        
        # Summary table
        print("\n" + "-"*80)
        print(f"{'Strategy':>20} | {'Net Cost':>12} | {'Max Profit':>12} | {'Max Loss':>12} | {'Risk/Reward':>12}")
        print("-"*80)
        
        for strategy in self.strategies:
            net_cost = sum(leg["cost"] for leg in strategy.legs)
            max_profit = strategy.calculate_max_profit()
            max_loss = strategy.calculate_max_loss()
            
            # Calculate risk/reward ratio
            if max_loss != 0:
                risk_reward = abs(max_profit / max_loss)
            else:
                risk_reward = float('inf')
            
            print(f"{strategy.name:>20} | ${net_cost:>11.2f} | ${max_profit:>11.2f} | ${max_loss:>11.2f} | {risk_reward:>11.2f}x")
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("="*80)
    
    async def run_demo(self):
        """Run the complete options trading demo"""
        try:
            print("="*80)
            print("COMPLETE OPTIONS TRADING STRATEGIES DEMO")
            print("="*80)
            print(f"\nGenerating sample options chain data...")
            
            # Generate sample chain
            chain_data = self.generate_sample_chain()
            self.analyzer = OptionsChainAnalyzer(chain_data)
            
            # Get first expiry for demo
            expiries = sorted(chain_data.keys())
            if not expiries:
                print("No expiries available")
                return
            
            expiry = expiries[0]
            print(f"Using expiry: {expiry}")
            print(f"Underlying price: ${self.underlying_price:.2f}")
            
            # Run all strategy demos
            self.demo_iron_condor(expiry)
            await asyncio.sleep(0.5)  # Small delay for readability
            
            self.demo_bull_put_spread(expiry)
            await asyncio.sleep(0.5)
            
            self.demo_covered_call(expiry)
            await asyncio.sleep(0.5)
            
            self.demo_butterfly_spread(expiry)
            await asyncio.sleep(0.5)
            
            self.demo_long_straddle(expiry)
            await asyncio.sleep(0.5)
            
            # Generate summary report
            self.generate_summary_report()
            
            # Save results to file
            self.save_results()
            
        except Exception as e:
            logger.error(f"Error in demo: {e}", exc_info=True)
    
    def save_results(self):
        """Save demo results to JSON file"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "underlying_symbol": "TLT",
                "underlying_price": self.underlying_price,
                "strategies": []
            }
            
            for strategy in self.strategies:
                strategy_data = {
                    "name": strategy.name,
                    "legs": strategy.legs,
                    "net_cost": sum(leg["cost"] for leg in strategy.legs),
                    "max_profit": strategy.calculate_max_profit(),
                    "max_loss": strategy.calculate_max_loss(),
                    "breakeven_points": strategy.get_breakeven_points()
                }
                results["strategies"].append(strategy_data)
            
            # Save to file
            filename = f"options_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main entry point"""
    demo = OptionsTradingDemo()
    asyncio.run(demo.run_demo())

if __name__ == "__main__":
    main()