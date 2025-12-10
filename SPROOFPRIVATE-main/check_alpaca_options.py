#!/usr/bin/env python3
"""
Check Alpaca Options Trading Capabilities
========================================
Quick script to check if options trading is available through Alpaca API
"""

import sys

# Hardcoded credentials
API_KEY = "PKCX98VZSJBQF79C1SD8"
SECRET_KEY = "KVLgbqFFlltuwszBbWhqHW6KyrzYO6raNb1y4Rjt"

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    print("✅ Alpaca SDK imported successfully")
    
    # Check available asset classes
    print("\nAvailable Asset Classes:")
    for asset_class in AssetClass:
        print(f"  - {asset_class.value}")
    
    # Initialize client
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
    # Get account info
    account = client.get_account()
    print(f"\n📊 Account Status: {account.status}")
    
    # Check for options-related attributes
    print("\nChecking for options capabilities...")
    options_attrs = ['options_trading_level', 'options_approved_level', 'options_trading_on']
    for attr in options_attrs:
        if hasattr(account, attr):
            print(f"  ✅ {attr}: {getattr(account, attr)}")
        else:
            print(f"  ❌ {attr}: Not available")
    
    # Try to get an options chain (if available)
    try:
        # This is hypothetical - checking if method exists
        if hasattr(client, 'get_option_contracts'):
            print("\n✅ Options trading methods available!")
            contracts = client.get_option_contracts('AAPL')
            print(f"Found {len(contracts)} option contracts for AAPL")
        else:
            print("\n❌ No direct options trading methods found")
            print("   Will use synthetic strategies instead")
    except Exception as e:
        print(f"\n❌ Options trading not available: {e}")
    
    # List available trading methods
    print("\nAvailable trading methods:")
    methods = [m for m in dir(client) if not m.startswith('_')]
    options_methods = [m for m in methods if 'option' in m.lower()]
    if options_methods:
        print("Options-specific methods:")
        for method in options_methods:
            print(f"  - {method}")
    else:
        print("  No options-specific methods found")
    
except ImportError as e:
    print(f"❌ Failed to import Alpaca SDK: {e}")
    print("\nPlease install the Alpaca SDK:")
    print("  pip install alpaca-py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)