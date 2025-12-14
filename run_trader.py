#!/usr/bin/env python3
"""
Startup script for Professional Crypto Trading Agent
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_agent import ProfessionalCryptoTrader
from config import TRADING_CONFIG

def main():
    print("🚀 Starting Professional Crypto Trading Agent...")
    print(f"Trading symbols: {TRADING_CONFIG['symbols']}")
    print(f"Check interval: {TRADING_CONFIG['check_interval_minutes']} minutes")
    print("-" * 50)
    
    # Initialize the trading agent
    trader = ProfessionalCryptoTrader(symbols=TRADING_CONFIG['symbols'])
    
    # Run in continuous mode
    try:
        trader.run_continuous(interval_minutes=TRADING_CONFIG['check_interval_minutes'])
    except KeyboardInterrupt:
        print("\n🛑 Trading agent stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running trading agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()