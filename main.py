#!/usr/bin/env python3
"""
Main entry point for the Enterprise-Grade AI Crypto Spot Trading System
"""

import sys
import os
import argparse
from datetime import datetime

# Add the workspace to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trading_system.trading_system import TradingSystem
from trading_system.config import AGENT_CONFIG, TRADING_PARAMS


def main():
    parser = argparse.ArgumentParser(description='Enterprise-Grade AI Crypto Spot Trading System')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Analysis timeframe (default: 1h)')
    parser.add_argument('--mode', type=str, choices=['single', 'continuous'], default='single', 
                       help='Run mode: single analysis or continuous trading (default: single)')
    parser.add_argument('--interval', type=int, default=300, help='Trading interval in seconds (default: 300 for 5 minutes)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    print(f"Starting Enterprise-Grade AI Crypto Spot Trading System")
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {args.mode}")
    print(f"Time: {datetime.now()}")
    print("-" * 60)
    
    # Create system configuration
    config = {
        'trading_params': TRADING_PARAMS,
        'agent_config': AGENT_CONFIG,
        'logging_config': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'trading_system.log'
        }
    }
    
    # Initialize trading system
    trading_system = TradingSystem(config)
    
    if args.mode == 'single':
        # Run a single analysis cycle
        print("Running single analysis cycle...")
        result = trading_system.run_analysis_cycle(args.symbol)
        
        print(f"\nDecision: {result['decision']['action']}")
        print(f"Confidence: {result['decision']['confidence']:.2f}")
        print(f"Reasoning: {result['decision']['reasoning']}")
        
        # Get system status
        status = trading_system.get_system_status()
        print(f"\nSystem Status:")
        print(f"Decision Summary - Buys: {status['decision_summary']['buy_count']}, "
              f"Sells: {status['decision_summary']['sell_count']}, "
              f"Holds: {status['decision_summary']['hold_count']}")
        print(f"Risk Level: {status['risk_summary']['drawdown_risk_level']}")
        print(f"Total Positions: {status['total_positions']}")
        
    elif args.mode == 'continuous':
        print(f"Starting continuous trading with {args.interval}s intervals...")
        print("Press Ctrl+C to stop")
        try:
            trading_system.start_continuous_trading(symbol=args.symbol, interval=args.interval)
        except KeyboardInterrupt:
            print("\nContinuous trading stopped by user")
    
    print(f"\nSystem finished at {datetime.now()}")


if __name__ == "__main__":
    main()