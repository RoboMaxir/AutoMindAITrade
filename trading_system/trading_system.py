"""
Main Trading System Controller
Coordinates all agents and manages the trading workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from datetime import datetime
import ccxt
import os

from trading_system.agents.technical_analysis_agent import TechnicalAnalysisAgent
from trading_system.agents.fundamental_agent import FundamentalIntelligenceAgent
from trading_system.agents.risk_management_agent import RiskManagementAgent
from trading_system.orchestrator.decision_orchestrator import DecisionOrchestrator
from trading_system.config import AGENT_CONFIG, TRADING_PARAMS, EXCHANGE_CONFIG

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main trading system that coordinates all agents and manages trading operations
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.trading_params = self.config.get('trading_params', TRADING_PARAMS)
        self.agent_config = self.config.get('agent_config', AGENT_CONFIG)
        
        # Initialize agents
        self.technical_agent = TechnicalAnalysisAgent(self.agent_config['technical_analysis'])
        self.fundamental_agent = FundamentalIntelligenceAgent(self.agent_config['fundamental_analysis'])
        self.risk_agent = RiskManagementAgent(self.agent_config['risk_management'])
        self.orchestrator = DecisionOrchestrator(self.agent_config['decision_orchestrator'])
        
        # Initialize exchange connection
        self.exchange = None
        self._initialize_exchange()
        
        # Trading state
        self.current_position = None
        self.position_history = []
        self.market_data_cache = {}
        self.last_analysis_time = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Trading System initialized successfully")
    
    def _setup_logging(self):
        """
        Setup system logging
        """
        log_config = self.config.get('logging_config', {})
        level = log_config.get('level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('log_file', 'trading_system.log')),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_exchange(self):
        """
        Initialize exchange connection
        """
        exchange_name = self.trading_params.get('default_exchange', 'binance')
        exchange_config = EXCHANGE_CONFIG.get(exchange_name, {})
        
        if not exchange_config.get('api_key'):
            logger.warning(f"No API key provided for {exchange_name}, running in simulation mode")
            self.exchange = None
            return
        
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['secret'],
                'sandbox': exchange_config.get('sandbox', True),
                'enableRateLimit': True,
            })
            logger.info(f"Connected to {exchange_name} exchange")
        except Exception as e:
            logger.error(f"Failed to connect to exchange {exchange_name}: {e}")
            self.exchange = None
    
    def fetch_market_data(self, symbol: str = None, timeframe: str = None, limit: int = 200) -> pd.DataFrame:
        """
        Fetch market data for analysis
        """
        if symbol is None:
            symbol = self.trading_params['default_symbol']
        if timeframe is None:
            timeframe = self.trading_params['timeframe']
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.market_data_cache:
            cached_data, cache_time = self.market_data_cache[cache_key]
            # Use cache if less than 5 minutes old
            if time.time() - cache_time < 300:  # 5 minutes
                return cached_data
        
        if self.exchange:
            try:
                # Fetch OHLCV data from exchange
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Cache the data
                self.market_data_cache[cache_key] = (df, time.time())
                
                return df
            except Exception as e:
                logger.error(f"Error fetching market data from exchange: {e}")
                # Return empty dataframe if exchange fails
                return pd.DataFrame()
        else:
            # Simulated data for testing
            logger.info("Using simulated market data (no exchange connection)")
            dates = pd.date_range(start='2023-01-01', periods=limit, freq='1h')
            np.random.seed(int(time.time()))
            
            # Create realistic OHLCV data
            base_price = 40000 + np.cumsum(np.random.randn(limit) * 50)
            high = base_price + np.random.rand(limit) * 100
            low = base_price - np.random.rand(limit) * 100
            close = base_price + np.random.randn(limit) * 30
            volume = np.random.randint(100, 1000, size=limit) * 1000
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': base_price - np.random.rand(limit) * 50,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
            # Cache the data
            self.market_data_cache[cache_key] = (df, time.time())
            
            return df
    
    def run_analysis_cycle(self, symbol: str = None) -> Dict:
        """
        Run a complete analysis cycle with all agents
        """
        logger.info(f"Starting analysis cycle for {symbol or self.trading_params['default_symbol']}")
        
        # Fetch market data
        market_data = self.fetch_market_data(symbol)
        if market_data.empty:
            logger.error("No market data available for analysis")
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No market data available'}
        
        # Technical Analysis
        logger.info("Running technical analysis...")
        technical_analysis = self.technical_agent.analyze_market_structure(market_data)
        tech_signal = self.technical_agent.generate_signal(technical_analysis)
        
        # Fundamental Analysis
        logger.info("Running fundamental analysis...")
        symbols_list = [symbol.split('/')[0]] if symbol else ['BTC', 'ETH']
        fundamental_analysis = self.fundamental_agent.analyze_news_intelligence(symbols_list)
        fund_signal = self.fundamental_agent.generate_signal(fundamental_analysis)
        
        # Risk Analysis
        logger.info("Running risk analysis...")
        market_context_for_risk = {
            'price': market_data['close'].iloc[-1],
            'atr': technical_analysis.get('atr', market_data['close'].iloc[-1] * 0.01),
            'volatility': technical_analysis.get('volatility_regime', 'normal')
        }
        risk_analysis = self.risk_agent.get_risk_summary()
        risk_signal = self.risk_agent.generate_signal(market_context_for_risk)
        
        # Combine all signals through orchestrator
        logger.info("Combining signals through orchestrator...")
        market_context = {
            'price': market_data['close'].iloc[-1],
            'trend': technical_analysis.get('trend_direction', 'sideways'),
            'volatility': technical_analysis.get('volatility_regime', 'normal'),
            'volume': market_data['volume'].iloc[-1]
        }
        
        final_decision = self.orchestrator.make_final_decision(
            {'signal': tech_signal},
            {'signal': fund_signal},
            {'signal': risk_signal},
            market_context
        )
        
        # Update last analysis time
        self.last_analysis_time = datetime.now()
        
        # Prepare comprehensive result
        result = {
            'timestamp': datetime.now(),
            'decision': final_decision,
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_analysis,
            'risk_analysis': risk_analysis,
            'market_data_summary': {
                'current_price': market_data['close'].iloc[-1],
                'price_change_24h': (
                    (market_data['close'].iloc[-1] - market_data['close'].iloc[-24]) 
                    / market_data['close'].iloc[-24] * 100
                    if len(market_data) > 24 else 0
                ),
                'volume_24h': market_data['volume'].iloc[-1]
            }
        }
        
        logger.info(f"Analysis cycle completed. Decision: {final_decision['action']} "
                   f"(Confidence: {final_decision['confidence']:.2f})")
        
        return result
    
    def execute_trade(self, decision: Dict, symbol: str = None) -> Optional[Dict]:
        """
        Execute trade based on decision (if not in simulation mode)
        """
        if symbol is None:
            symbol = self.trading_params['default_symbol']
        
        action = decision['action']
        confidence = decision['confidence']
        
        # Check if trade should be executed based on confidence threshold
        if confidence < self.agent_config['decision_orchestrator']['confidence_threshold']:
            logger.info(f"Trade not executed: confidence {confidence:.2f} below threshold")
            return None
        
        if not self.exchange:
            logger.info(f"SIMULATION MODE: Would execute {action} for {symbol}")
            # Simulate trade execution
            simulated_result = {
                'id': f"sim_{int(time.time())}",
                'symbol': symbol,
                'order_type': 'market',
                'side': action.lower(),
                'amount': 0.01,  # Simulated amount
                'price': self.fetch_market_data(symbol)['close'].iloc[-1],
                'timestamp': datetime.now(),
                'status': 'filled',
                'simulated': True
            }
            self.position_history.append(simulated_result)
            return simulated_result
        
        try:
            # Calculate position size based on risk management
            current_price = self.fetch_market_data(symbol)['close'].iloc[-1]
            stop_loss = self._calculate_stop_loss(current_price, decision)
            
            position_size = self.risk_agent.calculate_position_size(
                entry_price=current_price,
                stop_loss_price=stop_loss,
                symbol=symbol
            )
            
            # Validate trade with risk management
            trade_request = {
                'symbol': symbol,
                'action': action,
                'position_size': position_size['position_size'],
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': self.risk_agent.calculate_take_profit(current_price, stop_loss)
            }
            
            is_valid, reason, validated_request = self.risk_agent.validate_trade(trade_request)
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {reason}")
                return None
            
            # Execute trade
            if action == 'BUY':
                order = self.exchange.create_market_buy_order(symbol, validated_request['position_size'])
            elif action == 'SELL':
                order = self.exchange.create_market_sell_order(symbol, validated_request['position_size'])
            else:
                logger.info("No trade executed for HOLD action")
                return None
            
            # Add to position history
            self.position_history.append(order)
            logger.info(f"Trade executed: {order}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _calculate_stop_loss(self, current_price: float, decision: Dict) -> float:
        """
        Calculate stop loss based on market conditions and decision
        """
        # Simple stop loss calculation - in practice this would be more sophisticated
        atr = 100  # Placeholder - would come from technical analysis
        if decision['action'] == 'BUY':
            return current_price - (atr * 2)  # 2x ATR below entry
        else:  # SELL/SHORT
            return current_price + (atr * 2)  # 2x ATR above entry
    
    def run_trading_cycle(self, symbol: str = None) -> Dict:
        """
        Run a complete trading cycle: analyze -> decide -> execute
        """
        logger.info("Starting trading cycle...")
        
        # Run analysis
        analysis_result = self.run_analysis_cycle(symbol)
        decision = analysis_result['decision']
        
        # Execute trade if needed
        execution_result = self.execute_trade(decision, symbol)
        
        # Update risk management with new information
        if execution_result:
            # Update account value if possible (in real system)
            pass
        
        result = {
            'analysis': analysis_result,
            'execution': execution_result,
            'cycle_completed': datetime.now()
        }
        
        logger.info(f"Trading cycle completed. Action: {decision['action']}")
        
        return result
    
    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status
        """
        if self.exchange:
            try:
                balance = self.exchange.fetch_balance()
                positions = self.exchange.fetch_positions() if hasattr(self.exchange, 'fetch_positions') else []
                
                return {
                    'balance': balance,
                    'positions': positions,
                    'position_history': self.position_history,
                    'current_equity': self._calculate_equity(balance, positions)
                }
            except Exception as e:
                logger.error(f"Error fetching portfolio status: {e}")
                return {'error': str(e)}
        else:
            # Simulated portfolio status
            return {
                'balance': {'total': {'USDT': 100000}, 'free': {'USDT': 95000}, 'used': {'USDT': 5000}},
                'positions': [],
                'position_history': self.position_history,
                'current_equity': 100000,
                'simulated': True
            }
    
    def _calculate_equity(self, balance: Dict, positions: List) -> float:
        """
        Calculate total equity
        """
        # Sum all balances and positions
        total = 0
        for currency, amount in balance.get('total', {}).items():
            # Convert to USD equivalent (simplified)
            total += amount  # In simulation, assume all values are in USD
        
        return total
    
    def start_continuous_trading(self, symbol: str = None, interval: int = 300):
        """
        Start continuous trading loop
        """
        logger.info(f"Starting continuous trading for {symbol or self.trading_params['default_symbol']} "
                   f"with {interval}s intervals")
        
        try:
            while True:
                try:
                    # Run trading cycle
                    result = self.run_trading_cycle(symbol)
                    
                    # Wait for next interval
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("Trading interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in continuous trading: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status
        """
        decision_summary = self.orchestrator.get_decision_summary()
        risk_summary = self.risk_agent.get_risk_summary()
        portfolio_status = self.get_portfolio_status()
        
        return {
            'timestamp': datetime.now(),
            'agents_status': {
                'technical': 'active',
                'fundamental': 'active',
                'risk_management': 'active',
                'orchestrator': 'active'
            },
            'decision_summary': decision_summary,
            'risk_summary': risk_summary,
            'portfolio_status': portfolio_status,
            'last_analysis': self.last_analysis_time,
            'total_positions': len(self.position_history)
        }


def main():
    """Main function to run the trading system"""
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
    
    # Run a single analysis cycle
    print("Running single analysis cycle...")
    result = trading_system.run_analysis_cycle()
    
    print(f"Decision: {result['decision']['action']}")
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


if __name__ == "__main__":
    main()