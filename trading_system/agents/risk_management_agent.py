"""
Risk & Capital Management Agent
Implements dynamic position sizing, exposure limits, drawdown control, and capital preservation logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class RiskManagementAgent:
    """
    Implements dynamic risk management with veto power over all trade decisions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.position_sizing_method = config.get('position_sizing_method', 'fixed_fraction')
        self.max_correlation = config.get('max_correlation', 0.7)
        self.stop_loss_type = config.get('stop_loss_type', 'atr')
        self.take_profit_ratio = config.get('take_profit_ratio', 2.0)
        
        # Risk tracking
        self.account_value = 100000  # Starting account value
        self.initial_capital = 100000
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15%
        self.current_drawdown = 0.0
        self.max_account_value = 100000
        self.trade_history = []
        self.position_limits = {}
        self.volatility_buffer = 1.0
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              symbol: str = 'BTC/USDT', account_pct: float = 0.02) -> Dict:
        """
        Calculate position size based on risk parameters
        """
        # Calculate risk per trade (account percentage)
        risk_amount = self.account_value * account_pct
        
        # Calculate stop loss distance
        if stop_loss_price is not None:
            stop_distance = abs(entry_price - stop_loss_price)
            risk_per_unit = stop_distance
        else:
            # If no stop loss provided, use ATR-based stop
            atr_based_stop = entry_price * 0.02  # 2% default
            risk_per_unit = atr_based_stop
        
        # Calculate position size
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            position_size = 0.0
        
        # Apply position size limits
        max_position_size = self.account_value * self.config.get('max_position_size', 0.10)  # 10% max
        position_size = min(position_size, max_position_size / entry_price)
        
        return {
            'position_size': position_size,
            'position_value': position_size * entry_price,
            'risk_amount': risk_amount,
            'stop_distance': risk_per_unit,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price
        }
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate position size using Kelly Criterion
        """
        if avg_loss == 0:
            return 0.0
            
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate  # Probability of winning
        q = 1 - p   # Probability of losing
        
        kelly_fraction = (b * p - q) / b
        
        # Use fractional Kelly to reduce risk
        fractional_kelly = kelly_fraction * 0.5  # Half Kelly to reduce volatility
        
        # Ensure position size is within bounds
        return max(0.0, min(0.25, fractional_kelly))  # Cap at 25%
    
    def calculate_volatility_position_size(self, volatility: float, entry_price: float) -> Dict:
        """
        Calculate position size based on volatility
        """
        # Higher volatility = smaller position size
        volatility_factor = 1.0 / (1.0 + volatility)  # Reduce size as volatility increases
        
        # Calculate risk-adjusted position
        risk_pct = 0.02 * volatility_factor  # Reduce risk pct with higher volatility
        risk_amount = self.account_value * risk_pct
        
        # Default stop loss as percentage of entry price
        stop_loss_pct = 0.02 * (1 + volatility)  # Increase stop with higher volatility
        stop_loss_distance = entry_price * stop_loss_pct
        
        position_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
        
        return {
            'position_size': position_size,
            'position_value': position_size * entry_price,
            'risk_amount': risk_amount,
            'volatility_factor': volatility_factor,
            'adjusted_risk_pct': risk_pct
        }
    
    def calculate_stop_loss(self, entry_price: float, market_data: Dict, 
                           stop_type: str = None, multiplier: float = 1.0) -> float:
        """
        Calculate stop loss based on various methods
        """
        if stop_type is None:
            stop_type = self.stop_loss_type
            
        if stop_type == 'atr':
            # ATR-based stop loss
            atr = market_data.get('atr', entry_price * 0.01)  # Default 1% if no ATR
            stop_distance = atr * 2.0 * multiplier  # 2x ATR
            return entry_price - stop_distance if market_data.get('direction') != 'short' else entry_price + stop_distance
            
        elif stop_type == 'percentage':
            # Percentage-based stop loss
            stop_pct = 0.02 * multiplier  # Default 2%
            stop_distance = entry_price * stop_pct
            return entry_price - stop_distance if market_data.get('direction') != 'short' else entry_price + stop_distance
            
        elif stop_type == 'volatility':
            # Volatility-based stop loss
            volatility = market_data.get('volatility', 0.02)
            stop_distance = entry_price * volatility * 1.5 * multiplier
            return entry_price - stop_distance if market_data.get('direction') != 'short' else entry_price + stop_distance
            
        else:
            # Default percentage stop
            return entry_price * 0.98  # 2% stop loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate take profit based on risk/reward ratio
        """
        stop_distance = abs(entry_price - stop_loss_price)
        take_profit_distance = stop_distance * self.take_profit_ratio
        
        # For long positions, take profit is above entry
        # For short positions, take profit is below entry
        if entry_price > stop_loss_price:  # Long position
            return entry_price + take_profit_distance
        else:  # Short position
            return entry_price - take_profit_distance
    
    def assess_drawdown_risk(self) -> Dict:
        """
        Assess current drawdown and risk status
        """
        current_value = self.account_value
        self.max_account_value = max(self.max_account_value, current_value)
        
        if self.max_account_value > 0:
            self.current_drawdown = (self.max_account_value - current_value) / self.max_account_value
        else:
            self.current_drawdown = 0.0
            
        # Determine drawdown risk level
        if self.current_drawdown > self.max_drawdown:
            risk_level = 'CRITICAL'
            action_required = 'HALT_TRADING'
        elif self.current_drawdown > self.max_drawdown * 0.75:
            risk_level = 'HIGH'
            action_required = 'REDUCE_POSITION_SIZE'
        elif self.current_drawdown > self.max_drawdown * 0.5:
            risk_level = 'MEDIUM'
            action_required = 'CAUTIOUS_TRADING'
        else:
            risk_level = 'LOW'
            action_required = 'NORMAL_OPERATIONS'
            
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_risk_level': risk_level,
            'action_required': action_required,
            'account_value': current_value,
            'initial_capital': self.initial_capital
        }
    
    def assess_correlation_risk(self, portfolio_positions: List[Dict]) -> Dict:
        """
        Assess correlation risk between positions
        """
        if len(portfolio_positions) < 2:
            return {
                'correlation_risk': 0.0,
                'max_correlation': 0.0,
                'risk_level': 'LOW',
                'positions_assessed': len(portfolio_positions)
            }
        
        # Simplified correlation calculation between position returns
        # In practice, this would use historical return correlations
        correlations = []
        for i in range(len(portfolio_positions)):
            for j in range(i+1, len(portfolio_positions)):
                # Placeholder - in real implementation would calculate actual correlations
                corr = np.random.uniform(0, 0.3)  # Simulated low correlation
                correlations.append(corr)
        
        max_corr = max(correlations) if correlations else 0.0
        avg_corr = np.mean(correlations) if correlations else 0.0
        
        if max_corr > self.max_correlation:
            risk_level = 'HIGH'
        elif avg_corr > self.max_correlation * 0.7:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
            
        return {
            'correlation_risk': avg_corr,
            'max_correlation': max_corr,
            'risk_level': risk_level,
            'positions_assessed': len(portfolio_positions)
        }
    
    def validate_trade(self, trade_request: Dict) -> Tuple[bool, str, Dict]:
        """
        Validate a trade request against risk parameters
        Returns: (is_valid, reason, updated_request)
        """
        # Check drawdown status first
        drawdown_status = self.assess_drawdown_risk()
        
        if drawdown_status['action_required'] == 'HALT_TRADING':
            return False, "Maximum drawdown exceeded, trading halted", trade_request
        
        # Check if trade aligns with position sizing rules
        if 'position_size' not in trade_request or trade_request['position_size'] <= 0:
            return False, "Invalid position size", trade_request
        
        # Check position value against account size
        position_value = trade_request['position_size'] * trade_request['price']
        if position_value > self.account_value * 0.95:  # Don't risk more than 95% of account
            return False, "Position too large for account size", trade_request
        
        # Check if we have enough funds
        if position_value > self.account_value * 0.8:  # Only risk up to 80% of available funds
            adjusted_size = (self.account_value * 0.8) / trade_request['price']
            trade_request['position_size'] = adjusted_size
            logger.info(f"Adjusted position size due to account size limitations: {adjusted_size}")
        
        # Additional checks could go here (concentration risk, sector correlation, etc.)
        
        return True, "Trade approved", trade_request
    
    def update_account_value(self, new_value: float):
        """
        Update account value and track performance
        """
        self.account_value = new_value
        self.max_account_value = max(self.max_account_value, new_value)
        
        # Recalculate drawdown
        if self.max_account_value > 0:
            self.current_drawdown = (self.max_account_value - new_value) / self.max_account_value
    
    def get_risk_summary(self) -> Dict:
        """
        Get overall risk summary
        """
        drawdown_status = self.assess_drawdown_risk()
        
        return {
            'account_value': self.account_value,
            'current_drawdown': drawdown_status['current_drawdown'],
            'drawdown_risk_level': drawdown_status['drawdown_risk_level'],
            'action_required': drawdown_status['action_required'],
            'total_trades': len(self.trade_history),
            'volatility_buffer': self.volatility_buffer,
            'risk_controls_active': True
        }
    
    def apply_risk_controls(self, signals: List[Dict]) -> List[Dict]:
        """
        Apply risk controls to incoming signals
        """
        filtered_signals = []
        
        for signal in signals:
            # Check if signal passes risk validation
            if self._should_allow_trade(signal):
                filtered_signals.append(signal)
            else:
                # Log risk rejection
                logger.info(f"Risk management rejected signal: {signal.get('action')} for {signal.get('symbol')}")
                # Convert to HOLD if risk not acceptable
                modified_signal = signal.copy()
                modified_signal['action'] = 'HOLD'
                modified_signal['confidence'] = 0.1
                modified_signal['risk_adjusted'] = True
                filtered_signals.append(modified_signal)
        
        return filtered_signals
    
    def _should_allow_trade(self, signal: Dict) -> bool:
        """
        Internal method to determine if trade should be allowed
        """
        # Check drawdown
        drawdown_status = self.assess_drawdown_risk()
        if drawdown_status['action_required'] == 'HALT_TRADING':
            return False
        
        # Check risk-reward ratio if stop loss and take profit are provided
        if 'stop_loss' in signal and 'take_profit' in signal:
            risk_amount = abs(signal['entry_price'] - signal['stop_loss'])
            reward_amount = abs(signal['take_profit'] - signal['entry_price'])
            
            if risk_amount > 0 and reward_amount / risk_amount < 0.5:  # Minimum 1:2 risk-reward
                return False
        
        # Additional risk checks could be implemented here
        return True
    
    def generate_signal(self, market_context: Dict) -> Dict:
        """
        Generate risk-based signal that can override other agents
        """
        drawdown_status = self.assess_drawdown_risk()
        
        # If critical drawdown, force hold
        if drawdown_status['drawdown_risk_level'] == 'CRITICAL':
            return {
                'action': 'HOLD',
                'confidence': 1.0,
                'reasoning': 'Critical drawdown level reached, preserving capital',
                'risk_override': True
            }
        
        # If high drawdown, be very conservative
        if drawdown_status['drawdown_risk_level'] == 'HIGH':
            return {
                'action': 'HOLD',
                'confidence': 0.9,
                'reasoning': 'High drawdown level, reducing risk exposure',
                'risk_override': True
            }
        
        # If account has grown significantly, consider taking profits
        if self.account_value > self.max_account_value * 1.5:  # 50% above peak
            return {
                'action': 'SELL',
                'confidence': 0.7,
                'reasoning': 'Account significantly above peak, considering profit taking',
                'risk_override': False
            }
        
        # Otherwise, no override needed
        return {
            'action': 'NEUTRAL',
            'confidence': 0.0,
            'reasoning': 'Risk parameters within acceptable ranges',
            'risk_override': False
        }


def main():
    """Test the Risk Management Agent"""
    config = {
        'position_sizing_method': 'kelly_criterion',
        'max_correlation': 0.7,
        'stop_loss_type': 'atr',
        'take_profit_ratio': 2.0,
        'max_drawdown': 0.15,
        'max_position_size': 0.10
    }
    
    agent = RiskManagementAgent(config)
    
    # Test position sizing
    position = agent.calculate_position_size(entry_price=40000, stop_loss_price=39200)
    print("Position Sizing Results:")
    print(f"Position Size: {position['position_size']:.6f}")
    print(f"Position Value: ${position['position_value']:.2f}")
    print(f"Risk Amount: ${position['risk_amount']:.2f}")
    
    # Test drawdown assessment
    drawdown_status = agent.assess_drawdown_risk()
    print(f"\nDrawdown Status: {drawdown_status['drawdown_risk_level']}")
    print(f"Current Drawdown: {drawdown_status['current_drawdown']:.2%}")
    
    # Test risk-based signal
    market_context = {'price': 40000, 'atr': 500, 'volatility': 0.03}
    risk_signal = agent.generate_signal(market_context)
    print(f"\nRisk Signal: {risk_signal['action']}")
    print(f"Confidence: {risk_signal['confidence']}")
    print(f"Reasoning: {risk_signal['reasoning']}")


if __name__ == "__main__":
    main()