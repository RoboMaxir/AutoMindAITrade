"""
Decision Orchestrator
Aggregates all agent outputs, resolves conflicts, and produces final trading decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DecisionOrchestrator:
    """
    Orchestrates decisions from all agents and produces final trading actions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.conflict_resolution = config.get('conflict_resolution', 'conservative')
        self.decision_history = []
        self.voting_weights = {
            'technical': 0.4,
            'fundamental': 0.3,
            'risk': 0.3
        }
        
    def aggregate_signals(self, technical_signal: Dict, fundamental_signal: Dict, 
                        risk_signal: Dict, market_context: Dict) -> Dict:
        """
        Aggregate signals from all agents
        """
        # Normalize confidence scores
        tech_conf = max(0.0, min(1.0, technical_signal.get('confidence', 0.5)))
        fund_conf = max(0.0, min(1.0, fundamental_signal.get('confidence', 0.5)))
        risk_conf = max(0.0, min(1.0, risk_signal.get('confidence', 0.5)))
        
        # Weighted voting based on confidence and agent importance
        weighted_votes = {}
        
        # Technical analysis vote
        if technical_signal['action'] == 'BUY':
            weighted_votes['BUY'] = weighted_votes.get('BUY', 0) + (tech_conf * self.voting_weights['technical'])
        elif technical_signal['action'] == 'SELL':
            weighted_votes['SELL'] = weighted_votes.get('SELL', 0) + (tech_conf * self.voting_weights['technical'])
        else:  # HOLD
            weighted_votes['HOLD'] = weighted_votes.get('HOLD', 0) + (tech_conf * self.voting_weights['technical'])
        
        # Fundamental analysis vote
        if fundamental_signal['action'] == 'BUY':
            weighted_votes['BUY'] = weighted_votes.get('BUY', 0) + (fund_conf * self.voting_weights['fundamental'])
        elif fundamental_signal['action'] == 'SELL':
            weighted_votes['SELL'] = weighted_votes.get('SELL', 0) + (fund_conf * self.voting_weights['fundamental'])
        else:  # HOLD or other recommendations
            weighted_votes['HOLD'] = weighted_votes.get('HOLD', 0) + (fund_conf * self.voting_weights['fundamental'])
        
        # Risk management vote (has veto power in some cases)
        if risk_signal.get('risk_override', False):
            # Risk signal overrides if it's a critical override
            if risk_signal['action'] == 'HOLD':
                weighted_votes = {'HOLD': 1.0}
            elif risk_signal['action'] == 'SELL':
                weighted_votes = {'SELL': 1.0}
        else:
            # Normal risk weighting
            if risk_signal['action'] == 'BUY':
                weighted_votes['BUY'] = weighted_votes.get('BUY', 0) + (risk_conf * self.voting_weights['risk'])
            elif risk_signal['action'] == 'SELL':
                weighted_votes['SELL'] = weighted_votes.get('SELL', 0) + (risk_conf * self.voting_weights['risk'])
            else:  # HOLD or NEUTRAL
                weighted_votes['HOLD'] = weighted_votes.get('HOLD', 0) + (risk_conf * self.voting_weights['risk'])
        
        # Determine final action based on weighted votes
        final_action = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_action]
        
        # Apply confidence threshold
        if final_confidence < self.confidence_threshold:
            final_action = 'HOLD'
            final_confidence = 0.3  # Lower confidence for hold decisions
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            technical_signal, fundamental_signal, risk_signal, 
            final_action, final_confidence
        )
        
        # Calculate overall confidence taking into account agreement between agents
        agreement_score = self._calculate_agreement_score(technical_signal, fundamental_signal, risk_signal)
        final_confidence = (final_confidence + agreement_score) / 2
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'weighted_votes': weighted_votes,
            'agreement_score': agreement_score,
            'timestamp': datetime.now(),
            'signals_input': {
                'technical': technical_signal,
                'fundamental': fundamental_signal,
                'risk': risk_signal
            }
        }
    
    def _generate_reasoning(self, tech_signal: Dict, fund_signal: Dict, risk_signal: Dict, 
                           final_action: str, final_confidence: float) -> str:
        """
        Generate human-readable reasoning for the decision
        """
        reasons = []
        
        # Add technical reasoning
        tech_reason = tech_signal.get('reasoning', 'No technical reasoning provided')
        reasons.append(f"Technical: {tech_reason}")
        
        # Add fundamental reasoning
        fund_reason = fund_signal.get('reasoning', 'No fundamental reasoning provided')
        reasons.append(f"Fundamental: {fund_reason}")
        
        # Add risk considerations
        risk_reason = risk_signal.get('reasoning', 'No risk reasoning provided')
        reasons.append(f"Risk: {risk_reason}")
        
        # Add final decision summary
        reasons.append(f"Final decision: {final_action} with confidence {final_confidence:.2f}")
        
        return " | ".join(reasons)
    
    def _calculate_agreement_score(self, tech_signal: Dict, fund_signal: Dict, risk_signal: Dict) -> float:
        """
        Calculate how much agents agree on the same action
        """
        actions = [tech_signal['action'], fund_signal['action'], risk_signal['action']]
        
        # Count how many agents agree with the final decision
        final_action = max(set(actions), key=actions.count)  # Most common action
        agreement_count = actions.count(final_action)
        
        # Calculate agreement score (0.0 to 1.0)
        agreement_score = agreement_count / len(actions)
        
        # Also factor in confidence levels
        avg_confidence = (tech_signal.get('confidence', 0.5) + 
                         fund_signal.get('confidence', 0.5) + 
                         risk_signal.get('confidence', 0.5)) / 3
        
        return (agreement_score + avg_confidence) / 2
    
    def resolve_conflicts(self, signals: List[Dict]) -> Dict:
        """
        Resolve conflicts between agent signals using configured strategy
        """
        tech_signal = signals[0] if len(signals) > 0 else {'action': 'HOLD', 'confidence': 0.0}
        fund_signal = signals[1] if len(signals) > 1 else {'action': 'HOLD', 'confidence': 0.0}
        risk_signal = signals[2] if len(signals) > 2 else {'action': 'HOLD', 'confidence': 0.0}
        
        if self.conflict_resolution == 'conservative':
            # Conservative approach: if any agent says HOLD or SELL, prefer caution
            if risk_signal.get('risk_override', False) and risk_signal['action'] in ['HOLD', 'SELL']:
                return risk_signal
            elif 'HOLD' in [tech_signal['action'], fund_signal['action'], risk_signal['action']]:
                return {
                    'action': 'HOLD',
                    'confidence': max(tech_signal.get('confidence', 0.0), 
                                    fund_signal.get('confidence', 0.0), 
                                    risk_signal.get('confidence', 0.0)),
                    'reasoning': 'Conservative conflict resolution - holding due to mixed signals'
                }
            else:
                # If all are BUY, go with highest confidence
                all_signals = [tech_signal, fund_signal, risk_signal]
                highest_conf_signal = max(all_signals, key=lambda x: x.get('confidence', 0.0))
                return highest_conf_signal
                
        elif self.conflict_resolution == 'aggressive':
            # Aggressive approach: favor BUY signals
            if 'BUY' in [tech_signal['action'], fund_signal['action']] and risk_signal['action'] != 'HOLD':
                # Find highest confidence BUY signal
                buy_signals = [s for s in [tech_signal, fund_signal] if s['action'] == 'BUY']
                if buy_signals:
                    return max(buy_signals, key=lambda x: x.get('confidence', 0.0))
            # Otherwise use standard weighted approach
            return self._weighted_conflict_resolution(tech_signal, fund_signal, risk_signal)
            
        else:  # weighted_average
            return self._weighted_conflict_resolution(tech_signal, fund_signal, risk_signal)
    
    def _weighted_conflict_resolution(self, tech_signal: Dict, fund_signal: Dict, risk_signal: Dict) -> Dict:
        """
        Resolve conflicts using weighted average approach
        """
        # Calculate weighted scores for each action
        weights = {
            'technical': self.voting_weights['technical'],
            'fundamental': self.voting_weights['fundamental'], 
            'risk': self.voting_weights['risk']
        }
        
        # Score each possible action
        action_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        if tech_signal['action'] in action_scores:
            action_scores[tech_signal['action']] += tech_signal.get('confidence', 0.5) * weights['technical']
        
        if fund_signal['action'] in action_scores:
            action_scores[fund_signal['action']] += fund_signal.get('confidence', 0.5) * weights['fundamental']
        
        if risk_signal['action'] in action_scores:
            if risk_signal.get('risk_override', False):
                # Risk override gets maximum weight
                action_scores[risk_signal['action']] = max(action_scores[risk_signal['action']], 1.0)
            else:
                action_scores[risk_signal['action']] += risk_signal.get('confidence', 0.5) * weights['risk']
        
        # Choose action with highest score
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': f'Weighted resolution - {final_action} scored {final_confidence:.2f}'
        }
    
    def apply_risk_veto(self, decision: Dict, risk_signal: Dict) -> Dict:
        """
        Apply risk management veto power
        """
        # Risk has ultimate veto power in critical situations
        if risk_signal.get('risk_override', False) and risk_signal['action'] in ['HOLD', 'SELL']:
            logger.info(f"Risk management veto exercised: {risk_signal['action']}")
            return {
                'action': risk_signal['action'],
                'confidence': risk_signal.get('confidence', decision['confidence']),
                'reasoning': f"RISK VETO: {risk_signal['reasoning']} | Original decision: {decision['action']}",
                'veto_exercised': True,
                'original_decision': decision['action']
            }
        
        return decision
    
    def make_final_decision(self, technical_analysis: Dict, fundamental_analysis: Dict, 
                          risk_management: Dict, market_context: Dict) -> Dict:
        """
        Main method to make final trading decision
        """
        # Extract signals from analysis results
        tech_signal = technical_analysis.get('signal', 
                                           {'action': 'HOLD', 'confidence': 0.3, 'reasoning': 'No technical signal'})
        fund_signal = fundamental_analysis.get('signal', 
                                            {'action': 'HOLD', 'confidence': 0.3, 'reasoning': 'No fundamental signal'})
        risk_signal = risk_management.get('signal', 
                                        {'action': 'NEUTRAL', 'confidence': 0.5, 'reasoning': 'No risk signal'})
        
        # Aggregate signals
        aggregated = self.aggregate_signals(tech_signal, fund_signal, risk_signal, market_context)
        
        # Apply risk veto if necessary
        final_decision = self.apply_risk_veto(aggregated, risk_signal)
        
        # Validate decision meets minimum requirements
        final_decision = self._validate_decision(final_decision)
        
        # Log decision
        self._log_decision(final_decision, market_context)
        
        return final_decision
    
    def _validate_decision(self, decision: Dict) -> Dict:
        """
        Validate that decision meets basic requirements
        """
        # Ensure confidence is in valid range
        decision['confidence'] = max(0.0, min(1.0, decision.get('confidence', 0.5)))
        
        # Ensure action is valid
        valid_actions = ['BUY', 'SELL', 'HOLD']
        if decision['action'] not in valid_actions:
            decision['action'] = 'HOLD'
            decision['confidence'] = 0.1
            decision['reasoning'] = f"Invalid action corrected to HOLD. Original: {decision['action']}"
        
        return decision
    
    def _log_decision(self, decision: Dict, market_context: Dict):
        """
        Log decision for audit trail
        """
        decision_record = {
            'timestamp': decision.get('timestamp', datetime.now()),
            'action': decision['action'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'market_context': market_context,
            'veto_exercised': decision.get('veto_exercised', False)
        }
        
        self.decision_history.append(decision_record)
        
        logger.info(f"Decision: {decision['action']} (Conf: {decision['confidence']:.2f}) - {decision['reasoning']}")
    
    def get_decision_summary(self) -> Dict:
        """
        Get summary of recent decisions
        """
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'average_confidence': 0.0,
                'veto_rate': 0.0
            }
        
        recent_decisions = self.decision_history[-50:]  # Last 50 decisions
        
        buy_count = sum(1 for d in recent_decisions if d['action'] == 'BUY')
        sell_count = sum(1 for d in recent_decisions if d['action'] == 'SELL')
        hold_count = sum(1 for d in recent_decisions if d['action'] == 'HOLD')
        avg_confidence = np.mean([d['confidence'] for d in recent_decisions])
        veto_rate = sum(1 for d in recent_decisions if d.get('veto_exercised', False)) / len(recent_decisions)
        
        return {
            'total_decisions': len(recent_decisions),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'average_confidence': avg_confidence,
            'veto_rate': veto_rate,
            'latest_decision': recent_decisions[-1] if recent_decisions else None
        }


def main():
    """Test the Decision Orchestrator"""
    config = {
        'confidence_threshold': 0.7,
        'conflict_resolution': 'conservative'
    }
    
    orchestrator = DecisionOrchestrator(config)
    
    # Create mock signals from agents
    technical_signal = {
        'action': 'BUY',
        'confidence': 0.8,
        'reasoning': 'Strong bullish trend with volume confirmation'
    }
    
    fundamental_signal = {
        'action': 'HOLD',
        'confidence': 0.6,
        'reasoning': 'Mixed news sentiment, regulatory uncertainty'
    }
    
    risk_signal = {
        'action': 'NEUTRAL',
        'confidence': 0.7,
        'reasoning': 'Risk parameters acceptable',
        'risk_override': False
    }
    
    market_context = {
        'price': 40000,
        'trend': 'bullish',
        'volatility': 'normal',
        'volume': 'high'
    }
    
    # Make a decision
    decision = orchestrator.make_final_decision(
        {'signal': technical_signal},
        {'signal': fundamental_signal}, 
        {'signal': risk_signal},
        market_context
    )
    
    print("Decision Orchestrator Results:")
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Veto Exercised: {decision.get('veto_exercised', False)}")
    
    # Get decision summary
    summary = orchestrator.get_decision_summary()
    print(f"\nDecision Summary:")
    print(f"Total Decisions: {summary['total_decisions']}")
    print(f"Buy: {summary['buy_count']}, Sell: {summary['sell_count']}, Hold: {summary['hold_count']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    print(f"Veto Rate: {summary['veto_rate']:.2f}")


if __name__ == "__main__":
    main()