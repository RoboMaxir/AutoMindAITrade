#!/usr/bin/env python3
"""
Professional Crypto Spot Trading Agent

This is a fully autonomous professional crypto spot trader and market analyst.
It operates exclusively in crypto spot markets and executes Buy and Sell orders only.
It does not trade futures, leverage, margin, forex, CFDs, or any derivative instruments.
It does not perform price prediction or forecasting.

The agent analyzes the market state and makes rational Buy/Sell decisions exactly 
as a disciplined professional human trader would.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class MarketCondition:
    """Represents current market condition"""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: float  # ATR or similar measure
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    momentum: float  # RSI or similar momentum indicator
    support_resistance: Dict[str, float]  # Key levels


@dataclass
class Position:
    """Represents current position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime


@dataclass
class TradeSignal:
    """Trade signal with analysis"""
    action: OrderType
    symbol: str
    quantity: float
    price: float
    confidence: float  # 0.0 to 1.0
    reason: str
    technical_analysis: Dict
    fundamental_analysis: Dict
    risk_assessment: Dict


class NewsSentimentAnalyzer:
    """Analyzes news and sentiment for trading decisions"""
    
    def __init__(self):
        self.sentiment_thresholds = {
            'positive': 0.6,
            'negative': -0.6,
            'neutral': (-0.3, 0.3)
        }
    
    def get_news_sentiment(self, symbol: str) -> Dict:
        """
        Collects and analyzes news sentiment
        In a real implementation, this would connect to news APIs
        """
        # Simulated news analysis - in production, connect to news APIs
        # such as CryptoPanic, RSS feeds, or social media APIs
        
        # For simulation purposes, return neutral sentiment
        return {
            'sentiment_score': 0.0,  # -1 to 1 scale
            'risk_signals': [],
            'market_reaction_readiness': 'neutral',
            'news_items': []
        }


class TechnicalAnalyzer:
    """Performs technical analysis on price data"""
    
    def __init__(self):
        pass
    
    def analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify candlestick patterns and price behavior"""
        patterns = {}
        
        # Add technical indicators
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['atr'] = self.calculate_atr(df)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Identify recent patterns
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish engulfing pattern
        if (prev.close < prev.open and  # previous red candle
            latest.close > latest.open and  # current green candle
            latest.close >= prev.open and  # current closes above previous open
            latest.open <= prev.close):    # current opens below previous close
            patterns['bullish_engulfing'] = True
        
        # Bearish engulfing pattern
        if (prev.close > prev.open and  # previous green candle
            latest.close < latest.open and  # current red candle
            latest.open >= prev.close and  # current opens above previous close
            latest.close <= prev.open):    # current closes below previous open
            patterns['bearish_engulfing'] = True
        
        # Pin bar (reversal signal)
        body_size = abs(latest.close - latest.open)
        total_range = abs(latest.high - latest.low)
        upper_wick = max(latest.open, latest.close) - latest.high
        lower_wick = latest.low - min(latest.open, latest.close)
        
        if body_size != 0 and (abs(upper_wick) / body_size > 2 or abs(lower_wick) / body_size > 2):
            patterns['pin_bar'] = True
        
        return {
            'patterns': patterns,
            'indicators': {
                'ema_9': latest['ema_9'],
                'ema_21': latest['ema_21'],
                'rsi': latest['rsi'],
                'atr': latest['atr'],
                'volume_sma': latest['volume_sma'],
                'current_volume': latest['volume']
            },
            'trend': self.identify_trend(df),
            'momentum': latest['rsi'],
            'volatility': latest['atr']
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0.0
    
    def identify_trend(self, df: pd.DataFrame) -> str:
        """Identify current market trend"""
        ema_9 = df['ema_9'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        price = df['close'].iloc[-1]
        
        if ema_9 > ema_21 and price > ema_9:
            return 'bullish'
        elif ema_9 < ema_21 and price < ema_9:
            return 'bearish'
        else:
            return 'sideways'


class RiskManager:
    """Manages risk and position sizing"""
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.15):
        self.max_position_size = max_position_size  # 10% max per trade
        self.max_drawdown = max_drawdown  # 15% max drawdown
        self.current_capital = 10000.0  # Starting capital
        self.total_pnl = 0.0
    
    def calculate_position_size(self, price: float, stop_loss_pct: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        # Risk 2% of capital per trade maximum
        risk_amount = self.current_capital * 0.02
        price_risk = price * stop_loss_pct
        max_quantity = risk_amount / price_risk if price_risk > 0 else 0
        
        # Apply max position size constraint
        max_capital_for_trade = self.current_capital * self.max_position_size
        max_by_capital = max_capital_for_trade / price
        
        # Return minimum of both constraints
        return min(max_quantity, max_by_capital)
    
    def update_capital(self, pnl: float):
        """Update capital after a trade"""
        self.total_pnl += pnl
        self.current_capital += pnl


class ExchangeInterface:
    """Interface to interact with crypto exchanges"""
    
    def __init__(self, exchange_name: str = "simulated"):
        self.exchange_name = exchange_name
        self.balance = {"USDT": 10000.0}  # Starting balance
        self.positions = {}  # Current holdings
        self.order_history = []
    
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Get kline/candlestick data from exchange
        This is a simulated version - in production connect to exchange API
        """
        # Simulate getting real market data
        # In production, this would call exchange API like Binance, Coinbase, etc.
        
        # Create simulated data for demonstration
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price movements
        base_price = 50000  # Starting price around $50k
        returns = np.random.normal(0.0005, 0.02, limit)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create OHLCV data
        ohlcv_data = []
        for i, ts in enumerate(timestamps):
            price = prices[i]
            # Add some variation for OHLC
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            # Ensure proper OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            ohlcv_data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(100, 1000)  # Random volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        return df
    
    def place_order(self, symbol: str, side: OrderType, quantity: float, price: float) -> Dict:
        """Place buy/sell order on exchange"""
        order_id = f"ORDER_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Update balance based on order
        if side == OrderType.BUY:
            cost = quantity * price
            if self.balance.get("USDT", 0) >= cost:
                self.balance["USDT"] -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:
                logger.warning(f"Insufficient funds for buy order: {cost}")
                return {"success": False, "reason": "insufficient_funds"}
        elif side == OrderType.SELL:
            if self.positions.get(symbol, 0) >= quantity:
                revenue = quantity * price
                self.balance["USDT"] += revenue
                self.positions[symbol] -= quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
            else:
                logger.warning(f"Insufficient position for sell order: {quantity}")
                return {"success": False, "reason": "insufficient_position"}
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now(),
            "status": "filled"
        }
        
        self.order_history.append(order)
        logger.info(f"Order placed: {side.value} {quantity} {symbol} @ ${price:.2f}")
        
        return {"success": True, "order": order}


class ProfessionalCryptoTrader:
    """Main trading agent implementing the professional trading logic"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTCUSDT"]  # Default to Bitcoin
        self.news_analyzer = NewsSentimentAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager()
        self.exchange = ExchangeInterface()
        self.current_positions = {}
        self.trade_log = []
        
        logger.info("Professional Crypto Trading Agent initialized")
        logger.info(f"Trading symbols: {self.symbols}")
    
    def get_market_condition(self, symbol: str) -> MarketCondition:
        """Analyze current market condition for a symbol"""
        df = self.exchange.get_klines(symbol)
        
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return MarketCondition("unknown", 0, "unknown", 0, {})
        
        tech_analysis = self.tech_analyzer.analyze_candlestick_patterns(df)
        
        return MarketCondition(
            trend=tech_analysis['trend'],
            volatility=tech_analysis['volatility'],
            volume_trend="increasing",  # Simplified for demo
            momentum=tech_analysis['momentum'],
            support_resistance={}  # Would need additional analysis
        )
    
    def analyze_fundamentals(self, symbol: str) -> Dict:
        """Analyze fundamental factors and news sentiment"""
        return self.news_analyzer.get_news_sentiment(symbol)
    
    def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Generate a trade signal based on comprehensive analysis"""
        logger.info(f"Analyzing {symbol} for trade signal...")
        
        # Get market data
        df = self.exchange.get_klines(symbol)
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return None
        
        # Technical analysis
        tech_analysis = self.tech_analyzer.analyze_candlestick_patterns(df)
        current_price = df['close'].iloc[-1]
        
        # Fundamental analysis
        fundamental_analysis = self.analyze_fundamentals(symbol)
        
        # Risk assessment
        risk_assessment = {
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.05,  # 5% take profit
            "position_risk": "controlled",
            "market_risk": "normal"
        }
        
        # Decision logic based on combined analysis
        action = OrderType.HOLD
        reason = "No clear signal"
        confidence = 0.0
        quantity = 0.0
        
        # Check fundamental bias first
        sentiment_score = fundamental_analysis.get('sentiment_score', 0)
        
        if sentiment_score < -0.6:  # Strong negative sentiment
            # Prohibit buy, consider sell if holding
            if symbol in self.current_positions and self.current_positions[symbol] > 0:
                action = OrderType.SELL
                reason = "Strong negative sentiment detected, selling position"
                confidence = 0.8
                quantity = self.current_positions[symbol]
        
        elif sentiment_score > 0.6:  # Strong positive sentiment
            # Allow buy only with technical confirmation
            if tech_analysis['trend'] == 'bullish' and tech_analysis['momentum'] < 70:  # Not overbought
                action = OrderType.BUY
                reason = "Positive sentiment with bullish technical setup"
                confidence = 0.7
                quantity = self.risk_manager.calculate_position_size(current_price)
        
        else:  # Neutral sentiment, rely on technicals
            if tech_analysis['trend'] == 'bullish':
                if tech_analysis['momentum'] < 70:  # Not overbought
                    # Look for bullish patterns
                    if 'bullish_engulfing' in tech_analysis['patterns'] or tech_analysis['indicators']['ema_9'] > tech_analysis['indicators']['ema_21']:
                        action = OrderType.BUY
                        reason = "Bullish technical setup identified"
                        confidence = 0.6
                        quantity = self.risk_manager.calculate_position_size(current_price)
            
            elif tech_analysis['trend'] == 'bearish':
                if tech_analysis['momentum'] > 30:  # Not oversold
                    # Look for bearish patterns
                    if 'bearish_engulfing' in tech_analysis['patterns'] or tech_analysis['indicators']['ema_9'] < tech_analysis['indicators']['ema_21']:
                        if symbol in self.current_positions and self.current_positions[symbol] > 0:
                            action = OrderType.SELL
                            reason = "Bearish technical setup identified"
                            confidence = 0.6
                            quantity = self.current_positions[symbol]
        
        # High uncertainty check
        if abs(sentiment_score) < 0.2 and tech_analysis['trend'] == 'sideways':
            action = OrderType.HOLD
            reason = "High uncertainty, staying in cash"
            confidence = 0.1
            quantity = 0.0
        
        if action != OrderType.HOLD:
            return TradeSignal(
                action=action,
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                confidence=confidence,
                reason=reason,
                technical_analysis=tech_analysis,
                fundamental_analysis=fundamental_analysis,
                risk_assessment=risk_assessment
            )
        
        return None
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute the trade signal on the exchange"""
        logger.info(f"Executing trade: {signal.action.value} {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
        logger.info(f"Reason: {signal.reason}")
        logger.info(f"Confidence: {signal.confidence:.2f}")
        
        order_result = self.exchange.place_order(
            symbol=signal.symbol,
            side=signal.action,
            quantity=signal.quantity,
            price=signal.price
        )
        
        if order_result["success"]:
            # Update internal positions
            if signal.action == OrderType.BUY:
                self.current_positions[signal.symbol] = self.current_positions.get(signal.symbol, 0) + signal.quantity
            elif signal.action == OrderType.SELL:
                self.current_positions[signal.symbol] = max(0, self.current_positions.get(signal.symbol, 0) - signal.quantity)
                if self.current_positions[signal.symbol] == 0:
                    del self.current_positions[signal.symbol]
            
            # Log the trade
            trade_record = {
                "timestamp": datetime.now(),
                "action": signal.action.value,
                "symbol": signal.symbol,
                "quantity": signal.quantity,
                "price": signal.price,
                "reason": signal.reason,
                "confidence": signal.confidence
            }
            self.trade_log.append(trade_record)
            
            logger.info(f"Trade executed successfully: {signal.action.value} {signal.symbol}")
            return True
        else:
            logger.error(f"Trade execution failed: {order_result.get('reason')}")
            return False
    
    def log_decision_process(self, symbol: str, market_state: MarketCondition, 
                           signal: Optional[TradeSignal]):
        """Log the decision-making process for transparency"""
        log_entry = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "market_state": {
                "trend": market_state.trend,
                "volatility": market_state.volatility,
                "momentum": market_state.momentum
            },
            "action": signal.action.value if signal else "HOLD",
            "reason": signal.reason if signal else "No clear signal",
            "confidence": signal.confidence if signal else 0.0
        }
        
        logger.info(f"Decision Log: {json.dumps(log_entry, indent=2, default=str)}")
    
    def run_single_cycle(self):
        """Run one complete trading cycle"""
        logger.info("Starting trading cycle...")
        
        for symbol in self.symbols:
            try:
                # Analyze market condition
                market_condition = self.get_market_condition(symbol)
                
                # Generate trade signal
                signal = self.generate_trade_signal(symbol)
                
                # Log decision process
                self.log_decision_process(symbol, market_condition, signal)
                
                # Execute trade if signal generated
                if signal:
                    success = self.execute_trade(signal)
                    if success:
                        logger.info(f"Successfully executed trade for {symbol}")
                    else:
                        logger.error(f"Failed to execute trade for {symbol}")
                else:
                    logger.info(f"No trade signal for {symbol}, holding position")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Log current portfolio status
        logger.info(f"Current positions: {self.current_positions}")
        logger.info(f"Current balance: {self.exchange.balance}")
        
        logger.info("Trading cycle completed.\n")
    
    def run_continuous(self, interval_minutes: int = 60):
        """Run the trading agent continuously"""
        logger.info(f"Starting continuous trading mode, checking every {interval_minutes} minutes...")
        
        try:
            while True:
                self.run_single_cycle()
                logger.info(f"Sleeping for {interval_minutes} minutes before next cycle...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Trading agent stopped by user.")
        except Exception as e:
            logger.error(f"Unexpected error in continuous mode: {str(e)}")


def main():
    """Main entry point"""
    logger.info("Initializing Professional Crypto Trading Agent...")
    
    # Initialize the trading agent with desired symbols
    symbols_to_trade = ["BTCUSDT", "ETHUSDT"]  # Add more symbols as needed
    trader = ProfessionalCryptoTrader(symbols=symbols_to_trade)
    
    # Run in continuous mode (checking every hour)
    # For initial testing, you might want to use shorter intervals
    trader.run_continuous(interval_minutes=60)


if __name__ == "__main__":
    main()