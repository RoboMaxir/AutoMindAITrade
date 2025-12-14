"""
Technical Analysis Agent
Analyzes market structure, price action, and technical indicators to assess market state
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Import technical analysis libraries
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("TA-Lib not installed, using custom implementations")

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent:
    """
    Analyzes market structure and technical indicators to evaluate current market state
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_period = config.get('lookback_period', 200)
        self.signal_threshold = config.get('signal_threshold', 0.6)
        
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market structure: trend, range, accumulation, distribution
        """
        # Calculate key technical indicators
        df_copy = df.copy()
        
        # Moving averages for trend identification
        df_copy['ema_8'] = self.calculate_ema(df_copy['close'], 8)
        df_copy['ema_21'] = self.calculate_ema(df_copy['close'], 21)
        df_copy['ema_55'] = self.calculate_ema(df_copy['close'], 55)
        
        # RSI for momentum
        df_copy['rsi'] = self.calculate_rsi(df_copy['close'])
        
        # ATR for volatility
        df_copy['atr'] = self.calculate_atr(df_copy['high'], df_copy['low'], df_copy['close'])
        
        # MACD for momentum trend
        df_copy['macd'], df_copy['macd_signal'], df_copy['macd_hist'] = self.calculate_macd(df_copy['close'])
        
        # Bollinger Bands for range identification
        df_copy['bb_upper'], df_copy['bb_middle'], df_copy['bb_lower'] = self.calculate_bollinger_bands(df_copy['close'])
        
        # Volume analysis
        df_copy['volume_sma'] = df_copy['volume'].rolling(window=20).mean()
        df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_sma']
        
        # Determine market structure
        current_price = df_copy['close'].iloc[-1]
        ema_8 = df_copy['ema_8'].iloc[-1]
        ema_21 = df_copy['ema_21'].iloc[-1]
        ema_55 = df_copy['ema_55'].iloc[-1]
        
        rsi_current = df_copy['rsi'].iloc[-1]
        volume_ratio = df_copy['volume_ratio'].iloc[-1]
        
        # Trend analysis
        trend_strength = self._calculate_trend_strength(df_copy)
        trend_direction = self._determine_trend_direction(df_copy)
        
        # Range analysis
        is_in_range = self._is_price_in_range(current_price, df_copy['bb_upper'].iloc[-1], df_copy['bb_lower'].iloc[-1])
        
        # Breakout analysis
        breakout_signal = self._analyze_breakout_signals(df_copy)
        
        # Market structure assessment
        market_state = self._assess_market_structure(
            trend_direction, trend_strength, is_in_range, 
            rsi_current, volume_ratio, breakout_signal
        )
        
        return {
            'market_state': market_state,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'is_in_range': is_in_range,
            'rsi_level': rsi_current,
            'volume_confirmation': volume_ratio > 1.0,
            'breakout_signal': breakout_signal,
            'support_resistance': self._identify_support_resistance(df_copy),
            'volatility_regime': self._assess_volatility_regime(df_copy),
            'confidence': self._calculate_technical_confidence(df_copy)
        }
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength based on moving average alignment"""
        ema_8 = df['ema_8'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_55 = df['ema_55'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Check alignment of EMAs
        aligned_bullish = ema_8 > ema_21 > ema_55 and close > ema_55
        aligned_bearish = ema_8 < ema_21 < ema_55 and close < ema_55
        
        if aligned_bullish:
            # Calculate strength based on how far apart the EMAs are
            strength = min(1.0, abs((ema_8 - ema_55) / ema_55) * 10)
            return strength
        elif aligned_bearish:
            strength = min(1.0, abs((ema_55 - ema_8) / ema_55) * 10)
            return -strength
        else:
            # Mixed signals - weaker trend
            if ema_8 > ema_21:
                return min(0.5, abs((ema_8 - ema_21) / ema_21) * 5)
            else:
                return max(-0.5, -abs((ema_21 - ema_8) / ema_8) * 5)
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction based on recent price action"""
        recent_prices = df['close'].tail(10)
        
        # Compare recent highs and lows
        recent_high = recent_prices.max()
        recent_low = recent_prices.min()
        current_price = df['close'].iloc[-1]
        
        # Use EMA crossover logic as well
        ema_8 = df['ema_8'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        
        if current_price > ema_8 > ema_21 and recent_high > df['close'].iloc[-5]:
            return 'bullish'
        elif current_price < ema_8 < ema_21 and recent_low < df['close'].iloc[-5]:
            return 'bearish'
        else:
            return 'sideways'
    
    def _is_price_in_range(self, price: float, bb_upper: float, bb_lower: float) -> bool:
        """Check if price is in ranging mode"""
        return bb_lower <= price <= bb_upper
    
    def _analyze_breakout_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze potential breakout signals"""
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Check for breakouts with volume confirmation
        breakout_up = current_price > bb_upper and volume_ratio > 1.5
        breakdown_down = current_price < bb_lower and volume_ratio > 1.5
        
        # Check for fake breakouts (no volume confirmation)
        fake_breakout_up = current_price > bb_upper and volume_ratio <= 1.0
        fake_breakdown_down = current_price < bb_lower and volume_ratio <= 1.0
        
        return {
            'breakout_up': breakout_up,
            'breakdown_down': breakdown_down,
            'fake_breakout_up': fake_breakout_up,
            'fake_breakdown_down': fake_breakdown_down,
            'volume_confirmed': volume_ratio > 1.0
        }
    
    def _assess_market_structure(self, trend_direction: str, trend_strength: float, 
                                is_in_range: bool, rsi: float, volume_ratio: float, 
                                breakout_signal: Dict) -> str:
        """Assess overall market structure"""
        if is_in_range:
            if rsi > 70:
                return 'distribution_zone'
            elif rsi < 30:
                return 'accumulation_zone'
            else:
                return 'range_bound'
        elif trend_direction == 'bullish':
            if trend_strength > 0.7:
                return 'strong_bull_trend'
            else:
                return 'weak_bull_trend'
        elif trend_direction == 'bearish':
            if trend_strength < -0.7:
                return 'strong_bear_trend'
            else:
                return 'weak_bear_trend'
        else:
            return 'transitional'
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        # Simple method: find local min/max in recent data
        recent_data = df.tail(50)
        
        # Find local minima (support) and maxima (resistance)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Use rolling windows to identify swing points
        resistance = highs.rolling(window=5, center=True).max().dropna().unique()
        support = lows.rolling(window=5, center=True).min().dropna().unique()
        
        current_price = df['close'].iloc[-1]
        
        # Get nearest support and resistance
        nearest_support = max([s for s in support if s < current_price], default=None)
        nearest_resistance = min([r for r in resistance if r > current_price], default=None)
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'supports': sorted([s for s in support if s < current_price][-3:]),
            'resistances': sorted([r for r in resistance if r > current_price][:3])
        }
    
    def _assess_volatility_regime(self, df: pd.DataFrame) -> str:
        """Assess current volatility regime"""
        atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(window=50).mean().iloc[-1]
        
        ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        if ratio > 1.5:
            return 'high_volatility'
        elif ratio < 0.7:
            return 'low_volatility'
        else:
            return 'normal_volatility'
    
    def _calculate_technical_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in technical analysis"""
        # Multiple confirming factors increase confidence
        confidences = []
        
        # Trend alignment
        ema_8 = df['ema_8'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_55 = df['ema_55'].iloc[-1]
        close = df['close'].iloc[-1]
        
        trend_alignment = (ema_8 > ema_21 > ema_55 and close > ema_55) or (ema_8 < ema_21 < ema_55 and close < ema_55)
        confidences.append(0.8 if trend_alignment else 0.3)
        
        # Momentum confirmation
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        
        if (close > ema_55 and rsi > 50 and macd > macd_signal) or \
           (close < ema_55 and rsi < 50 and macd < macd_signal):
            confidences.append(0.9)
        else:
            confidences.append(0.5)
        
        # Volume confirmation
        volume_ratio = df['volume_ratio'].iloc[-1]
        confidences.append(min(1.0, volume_ratio * 0.6))
        
        return sum(confidences) / len(confidences)
    
    def generate_signal(self, market_analysis: Dict) -> Dict:
        """
        Generate buy/sell/hold signal based on market analysis
        Returns: dict with 'action' and 'confidence' keys
        """
        market_state = market_analysis['market_state']
        trend_direction = market_analysis['trend_direction']
        rsi_level = market_analysis['rsi_level']
        volume_confirmed = market_analysis['volume_confirmation']
        breakout_signal = market_analysis['breakout_signal']
        confidence = market_analysis['confidence']
        
        # Logic for generating signals
        if market_state in ['strong_bull_trend', 'accumulation_zone']:
            if trend_direction == 'bullish' and rsi_level < 70:  # Not overbought
                if volume_confirmed or not breakout_signal['fake_breakout_up']:
                    action = 'BUY'
                else:
                    action = 'HOLD'  # Fake breakout
            else:
                action = 'HOLD'
        elif market_state in ['strong_bear_trend', 'distribution_zone']:
            if trend_direction == 'bearish' and rsi_level > 30:  # Not oversold
                if volume_confirmed or not breakout_signal['fake_breakdown_down']:
                    action = 'SELL'
                else:
                    action = 'HOLD'  # Fake breakdown
            else:
                action = 'HOLD'
        elif market_state in ['range_bound', 'transitional']:
            # More conservative in ranging markets
            if rsi_level < 30:  # Oversold
                action = 'BUY'
            elif rsi_level > 70:  # Overbought
                action = 'SELL'
            else:
                action = 'HOLD'
        else:
            action = 'HOLD'
        
        # Adjust confidence based on market conditions
        adjusted_confidence = confidence
        
        # Lower confidence in transitional/range-bound markets
        if market_state in ['range_bound', 'transitional']:
            adjusted_confidence *= 0.7
            
        # Increase confidence in strong trending markets
        if market_state in ['strong_bull_trend', 'strong_bear_trend']:
            adjusted_confidence = min(1.0, adjusted_confidence * 1.2)
        
        return {
            'action': action,
            'confidence': max(0.0, min(1.0, adjusted_confidence)),
            'reasoning': f"Market state: {market_state}, Trend: {trend_direction}, RSI: {rsi_level:.2f}"
        }


def main():
    """Test the Technical Analysis Agent"""
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    # Create realistic OHLCV data
    prices = 40000 + np.cumsum(np.random.randn(200) * 100)  # Starting around $40k
    high = prices + np.random.rand(200) * 150
    low = prices - np.random.rand(200) * 150
    close = prices + np.random.randn(200) * 50
    volume = np.random.randint(100, 1000, size=200) * 1000
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.rand(200) * 50,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Initialize agent
    config = {
        'lookback_period': 200,
        'signal_threshold': 0.6
    }
    
    agent = TechnicalAnalysisAgent(config)
    
    # Run analysis
    analysis = agent.analyze_market_structure(df)
    signal = agent.generate_signal(analysis)
    
    print("Technical Analysis Results:")
    print(f"Market State: {analysis['market_state']}")
    print(f"Trend Direction: {analysis['trend_direction']}")
    print(f"Trend Strength: {analysis['trend_strength']:.2f}")
    print(f"RSI Level: {analysis['rsi_level']:.2f}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print("\nSignal:")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Reasoning: {signal['reasoning']}")


if __name__ == "__main__":
    main()