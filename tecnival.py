import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import ccxt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import warnings
warnings.filterwarnings("ignore")

# ————————————————————————————————————————————————————————
# 🔹 1. تنظیمات سیستم
# ————————————————————————————————————————————————————————

ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "4h"
LOOKBACK = 60  # تعداد کندل‌های قبلی برای آموزش LSTM
PREDICT_HORIZON = 6  # پیش‌بینی ۶ کندل (۲۴ ساعت برای 4h)

# ————————————————————————————————————————————————————————
# 🔹 2. کلاس LSTM برای پیش‌بینی قیمت
# ————————————————————————————————————————————————————————

class LSTM_Price_Predictor:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def create_features(self, df):
        """استخراج ویژگی‌های پیشرفته برای LSTM"""
        df = df.copy()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        df['macd'] = MACD(df['close']).macd()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['volume_ma']
        return df[['close', 'rsi', 'ema20', 'ema50', 'macd', 'atr', 'vol_ratio']].dropna()

    def prepare_data(self, df):
        features = self.create_features(df)
        X, y = [], []
        for i in range(LOOKBACK, len(features) - PREDICT_HORIZON):
            X.append(features.iloc[i-LOOKBACK:i].values)
            future_close = features.iloc[i + PREDICT_HORIZON]['close']
            y.append([future_close])
        X, y = np.array(X), np.array(y)
        self.scaler_X.fit(X.reshape(-1, X.shape[-1]))
        self.scaler_y.fit(y)
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler_y.transform(y)
        return X_scaled, y_scaled

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, df, epochs=50, verbose=0):
        X, y = self.prepare_data(df)
        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=verbose)
        loss = self.model.evaluate(X, y, verbose=0)
        print(f"✅ LSTM Training Loss: {loss:.6f} | MAE: {np.sqrt(loss):.4f}")

    def predict(self, df):
        features = self.create_features(df)
        recent = features.iloc[-LOOKBACK:].values
        recent_scaled = self.scaler_X.transform(recent)
        X_input = recent_scaled.reshape(1, LOOKBACK, -1)
        pred_scaled = self.model.predict(X_input, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0][0]
        current = features.iloc[-1]['close']
        direction = "UP" if pred > current else "DOWN"
        confidence = abs(pred - current) / current
        return pred, direction, confidence

# ————————————————————————————————————————————————————————
# 🔹 3. تحلیل الگوهای تکنیکال هوشمند
# ————————————————————————————————————————————————————————

def detect_patterns(df):
    """تشخیص الگوهای کلاسیک بدون نیاز به تصویر (rule-based + ML-like)"""
    patterns = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # 🔍 الگوی سر و شانه (ساده‌شده)
    # پیدا کردن peakها و valleyها
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(closes, distance=10)
    valleys, _ = find_peaks(-closes, distance=10)

    # الگوی دوقلو بالا (Double Top)
    if len(peaks) >= 2:
        p1, p2 = closes[peaks[-2]], closes[peaks[-1]]
        if abs(p1 - p2) / p1 < 0.02 and peaks[-1] - peaks[-2] > 5:
            patterns.append({"name": "Double Top", "confidence": 0.85, "type": "BEARISH"})

    # الگوی دوقلو پایین (Double Bottom)
    if len(valleys) >= 2:
        v1, v2 = closes[valleys[-2]], closes[valleys[-1]]
        if abs(v1 - v2) / v1 < 0.02 and valleys[-1] - valleys[-2] > 5:
            patterns.append({"name": "Double Bottom", "confidence": 0.82, "type": "BULLISH"})

    # مثلث نزولی (Lower Highs + Flat Low)
    if len(peaks) >= 3 and len(valleys) >= 3:
        recent_peaks = [closes[i] for i in peaks[-3:]]
        recent_valleys = [closes[i] for i in valleys[-3:]]
        if (recent_peaks[0] > recent_peaks[1] > recent_peaks[2] and
            abs(recent_valleys[0] - recent_valleys[1]) < 0.01 * recent_valleys[0]):
            patterns.append({"name": "Descending Triangle", "confidence": 0.78, "type": "BEARISH"})

    return patterns

# ————————————————————————————————————————————————————————
# 🔹 4. کلاس اسکنر نهایی
# ————————————————————————————————————————————————————————

class DeepTrade_AI:
    def __init__(self, exchange_name="wazirx"):
        self.exchange = getattr(ccxt, exchange_name)()
        self.lstm = LSTM_Price_Predictor()
        self.signals = []

    def fetch_data(self, symbol, limit=200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)
            return df
        except Exception as e:
            print(f"❌ {symbol}: {e}")
            return None

    def analyze_asset(self, symbol):
        print(f"\n🧠 Analyzing {symbol}...")
        df = self.fetch_data(symbol)
        if df is None or len(df) < LOOKBACK + 10:
            return None

        # مرحله ۱: آموزش و پیش‌بینی LSTM
        print("   → Training LSTM...")
        self.lstm.train(df, epochs=30, verbose=0)
        pred_price, direction, confidence_lstm = self.lstm.predict(df)
        current_price = df['close'].iloc[-1]
        pct_change = (pred_price - current_price) / current_price * 100

        # مرحله ۲: تشخیص الگوها
        patterns = detect_patterns(df)
        pattern_signal = None
        pattern_conf = 0.0
        if patterns:
            best = max(patterns, key=lambda x: x['confidence'])
            pattern_signal = best['type']
            pattern_conf = best['confidence']
            print(f"   → Pattern Detected: {best['name']} ({best['type']}) | Conf: {pattern_conf:.2f}")

        # مرحله ۳: تحلیل تکنیکال کلاسیک
        rsi = RSIIndicator(df['close'], 14).rsi().iloc[-1]
        ema20 = EMAIndicator(df['close'], 20).ema_indicator().iloc[-1]
        ema50 = EMAIndicator(df['close'], 50).ema_indicator().iloc[-1]
        trend = "BULLISH" if ema20 > ema50 else "BEARISH"

        # مرحله ۴: ترکیب سیگنال‌ها (Confluence)
        signal = None
        confidence = 0.0

        # قوانین تصمیم‌گیری هوشمند:
        if direction == "UP" and trend == "BULLISH" and rsi < 65:
            signal = "BUY"
            confidence = (confidence_lstm * 0.5) + (pattern_conf * 0.3) + (0.2 if rsi < 55 else 0.1)
        elif direction == "DOWN" and trend == "BEARISH" and rsi > 35:
            signal = "SELL"
            confidence = (confidence_lstm * 0.5) + (pattern_conf * 0.3) + (0.2 if rsi > 45 else 0.1)

        if signal and confidence > 0.6:
            return {
                "symbol": symbol,
                "signal": signal,
                "confidence": round(confidence, 2),
                "price": round(current_price, 2),
                "pred_price": round(pred_price, 2),
                "pct_change": round(pct_change, 2),
                "rsi": round(rsi, 1),
                "trend": trend,
                "pattern": patterns[0]['name'] if patterns else "None"
            }
        return None

    def run(self):
        print("🚀 DeepTrade AI — Deep Learning Market Scanner")
        for symbol in ASSETS:
            result = self.analyze_asset(symbol)
            if result:
                self.signals.append(result)
                print(f"✅ {symbol} → {result['signal']} | {result['confidence']:.2f} | Δ{result['pct_change']:+.2f}%")
            time.sleep(1.5)  # جلوگیری از محدودیت API
        return self.signals

# ————————————————————————————————————————————————————————
# 🔹 5. اجرای سیستم
# ————————————————————————————————————————————————————————

if __name__ == "__main__":
    scanner = DeepTrade_AI(exchange_name="wazirx")
    signals = scanner.run()

    print("\n" + "="*90)
    print("📊 DEEP LEARNING SIGNALS REPORT")
    print("="*90)
    if signals:
        report = pd.DataFrame(signals)
        print(report.to_string(index=False, float_format="%.2f"))
    else:
        print("❌ No high-confidence signals generated.")
    print("="*90)