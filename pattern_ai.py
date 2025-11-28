import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import ccxt
import warnings
warnings.filterwarnings("ignore")

class PatternMemory:
    def __init__(self, name="trade_patterns_v1"):
        self.name = name
        self.patterns = []  # لیست الگوها: {fingerprint, future_return, metadata}
        self.scaler = StandardScaler()
        self.nn = None  # Nearest Neighbors برای جستجوی شبیه‌ترین الگو

    # مرحله ۱: استخراج "سرنگشت الگو" از یک قطعه نمودار
    def encode_pattern(self, df_segment):
        """تبدیل یک تکه نمودار (مثلاً ۱۵ کندل) به یک بردار عددی ثابت (fingerprint)"""
        closes = df_segment['close'].values
        highs = df_segment['high'].values
        lows = df_segment['low'].values
        volumes = df_segment['volume'].values

        # Normalization نسبی: تبدیل به درصد نسبت به اولین قیمت
        base = closes[0]
        norm_close = (closes - base) / base
        norm_high = (highs - base) / base
        norm_low = (lows - base) / base
        norm_vol = volumes / volumes.mean() if volumes.mean() != 0 else np.ones_like(volumes)

        # ویژگی‌های استخراج‌شده:
        features = [
            # شکل کلی
            norm_close[-1],                         # بازگشت/ادامه؟
            np.max(norm_high),                      # سقف نسبی
            np.min(norm_low),                       # کف نسبی
            np.std(norm_close),                     # نوسان
            # RSI و EMA داخلی
            RSIIndicator(pd.Series(closes), 14).rsi().iloc[-1] / 100,  # نرمال‌شده
            (EMAIndicator(pd.Series(closes), 5).ema_indicator().iloc[-1] - base) / base,
            (EMAIndicator(pd.Series(closes), 20).ema_indicator().iloc[-1] - base) / base,
            # حجم
            norm_vol[-1],
            np.mean(norm_vol[-5:]),                 # حجم ۵ کندل اخیر
        ]
        return np.array(features, dtype=np.float32)

    # مرحله ۲: آموزش حافظه روی داده‌های تاریخی
    def learn_from_history(self, df, window_size=15, future_horizon=5):
        """
        df: داده تاریخی (OHLCV)
        window_size: طول الگو (مثلاً ۱۵ کندل)
        future_horizon: چند کندل جلوتر را بررسی کنیم (مثلاً ۵ کندل = ۲۰ ساعت برای 4h)
        """
        print(f"🧠 Training Pattern Memory on {len(df)} candles...")
        for i in range(window_size, len(df) - future_horizon):
            segment = df.iloc[i - window_size:i]
            future_price = df.iloc[i + future_horizon]['close']
            current_price = df.iloc[i - 1]['close']
            future_return = (future_price - current_price) / current_price  # درصد تغییر

            fingerprint = self.encode_pattern(segment)
            self.patterns.append({
                'fingerprint': fingerprint,
                'future_return': future_return,
                'timestamp': df.iloc[i]['timestamp'],
                'symbol': getattr(df, 'symbol', 'N/A'),
                'avg_volume': segment['volume'].mean(),
                'volatility': segment['high'].max() - segment['low'].min()
            })

        # آماده‌سازی جستجوی شبیه‌ترین الگو
        X = np.array([p['fingerprint'] for p in self.patterns])
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X_scaled)
        print(f"✅ Pattern Memory trained on {len(self.patterns)} historical patterns.")

    # مرحله ۳: پرس‌وجو — "این الگوی جدید شبیه چه چیزهایی توی حافظمه؟"
    def query(self, current_segment):
        fp = self.encode_pattern(current_segment).reshape(1, -1)
        fp_scaled = self.scaler.transform(fp)
        distances, indices = self.nn.kneighbors(fp_scaled)

        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            pat = self.patterns[idx]
            similar.append({
                'similarity': 1 - dist,  # هرچه نزدیک‌تر به ۱، شبیه‌تر
                'future_return': pat['future_return'],
                'timestamp': pat['timestamp'],
                'symbol': pat['symbol']
            })
        return similar

    # ذخیره/بارگذاری حافظه
    def save(self, path=None):
        path = path or f"{self.name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'patterns': self.patterns,
                'scaler': self.scaler,
                'nn': self.nn
            }, f)
        print(f"💾 Pattern Memory saved to {path}")

    def load(self, path=None):
        path = path or f"{self.name}.pkl"
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.patterns = data['patterns']
                self.scaler = data['scaler']
                self.nn = data['nn']
            print(f"📂 Pattern Memory loaded from {path} ({len(self.patterns)} patterns)")
            return True
        except:
            print("⚠️ No saved memory found. Training from scratch.")
            return False