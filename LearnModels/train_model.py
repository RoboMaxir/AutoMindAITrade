import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ———————————————————————————————————————————————————————————————
# 🔷 بارگذاری داده‌های محلی (بدون نیاز به اینترنت)
# ———————————————————————————————————————————————————————————————

def load_local_data(symbol="BTC/USDT", data_dir="historical_data"):
    """بارگذاری داده‌ها از فایل‌های CSV ذخیره‌شده"""
    filename = f"{symbol.replace('/', '_')}_4h_3years.csv"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"فایل داده یافت نشد: {filepath}")
    
    print(f"📥 بارگذاری داده‌ها از: {filepath}")
    df = pd.read_csv(filepath)
    
    # تبدیل timestamp به datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ داده‌ها بارگذاری شدند:")
    print(f"   - تعداد کندل‌ها: {len(df):,}")
    print(f"   - بازه زمانی: {df['timestamp'].min()} تا {df['timestamp'].max()}")
    print(f"   - آخرین قیمت: {df['close'].iloc[-1]:,.2f} USDT")
    
    return df

# ———————————————————————————————————————————————————————————————
# 🔷 پیش‌پردازش داده‌ها
# ———————————————————————————————————————————————————————————————

def preprocess_data(df):
    """ایجاد ویژگی‌های تکنیکال و آماده‌سازی برای LSTM"""
    print("\n🔧 پیش‌پردازش داده‌ها...")
    
    # ایجاد کپی برای جلوگیری از تغییر داده اصلی
    data = df.copy()
    
    # محاسبه اندیکاتورهای تکنیکال
    data['rsi'] = RSIIndicator(data['close'], window=14).rsi()
    data['ema20'] = EMAIndicator(data['close'], window=20).ema_indicator()
    data['ema50'] = EMAIndicator(data['close'], window=50).ema_indicator()
    data['macd'] = (EMAIndicator(data['close'], window=12).ema_indicator() - 
                   EMAIndicator(data['close'], window=26).ema_indicator())
    data['atr'] = AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['return'] = data['close'].pct_change()
    
    # حذف ردیف‌های با داده‌های مفقوده
    data = data.dropna()
    
    print(f"✅ ویژگی‌های تکنیکال ایجاد شدند:")
    print(f"   - ویژگی‌های اضافه‌شده: RSI, EMA20/50, MACD, ATR, Volume Ratio")
    print(f"   - تعداد نمونه‌های نهایی: {len(data):,}")
    
    return data

# ———————————————————————————————————————————————————————————————
# 🔷 آماده‌سازی داده برای LSTM
# ———————————————————————————————————————————————————————————————

def prepare_lstm_data(data, lookback=60, predict_horizon=5):
    """
    آماده‌سازی داده برای آموزش LSTM
    lookback: تعداد کندل‌های گذشته برای پیش‌بینی
    predict_horizon: پیش‌بینی چند کندل آینده
    """
    print(f"\n🧠 آماده‌سازی داده برای LSTM (lookback={lookback}, horizon={predict_horizon})...")
    
    # انتخاب ویژگی‌های ورودی
    feature_cols = ['close', 'rsi', 'ema20', 'ema50', 'macd', 'atr', 'volume_ratio', 'return']
    
    # مقیاس‌گذاری داده‌ها
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols])
    
    X, y = [], []
    
    for i in range(lookback, len(scaled_data) - predict_horizon):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i + predict_horizon][0])  # فقط قیمت close
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ داده‌ها آماده شدند:")
    print(f"   - شکل X: {X.shape}")
    print(f"   - شکل y: {y.shape}")
    
    return X, y, scaler

# ———————————————————————————————————————————————————————————————
# 🔷 ساخت و آموزش مدل LSTM
# ———————————————————————————————————————————————————————————————

def build_and_train_lstm(X, y, symbol="BTC/USDT"):
    """ساخت و آموزش مدل LSTM با Early Stopping"""
    print(f"\n⚡️ ساخت و آموزش مدل LSTM برای {symbol}...")
    
    # تقسیم داده‌ها به آموزش و اعتبارسنجی
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"📊 تقسیم داده‌ها:")
    print(f"   - آموزش: {len(X_train):,} نمونه")
    print(f"   - اعتبارسنجی: {len(X_val):,} نمونه")
    
    # ساخت مدل
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks برای ذخیره بهترین مدل و توقف زودهنگام
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'model_{symbol.replace("/", "_")}_best.h5', save_best_only=True, verbose=1)
    ]
    
    # آموزش مدل
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # ارزیابی مدل
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n✅ آموزش کامل شد!")
    print(f"   - خطای آموزش (MAE): {train_mae:.4f}")
    print(f"   - خطای اعتبارسنجی (MAE): {val_mae:.4f}")
    
    return model, history

# ———————————————————————————————————————————————————————————————
# 🔷 ترسیم نمودارهای آموزش
# ———————————————————————————————————————————————————————————————

def plot_training_history(history, symbol="BTC/USDT"):
    """ترسیم نمودارهای آموزش برای تحلیل عملکرد"""
    plt.figure(figsize=(12, 6))
    
    # نمودار loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='آموزش')
    plt.plot(history.history['val_loss'], label='اعتبارسنجی')
    plt.title(f'Loss - {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # نمودار MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='آموزش')
    plt.plot(history.history['val_mae'], label='اعتبارسنجی')
    plt.title(f'MAE - {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plot_path = f'training_history_{symbol.replace("/", "_")}.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"📈 نمودارهای آموزش ذخیره شدند: {plot_path}")
    return plot_path

# ———————————————————————————————————————————————————————————————
# ▶️ اجرای کامل آموزش
# ———————————————————————————————————————————————————————————————

if __name__ == "__main__":
    # تنظیمات
    SYMBOL = "BTC/USDT"
    DATA_DIR = "historical_data"
    
    print(f"{'='*80}")
    print("🚀 آموزش مدل هوش مصنوعی برای ترید خودکار")
    print(f"   - نماد: {SYMBOL}")
    print(f"   - داده‌ها: {DATA_DIR}")
    print(f"{'='*80}")
    
    # ایجاد پوشه‌های خروجی
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    try:
        # 1. بارگذاری داده‌ها
        df = load_local_data(SYMBOL, DATA_DIR)
        
        # 2. پیش‌پردازش
        processed_data = preprocess_data(df)
        
        # 3. آماده‌سازی داده برای LSTM
        X, y, scaler = prepare_lstm_data(processed_data)
        
        # 4. آموزش مدل
        model, history = build_and_train_lstm(X, y, SYMBOL)
        
        # 5. ترسیم نمودارها
        plot_path = plot_training_history(history, SYMBOL)
        
        # 6. ذخیره اسکیلر
        import pickle
        scaler_path = f'models/scaler_{SYMBOL.replace("/", "_")}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 اسکیلر ذخیره شد: {scaler_path}")
        
        # 7. خلاصه نهایی
        print(f"\n{'='*80}")
        print("🎉 آموزش با موفقیت به پایان رسید!")
        print(f"   - مدل ذخیره شد: model_{SYMBOL.replace('/', '_')}_best.h5")
        print(f"   - نمودارها: {plot_path}")
        print(f"   - اسکیلر: {scaler_path}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ خطای کامل: {str(e)}")
        print("💡 راهکارهای پیشنهادی:")
        print("   - بررسی وجود فایل داده در پوشه historical_data")
        print("   - نصب کتابخانه‌های مورد نیاز: pip install tensorflow ta scikit-learn matplotlib")