import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
import logging
import sys
import io
import ccxt

# ———————————————————————————————————————————————————————————————
# 🔷 رفع مشکلات شبکه و کدگذاری
# ———————————————————————————————————————————————————————————————

# رفع مشکل کدگذاری در ویندوز
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# تنظیمات شبکه برای دسترسی به صرافی‌های ایرانی
socket.setdefaulttimeout(60)  # افزایش زمان تایم‌اوت

# ———————————————————————————————————————————————————————————————
# 🔷 کلاس جمع‌آوری داده‌های واقعی
# ———————————————————————————————————————————————————————————————

class IranianRealDataCollector:
    def __init__(self, output_dir="iranian_real_data"):
        self.output_dir = output_dir
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'sources': {},
            'collection_stats': {
                'successful_symbols': 0,
                'failed_symbols': 0,
                'total_candles': 0
            }
        }
        
        # تنظیمات APIهای قابل دسترس در ایران
        self.exchanges = {
            'kucoin': {
                'instance': ccxt.kucoin(),
                'rate_limit': 2.0,
                'enabled': True
            },
            'coinex': {
                'instance': ccxt.coinex(),
                'rate_limit': 2.5,
                'enabled': True
            },
            'nobitex': {
                'instance': ccxt.nobitex(),
                'rate_limit': 3.0,
                'enabled': False  # برای استفاده نیاز به API key دارد
            },
            'wallex': {
                'instance': ccxt.wallex(),
                'rate_limit': 3.0,
                'enabled': False  # برای استفاده نیاز به API key دارد
            }
        }
        
        # ایجاد پوشه‌های ضروری
        self._create_directories()
    
    def _create_directories(self):
        """ایجاد ساختار پوشه‌های استاندارد"""
        directories = [
            f"{self.output_dir}/crypto",
            f"{self.output_dir}/stocks",
            f"{self.output_dir}/forex",
            f"{self.output_dir}/metadata"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"✅ پوشه ایجاد شد: {directory}")
    
    # ———————————————————————————————————————————————————————————————
    # 🔷 جمع‌آوری داده‌های واقعی از صرافی‌های ایرانی
    # ———————————————————————————————————————————————————————————————

    def fetch_crypto_data(self, symbol, timeframe='4h', days=730):
        """جمع‌آوری داده‌های واقعی از صرافی‌های قابل دسترس"""
        logging.info(f"\n🔄 جمع‌آوری داده‌های واقعی برای {symbol} ({timeframe})")
        logging.info("🔄 در حال تلاش با صرافی‌های قابل دسترس در ایران...")
        
        all_data = []
        successful_exchange = None
        
        # تلاش با صرافی‌های مختلف به ترتیب اولویت
        for exchange_name, exchange_config in self.exchanges.items():
            if not exchange_config['enabled']:
                continue
                
            try:
                logging.info(f"   📥 دریافت داده از {exchange_name}...")
                
                # محاسبه تایم‌استمپ‌ها
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                since = int(start_time.timestamp() * 1000)
                
                # دریافت داده‌ها
                ohlcv = exchange_config['instance'].fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=10000
                )
                
                if ohlcv and len(ohlcv) > 100:
                    all_data = ohlcv
                    successful_exchange = exchange_name
                    logging.info(f"   ✅ {exchange_name}: {len(ohlcv)} کندل دریافت شد")
                    break
                
                time.sleep(exchange_config['rate_limit'])
                
            except Exception as e:
                logging.warning(f"   ⚠️ {exchange_name} خطا: {str(e)}")
                time.sleep(exchange_config['rate_limit'])
                continue
        
        if not all_
            logging.error(f"❌ هیچ داده‌ای برای {symbol} دریافت نشد")
            return None
        
        # ایجاد DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        logging.info(f"✅ داده‌های واقعی {symbol} آماده شدند:")
        logging.info(f"   - منبع: {successful_exchange}")
        logging.info(f"   - تعداد کندل‌ها: {len(df):,}")
        logging.info(f"   - بازه زمانی: {df['timestamp'].min()} تا {df['timestamp'].max()}")
        
        return df
    
    # ———————————————————————————————————————————————————————————————
    # 🔷 ذخیره داده‌های واقعی
    # ———————————————————————————————————————————————————————————————

    def save_real_data(self, df, symbol, market_type, source):
        """ذخیره داده‌های واقعی در فرمت‌های استاندارد"""
        try:
            # ساخت نام فایل ایمن
            safe_symbol = symbol.replace('/', '_').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d')
            base_filename = f"{safe_symbol}_{timestamp}"
            
            # مسیر ذخیره‌سازی
            market_dir = {
                'crypto': f"{self.output_dir}/crypto",
                'stocks': f"{self.output_dir}/stocks",
                'forex': f"{self.output_dir}/forex"
            }.get(market_type, f"{self.output_dir}/other")
            
            # 1. ذخیره Parquet
            parquet_path = f"{market_dir}/{base_filename}.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)
            
            # 2. ذخیره CSV
            csv_path = f"{market_dir}/{base_filename}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # به‌روزرسانی متادیتا
            file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            self.metadata['sources'][symbol] = {
                'exchange': source,
                'market_type': market_type,
                'parquet_path': parquet_path,
                'csv_path': csv_path,
                'file_size_mb': round(file_size_mb, 2),
                'candle_count': len(df),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            }
            
            logging.info(f"💾 {symbol} ذخیره شد:")
            logging.info(f"   - Parquet: {parquet_path} ({file_size_mb:.2f} MB)")
            logging.info(f"   - CSV: {csv_path}")
            logging.info(f"   - منبع: {source}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ خطای ذخیره‌سازی {symbol}: {str(e)}")
            return False
    
    # ———————————————————————————————————————————————————————————————
    # 🔷 جمع‌آوری کامل داده‌ها
    # ———————————————————————————————————————————————————————————————

    def collect_all_data(self):
        """جمع‌آوری داده‌های واقعی از صرافی‌های قابل دسترس در ایران"""
        logging.info("=" * 80)
        logging.info("🚀 شروع جمع‌آوری داده‌های واقعی از صرافی‌های قابل دسترس در ایران")
        logging.info("✅ استفاده از KuCoin و CoinEx به عنوان منابع اصلی")
        logging.info("✅ بدون نیاز به VPN یا دور زدن محدودیت‌ها")
        logging.info("=" * 80)
        
        # نمادهای پیشنهادی
        SYMBOLS_CONFIG = {
            'crypto': [
                'BTC/USDT',
                'ETH/USDT', 
                'SOL/USDT',
                'BNB/USDT',
                'ADA/USDT',
                'XRP/USDT'
            ]
        }
        
        for market_type, symbols in SYMBOLS_CONFIG.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"📈 بازار: {market_type.upper()}")
            logging.info(f"{'='*60}")
            
            for symbol in symbols:
                try:
                    logging.info(f"\n🔍 پردازش نماد: {symbol}")
                    
                    if market_type == 'crypto':
                        df = self.fetch_crypto_data(symbol, timeframe='4h', days=730)
                    else:
                        logging.warning(f"⚠️ نوع بازار پشتیبانی نشده: {market_type}")
                        continue
                    
                    if df is not None and not df.empty:
                        if self.save_real_data(df, symbol, market_type, 'kucoin_or_coinex'):
                            self.metadata['collection_stats']['successful_symbols'] += 1
                            self.metadata['collection_stats']['total_candles'] += len(df)
                    else:
                        self.metadata['collection_stats']['failed_symbols'] += 1
                        logging.error(f"❌ شکست در جمع‌آوری داده برای {symbol}")
                    
                    # تأخیر بین درخواست‌ها برای جلوگیری از محدودیت
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    logging.info("\n🛑 عملیات توسط کاربر متوقف شد.")
                    return
                except Exception as e:
                    logging.error(f"❌ خطای غیرمنتظره در {symbol}: {str(e)}")
                    self.metadata['collection_stats']['failed_symbols'] += 1
        
        # ذخیره متادیتا
        self._save_metadata()
        
        logging.info(f"\n{'='*80}")
        logging.info("🎉 جمع‌آوری داده‌های واقعی با موفقیت به پایان رسید!")
        logging.info(f"   - نمادهای موفق: {self.metadata['collection_stats']['successful_symbols']}")
        logging.info(f"   - نمادهای شکست‌خورده: {self.metadata['collection_stats']['failed_symbols']}")
        logging.info(f"   - کل کندل‌ها: {self.metadata['collection_stats']['total_candles']:,}")
        logging.info(f"   - داده‌ها در: {os.path.abspath(self.output_dir)}")
        logging.info(f"{'='*80}")
    
    def _save_metadata(self):
        """ذخیره متادیتا و گزارش‌ها"""
        try:
            metadata_path = f"{self.output_dir}/metadata/dataset_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            logging.info(f"📄 متادیتا ذخیره شد: {metadata_path}")
            
        except Exception as e:
            logging.error(f"❌ خطای ذخیره متادیتا: {str(e)}")

# ———————————————————————————————————————————————————————————————
# ▶️ اجرای کد
# ———————————————————————————————————————————————————————————————

if __name__ == "__main__":
    # تنظیمات لاگینگ
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('real_data_collection.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print("=" * 80)
    print("=== سیستم جمع‌آوری داده‌های واقعی از صرافی‌های ایرانی ===")
    print("✅ استفاده از KuCoin و CoinEx به عنوان منابع اصلی")
    print("✅ بدون نیاز به VPN یا دور زدن محدودیت‌ها")
    print("✅ داده‌های واقعی و کیفیت‌دار برای آموزش مدل‌های هوش مصنوعی")
    print("=" * 80)
    
    # ایجاد و اجرای سیستم
    collector = IranianRealDataCollector(output_dir="iranian_real_data")
    collector.collect_all_data()
    
    print("\n✅ سیستم با موفقیت اجرا شد!")
    print("💡 داده‌های واقعی آماده‌اند:")
    print("   - می‌توانید این داده‌ها را مستقیماً در Google Colab بارگذاری کنید")
    print("   - برای آموزش مدل‌های LSTM و Transformer استفاده کنید")
    print("   - سیگنال‌های معاملاتی دقیق تولید کنید")