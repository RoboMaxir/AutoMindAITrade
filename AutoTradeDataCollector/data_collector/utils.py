import os
import logging
import pandas as pd
from datetime import datetime


def setup_logging():
    """
    تنظیم سیستم لاگینگ
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )


def validate_symbol(symbol: str) -> bool:
    """
    اعتبارسنجی نماد معاملاتی
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # بررسی فرمت نماد (مثلاً BTC/USDT)
    parts = symbol.split("/")
    if len(parts) != 2:
        return False
    
    base, quote = parts
    if not base or not quote:
        return False
    
    # بررسی اینکه نام نماد فقط شامل حروف بزرگ انگلیسی باشد
    if not (base.isalpha() and quote.isalpha()):
        return False
    
    return True


def clean_symbol(symbol: str) -> str:
    """
    پاک‌سازی نماد معاملاتی
    """
    return symbol.replace("/", "_").replace(":", "_")


def ensure_directory_exists(path: str):
    """
    اطمینان از وجود دایرکتوری
    """
    os.makedirs(path, exist_ok=True)


def save_parquet(df: pd.DataFrame, filepath: str):
    """
    ذخیره داده‌ها در فرمت Parquet
    """
    df.to_parquet(filepath, compression="snappy", index=False)


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    بارگذاری داده‌ها از فرمت Parquet
    """
    return pd.read_parquet(filepath)


def get_current_time_iso():
    """
    دریافت زمان فعلی در فرمت ISO
    """
    return datetime.now().isoformat()


def calculate_time_difference(start_time: datetime, end_time: datetime):
    """
    محاسبه تفاوت زمانی
    """
    return end_time - start_time