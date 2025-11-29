import logging
from data_collector.crypto_collector import CryptoCollector
from data_collector.metadata_manager import MetadataManager
from data_collector.utils import setup_logging


if __name__ == "__main__":
    # تنظیم سیستم لاگینگ
    setup_logging()

    # ایجاد مدیر متادیتا
    metadata = MetadataManager()

    # ایجاد جمع‌آور داده‌های کریپتو
    collector = CryptoCollector(
        output_dir="datasets/crypto",
        metadata_manager=metadata
    )

    # لیست نمادهای مورد نظر
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT"
    ]

    # جمع‌آوری داده‌ها
    collector.collect(symbols, timeframe="1h", days=700)
    
    logging.info("✅ فرآیند جمع‌آوری داده‌ها با موفقیت انجام شد!")