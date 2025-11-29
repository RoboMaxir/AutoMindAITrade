import os
import time
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import ccxt


class CryptoCollector:
    """
    سیستم حرفه‌ای جمع‌آوری دیتای بازار کریپتو
    """

    def __init__(self, output_dir="datasets/crypto", metadata_manager=None):

        self.output_dir = output_dir
        self.metadata = metadata_manager
        os.makedirs(self.output_dir, exist_ok=True)

        # صرافی‌ها
        self.exchanges = {
            "kucoin": ccxt.kucoin(),
            "coinex": ccxt.coinex(),
        }

    # ----------------------------------------------------------------------
    def fetch(self, symbol: str, timeframe="1h", days=365):
        """
        دریافت کندل‌های واقعی از چند صرافی مختلف
        """

        logging.info(f"🔍 دریافت دیتا برای: {symbol} — {timeframe}")

        all_data = None
        used_exchange = None

        for name, ex in self.exchanges.items():
            try:
                logging.info(f"📡 تلاش با {name} ...")

                since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

                ohlcv = ex.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=10000
                )

                if ohlcv and len(ohlcv) > 50:
                    used_exchange = name
                    all_data = ohlcv
                    break

            except Exception as e:
                logging.warning(f"⚠️ خطا در {name}: {e}")
                time.sleep(1)
                continue

        if all_data is None:
            logging.error(f"❌ دریافت نشد: {symbol}")
            return None, None

        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol

        logging.info(f"✅ دریافت شد: {symbol} — {len(df)} کندل")

        return df, used_exchange

    # ----------------------------------------------------------------------
    def save(self, df, symbol, exchange):
        """
        ذخیره استاندارد دیتاست
        """

        safe_symbol = symbol.replace("/", "_")
        file_name = f"{safe_symbol}.parquet"
        full_path = os.path.join(self.output_dir, file_name)

        df.to_parquet(full_path, compression="snappy", index=False)

        logging.info(f"💾 ذخیره شد: {full_path}")

        if self.metadata:
            self.metadata.update(symbol, exchange, full_path, len(df))

        return full_path

    # ----------------------------------------------------------------------
    def collect(self, symbol_list, timeframe="1h", days=365):
        """
        جمع‌آوری کامل دیتا برای لیست نمادها
        """

        for symbol in symbol_list:
            df, ex = self.fetch(symbol, timeframe, days)
            if df is not None:
                self.save(df, symbol, ex)
            time.sleep(2)

        logging.info("🎉 جمع‌آوری کامل شد!")