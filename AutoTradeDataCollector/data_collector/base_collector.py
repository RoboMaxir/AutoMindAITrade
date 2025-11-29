import os
import time
import json
import logging
from datetime import datetime, timedelta

import pandas as pd


class BaseCollector:
    """
    کلاس پایه برای جمع‌آوری داده‌ها
    """

    def __init__(self, output_dir="datasets", metadata_manager=None):
        self.output_dir = output_dir
        self.metadata_manager = metadata_manager
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch(self, symbol: str, timeframe="1h", days=365):
        """
        متود پایه برای دریافت داده‌ها - باید در کلاس‌های فرزند پیاده‌سازی شود
        """
        raise NotImplementedError("این متود باید در کلاس فرزند پیاده‌سازی شود")

    def save(self, df, symbol, exchange):
        """
        ذخیره داده‌ها
        """
        raise NotImplementedError("این متود باید در کلاس فرزند پیاده‌سازی شود")

    def collect(self, symbol_list, timeframe="1h", days=365):
        """
        جمع‌آوری کامل داده‌ها برای لیست نمادها
        """
        raise NotImplementedError("این متود باید در کلاس فرزند پیاده‌سازی شود")