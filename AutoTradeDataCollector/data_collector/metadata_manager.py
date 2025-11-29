import json
import os
from datetime import datetime


class MetadataManager:

    def __init__(self, path="datasets/metadata/crypto_metadata.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "created_at": datetime.now().isoformat(),
                "symbols": {}
            }
            self._save()

    def update(self, symbol, exchange, path, count):
        self.data["symbols"][symbol] = {
            "exchange": exchange,
            "file": path,
            "candle_count": count,
            "updated_at": datetime.now().isoformat()
        }
        self._save()

    def get_metadata(self, symbol=None):
        if symbol:
            return self.data["symbols"].get(symbol)
        return self.data

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)