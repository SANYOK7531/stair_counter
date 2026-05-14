import json
import os
import threading
from datetime import datetime
from pathlib import Path


class DailyStats:
    """
    Per-day, per-hour visit counters persisted in a single JSON file:
        { "YYYY-MM-DD": { "HH": { "left": N, "right": N } } }

    Writes are atomic (tmp file + os.replace). The lock serialises mutations.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[stats] failed to read {self.file_path}: {e}")
            print("[stats] starting from empty state, file will be overwritten")
            return {}

    def _save(self) -> None:
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.file_path)
        except OSError as e:
            print(f"[stats] write error: {e}")

    def increment(self, counter_name: str, when: datetime | None = None) -> None:
        if when is None:
            when = datetime.now()

        date_key = when.strftime("%Y-%m-%d")
        hour_key = when.strftime("%H")

        with self._lock:
            day = self._data.setdefault(date_key, {})
            hour = day.setdefault(hour_key, {})
            hour[counter_name] = hour.get(counter_name, 0) + 1
            self._save()

    def decrement(self, counter_name: str, when: datetime | None = None) -> None:
        """
        Roll back an increment. `when` should be the timestamp of the original
        increment so the bucket matches even if the rollback crosses an hour
        boundary. No-op if the bucket is missing or already zero.
        """
        if when is None:
            when = datetime.now()

        date_key = when.strftime("%Y-%m-%d")
        hour_key = when.strftime("%H")

        with self._lock:
            day = self._data.get(date_key)
            if day is None:
                return
            hour = day.get(hour_key)
            if hour is None:
                return
            current = hour.get(counter_name, 0)
            if current <= 0:
                return
            new_value = current - 1
            if new_value == 0:
                hour.pop(counter_name, None)
            else:
                hour[counter_name] = new_value
            if not hour:
                day.pop(hour_key, None)
            if not day:
                self._data.pop(date_key, None)
            self._save()

    def today_total(self, counter_name: str | None = None) -> int:
        date_key = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            day = self._data.get(date_key, {})
            total = 0
            for hour_data in day.values():
                if counter_name is None:
                    total += sum(hour_data.values())
                else:
                    total += hour_data.get(counter_name, 0)
            return total

    def reset_today(self) -> None:
        date_key = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if date_key in self._data:
                self._data.pop(date_key, None)
                self._save()
