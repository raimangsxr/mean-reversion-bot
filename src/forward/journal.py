from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


class ForwardJournalError(RuntimeError):
    pass


class ForwardJournal:
    """Append-only JSONL journal with idempotency keys.

    Forward validation is only useful if observations are auditable. Unlike the
    execution journal, failures here are fail-closed: callers should not place a
    new order when its signal/decision cannot be persisted.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._seen = self._load_seen_keys()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_seen_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        seen: set[str] = set()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    key = record.get("idempotency_key")
                    if key:
                        seen.add(str(key))
        except Exception as exc:
            raise ForwardJournalError(
                f"No se puede leer el forward journal {self.path}: {exc}"
            ) from exc
        return seen

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            raise ForwardJournalError(
                f"No se puede leer el forward journal {self.path}: {exc}"
            ) from exc
        return records

    def append_once(self, event: str, idempotency_key: str, **payload: Any) -> bool:
        if idempotency_key in self._seen:
            return False
        record = {
            "recorded_at": self._now_iso(),
            "event": event,
            "idempotency_key": idempotency_key,
            **payload,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            raise ForwardJournalError(
                f"No se puede escribir el forward journal {self.path}: {exc}"
            ) from exc
        self._seen.add(idempotency_key)
        return True
