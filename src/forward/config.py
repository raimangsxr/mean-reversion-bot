from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_config_sha256(values: dict[str, Any]) -> str:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
