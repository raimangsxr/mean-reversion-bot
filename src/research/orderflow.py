from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from config.settings import (
    ORDERFLOW_ABSORPTION_TOLERANCE_BPS,
    ORDERFLOW_ADVERSE_ACCEPTANCE_BPS,
    ORDERFLOW_DECISION_SECONDS,
    ORDERFLOW_IMBALANCE_THRESHOLD,
)


# Preregistered exploratory contract. Only the 60s window is used for the
# counterfactual veto diagnostic. The other windows are descriptive only.
FEATURE_WINDOWS_SECONDS = (30, 60, 180, 420)
PRIMARY_DECISION_WINDOW_SECONDS = ORDERFLOW_DECISION_SECONDS
FLOW_IMBALANCE_THRESHOLD = ORDERFLOW_IMBALANCE_THRESHOLD
ADVERSE_ACCEPTANCE_BPS = ORDERFLOW_ADVERSE_ACCEPTANCE_BPS
ABSORPTION_TOLERANCE_BPS = ORDERFLOW_ABSORPTION_TOLERANCE_BPS
LARGE_TRADE_QUANTILE = 0.90

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
AGG_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_maker",
)


@dataclass(frozen=True)
class FlowWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = _utc(self.start)
        end = _utc(self.end)
        if end <= start:
            raise ValueError("FlowWindow.end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class ArchiveFetchResult:
    symbol: str
    day: str
    rows: int
    cache_path: str
    archive_url: str
    zip_sha256: str
    downloaded: bool


def _utc(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def market_id(symbol: str) -> str:
    contract = symbol.split(":", 1)[0]
    if "/" not in contract:
        raise ValueError(f"Unsupported CCXT symbol: {symbol!r}")
    base, quote = contract.split("/", 1)
    return f"{base}{quote}".replace("-", "").upper()


def _slug(symbol: str) -> str:
    return "_".join(part for part in symbol.replace(":", "/").split("/") if part).lower()


def _timestamp_to_ms(raw: str | int | float) -> int:
    value = int(float(raw))
    magnitude = abs(value)
    if magnitude >= 10**17:  # nanoseconds
        return value // 1_000_000
    if magnitude >= 10**14:  # microseconds
        return value // 1_000
    return value  # milliseconds for Futures archive today


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Invalid buyer_maker value: {value!r}")


def _archive_url(symbol: str, day: date) -> str:
    mid = market_id(symbol)
    ds = day.isoformat()
    return f"{BINANCE_VISION_BASE}/{mid}/{mid}-aggTrades-{ds}.zip"


def _cache_paths(cache_dir: str | Path, symbol: str, day: date) -> tuple[Path, Path]:
    root = Path(cache_dir) / "aggtrades" / _slug(symbol)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / day.isoformat()
    return stem.with_suffix(".csv.gz"), stem.with_suffix(".meta.json")


def _merge_windows(windows: list[FlowWindow]) -> list[FlowWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start)
    merged: list[FlowWindow] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = FlowWindow(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def _window_ms(windows: list[FlowWindow]) -> list[tuple[int, int]]:
    return [
        (int(w.start.timestamp() * 1000), int(w.end.timestamp() * 1000))
        for w in _merge_windows(windows)
    ]


def _meta_covers(meta: dict, windows: list[FlowWindow]) -> bool:
    try:
        cached = [FlowWindow(item["start"], item["end"]) for item in meta["windows"]]
    except (KeyError, TypeError, ValueError):
        return False
    cached_ms = _window_ms(cached)
    for start_ms, end_ms in _window_ms(windows):
        if not any(cs <= start_ms and ce >= end_ms for cs, ce in cached_ms):
            return False
    return True


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mean-reversion-bot-orderflow/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Binance Vision HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot download Binance Vision archive {url}: {exc}") from exc


def _download_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "mean-reversion-bot-orderflow/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Cannot download checksum {url}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(zip_path: Path, archive_url: str) -> str:
    actual = _sha256_file(zip_path)
    payload = _download_text(archive_url + ".CHECKSUM")
    expected = payload.strip().split()[0].lower() if payload.strip() else ""
    if len(expected) != 64:
        raise RuntimeError(f"Invalid CHECKSUM payload for {archive_url}")
    if actual.lower() != expected:
        raise RuntimeError(
            f"Checksum mismatch for {archive_url}: {actual} != {expected}"
        )
    return actual


def _row_is_header(row: list[str]) -> bool:
    if len(row) < 7:
        return True
    try:
        int(float(row[0]))
        float(row[1])
        float(row[2])
        _timestamp_to_ms(row[5])
    except (TypeError, ValueError):
        return True
    return False


def _extract_intervals_from_zip(zip_path: Path, windows: list[FlowWindow]) -> pd.DataFrame:
    intervals = _window_ms(windows)
    if not intervals:
        return empty_aggtrades()
    min_ms = min(v[0] for v in intervals)
    max_ms = max(v[1] for v in intervals)
    records: list[tuple[int, float, float, int, int, int, bool]] = []

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise RuntimeError(f"Empty Binance Vision zip: {zip_path}")
        # Binance publishes a single CSV per daily archive. If metadata files are
        # ever added, prefer the CSV member explicitly.
        member = next((m for m in members if m.lower().endswith(".csv")), members[0])
        with archive.open(member, "r") as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.reader(text)
            for row in reader:
                if _row_is_header(row):
                    continue
                ts_ms = _timestamp_to_ms(row[5])
                if ts_ms < min_ms:
                    continue
                if ts_ms > max_ms:
                    break
                if not any(start <= ts_ms <= end for start, end in intervals):
                    continue
                records.append(
                    (
                        int(float(row[0])),
                        float(row[1]),
                        float(row[2]),
                        int(float(row[3])),
                        int(float(row[4])),
                        ts_ms,
                        _as_bool(row[6]),
                    )
                )

    if not records:
        return empty_aggtrades()
    df = pd.DataFrame.from_records(records, columns=AGG_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="agg_trade_id", keep="last").sort_values("timestamp")
    df["quote_notional"] = df["price"] * df["quantity"]
    return df.reset_index(drop=True)


def empty_aggtrades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agg_trade_id": pd.Series(dtype="int64"),
            "price": pd.Series(dtype="float64"),
            "quantity": pd.Series(dtype="float64"),
            "first_trade_id": pd.Series(dtype="int64"),
            "last_trade_id": pd.Series(dtype="int64"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "buyer_maker": pd.Series(dtype="bool"),
            "quote_notional": pd.Series(dtype="float64"),
        }
    )


def load_cached_aggtrades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_aggtrades()
    df = pd.read_csv(path, compression="gzip")
    if df.empty:
        return empty_aggtrades()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["buyer_maker"] = df["buyer_maker"].map(_as_bool)
    for column in ("price", "quantity", "quote_notional"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["price", "quantity"]).sort_values("timestamp").reset_index(drop=True)


def ensure_aggtrade_windows(
    symbol: str,
    windows: list[FlowWindow],
    *,
    cache_dir: str | Path,
    no_download: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[ArchiveFetchResult]]:
    """Ensure compact aggTrade cache for the requested event windows.

    The full daily Binance Vision ZIP is temporary: only rows inside the union
    of requested windows are persisted. Metadata records the exact covered
    windows and archive SHA-256 so subsequent runs are deterministic.
    """
    if not windows:
        return empty_aggtrades(), []

    grouped: dict[date, list[FlowWindow]] = {}
    for window in windows:
        start_day = window.start.date()
        end_day = window.end.date()
        if start_day != end_day:
            midnight = _utc(pd.Timestamp(end_day))
            grouped.setdefault(start_day, []).append(FlowWindow(window.start, midnight - pd.Timedelta(milliseconds=1)))
            grouped.setdefault(end_day, []).append(FlowWindow(midnight, window.end))
        else:
            grouped.setdefault(start_day, []).append(window)

    frames: list[pd.DataFrame] = []
    manifest: list[ArchiveFetchResult] = []
    for day, requested in sorted(grouped.items()):
        requested = _merge_windows(requested)
        cache_path, meta_path = _cache_paths(cache_dir, symbol, day)
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}

        valid_cache = cache_path.exists() and _meta_covers(meta, requested) and not refresh
        downloaded = False
        if valid_cache:
            day_df = load_cached_aggtrades(cache_path)
            archive_url = str(meta.get("archive_url") or _archive_url(symbol, day))
            zip_sha = str(meta.get("zip_sha256") or "")
        else:
            if no_download:
                raise RuntimeError(
                    f"Missing aggTrade cache for {symbol} {day}; rerun without --no-download."
                )
            # Preserve previously-cached windows when a new causal experiment asks
            # for additional intraday intervals on the same archive day. Without
            # this union a cache extension would silently discard the older 420s
            # diagnostic windows from the first order-flow study.
            cached_windows: list[FlowWindow] = []
            if cache_path.exists():
                try:
                    cached_windows = [
                        FlowWindow(item["start"], item["end"])
                        for item in meta.get("windows", [])
                    ]
                except (KeyError, TypeError, ValueError):
                    cached_windows = []
            extraction_windows = _merge_windows([*cached_windows, *requested])

            archive_url = _archive_url(symbol, day)
            with tempfile.TemporaryDirectory(prefix="aggtrades-") as tmp:
                zip_path = Path(tmp) / os.path.basename(archive_url)
                _download(archive_url, zip_path)
                zip_sha = _verify_checksum(zip_path, archive_url)
                day_df = _extract_intervals_from_zip(zip_path, extraction_windows)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            day_df.to_csv(cache_path, index=False, compression="gzip")
            meta = {
                "schema": 1,
                "source": "binance_vision_usdm_daily_aggTrades",
                "symbol": symbol,
                "market_id": market_id(symbol),
                "day": day.isoformat(),
                "archive_url": archive_url,
                "zip_sha256": zip_sha,
                "rows": int(len(day_df)),
                "windows": [
                    {"start": w.start.isoformat(), "end": w.end.isoformat()}
                    for w in extraction_windows
                ],
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            downloaded = True

        frames.append(day_df)
        manifest.append(
            ArchiveFetchResult(
                symbol=symbol,
                day=day.isoformat(),
                rows=int(len(day_df)),
                cache_path=str(cache_path),
                archive_url=archive_url,
                zip_sha256=zip_sha,
                downloaded=downloaded,
            )
        )

    merged = pd.concat(frames, ignore_index=True) if frames else empty_aggtrades()
    if not merged.empty:
        merged = merged.drop_duplicates(subset="agg_trade_id", keep="last").sort_values("timestamp")
    return merged.reset_index(drop=True), manifest


def _imbalance(signed_quote: pd.Series, quote: pd.Series) -> float:
    total = float(quote.sum())
    return float(signed_quote.sum() / total) if total > 0 else float("nan")


def compute_flow_features(
    trades: pd.DataFrame,
    *,
    start: pd.Timestamp,
    side: str,
    window_seconds: int,
) -> dict[str, float | int]:
    start = _utc(start)
    end = start + pd.Timedelta(seconds=int(window_seconds))
    window = trades.loc[(trades["timestamp"] >= start) & (trades["timestamp"] <= end)].copy()
    prefix = f"of_{window_seconds}s_"
    if window.empty:
        return {
            prefix + "trades": 0,
            prefix + "quote_volume": 0.0,
            prefix + "side_imbalance": float("nan"),
            prefix + "side_return_bps": float("nan"),
            prefix + "mae_bps": float("nan"),
            prefix + "mfe_bps": float("nan"),
            prefix + "first_half_side_imbalance": float("nan"),
            prefix + "second_half_side_imbalance": float("nan"),
            prefix + "imbalance_change": float("nan"),
            prefix + "large_side_imbalance": float("nan"),
        }

    direction = 1.0 if side == "long" else -1.0
    if side not in {"long", "short"}:
        raise ValueError(f"Invalid side: {side!r}")
    quote = window["quote_notional"].astype(float)
    aggressor = np.where(window["buyer_maker"].astype(bool), -1.0, 1.0)
    signed_quote = pd.Series(aggressor * quote.to_numpy(dtype=float), index=window.index)
    raw_imb = _imbalance(signed_quote, quote)
    side_imb = direction * raw_imb

    prices = window["price"].astype(float)
    first_price = float(prices.iloc[0])
    relative = direction * (prices / first_price - 1.0) * 10_000.0
    midpoint = start + pd.Timedelta(seconds=window_seconds / 2.0)
    first_mask = window["timestamp"] < midpoint
    second_mask = ~first_mask
    first_imb = _imbalance(signed_quote.loc[first_mask], quote.loc[first_mask]) if first_mask.any() else float("nan")
    second_imb = _imbalance(signed_quote.loc[second_mask], quote.loc[second_mask]) if second_mask.any() else float("nan")
    first_side = direction * first_imb if math.isfinite(first_imb) else float("nan")
    second_side = direction * second_imb if math.isfinite(second_imb) else float("nan")

    threshold = float(quote.quantile(LARGE_TRADE_QUANTILE)) if len(quote) > 1 else float(quote.iloc[0])
    large_mask = quote >= threshold
    large_imb = _imbalance(signed_quote.loc[large_mask], quote.loc[large_mask])

    return {
        prefix + "trades": int(len(window)),
        prefix + "quote_volume": float(quote.sum()),
        prefix + "side_imbalance": float(side_imb),
        prefix + "side_return_bps": float(relative.iloc[-1]),
        prefix + "mae_bps": float(relative.min()),
        prefix + "mfe_bps": float(relative.max()),
        prefix + "first_half_side_imbalance": float(first_side),
        prefix + "second_half_side_imbalance": float(second_side),
        prefix + "imbalance_change": float(second_side - first_side)
        if math.isfinite(first_side) and math.isfinite(second_side)
        else float("nan"),
        prefix + "large_side_imbalance": float(direction * large_imb)
        if math.isfinite(large_imb)
        else float("nan"),
    }


def classify_primary_window(row: pd.Series | dict) -> str:
    prefix = f"of_{PRIMARY_DECISION_WINDOW_SECONDS}s_"
    imbalance = float(row.get(prefix + "side_imbalance", float("nan")))
    side_return = float(row.get(prefix + "side_return_bps", float("nan")))
    trades = int(row.get(prefix + "trades", 0) or 0)
    if trades <= 0 or not math.isfinite(imbalance) or not math.isfinite(side_return):
        return "missing"
    if imbalance <= -FLOW_IMBALANCE_THRESHOLD and side_return <= -ADVERSE_ACCEPTANCE_BPS:
        return "adverse_acceptance"
    if imbalance <= -FLOW_IMBALANCE_THRESHOLD and side_return >= -ABSORPTION_TOLERANCE_BPS:
        return "adverse_absorption"
    if imbalance >= FLOW_IMBALANCE_THRESHOLD and side_return >= 0.0:
        return "aligned_support"
    return "mixed"


def add_orderflow_features(
    trades_to_label: pd.DataFrame,
    aggtrades_by_symbol: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict] = []
    for item in trades_to_label.to_dict("records"):
        symbol = str(item["symbol"])
        side = str(item["side"])
        decision_start = _utc(item["decision_start"])
        features: dict[str, object] = dict(item)
        data = aggtrades_by_symbol.get(symbol, empty_aggtrades())
        for seconds in FEATURE_WINDOWS_SECONDS:
            features.update(
                compute_flow_features(
                    data, start=decision_start, side=side, window_seconds=seconds
                )
            )
        features["orderflow_class"] = classify_primary_window(features)
        features["orderflow_veto"] = features["orderflow_class"] == "adverse_acceptance"
        records.append(features)
    return pd.DataFrame.from_records(records)


def profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def summarize_orderflow(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if features.empty:
        return pd.DataFrame(), {
            "trades": 0,
            "baseline_pf": None,
            "kept_pf_diagnostic": None,
            "note": "No trades were available for order-flow analysis.",
        }

    rows: list[dict] = []
    for label, group in features.groupby("orderflow_class", dropna=False, sort=True):
        rows.append(
            {
                "orderflow_class": str(label),
                "trades": int(len(group)),
                "wins": int((group["pnl"] > 0).sum()),
                "losses": int((group["pnl"] < 0).sum()),
                "win_rate": float((group["pnl"] > 0).mean()),
                "stop_rate": float(group["exit_reason"].astype(str).str.contains("stop", case=False).mean()),
                "pf": profit_factor(group["pnl"]),
                "pnl": float(group["pnl"].sum()),
                "median_side_imbalance_60s": float(group["of_60s_side_imbalance"].median()),
                "median_side_return_bps_60s": float(group["of_60s_side_return_bps"].median()),
            }
        )
    summary = pd.DataFrame.from_records(rows)

    veto = features["orderflow_veto"].astype(bool)
    losses = features["pnl"] < 0
    stops = features["exit_reason"].astype(str).str.contains("stop", case=False)
    kept = features.loc[~veto]
    diagnostic = {
        "schema": 1,
        "status": "diagnostic_only_not_executable_backtest",
        "primary_window_seconds": PRIMARY_DECISION_WINDOW_SECONDS,
        "preregistered_thresholds": {
            "flow_imbalance": FLOW_IMBALANCE_THRESHOLD,
            "adverse_acceptance_bps": ADVERSE_ACCEPTANCE_BPS,
            "absorption_tolerance_bps": ABSORPTION_TOLERANCE_BPS,
        },
        "trades": int(len(features)),
        "baseline_pf": profit_factor(features["pnl"]),
        "baseline_win_rate": float((features["pnl"] > 0).mean()),
        "baseline_stop_rate": float(stops.mean()),
        "vetoed_trades": int(veto.sum()),
        "vetoed_losers": int((veto & losses).sum()),
        "vetoed_winners": int((veto & ~losses).sum()),
        "veto_loss_precision": float((veto & losses).sum() / veto.sum()) if veto.any() else None,
        "loss_recall": float((veto & losses).sum() / losses.sum()) if losses.any() else None,
        "stop_recall": float((veto & stops).sum() / stops.sum()) if stops.any() else None,
        "kept_trades": int(len(kept)),
        "kept_pf_diagnostic": profit_factor(kept["pnl"]) if not kept.empty else None,
        "kept_win_rate_diagnostic": float((kept["pnl"] > 0).mean()) if not kept.empty else None,
        "causality_warning": (
            "The veto is evaluated against the original V12 fills. A live order-flow gate would "
            "delay/re-route entry, so kept_pf_diagnostic is discrimination evidence only, not a "
            "tradable performance estimate. If discrimination is useful, the next experiment must "
            "simulate the delayed entry path explicitly."
        ),
    }
    return summary, diagnostic
