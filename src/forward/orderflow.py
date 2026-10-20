from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
import time
from typing import Deque, Iterable

import pandas as pd

try:
    import websocket  # type: ignore
except ImportError:  # pragma: no cover
    websocket = None

from src.research.orderflow import compute_flow_features, classify_primary_window


BINANCE_USDM_MARKET_STREAM_BASE = "wss://fstream.binance.com/market/stream"


@dataclass(frozen=True)
class LiveAggTrade:
    symbol: str
    agg_trade_id: int
    price: float
    quantity: float
    trade_time: datetime
    event_time: datetime
    buyer_maker: bool

    @property
    def quote_notional(self) -> float:
        return self.price * self.quantity


@dataclass(frozen=True)
class FlowDecision:
    symbol: str
    side: str
    start: datetime
    end: datetime
    reference_price: float
    orderflow_class: str
    veto: bool
    features: dict[str, float | int]


class AggTradeCollector:
    """Continuously collect mainnet USD-M aggTrade market data in memory.

    Strategy candles and executed order flow are public mainnet evidence even when
    orders are sent only to Binance Demo. The collector deliberately does not
    backfill gaps. A disconnect resets ``connected_since`` and any signal whose
    observation window crosses that gap must fail closed.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        retention_seconds: int = 15 * 60,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self.symbols = tuple(symbols)
        if not self.symbols:
            raise ValueError("AggTradeCollector requires at least one symbol")
        self.retention_seconds = int(retention_seconds)
        if self.retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)

        self._market_to_symbol = {self._market_id(s): s for s in self.symbols}
        self._buffers: dict[str, Deque[LiveAggTrade]] = {
            symbol: deque() for symbol in self.symbols
        }
        self._last_id: dict[str, int] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._app = None
        self._connected_since: datetime | None = None
        self._last_event_received_at: datetime | None = None
        self._last_error: str | None = None

    @staticmethod
    def _market_id(symbol: str) -> str:
        contract = symbol.split(":", 1)[0]
        base, quote = contract.split("/", 1)
        return f"{base}{quote}".replace("-", "").upper()

    @property
    def url(self) -> str:
        streams = "/".join(
            f"{self._market_id(symbol).lower()}@aggTrade" for symbol in self.symbols
        )
        return f"{BINANCE_USDM_MARKET_STREAM_BASE}?streams={streams}"

    @staticmethod
    def _utc_from_ms(value: int | float | str) -> datetime:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def start(self) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is not installed. Run pip install -r requirements.txt"
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="aggtrade-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        app = self._app
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

    def _run(self) -> None:  # pragma: no cover - network loop covered through callbacks
        delay = self.reconnect_initial_seconds
        while not self._stop.is_set():
            try:
                self._app = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._app.run_forever()
            except Exception as exc:
                self._on_error(None, exc)
            finally:
                with self._lock:
                    self._connected_since = None
                self._app = None
            if self._stop.is_set():
                break
            time.sleep(delay)
            delay = min(self.reconnect_max_seconds, max(1.0, delay * 2.0))
            if self._connected_since is not None:
                delay = self.reconnect_initial_seconds

    def _on_open(self, _app) -> None:
        with self._lock:
            self._connected_since = datetime.now(timezone.utc)
            self._last_error = None

    def _on_close(self, _app, _status=None, _message=None) -> None:
        with self._lock:
            self._connected_since = None

    def _on_error(self, _app, error) -> None:
        with self._lock:
            self._last_error = str(error)

    def _on_message(self, _app, raw: str) -> None:
        try:
            payload = json.loads(raw)
            data = payload.get("data", payload)
            if data.get("e") != "aggTrade":
                return
            market = str(data["s"]).upper()
            symbol = self._market_to_symbol.get(market)
            if symbol is None:
                return
            trade = LiveAggTrade(
                symbol=symbol,
                agg_trade_id=int(data["a"]),
                price=float(data["p"]),
                quantity=float(data["q"]),
                trade_time=self._utc_from_ms(data["T"]),
                event_time=self._utc_from_ms(data["E"]),
                buyer_maker=bool(data["m"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._on_error(_app, f"invalid aggTrade payload: {exc}")
            return

        received = datetime.now(timezone.utc)
        cutoff = received.timestamp() - self.retention_seconds
        with self._lock:
            last_id = self._last_id.get(symbol)
            if last_id is not None and trade.agg_trade_id <= last_id:
                return
            self._last_id[symbol] = trade.agg_trade_id
            buf = self._buffers[symbol]
            buf.append(trade)
            while buf and buf[0].trade_time.timestamp() < cutoff:
                buf.popleft()
            self._last_event_received_at = received

    def wait_until_connected(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                if self._connected_since is not None:
                    return True
            time.sleep(0.05)
        return False

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "connected": self._connected_since is not None,
                "connected_since": self._connected_since.isoformat()
                if self._connected_since
                else None,
                "last_event_received_at": self._last_event_received_at.isoformat()
                if self._last_event_received_at
                else None,
                "last_error": self._last_error,
                "buffered": {symbol: len(buf) for symbol, buf in self._buffers.items()},
            }

    def _window(self, symbol: str, start: datetime, end: datetime) -> list[LiveAggTrade]:
        start = self._utc(start)
        end = self._utc(end)
        with self._lock:
            return [
                item
                for item in self._buffers.get(symbol, ())
                if start <= item.trade_time <= end
            ]

    def has_complete_window(self, symbol: str, start: datetime, end: datetime) -> bool:
        """Return True only if no reconnect can have created a blind spot.

        ``connected_since <= start`` proves this connection predates the window;
        requiring a trade timestamp at/after ``end`` proves the stream has advanced
        beyond the causal decision boundary.
        """
        start = self._utc(start)
        end = self._utc(end)
        with self._lock:
            connected_since = self._connected_since
            if connected_since is None or connected_since > start:
                return False
            buf = self._buffers.get(symbol)
            if not buf:
                return False
            return buf[-1].trade_time >= end

    def frame(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = self._window(symbol, start, end)
        if not rows:
            return pd.DataFrame(
                columns=(
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "timestamp",
                    "buyer_maker",
                    "quote_notional",
                )
            )
        return pd.DataFrame.from_records(
            [
                {
                    "agg_trade_id": row.agg_trade_id,
                    "price": row.price,
                    "quantity": row.quantity,
                    "first_trade_id": row.agg_trade_id,
                    "last_trade_id": row.agg_trade_id,
                    "timestamp": pd.Timestamp(row.trade_time),
                    "buyer_maker": row.buyer_maker,
                    "quote_notional": row.quote_notional,
                }
                for row in rows
            ]
        )

    def decide(
        self,
        symbol: str,
        *,
        side: str,
        start: datetime,
        window_seconds: int,
    ) -> FlowDecision:
        start = self._utc(start)
        end = start + pd.Timedelta(seconds=int(window_seconds)).to_pytimedelta()
        if not self.has_complete_window(symbol, start, end):
            raise RuntimeError(f"{symbol}: incomplete aggTrade coverage for order-flow window")
        frame = self.frame(symbol, start, end)
        features = compute_flow_features(
            frame,
            start=pd.Timestamp(start),
            side=side,
            window_seconds=int(window_seconds),
        )
        flow_class = classify_primary_window(features)
        if flow_class == "missing" or frame.empty:
            raise RuntimeError(f"{symbol}: order-flow window contains no usable aggTrades")
        reference_price = float(frame.iloc[-1]["price"])
        return FlowDecision(
            symbol=symbol,
            side=side,
            start=start,
            end=end,
            reference_price=reference_price,
            orderflow_class=flow_class,
            veto=flow_class == "adverse_acceptance",
            features=features,
        )
