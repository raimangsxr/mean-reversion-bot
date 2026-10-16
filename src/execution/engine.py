from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
from typing import Any, Optional

import ccxt

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    EXECUTION_JOURNAL_FILE,
    LEVERAGE,
    MARGIN_MODE,
    STATE_FILE,
    TAKE_PROFIT_ORDER_MODE,
    TRIGGER_PRICE_TYPE,
    USE_TESTNET,
)


class ExecutionError(RuntimeError):
    pass


class ProtectionError(ExecutionError):
    pass


@dataclass
class AccountSnapshot:
    equity: float
    free_margin: float


@dataclass
class PositionSnapshot:
    side: str
    contracts: float
    entry_price: float
    notional: float


@dataclass
class BotState:
    status: str = "flat"
    symbol: Optional[str] = None
    side: Optional[str] = None
    entry_order_id: Optional[str] = None
    entry_client_order_id: Optional[str] = None
    # market para la entrada entrada/fallback V8 final; limit para la entrada pasiva V8.
    entry_order_kind: Optional[str] = None
    entry_time: Optional[str] = None
    entry_price: Optional[float] = None
    amount: Optional[float] = None

    # V8: entrada LIMIT persistente. Estos campos solo viven mientras status
    # esta en entry_pending*. Al proteger el fill se limpian.
    # V8 final usa dos referencias distintas:
    # - pending_reference_price: close de la vela de senal; ancla la LIMIT -offset.
    # - pending_fallback_reference_price: precio observado en t=0; ancla el cap no-chase.
    pending_reference_price: Optional[float] = None
    pending_fallback_reference_price: Optional[float] = None
    pending_limit_price: Optional[float] = None
    pending_expires_at: Optional[str] = None
    pending_stop_distance: Optional[float] = None
    pending_risk_reward_ratio: Optional[float] = None
    pending_use_take_profit: Optional[bool] = None
    pending_fallback_cap_bps: Optional[float] = None
    pending_fallback_max_lateness_seconds: Optional[int] = None
    pending_journal_context: Optional[dict[str, Any]] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_order_id: Optional[str] = None
    stop_client_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    take_profit_client_order_id: Optional[str] = None
    # "limit" para TP maker; "trigger" para el TP market legado.
    take_profit_order_kind: Optional[str] = None
    take_profit_last_filled: float = 0.0
    protection_mode: Optional[str] = None
    last_signal_timestamp: Optional[str] = None
    updated_at: Optional[str] = None
    note: Optional[str] = None


class JsonStateStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return BotState(**raw)
        except Exception as exc:
            raise ExecutionError(
                f"No se puede leer el estado persistido {self.path}: {exc}"
            ) from exc

    def save(self, state: BotState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        tmp.replace(self.path)


class BinanceExecutionEngine:
    """Motor Binance USD-M Futures para One-way Mode.

    Permite dos formas de proteccion:
    - fixed_rr: STOP_MARKET reduceOnly + TP LIMIT reduceOnly post-only (maker).
    - stop_only: solo SL reduceOnly; la salida normal se hace por senal z/timeout.

    El TP trigger-market anterior se conserva como modo de compatibilidad mediante
    TAKE_PROFIT_ORDER_MODE=trigger_market.
    """

    BOT_ORDER_PREFIXES = ("zsl_", "ztp_")
    ENTRY_PENDING_STATUSES = {
        "entry_pending_submitting",
        "entry_pending",
        "entry_pending_unknown",
    }
    ENTRY_AMBIGUITY_GRACE_SECONDS = 30

    def __init__(
        self,
        state_file: str = STATE_FILE,
        leverage: int = LEVERAGE,
        margin_mode: str = MARGIN_MODE,
        take_profit_order_mode: str = TAKE_PROFIT_ORDER_MODE,
        journal_file: str = EXECUTION_JOURNAL_FILE,
    ) -> None:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            raise ExecutionError("Faltan BINANCE_API_KEY/BINANCE_SECRET_KEY.")

        if take_profit_order_mode not in {"maker_limit", "trigger_market"}:
            raise ValueError(
                "take_profit_order_mode debe ser maker_limit o trigger_market."
            )

        self.leverage = leverage
        self.margin_mode = margin_mode
        self.take_profit_order_mode = take_profit_order_mode
        self.state_store = JsonStateStore(state_file)
        self.journal_path = Path(journal_file)

        self.exchange = ccxt.binance(
            {
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_SECRET_KEY,
                "enableRateLimit": True,
                "timeout": 15_000,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                },
            }
        )
        if USE_TESTNET:
            self.exchange.enable_demo_trading(True)

        self.exchange.load_markets()

        try:
            self.exchange.load_time_difference()
        except Exception as exc:
            print(
                "Aviso: no se pudo sincronizar inicialmente "
                f"el reloj con Binance: {exc}"
            )

    @staticmethod
    def _client_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:20]}"

    @staticmethod
    def _is_noop_error(exc: Exception, code: str, text: str) -> bool:
        message = str(exc).lower()
        return code in message or text.lower() in message

    @staticmethod
    def _is_timestamp_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "-1021" in message
            or "timestamp for this request is outside of the recvwindow" in message
            or "timestamp for this request was" in message
        )

    def _resync_time(self) -> None:
        try:
            self.exchange.load_time_difference()
        except Exception as exc:
            raise ExecutionError(
                f"No se pudo resincronizar el reloj con Binance: {exc}"
            ) from exc

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _journal(self, event: str, **payload: Any) -> None:
        """Append-only JSONL telemetry. A journal failure never blocks safety."""
        record = {
            "recorded_at": self._now_iso(),
            "event": event,
            **payload,
        }
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, separators=(",", ":")))
                handle.write("\n")
        except Exception as exc:
            print(f"Aviso: no se pudo escribir execution journal: {exc}")

    def _market_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "best_bid": None,
            "best_ask": None,
            "mid_price": None,
        }
        try:
            book = self.exchange.fetch_order_book(symbol, 5)
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid = float(bids[0][0]) if bids else None
            ask = float(asks[0][0]) if asks else None
            snapshot["best_bid"] = bid
            snapshot["best_ask"] = ask
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                snapshot["mid_price"] = (bid + ask) / 2.0
        except Exception as exc:
            snapshot["snapshot_error"] = str(exc)
        return snapshot

    @staticmethod
    def _entry_slippage_bps(side: str, fill: float, reference: Optional[float]) -> Optional[float]:
        if reference is None or reference <= 0 or fill <= 0:
            return None
        if side == "long":
            return (fill - reference) / reference * 10_000.0
        return (reference - fill) / reference * 10_000.0

    @staticmethod
    def _exit_slippage_bps(side: str, fill: float, reference: Optional[float]) -> Optional[float]:
        """Positive means adverse execution versus the intended exit reference."""
        if reference is None or reference <= 0 or fill <= 0:
            return None
        if side == "long":
            return (reference - fill) / reference * 10_000.0
        return (fill - reference) / reference * 10_000.0

    @staticmethod
    def _order_fee_payload(order: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not order:
            return {"fee": None, "fees": []}
        return {
            "fee": order.get("fee"),
            "fees": order.get("fees") or [],
        }

    def _fetch_trade_details(
        self,
        symbol: str,
        order_id: Optional[str],
        since_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Best-effort fill/commission enrichment from user trades."""
        if not order_id:
            return {}
        try:
            trades = self.exchange.fetch_my_trades(symbol, since_ms, 100)
        except Exception as exc:
            return {"trade_lookup_error": str(exc)}

        matched = [
            trade
            for trade in trades
            if str(trade.get("order") or trade.get("orderId") or "") == str(order_id)
        ]
        if not matched:
            return {}

        amount = sum(float(trade.get("amount") or 0.0) for trade in matched)
        weighted = sum(
            float(trade.get("amount") or 0.0) * float(trade.get("price") or 0.0)
            for trade in matched
        )
        fees = [trade.get("fee") for trade in matched if trade.get("fee")]
        liquidity = sorted(
            {
                str(trade.get("takerOrMaker"))
                for trade in matched
                if trade.get("takerOrMaker")
            }
        )
        return {
            "trade_count": len(matched),
            "trade_amount": amount,
            "trade_vwap": (weighted / amount) if amount > 0 else None,
            "trade_fees": fees,
            "liquidity": liquidity,
        }

    @staticmethod
    def _take_profit_kind(state: BotState) -> Optional[str]:
        if state.take_profit_order_kind in {"limit", "trigger"}:
            return state.take_profit_order_kind
        if not state.take_profit_order_id:
            return None
        # Migracion defensiva de estados persistidos anteriores.
        if state.protection_mode == "stop_and_tp_maker":
            return "limit"
        return "trigger"

    def configure_account(self, symbol: str) -> None:
        market = self.exchange.market(symbol)
        if not market.get("contract") or not market.get("linear"):
            raise ExecutionError(
                f"{symbol} no es un contrato lineal USD-M Futures de Binance."
            )

        try:
            self.exchange.set_position_mode(False, symbol)
        except Exception as exc:
            if not self._is_noop_error(exc, "-4059", "no need to change position side"):
                raise ExecutionError(f"No se pudo activar One-way Mode: {exc}") from exc

        try:
            self.exchange.set_margin_mode(self.margin_mode, symbol)
        except Exception as exc:
            if not self._is_noop_error(exc, "-4046", "no need to change margin type"):
                raise ExecutionError(
                    f"No se pudo configurar margin mode={self.margin_mode}: {exc}"
                ) from exc

        try:
            self.exchange.set_leverage(self.leverage, symbol)
        except Exception as exc:
            raise ExecutionError(
                f"No se pudo configurar leverage={self.leverage}x: {exc}"
            ) from exc

    def mark_signal_processed(self, timestamp: str) -> None:
        state = self.state_store.load()
        state.last_signal_timestamp = timestamp
        self.state_store.save(state)

    def was_signal_processed(self, timestamp: str) -> bool:
        return self.state_store.load().last_signal_timestamp == timestamp

    def _set_flat(self, note: str) -> BotState:
        previous = self.state_store.load()
        state = BotState(
            status="flat",
            last_signal_timestamp=previous.last_signal_timestamp,
            note=note,
        )
        self.state_store.save(state)
        return state

    def get_account_snapshot(self, currency: str = "USDT") -> AccountSnapshot:
        try:
            balance = self.exchange.fetch_balance()
        except Exception as exc:
            if not self._is_timestamp_error(exc):
                raise ExecutionError(
                    f"Error obteniendo balance Futures: {exc}"
                ) from exc

            self._resync_time()
            try:
                balance = self.exchange.fetch_balance()
            except Exception as retry_exc:
                raise ExecutionError(
                    "Error obteniendo balance Futures tras "
                    f"resincronizar el reloj: {retry_exc}"
                ) from retry_exc

        info = balance.get("info") or {}

        def number(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        equity = (
            number(info.get("totalMarginBalance"))
            or number(info.get("totalWalletBalance"))
            or number((balance.get("total") or {}).get(currency))
            or 0.0
        )
        free_margin = (
            number(info.get("availableBalance"))
            or number((balance.get("free") or {}).get(currency))
            or 0.0
        )
        if equity <= 0 or free_margin <= 0:
            raise ExecutionError(
                f"Balance Futures insuficiente/no legible: equity={equity}, free={free_margin}."
            )
        return AccountSnapshot(equity=equity, free_margin=free_margin)

    def get_market_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close") or ticker.get("ask")
            if price is None or float(price) <= 0:
                raise ValueError("ticker sin precio utilizable")
            return float(price)
        except Exception as exc:
            raise ExecutionError(f"No se pudo obtener precio de {symbol}: {exc}") from exc

    def normalize_amount(self, symbol: str, base_amount: float, price: float) -> float:
        market = self.exchange.market(symbol)
        contract_size = float(market.get("contractSize") or 1.0)
        raw_contracts = base_amount / contract_size
        contracts = float(self.exchange.amount_to_precision(symbol, raw_contracts))
        if contracts <= 0:
            raise ExecutionError("La cantidad queda a cero al aplicar precision.")

        limits = market.get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min")
        if min_amount is not None and contracts < float(min_amount):
            raise ExecutionError(
                f"Cantidad {contracts} inferior al minimo ({min_amount})."
            )

        base_qty = contracts * contract_size
        notional = base_qty * price
        min_cost = (limits.get("cost") or {}).get("min")
        if min_cost is not None and notional < float(min_cost):
            raise ExecutionError(
                f"Notional {notional:.4f} inferior al minimo ({min_cost})."
            )
        return contracts

    def _fetch_open_position(self, symbol: str) -> Optional[PositionSnapshot]:
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as exc:
            if not self._is_timestamp_error(exc):
                raise ExecutionError(
                    f"No se pudieron consultar posiciones: {exc}"
                ) from exc

            self._resync_time()
            try:
                positions = self.exchange.fetch_positions([symbol])
            except Exception as retry_exc:
                raise ExecutionError(
                    "No se pudieron consultar posiciones tras "
                    f"resincronizar el reloj: {retry_exc}"
                ) from retry_exc

        active: list[PositionSnapshot] = []
        for position in positions:
            contracts = abs(float(position.get("contracts") or 0.0))
            if contracts <= 0:
                continue
            side = str(position.get("side") or "").lower()
            if side not in {"long", "short"}:
                raise ExecutionError(f"Posicion con side no reconocido: {position}")
            active.append(
                PositionSnapshot(
                    side=side,
                    contracts=contracts,
                    entry_price=float(position.get("entryPrice") or 0.0),
                    notional=abs(float(position.get("notional") or 0.0)),
                )
            )

        if len(active) > 1:
            raise ExecutionError(
                "Se detectaron varias posiciones activas. El bot exige One-way Mode."
            )
        return active[0] if active else None

    def _average_fill_price(self, order: dict[str, Any], symbol: str) -> Optional[float]:
        average = order.get("average") or order.get("price")
        if average is not None and float(average) > 0:
            return float(average)

        filled = float(order.get("filled") or 0.0)
        cost = float(order.get("cost") or 0.0)
        contract_size = float(self.exchange.market(symbol).get("contractSize") or 1.0)
        if filled > 0 and cost > 0:
            return cost / (filled * contract_size)

        info = order.get("info") or {}
        executed_qty = float(info.get("executedQty") or 0.0)
        cumulative_quote = float(
            info.get("cumQuote")
            or info.get("cumQuoteQty")
            or info.get("cummulativeQuoteQty")
            or 0.0
        )
        if executed_qty > 0 and cumulative_quote > 0:
            return cumulative_quote / executed_qty
        return None

    def _resolve_market_fill(
        self,
        order: dict[str, Any],
        symbol: str,
    ) -> tuple[float, float]:
        resolved = order
        order_id = order.get("id")
        if order_id:
            try:
                resolved = self.exchange.fetch_order(order_id, symbol)
            except Exception:
                resolved = order

        filled = float(resolved.get("filled") or order.get("filled") or 0.0)
        average = self._average_fill_price(resolved, symbol) or self._average_fill_price(
            order, symbol
        )
        if filled <= 0 or average is None or average <= 0:
            raise ExecutionError(
                "La orden fue enviada, pero no se puede confirmar fill/precio."
            )
        return filled, average

    def _create_trigger_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        trigger_price: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        params = {
            "stopPrice": float(self.exchange.price_to_precision(symbol, trigger_price)),
            "reduceOnly": True,
            "workingType": TRIGGER_PRICE_TYPE,
            "clientOrderId": client_order_id,
        }
        return self.exchange.create_order(
            symbol,
            order_type,
            side,
            amount,
            None,
            params,
        )

    def _create_take_profit_limit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        precise_price = float(self.exchange.price_to_precision(symbol, price))
        return self.exchange.create_order(
            symbol,
            "limit",
            side,
            amount,
            precise_price,
            {
                "reduceOnly": True,
                "postOnly": True,
                "clientOrderId": client_order_id,
            },
        )

    def _create_entry_limit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        precise_price = float(self.exchange.price_to_precision(symbol, price))
        order_side = "buy" if side == "long" else "sell"
        return self.exchange.create_order(
            symbol,
            "limit",
            order_side,
            amount,
            precise_price,
            {
                "postOnly": True,
                "clientOrderId": client_order_id,
            },
        )

    @staticmethod
    def _order_client_id(order: dict[str, Any]) -> str:
        return str(
            order.get("clientOrderId")
            or (order.get("info") or {}).get("clientOrderId")
            or ""
        )

    def _find_open_regular_order_by_client_id(
        self,
        symbol: str,
        client_order_id: str,
    ) -> Optional[dict[str, Any]]:
        orders = self.exchange.fetch_open_orders(symbol)
        for order in orders:
            if self._order_client_id(order) == client_order_id:
                return order
        return None

    @staticmethod
    def _clear_pending_entry_fields(state: BotState) -> None:
        state.pending_reference_price = None
        state.pending_fallback_reference_price = None
        state.pending_limit_price = None
        state.pending_expires_at = None
        state.pending_stop_distance = None
        state.pending_risk_reward_ratio = None
        state.pending_use_take_profit = None
        state.pending_fallback_cap_bps = None
        state.pending_fallback_max_lateness_seconds = None
        state.pending_journal_context = None

    @staticmethod
    def _iso_to_utc(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def pending_entry_remaining_seconds(self, state: Optional[BotState] = None) -> Optional[float]:
        state = state or self.state_store.load()
        if state.status not in self.ENTRY_PENDING_STATUSES:
            return None
        expires_at = self._iso_to_utc(state.pending_expires_at)
        if expires_at is None:
            return None
        return (expires_at - datetime.now(timezone.utc)).total_seconds()

    def _fetch_trigger_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return self.exchange.fetch_order(order_id, symbol, {"trigger": True})

    def _fetch_trigger_order_status(self, symbol: str, order_id: str) -> str:
        order = self._fetch_trigger_order(symbol, order_id)
        return str(order.get("status") or "unknown").lower()

    def _fetch_regular_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return self.exchange.fetch_order(order_id, symbol)

    def _fetch_regular_order_status(self, symbol: str, order_id: str) -> str:
        order = self._fetch_regular_order(symbol, order_id)
        return str(order.get("status") or "unknown").lower()

    def _fetch_bot_trigger_orders(self, symbol: str) -> list[dict[str, Any]]:
        orders = self.exchange.fetch_open_orders(
            symbol,
            None,
            None,
            {"trigger": True},
        )
        result: list[dict[str, Any]] = []
        for order in orders:
            client_id = str(
                order.get("clientOrderId")
                or (order.get("info") or {}).get("clientOrderId")
                or ""
            )
            if client_id.startswith(self.BOT_ORDER_PREFIXES):
                result.append(order)
        return result

    def _fetch_bot_regular_tp_orders(self, symbol: str) -> list[dict[str, Any]]:
        orders = self.exchange.fetch_open_orders(symbol)
        result: list[dict[str, Any]] = []
        for order in orders:
            client_id = str(
                order.get("clientOrderId")
                or (order.get("info") or {}).get("clientOrderId")
                or ""
            )
            if client_id.startswith("ztp_"):
                result.append(order)
        return result

    def _fetch_bot_regular_entry_orders(self, symbol: str) -> list[dict[str, Any]]:
        orders = self.exchange.fetch_open_orders(symbol)
        result: list[dict[str, Any]] = []
        for order in orders:
            if self._order_client_id(order).startswith("zentry_"):
                result.append(order)
        return result

    def _cancel_bot_entry_orders(self, symbol: str) -> None:
        state = self.state_store.load()
        order_ids: set[str] = set()
        if state.entry_order_kind == "limit" and state.entry_order_id:
            order_ids.add(str(state.entry_order_id))

        try:
            for order in self._fetch_bot_regular_entry_orders(symbol):
                if order.get("id") is not None:
                    order_ids.add(str(order["id"]))
        except Exception as exc:
            raise ExecutionError(
                f"No se pudieron enumerar entradas LIMIT del bot: {exc}"
            ) from exc

        for order_id in order_ids:
            self._cancel_order_safely(symbol, order_id, trigger=False)

    def _cancel_order_safely(
        self,
        symbol: str,
        order_id: str,
        *,
        trigger: bool,
    ) -> None:
        params = {"trigger": True} if trigger else {}
        try:
            self.exchange.cancel_order(order_id, symbol, params)
        except ccxt.OrderNotFound:
            pass
        except Exception as exc:
            text = str(exc).lower()
            if "-2011" in text or "unknown order" in text:
                return
            raise ExecutionError(
                f"No se pudo cancelar order {order_id} (trigger={trigger}): {exc}"
            ) from exc

    def _cancel_bot_protection_orders(self, symbol: str) -> None:
        state = self.state_store.load()
        trigger_ids: set[str] = set()
        regular_ids: set[str] = set()

        if state.stop_order_id:
            trigger_ids.add(str(state.stop_order_id))
        if state.take_profit_order_id:
            kind = self._take_profit_kind(state)
            if kind == "limit":
                regular_ids.add(str(state.take_profit_order_id))
            else:
                trigger_ids.add(str(state.take_profit_order_id))

        try:
            for order in self._fetch_bot_trigger_orders(symbol):
                if order.get("id") is not None:
                    trigger_ids.add(str(order["id"]))
        except Exception as exc:
            raise ExecutionError(
                f"No se pudieron enumerar ordenes condicionales del bot: {exc}"
            ) from exc

        try:
            for order in self._fetch_bot_regular_tp_orders(symbol):
                if order.get("id") is not None:
                    regular_ids.add(str(order["id"]))
        except Exception as exc:
            raise ExecutionError(
                f"No se pudieron enumerar TPs LIMIT del bot: {exc}"
            ) from exc

        for order_id in trigger_ids:
            self._cancel_order_safely(symbol, order_id, trigger=True)
        for order_id in regular_ids:
            self._cancel_order_safely(symbol, order_id, trigger=False)

    def _emergency_close(self, symbol: str) -> None:
        position = self._fetch_open_position(symbol)
        if position is None:
            return

        close_side = "sell" if position.side == "long" else "buy"
        amount = float(self.exchange.amount_to_precision(symbol, position.contracts))
        if amount <= 0:
            raise ProtectionError("Cantidad invalida para cierre de emergencia.")

        client_id = self._client_id("zpanic")
        market = self._market_snapshot(symbol)
        submit_time = self._now_iso()
        try:
            close_order = self.exchange.create_order(
                symbol,
                "market",
                close_side,
                amount,
                None,
                {"reduceOnly": True, "clientOrderId": client_id},
            )
        except Exception as exc:
            self._journal(
                "emergency_close_failed",
                symbol=symbol,
                side=position.side,
                amount=amount,
                submit_time=submit_time,
                market=market,
                error=str(exc),
            )
            raise ProtectionError(f"Fallo el cierre de emergencia: {exc}") from exc

        try:
            filled, fill_price = self._resolve_market_fill(close_order, symbol)
            self._journal(
                "emergency_close_fill",
                symbol=symbol,
                side=position.side,
                order_id=close_order.get("id"),
                client_order_id=client_id,
                amount=filled,
                fill_price=fill_price,
                submit_time=submit_time,
                market=market,
                exit_slippage_bps=self._exit_slippage_bps(
                    position.side,
                    fill_price,
                    market.get("mid_price"),
                ),
                expected_liquidity="taker",
                **self._order_fee_payload(close_order),
            )
        except Exception as exc:
            self._journal(
                "emergency_close_fill_unknown",
                symbol=symbol,
                side=position.side,
                order_id=close_order.get("id"),
                submit_time=submit_time,
                error=str(exc),
            )

        for _ in range(6):
            time.sleep(0.5)
            if self._fetch_open_position(symbol) is None:
                return
        raise ProtectionError(
            "Se envio el cierre de emergencia, pero la posicion sigue abierta."
        )

    def _recover_owned_position(self, symbol: str, reason: str) -> dict[str, Any]:
        state = self.state_store.load()
        try:
            # Una LIMIT de entrada remanente podria reabrir posicion despues del
            # cierre de emergencia. Siempre se limpia antes y despues.
            self._cancel_bot_entry_orders(symbol)
            self._emergency_close(symbol)
            self._cancel_bot_entry_orders(symbol)
            self._cancel_bot_protection_orders(symbol)
            flat = self._set_flat(f"Recuperacion automatica: {reason}")
            return {
                "known": True,
                "has_position": False,
                "state": flat,
                "reason": reason,
            }
        except Exception as exc:
            state.status = "critical"
            state.note = f"{reason}. Recuperacion automatica fallo: {exc}"
            self.state_store.save(state)
            return {
                "known": False,
                "has_position": True,
                "state": state,
                "reason": state.note,
            }

    def _fetch_take_profit_order(
        self,
        symbol: str,
        state: BotState,
    ) -> Optional[dict[str, Any]]:
        if not state.take_profit_order_id:
            return None
        kind = self._take_profit_kind(state)
        if kind == "limit":
            return self._fetch_regular_order(symbol, state.take_profit_order_id)
        return self._fetch_trigger_order(symbol, state.take_profit_order_id)

    def _journal_flat_reconciliation(self, symbol: str, state: BotState) -> None:
        """Best-effort identify which protective order closed the position."""
        stop_order: Optional[dict[str, Any]] = None
        tp_order: Optional[dict[str, Any]] = None
        stop_error: Optional[str] = None
        tp_error: Optional[str] = None

        if state.stop_order_id:
            try:
                stop_order = self._fetch_trigger_order(symbol, state.stop_order_id)
            except Exception as exc:
                stop_error = str(exc)
        if state.take_profit_order_id:
            try:
                tp_order = self._fetch_take_profit_order(symbol, state)
            except Exception as exc:
                tp_error = str(exc)

        def status(order: Optional[dict[str, Any]]) -> str:
            return str((order or {}).get("status") or "unknown").lower()

        closed_statuses = {"closed", "filled"}
        stop_status = status(stop_order)
        tp_status = status(tp_order)

        exit_reason = "unknown"
        exit_order = None
        reference_price: Optional[float] = None
        if tp_status in closed_statuses:
            exit_reason = "take_profit"
            exit_order = tp_order
            reference_price = state.take_profit_price
        elif stop_status in closed_statuses:
            exit_reason = "stop_loss"
            exit_order = stop_order
            reference_price = state.stop_price

        fill_price = (
            self._average_fill_price(exit_order, symbol)
            if exit_order is not None
            else None
        )
        order_id = (str(exit_order.get("id")) if exit_order and exit_order.get("id") else None)
        since_ms = None
        if state.entry_time:
            try:
                since_ms = int(datetime.fromisoformat(state.entry_time).timestamp() * 1000)
            except Exception:
                since_ms = None
        trade_details = self._fetch_trade_details(symbol, order_id, since_ms)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "position_side": state.side,
            "exit_reason": exit_reason,
            "entry_order_id": state.entry_order_id,
            "entry_price": state.entry_price,
            "entry_time": state.entry_time,
            "stop_order_id": state.stop_order_id,
            "stop_status": stop_status,
            "stop_error": stop_error,
            "take_profit_order_id": state.take_profit_order_id,
            "take_profit_order_kind": self._take_profit_kind(state),
            "take_profit_status": tp_status,
            "take_profit_error": tp_error,
            "exit_order_id": order_id,
            "planned_exit_price": reference_price,
            "actual_exit_price": fill_price,
            "trade_details": trade_details,
            **self._order_fee_payload(exit_order),
        }
        if fill_price is not None:
            payload["exit_slippage_bps"] = self._exit_slippage_bps(
                state.side or "long",
                fill_price,
                reference_price,
            )
        self._journal("position_closed_detected", **payload)

    def _protect_filled_entry(
        self,
        *,
        symbol: str,
        side: str,
        filled: float,
        fill_price: float,
        entry_order_id: Optional[str],
        entry_client_order_id: Optional[str],
        entry_order_kind: str,
        stop_distance: float,
        risk_reward_ratio: float,
        use_take_profit: bool,
        journal_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Instala SL/TP a partir del fill real, comun a MARKET y LIMIT V8."""
        if side == "long":
            stop_price = fill_price - stop_distance
            take_profit_price = (
                fill_price + stop_distance * risk_reward_ratio
                if use_take_profit
                else None
            )
            exit_side = "sell"
        else:
            stop_price = fill_price + stop_distance
            take_profit_price = (
                fill_price - stop_distance * risk_reward_ratio
                if use_take_profit
                else None
            )
            exit_side = "buy"

        if stop_price <= 0 or (
            take_profit_price is not None and take_profit_price <= 0
        ):
            state = self.state_store.load()
            state.status = "critical"
            state.note = "Entrada abierta con niveles de proteccion invalidos."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        state = self.state_store.load()
        state.status = "entry_filled"
        state.entry_order_id = entry_order_id
        state.entry_client_order_id = entry_client_order_id
        state.entry_order_kind = entry_order_kind
        state.entry_time = self._now_iso()
        state.entry_price = fill_price
        state.amount = filled
        state.stop_price = stop_price
        state.take_profit_price = take_profit_price
        state.take_profit_order_kind = None
        state.take_profit_last_filled = 0.0
        state.protection_mode = (
            "stop_and_tp_maker"
            if use_take_profit and self.take_profit_order_mode == "maker_limit"
            else "stop_and_tp_trigger"
            if use_take_profit
            else "stop_only"
        )
        self._clear_pending_entry_fields(state)
        state.note = "Entrada ejecutada; creando proteccion reduceOnly."
        self.state_store.save(state)

        stop_client_id = self._client_id("zsl")
        stop_order: Optional[dict[str, Any]] = None
        tp_order: Optional[dict[str, Any]] = None

        try:
            stop_order = self._create_trigger_order(
                symbol=symbol,
                order_type="stop_market",
                side=exit_side,
                amount=filled,
                trigger_price=stop_price,
                client_order_id=stop_client_id,
            )
            state.status = "protecting"
            state.stop_order_id = (
                str(stop_order.get("id")) if stop_order.get("id") is not None else None
            )
            state.stop_client_order_id = stop_client_id
            self.state_store.save(state)

            if use_take_profit and take_profit_price is not None:
                tp_client_id = self._client_id("ztp")
                if self.take_profit_order_mode == "maker_limit":
                    tp_order = self._create_take_profit_limit(
                        symbol=symbol,
                        side=exit_side,
                        amount=filled,
                        price=take_profit_price,
                        client_order_id=tp_client_id,
                    )
                    state.take_profit_order_kind = "limit"
                else:
                    tp_order = self._create_trigger_order(
                        symbol=symbol,
                        order_type="take_profit_market",
                        side=exit_side,
                        amount=filled,
                        trigger_price=take_profit_price,
                        client_order_id=tp_client_id,
                    )
                    state.take_profit_order_kind = "trigger"
                state.take_profit_order_id = (
                    str(tp_order.get("id")) if tp_order.get("id") is not None else None
                )
                state.take_profit_client_order_id = tp_client_id
        except Exception as exc:
            state = self.state_store.load()
            state.status = "critical"
            state.note = f"Proteccion incompleta/ambigua: {exc}"
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            if recovery.get("known") and not recovery.get("has_position"):
                raise ProtectionError(
                    "La proteccion fallo; la posicion fue cerrada automaticamente."
                ) from exc
            raise ProtectionError(recovery.get("reason", state.note)) from exc

        if not state.stop_order_id:
            state.status = "critical"
            state.note = "Binance creo la proteccion sin devolver ID de stop."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        if use_take_profit and not state.take_profit_order_id:
            state.status = "critical"
            state.note = "Binance no devolvio ID del take profit."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        state.status = "protected"
        if use_take_profit:
            state.note = (
                "Posicion Futures protegida con STOP_MARKET + TP LIMIT maker reduceOnly."
                if state.take_profit_order_kind == "limit"
                else "Posicion Futures protegida con SL/TP trigger reduceOnly."
            )
        else:
            state.note = "Posicion Futures protegida con stop reduceOnly; salida por senal/timeout."
        self.state_store.save(state)

        self._journal(
            "protection_created",
            symbol=symbol,
            side=side,
            amount=filled,
            entry_order_id=state.entry_order_id,
            entry_order_kind=entry_order_kind,
            entry_price=fill_price,
            stop_order_id=state.stop_order_id,
            stop_client_order_id=state.stop_client_order_id,
            stop_price=stop_price,
            stop_type="STOP_MARKET",
            take_profit_order_id=state.take_profit_order_id,
            take_profit_client_order_id=state.take_profit_client_order_id,
            take_profit_price=take_profit_price,
            take_profit_order_kind=state.take_profit_order_kind,
            take_profit_post_only=(state.take_profit_order_kind == "limit"),
            protection_mode=state.protection_mode,
            context=journal_context or {},
        )

        return {
            "stop": stop_order,
            "take_profit": tp_order,
            "state": state,
        }

    def _pending_order_for_state(
        self,
        symbol: str,
        state: BotState,
    ) -> Optional[dict[str, Any]]:
        if state.entry_order_id:
            return self._fetch_regular_order(symbol, state.entry_order_id)
        if state.entry_client_order_id:
            order = self._find_open_regular_order_by_client_id(
                symbol,
                state.entry_client_order_id,
            )
            if order is not None and order.get("id") is not None:
                state.entry_order_id = str(order["id"])
                state.status = "entry_pending"
                self.state_store.save(state)
            return order
        return None

    def _cancel_pending_order_verified(
        self,
        symbol: str,
        state: BotState,
    ) -> Optional[dict[str, Any]]:
        order = self._pending_order_for_state(symbol, state)
        if order is None:
            return None
        order_id = str(order.get("id") or state.entry_order_id or "")
        if not order_id:
            raise ExecutionError("La entrada pendiente no tiene order id verificable.")

        status = str(order.get("status") or "unknown").lower()
        if status in {"open", "new", "partially_filled", "partial"}:
            self._cancel_order_safely(symbol, order_id, trigger=False)

        try:
            resolved = self._fetch_regular_order(symbol, order_id)
        except Exception as exc:
            raise ExecutionError(
                f"No se pudo verificar la cancelacion de la entrada pendiente: {exc}"
            ) from exc

        final_status = str(resolved.get("status") or "unknown").lower()
        if final_status in {"open", "new", "partially_filled", "partial"}:
            raise ExecutionError(
                f"La entrada pendiente sigue abierta tras cancelar: {final_status}."
            )
        return resolved

    def _adopt_pending_fill(
        self,
        symbol: str,
        state: BotState,
        position: PositionSnapshot,
        order: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Cancela remanente de un fill parcial y protege lo ya ejecutado."""
        context = dict(state.pending_journal_context or {})
        stop_distance = float(state.pending_stop_distance or 0.0)
        rr = float(state.pending_risk_reward_ratio or 0.0)
        use_take_profit = bool(state.pending_use_take_profit)
        limit_reference = state.pending_reference_price
        fallback_reference = (
            state.pending_fallback_reference_price
            if state.pending_fallback_reference_price is not None
            else limit_reference
        )
        limit_price = state.pending_limit_price
        client_id = state.entry_client_order_id

        if stop_distance <= 0 or rr <= 0:
            raise ExecutionError(
                "La entrada pendiente tiene parametros de proteccion incompletos."
            )

        # Si la LIMIT sigue abierta/partial, primero bloqueamos nuevos fills.
        if state.entry_order_id or client_id:
            try:
                order = self._cancel_pending_order_verified(symbol, state) or order
            except Exception as exc:
                state.status = "critical"
                state.note = f"Fill parcial detectado pero no se pudo cancelar remanente: {exc}"
                self.state_store.save(state)
                recovery = self._recover_owned_position(symbol, state.note)
                raise ProtectionError(recovery.get("reason", state.note)) from exc

        # Releer posicion despues de cancelar para capturar cualquier fill en carrera.
        refreshed = self._fetch_open_position(symbol)
        if refreshed is None:
            raise ExecutionError(
                "La LIMIT figura ejecutada/parcial pero Binance no muestra posicion."
            )
        position = refreshed
        fill_price = float(position.entry_price or 0.0)
        if fill_price <= 0 and order is not None:
            fill_price = float(self._average_fill_price(order, symbol) or 0.0)
        if fill_price <= 0:
            raise ExecutionError("No se puede determinar el precio real del fill LIMIT.")

        order_id = state.entry_order_id
        self._journal(
            "entry_fill",
            symbol=symbol,
            side=state.side,
            order_id=order_id,
            client_order_id=client_id,
            fill_time=self._now_iso(),
            requested_amount=state.amount,
            filled_amount=position.contracts,
            fill_price=fill_price,
            expected_entry_price=fallback_reference,
            entry_limit_reference_price=limit_reference,
            fallback_reference_price=fallback_reference,
            entry_limit_price=limit_price,
            entry_source="limit_maker",
            entry_slippage_bps=self._entry_slippage_bps(
                state.side or "long",
                fill_price,
                fallback_reference,
            ),
            expected_liquidity="maker",
            trade_details=self._fetch_trade_details(symbol, order_id),
            context=context,
            **self._order_fee_payload(order),
        )

        protected = self._protect_filled_entry(
            symbol=symbol,
            side=state.side or position.side,
            filled=position.contracts,
            fill_price=fill_price,
            entry_order_id=order_id,
            entry_client_order_id=client_id,
            entry_order_kind="limit",
            stop_distance=stop_distance,
            risk_reward_ratio=rr,
            use_take_profit=use_take_profit,
            journal_context=context,
        )
        protected["position"] = position
        return protected

    def submit_passive_entry(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        reference_price: float,
        limit_offset_bps: float,
        wait_seconds: int,
        fallback_cap_bps: float,
        fallback_max_lateness_seconds: int,
        stop_distance: float,
        risk_reward_ratio: float,
        use_take_profit: bool,
        signal_timestamp: str,
        fallback_reference_price: Optional[float] = None,
        journal_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Crea la entrada V8 final persistente.

        V8 final usa dos referencias distintas:
        - reference_price: close de la vela de senal; ancla la LIMIT +/- offset.
        - fallback_reference_price: precio observado al iniciar (t=0); ancla el
          cap no-chase del fallback.

        Si la LIMIT ya seria marketable en t=0, no se fuerza una orden postOnly
        que Binance rechazaria. Se persiste un estado wait-only y, al expirar,
        se evalua el mismo fallback condicional que en el research V8 final.
        """
        if side not in {"long", "short"}:
            raise ValueError("side debe ser long o short.")

        fallback_reference = float(
            fallback_reference_price
            if fallback_reference_price is not None
            else reference_price
        )
        if min(
            amount,
            reference_price,
            fallback_reference,
            limit_offset_bps,
            stop_distance,
            risk_reward_ratio,
        ) <= 0:
            raise ValueError("Parametros de entrada LIMIT invalidos.")
        if (
            wait_seconds <= 0
            or fallback_cap_bps < 0
            or fallback_max_lateness_seconds < 0
        ):
            raise ValueError("Parametros temporales/cap de entrada LIMIT invalidos.")

        current = self.get_position_state(symbol)
        if not current.get("known") or current.get("has_position"):
            raise ExecutionError(
                f"No es seguro crear entrada pendiente: "
                f"{current.get('reason', 'posicion existente')}."
            )
        if current.get("entry_pending"):
            raise ExecutionError("Ya existe una entrada LIMIT pendiente.")

        if side == "long":
            raw_limit = reference_price * (1.0 - limit_offset_bps / 10_000.0)
        else:
            raw_limit = reference_price * (1.0 + limit_offset_bps / 10_000.0)
        limit_price = float(self.exchange.price_to_precision(symbol, raw_limit))
        if limit_price <= 0:
            raise ExecutionError("Precio LIMIT de entrada invalido tras precision.")

        # Misma condicion conservadora que V8 final: la orden solo se considera
        # postOnly si puede descansar desde t=0. Si ya seria marketable, se
        # espera sin orden hasta la decision de fallback.
        resting_possible = (
            fallback_reference > limit_price
            if side == "long"
            else fallback_reference < limit_price
        )

        now = datetime.now(timezone.utc)
        expires_at = now.timestamp() + wait_seconds
        expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
        client_id = self._client_id("zentry") if resting_possible else None

        state = self.state_store.load()
        state.status = "entry_pending_submitting" if resting_possible else "entry_pending"
        state.symbol = symbol
        state.side = side
        state.amount = amount
        state.entry_order_id = None
        state.entry_client_order_id = client_id
        state.entry_order_kind = "limit" if resting_possible else "wait"
        state.pending_reference_price = reference_price
        state.pending_fallback_reference_price = fallback_reference
        state.pending_limit_price = limit_price
        state.pending_expires_at = expires_iso
        state.pending_stop_distance = stop_distance
        state.pending_risk_reward_ratio = risk_reward_ratio
        state.pending_use_take_profit = use_take_profit
        state.pending_fallback_cap_bps = fallback_cap_bps
        state.pending_fallback_max_lateness_seconds = fallback_max_lateness_seconds
        state.pending_journal_context = dict(journal_context or {})
        state.last_signal_timestamp = signal_timestamp
        state.protection_mode = (
            "stop_and_tp_maker"
            if use_take_profit and self.take_profit_order_mode == "maker_limit"
            else "stop_and_tp_trigger"
            if use_take_profit
            else "stop_only"
        )

        if not resting_possible:
            state.note = (
                "V8 final: LIMIT no puede descansar postOnly en t=0; "
                "esperando decision de fallback."
            )
            self.state_store.save(state)
            self._journal(
                "entry_pending_wait_only",
                symbol=symbol,
                side=side,
                amount=amount,
                limit_reference_price=reference_price,
                fallback_reference_price=fallback_reference,
                limit_price=limit_price,
                limit_offset_bps=limit_offset_bps,
                expires_at=expires_iso,
                fallback_cap_bps=fallback_cap_bps,
                fallback_max_lateness_seconds=fallback_max_lateness_seconds,
                context=journal_context or {},
            )
            return {"order": None, "state": state}

        state.note = "Enviando entrada LIMIT postOnly V8 final."
        self.state_store.save(state)

        submit_time = self._now_iso()
        try:
            order = self._create_entry_limit(
                symbol=symbol,
                side=side,
                amount=amount,
                price=limit_price,
                client_order_id=client_id or "",
            )
        except ccxt.NetworkError as exc:
            state.status = "entry_pending_unknown"
            state.note = "Resultado ambiguo al crear LIMIT de entrada; se reconciliara."
            self.state_store.save(state)
            self._journal(
                "entry_pending_ambiguous",
                symbol=symbol,
                side=side,
                amount=amount,
                limit_reference_price=reference_price,
                fallback_reference_price=fallback_reference,
                limit_price=limit_price,
                client_order_id=client_id,
                submit_time=submit_time,
                context=journal_context or {},
                error=str(exc),
            )
            raise ExecutionError(state.note) from exc
        except Exception as exc:
            error_text = str(exc).lower()
            post_only_rejection = (
                "-5022" in error_text
                or "post only" in error_text
                or "post-only" in error_text
                or "could not be executed as maker" in error_text
            )
            if post_only_rejection:
                # El research V8 final no fuerza un fill taker si la LIMIT no puede
                # descansar en t=0. Conservamos la senal en wait-only y
                # dejamos que el fallback/no-chase decida al expirar.
                state.status = "entry_pending"
                state.entry_order_id = None
                state.entry_client_order_id = None
                state.entry_order_kind = "wait"
                state.note = (
                    "V8 final: Binance rechazo postOnly por marketable; "
                    "esperando decision de fallback."
                )
                self.state_store.save(state)
                self._journal(
                    "entry_pending_wait_only",
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    limit_reference_price=reference_price,
                    fallback_reference_price=fallback_reference,
                    limit_price=limit_price,
                    limit_offset_bps=limit_offset_bps,
                    expires_at=expires_iso,
                    fallback_cap_bps=fallback_cap_bps,
                    fallback_max_lateness_seconds=fallback_max_lateness_seconds,
                    context=journal_context or {},
                    reason="post_only_rejected_marketable",
                    error=str(exc),
                )
                return {"order": None, "state": state}

            self._journal(
                "entry_pending_rejected",
                symbol=symbol,
                side=side,
                amount=amount,
                limit_reference_price=reference_price,
                fallback_reference_price=fallback_reference,
                limit_price=limit_price,
                client_order_id=client_id,
                submit_time=submit_time,
                context=journal_context or {},
                error=str(exc),
            )
            self._set_flat(f"LIMIT de entrada rechazada/no enviada: {exc}")
            raise ExecutionError(f"No se pudo crear LIMIT de entrada: {exc}") from exc

        order_id = str(order.get("id")) if order.get("id") is not None else None
        if not order_id:
            state.status = "entry_pending_unknown"
            state.note = "Binance creo la LIMIT de entrada sin devolver order id."
            self.state_store.save(state)
            raise ExecutionError(state.note)

        state.entry_order_id = order_id
        state.status = "entry_pending"
        state.note = "Entrada LIMIT postOnly V8 final pendiente de fill."
        self.state_store.save(state)
        self._journal(
            "entry_pending_created",
            symbol=symbol,
            side=side,
            order_id=order_id,
            client_order_id=client_id,
            amount=amount,
            limit_reference_price=reference_price,
            fallback_reference_price=fallback_reference,
            limit_price=limit_price,
            limit_offset_bps=limit_offset_bps,
            expires_at=expires_iso,
            fallback_cap_bps=fallback_cap_bps,
            fallback_max_lateness_seconds=fallback_max_lateness_seconds,
            expected_liquidity="maker",
            context=journal_context or {},
        )
        return {"order": order, "state": state}

    def resolve_pending_entry(self, symbol: str) -> dict[str, Any]:
        """Expira la entrada V8 final y aplica fallback MARKET solo dentro del cap."""
        state = self.state_store.load()
        if state.status not in self.ENTRY_PENDING_STATUSES:
            return {"action": "not_pending", "state": state}

        position = self._fetch_open_position(symbol)
        if position is not None:
            if state.entry_order_kind == "wait":
                return {
                    "action": "unknown_position",
                    "state": state,
                    "reason": (
                        "Aparecio una posicion mientras V8 final estaba en wait-only; "
                        "no puede atribuirse a una orden del bot."
                    ),
                }
            protected = self._adopt_pending_fill(symbol, state, position)
            return {"action": "limit_filled", **protected}

        remaining = self.pending_entry_remaining_seconds(state)
        if remaining is None:
            raise ExecutionError("Entrada pendiente sin expiracion valida.")
        if remaining > 0:
            return {
                "action": "pending",
                "remaining_seconds": remaining,
                "state": state,
            }

        # Capturar parametros antes de que _set_flat limpie el pending state.
        limit_reference = float(state.pending_reference_price or 0.0)
        fallback_reference = float(
            state.pending_fallback_reference_price
            if state.pending_fallback_reference_price is not None
            else state.pending_reference_price
            or 0.0
        )
        cap_bps = float(state.pending_fallback_cap_bps or 0.0)
        max_lateness = int(state.pending_fallback_max_lateness_seconds or 0)
        stop_distance = float(state.pending_stop_distance or 0.0)
        rr = float(state.pending_risk_reward_ratio or 0.0)
        use_take_profit = bool(state.pending_use_take_profit)
        amount = float(state.amount or 0.0)
        side = state.side or ""
        context = dict(state.pending_journal_context or {})
        expires_at = self._iso_to_utc(state.pending_expires_at)

        if (
            limit_reference <= 0
            or fallback_reference <= 0
            or stop_distance <= 0
            or rr <= 0
            or amount <= 0
        ):
            raise ExecutionError(
                "Entrada pendiente sin parametros suficientes para fallback."
            )

        now = datetime.now(timezone.utc)
        lateness = (
            (now - expires_at).total_seconds()
            if expires_at is not None
            else float("inf")
        )

        # Tomamos la decision de fallback con el precio observado al expirar
        # ANTES de cancelar. Si el ticker falla, una LIMIT real sigue viva y
        # podemos reintentar sin perder innecesariamente la senal.
        current_price: Optional[float] = None
        cap_price: Optional[float] = None
        within_cap = False
        if lateness <= max_lateness:
            current_price = self.get_market_price(symbol)
            if side == "long":
                cap_price = fallback_reference * (1.0 + cap_bps / 10_000.0)
                within_cap = current_price <= cap_price
            elif side == "short":
                cap_price = fallback_reference * (1.0 - cap_bps / 10_000.0)
                within_cap = current_price >= cap_price
            else:
                raise ExecutionError(f"Side pendiente no valido: {side}")

        resolved_order = None
        if state.entry_order_kind == "limit":
            resolved_order = self._cancel_pending_order_verified(symbol, state)

            # Evitar carrera cancel/fill: tras cancelar, comprobar varias veces
            # si aparecio posicion. Si aparece, se protege y NO se envia fallback.
            for _ in range(4):
                position = self._fetch_open_position(symbol)
                if position is not None:
                    protected = self._adopt_pending_fill(
                        symbol,
                        state,
                        position,
                        order=resolved_order,
                    )
                    return {"action": "limit_filled", **protected}
                time.sleep(0.2)

        if lateness > max_lateness:
            flat = self._set_flat(
                f"Entrada V8 final cancelada: fallback demasiado tarde ({lateness:.0f}s)."
            )
            self._journal(
                "entry_cancelled_late",
                symbol=symbol,
                side=side,
                limit_reference_price=limit_reference,
                fallback_reference_price=fallback_reference,
                lateness_seconds=lateness,
                max_lateness_seconds=max_lateness,
                context=context,
            )
            return {"action": "cancelled_late", "state": flat}

        assert current_price is not None and cap_price is not None
        if not within_cap:
            flat = self._set_flat(
                "Entrada V8 final cancelada: precio fuera del cap no-chase."
            )
            self._journal(
                "entry_cancelled_no_chase",
                symbol=symbol,
                side=side,
                limit_reference_price=limit_reference,
                fallback_reference_price=fallback_reference,
                current_price=current_price,
                cap_price=cap_price,
                fallback_cap_bps=cap_bps,
                context=context,
            )
            return {
                "action": "cancelled_no_chase",
                "state": flat,
                "current_price": current_price,
                "cap_price": cap_price,
            }

        self._journal(
            "entry_fallback_selected",
            symbol=symbol,
            side=side,
            limit_reference_price=limit_reference,
            fallback_reference_price=fallback_reference,
            current_price=current_price,
            cap_price=cap_price,
            fallback_cap_bps=cap_bps,
            context=context,
        )
        self._set_flat(
            "Entrada V8 final expirada; ejecutando fallback MARKET dentro del cap."
        )
        fallback_context = {
            **context,
            "entry_source": "market_fallback_conditional",
            "entry_reference_price": fallback_reference,
            "entry_limit_reference_price": limit_reference,
            "fallback_market_price": current_price,
            "fallback_cap_bps": cap_bps,
        }
        result = self.execute_trade_with_protection(
            symbol=symbol,
            side=side,
            amount=amount,
            stop_distance=stop_distance,
            risk_reward_ratio=rr,
            use_take_profit=use_take_profit,
            journal_context=fallback_context,
        )
        return {
            "action": "fallback_market",
            **result,
            "current_price": current_price,
            "cap_price": cap_price,
        }

    def get_position_state(self, symbol: str) -> dict[str, Any]:
        """Tri-state flat/open/unknown; un error nunca se interpreta como flat."""
        state = self.state_store.load()
        try:
            position = self._fetch_open_position(symbol)
        except Exception as exc:
            return {
                "known": False,
                "has_position": None,
                "state": state,
                "reason": str(exc),
            }

        if state.status in self.ENTRY_PENDING_STATUSES:
            # V8 final wait-only: la LIMIT no pudo descansar postOnly en t=0, por
            # lo que no existe una orden del bot que pueda atribuir una posicion.
            if state.entry_order_kind == "wait":
                if position is not None:
                    return {
                        "known": False,
                        "has_position": True,
                        "state": state,
                        "position": position,
                        "reason": (
                            "Aparecio una posicion durante un wait-only V8 final; "
                            "no se puede atribuir al bot de forma segura."
                        ),
                    }
                return {
                    "known": True,
                    "has_position": False,
                    "state": state,
                    "entry_pending": True,
                    "pending_order_status": "wait_only",
                    "pending_order": None,
                }

            if position is not None:
                try:
                    protected = self._adopt_pending_fill(symbol, state, position)
                    refreshed = self._fetch_open_position(symbol) or position
                    return {
                        "known": True,
                        "has_position": True,
                        "state": protected["state"],
                        "position": refreshed,
                        "entry_pending": False,
                    }
                except Exception as exc:
                    current = self.state_store.load()
                    return {
                        "known": False,
                        "has_position": True,
                        "state": current,
                        "position": position,
                        "reason": f"Fill LIMIT detectado pero no pudo protegerse: {exc}",
                    }

            try:
                pending_order = self._pending_order_for_state(symbol, state)
            except Exception as exc:
                return {
                    "known": False,
                    "has_position": False,
                    "state": state,
                    "reason": f"No se pudo verificar entrada LIMIT pendiente: {exc}",
                }

            if pending_order is None:
                # Tras un timeout de create_order puede existir una pequena carrera:
                # la orden ya se ejecuto pero positionRisk aun no la refleja. Durante
                # unos segundos preferimos UNKNOWN a declarar flat prematuramente.
                updated_at = self._iso_to_utc(state.updated_at)
                age = (
                    (datetime.now(timezone.utc) - updated_at).total_seconds()
                    if updated_at is not None
                    else float("inf")
                )
                if age < self.ENTRY_AMBIGUITY_GRACE_SECONDS:
                    return {
                        "known": False,
                        "has_position": False,
                        "state": state,
                        "reason": (
                            "Entrada LIMIT ambigua sin orden/posicion visible; "
                            f"reconciliacion en gracia ({age:.0f}s)."
                        ),
                    }

                flat = self._set_flat(
                    "No hay posicion ni LIMIT pendiente tras reconciliacion."
                )
                self._journal(
                    "entry_pending_missing",
                    symbol=symbol,
                    side=state.side,
                    client_order_id=state.entry_client_order_id,
                )
                return {
                    "known": True,
                    "has_position": False,
                    "state": flat,
                    "entry_pending": False,
                }

            pending_status = str(pending_order.get("status") or "unknown").lower()
            pending_filled = float(pending_order.get("filled") or 0.0)
            if pending_filled > 0:
                return {
                    "known": False,
                    "has_position": False,
                    "state": state,
                    "reason": (
                        "La LIMIT muestra fill parcial/completo pero la posicion aun "
                        "no es visible; se reintentara antes de cualquier accion."
                    ),
                }

            if pending_status in {"open", "new"}:
                if state.status != "entry_pending":
                    state.status = "entry_pending"
                    state.entry_order_id = (
                        str(pending_order.get("id"))
                        if pending_order.get("id") is not None
                        else state.entry_order_id
                    )
                    state.note = "Entrada LIMIT postOnly pendiente de fill."
                    self.state_store.save(state)
                return {
                    "known": True,
                    "has_position": False,
                    "state": state,
                    "entry_pending": True,
                    "pending_order_status": pending_status,
                    "pending_order": pending_order,
                }

            if pending_status in {"closed", "filled"}:
                return {
                    "known": False,
                    "has_position": False,
                    "state": state,
                    "reason": (
                        "La LIMIT figura ejecutada pero la posicion no es visible todavia."
                    ),
                }

            if pending_status in {"canceled", "cancelled", "rejected", "expired"}:
                flat = self._set_flat(
                    f"Entrada LIMIT termino sin posicion: {pending_status}."
                )
                return {
                    "known": True,
                    "has_position": False,
                    "state": flat,
                    "entry_pending": False,
                }

            return {
                "known": False,
                "has_position": False,
                "state": state,
                "reason": f"Estado no reconocido de LIMIT pendiente: {pending_status}",
            }

        if position is None:
            if state.status != "flat" or state.stop_order_id or state.take_profit_order_id:
                self._journal_flat_reconciliation(symbol, state)
                try:
                    self._cancel_bot_entry_orders(symbol)
                    self._cancel_bot_protection_orders(symbol)
                except Exception as exc:
                    state.status = "critical"
                    state.note = f"Posicion cerrada pero quedan ordenes sin reconciliar: {exc}"
                    self.state_store.save(state)
                    return {
                        "known": False,
                        "has_position": False,
                        "state": state,
                        "reason": state.note,
                    }
                state = self._set_flat("Binance confirma que no hay posicion abierta.")
            return {"known": True, "has_position": False, "state": state}

        if state.status == "flat":
            return {
                "known": False,
                "has_position": True,
                "state": state,
                "position": position,
                "reason": (
                    "Existe una posicion Futures que no pertenece al estado del bot. "
                    "No se tocara automaticamente."
                ),
            }

        if state.status in {
            "entry_submitting",
            "entry_unknown",
            "entry_filled",
            "protecting",
            "closing",
            "critical",
        }:
            return self._recover_owned_position(
                symbol,
                f"Posicion del bot detectada en estado inseguro '{state.status}'",
            )

        if state.status != "protected":
            return {
                "known": False,
                "has_position": True,
                "state": state,
                "position": position,
                "reason": f"Estado local desconocido: {state.status}",
            }

        if state.side != position.side:
            return self._recover_owned_position(
                symbol,
                f"Side local={state.side} no coincide con Binance={position.side}",
            )

        if not state.stop_order_id:
            return self._recover_owned_position(
                symbol,
                "La posicion figura protegida pero falta el stop reduceOnly",
            )

        try:
            stop_status = self._fetch_trigger_order_status(symbol, state.stop_order_id)
        except Exception as exc:
            return {
                "known": False,
                "has_position": True,
                "state": state,
                "position": position,
                "reason": f"No se pudo verificar el stop: {exc}",
            }

        if stop_status != "open":
            return self._recover_owned_position(
                symbol,
                f"Stop no abierto mientras la posicion sigue viva: SL={stop_status}",
            )

        tp_status: Optional[str] = None
        if state.take_profit_order_id:
            try:
                tp_kind = self._take_profit_kind(state)
                if tp_kind == "limit":
                    tp_order = self._fetch_regular_order(
                        symbol,
                        state.take_profit_order_id,
                    )
                    tp_status = str(tp_order.get("status") or "unknown").lower()
                    tp_filled = float(tp_order.get("filled") or 0.0)
                    if tp_filled > float(state.take_profit_last_filled or 0.0):
                        self._journal(
                            "take_profit_partial_fill",
                            symbol=symbol,
                            side=state.side,
                            order_id=state.take_profit_order_id,
                            cumulative_filled=tp_filled,
                            previous_filled=state.take_profit_last_filled,
                            remaining=tp_order.get("remaining"),
                            average_fill_price=self._average_fill_price(tp_order, symbol),
                            expected_liquidity="maker",
                            trade_details=self._fetch_trade_details(
                                symbol,
                                state.take_profit_order_id,
                            ),
                            **self._order_fee_payload(tp_order),
                        )
                        state.take_profit_last_filled = tp_filled
                        self.state_store.save(state)
                else:
                    tp_status = self._fetch_trigger_order_status(
                        symbol,
                        state.take_profit_order_id,
                    )
            except Exception as exc:
                return {
                    "known": False,
                    "has_position": True,
                    "state": state,
                    "position": position,
                    "reason": f"No se pudo verificar el TP: {exc}",
                }
            if tp_status != "open":
                return self._recover_owned_position(
                    symbol,
                    f"TP no abierto mientras la posicion sigue viva: TP={tp_status}",
                )

        return {
            "known": True,
            "has_position": True,
            "state": state,
            "position": position,
            "stop_status": stop_status,
            "take_profit_status": tp_status,
        }

    def close_position_market(self, symbol: str, reason: str) -> BotState:
        """Cierra la posicion del bot en mercado y elimina SL/TP restantes."""
        state = self.state_store.load()
        position = self._fetch_open_position(symbol)
        if position is None:
            self._cancel_bot_protection_orders(symbol)
            return self._set_flat(f"Cierre solicitado sin posicion: {reason}")

        if state.status != "protected" or state.side != position.side:
            raise ExecutionError(
                "No es seguro cerrar por senal: el estado local no coincide con Binance."
            )

        state.status = "closing"
        state.note = reason
        self.state_store.save(state)

        close_side = "sell" if position.side == "long" else "buy"
        amount = float(self.exchange.amount_to_precision(symbol, position.contracts))
        if amount <= 0:
            state.status = "critical"
            state.note = "Cantidad invalida para cierre por senal."
            self.state_store.save(state)
            raise ExecutionError(state.note)

        client_id = self._client_id("zexit")
        market = self._market_snapshot(symbol)
        submit_time = self._now_iso()
        try:
            close_order = self.exchange.create_order(
                symbol,
                "market",
                close_side,
                amount,
                None,
                {
                    "reduceOnly": True,
                    "clientOrderId": client_id,
                },
            )
        except Exception as exc:
            state.status = "critical"
            state.note = f"Cierre market ambiguo/fallido: {reason}: {exc}"
            self.state_store.save(state)
            self._journal(
                "market_exit_failed",
                symbol=symbol,
                side=position.side,
                reason=reason,
                amount=amount,
                submit_time=submit_time,
                market=market,
                error=str(exc),
            )
            raise ExecutionError(state.note) from exc

        try:
            filled, fill_price = self._resolve_market_fill(close_order, symbol)

            close_order_id = (
                str(close_order.get("id"))
                if close_order.get("id") is not None
                else None
            )
            try:
                since_ms = (
                    int(datetime.fromisoformat(submit_time).timestamp() * 1000)
                    - 60_000
                )
            except Exception:
                since_ms = None

            trade_details = self._fetch_trade_details(
                symbol,
                close_order_id,
                since_ms,
            )

            self._journal(
                "market_exit_fill",
                symbol=symbol,
                side=position.side,
                reason=reason,
                order_id=close_order_id,
                client_order_id=client_id,
                amount=filled,
                fill_price=fill_price,
                submit_time=submit_time,
                market=market,
                exit_slippage_bps=self._exit_slippage_bps(
                    position.side,
                    fill_price,
                    market.get("mid_price"),
                ),
                expected_liquidity="taker",
                trade_details=trade_details,
                **self._order_fee_payload(close_order),
            )
        except Exception as exc:
            self._journal(
                "market_exit_fill_unknown",
                symbol=symbol,
                side=position.side,
                reason=reason,
                order_id=close_order.get("id"),
                submit_time=submit_time,
                error=str(exc),
            )

        for _ in range(6):
            time.sleep(0.5)
            if self._fetch_open_position(symbol) is None:
                break
        else:
            state.status = "critical"
            state.note = f"Cierre enviado pero la posicion sigue abierta: {reason}"
            self.state_store.save(state)
            raise ExecutionError(state.note)

        try:
            self._cancel_bot_protection_orders(symbol)
        except Exception as exc:
            state.status = "critical"
            state.note = f"Posicion cerrada pero fallo limpieza de SL/TP: {exc}"
            self.state_store.save(state)
            raise ExecutionError(state.note) from exc

        return self._set_flat(reason)

    def execute_trade_with_protection(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_distance: float,
        risk_reward_ratio: float,
        use_take_profit: bool = True,
        journal_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if side not in {"long", "short"}:
            raise ValueError("side debe ser 'long' o 'short'.")
        if amount <= 0 or stop_distance <= 0 or risk_reward_ratio <= 0:
            raise ValueError(
                "amount, stop_distance y risk_reward_ratio deben ser positivos."
            )

        current = self.get_position_state(symbol)
        if not current.get("known") or current.get("has_position"):
            raise ExecutionError(
                f"No es seguro abrir posicion: {current.get('reason', 'posicion existente')}."
            )

        state = self.state_store.load()
        entry_client_id = self._client_id("zentry")
        state.status = "entry_submitting"
        state.symbol = symbol
        state.side = side
        state.amount = amount
        state.entry_client_order_id = entry_client_id
        state.entry_order_kind = "market"
        state.take_profit_order_kind = None
        state.take_profit_last_filled = 0.0
        if use_take_profit:
            state.protection_mode = (
                "stop_and_tp_maker"
                if self.take_profit_order_mode == "maker_limit"
                else "stop_and_tp_trigger"
            )
        else:
            state.protection_mode = "stop_only"
        state.note = "Enviando entrada Futures."
        self.state_store.save(state)

        entry_side = "buy" if side == "long" else "sell"
        entry_market = self._market_snapshot(symbol)
        entry_submit_time = self._now_iso()
        try:
            entry = self.exchange.create_order(
                symbol,
                "market",
                entry_side,
                amount,
                None,
                {"clientOrderId": entry_client_id},
            )
        except ccxt.NetworkError as exc:
            state.status = "entry_unknown"
            state.note = (
                "Resultado de entrada ambiguo por error de red; se reconciliara "
                "antes de cualquier nueva operacion."
            )
            self.state_store.save(state)
            self._journal(
                "entry_ambiguous",
                symbol=symbol,
                side=side,
                amount=amount,
                submit_time=entry_submit_time,
                market=entry_market,
                context=journal_context or {},
                error=str(exc),
            )
            raise ExecutionError(state.note) from exc
        except Exception as exc:
            self._journal(
                "entry_rejected",
                symbol=symbol,
                side=side,
                amount=amount,
                submit_time=entry_submit_time,
                market=entry_market,
                context=journal_context or {},
                error=str(exc),
            )
            self._set_flat(f"Entrada rechazada/no enviada: {exc}")
            raise ExecutionError(f"No se pudo abrir {side}: {exc}") from exc

        try:
            filled, fill_price = self._resolve_market_fill(entry, symbol)
        except Exception as exc:
            state = self.state_store.load()
            state.status = "entry_unknown"
            state.entry_order_id = str(entry.get("id")) if entry.get("id") else None
            state.note = f"Entrada enviada pero fill no confirmable: {exc}"
            self.state_store.save(state)
            self._journal(
                "entry_fill_unknown",
                symbol=symbol,
                side=side,
                order_id=state.entry_order_id,
                submit_time=entry_submit_time,
                market=entry_market,
                context=journal_context or {},
                error=str(exc),
            )
            raise ExecutionError(state.note) from exc

        entry_fill_time = self._now_iso()
        entry_order_id = str(entry.get("id")) if entry.get("id") else None
        expected_reference = None
        if journal_context is not None:
            try:
                expected_reference = float(journal_context.get("expected_entry_price"))
            except (TypeError, ValueError):
                expected_reference = None
        slippage_reference = entry_market.get("mid_price") or expected_reference
        try:
            since_ms = int(datetime.fromisoformat(entry_submit_time).timestamp() * 1000) - 60_000
        except Exception:
            since_ms = None
        self._journal(
            "entry_fill",
            symbol=symbol,
            side=side,
            order_id=entry_order_id,
            client_order_id=entry_client_id,
            submit_time=entry_submit_time,
            fill_time=entry_fill_time,
            requested_amount=amount,
            filled_amount=filled,
            fill_price=fill_price,
            expected_entry_price=expected_reference,
            market=entry_market,
            entry_slippage_bps=self._entry_slippage_bps(
                side,
                fill_price,
                slippage_reference,
            ),
            expected_liquidity="taker",
            trade_details=self._fetch_trade_details(symbol, entry_order_id, since_ms),
            context=journal_context or {},
            **self._order_fee_payload(entry),
        )

        if side == "long":
            stop_price = fill_price - stop_distance
            take_profit_price = (
                fill_price + stop_distance * risk_reward_ratio
                if use_take_profit
                else None
            )
            exit_side = "sell"
        else:
            stop_price = fill_price + stop_distance
            take_profit_price = (
                fill_price - stop_distance * risk_reward_ratio
                if use_take_profit
                else None
            )
            exit_side = "buy"

        if stop_price <= 0 or (
            take_profit_price is not None and take_profit_price <= 0
        ):
            state = self.state_store.load()
            state.status = "critical"
            state.note = "Entrada abierta con niveles de proteccion invalidos."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        state = self.state_store.load()
        state.status = "entry_filled"
        state.entry_order_id = entry_order_id
        state.entry_time = entry_fill_time
        state.entry_price = fill_price
        state.amount = filled
        state.stop_price = stop_price
        state.take_profit_price = take_profit_price
        state.note = "Entrada ejecutada; creando proteccion reduceOnly."
        self.state_store.save(state)

        stop_client_id = self._client_id("zsl")
        stop_order: Optional[dict[str, Any]] = None
        tp_order: Optional[dict[str, Any]] = None

        try:
            stop_order = self._create_trigger_order(
                symbol=symbol,
                order_type="stop_market",
                side=exit_side,
                amount=filled,
                trigger_price=stop_price,
                client_order_id=stop_client_id,
            )
            state.status = "protecting"
            state.stop_order_id = (
                str(stop_order.get("id")) if stop_order.get("id") is not None else None
            )
            state.stop_client_order_id = stop_client_id
            self.state_store.save(state)

            if use_take_profit and take_profit_price is not None:
                tp_client_id = self._client_id("ztp")
                if self.take_profit_order_mode == "maker_limit":
                    tp_order = self._create_take_profit_limit(
                        symbol=symbol,
                        side=exit_side,
                        amount=filled,
                        price=take_profit_price,
                        client_order_id=tp_client_id,
                    )
                    state.take_profit_order_kind = "limit"
                else:
                    tp_order = self._create_trigger_order(
                        symbol=symbol,
                        order_type="take_profit_market",
                        side=exit_side,
                        amount=filled,
                        trigger_price=take_profit_price,
                        client_order_id=tp_client_id,
                    )
                    state.take_profit_order_kind = "trigger"
                state.take_profit_order_id = (
                    str(tp_order.get("id")) if tp_order.get("id") is not None else None
                )
                state.take_profit_client_order_id = tp_client_id
        except Exception as exc:
            state = self.state_store.load()
            state.status = "critical"
            state.note = f"Proteccion incompleta/ambigua: {exc}"
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            if recovery.get("known") and not recovery.get("has_position"):
                raise ProtectionError(
                    "La proteccion fallo; la posicion fue cerrada automaticamente."
                ) from exc
            raise ProtectionError(recovery.get("reason", state.note)) from exc

        if not state.stop_order_id:
            state.status = "critical"
            state.note = "Binance creo la proteccion sin devolver ID de stop."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        if use_take_profit and not state.take_profit_order_id:
            state.status = "critical"
            state.note = "Binance no devolvio ID del take profit."
            self.state_store.save(state)
            recovery = self._recover_owned_position(symbol, state.note)
            raise ProtectionError(recovery.get("reason", state.note))

        state.status = "protected"
        if use_take_profit:
            state.note = (
                "Posicion Futures protegida con STOP_MARKET + TP LIMIT maker reduceOnly."
                if state.take_profit_order_kind == "limit"
                else "Posicion Futures protegida con SL/TP trigger reduceOnly."
            )
        else:
            state.note = "Posicion Futures protegida con stop reduceOnly; salida por senal/timeout."
        self.state_store.save(state)

        self._journal(
            "protection_created",
            symbol=symbol,
            side=side,
            amount=filled,
            entry_order_id=state.entry_order_id,
            entry_price=fill_price,
            stop_order_id=state.stop_order_id,
            stop_client_order_id=state.stop_client_order_id,
            stop_price=stop_price,
            stop_type="STOP_MARKET",
            take_profit_order_id=state.take_profit_order_id,
            take_profit_client_order_id=state.take_profit_client_order_id,
            take_profit_price=take_profit_price,
            take_profit_order_kind=state.take_profit_order_kind,
            take_profit_post_only=(state.take_profit_order_kind == "limit"),
            protection_mode=state.protection_mode,
            context=journal_context or {},
        )

        return {
            "entry": entry,
            "stop": stop_order,
            "take_profit": tp_order,
            "state": state,
        }
