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
    entry_time: Optional[str] = None
    entry_price: Optional[float] = None
    amount: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_order_id: Optional[str] = None
    stop_client_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    take_profit_client_order_id: Optional[str] = None
    # "limit" para V7 maker TP; "trigger" para el TP market legado.
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
    - fixed_rr: STOP_MARKET reduceOnly + TP LIMIT reduceOnly post-only (V7).
    - stop_only: solo SL reduceOnly; la salida normal se hace por senal z/timeout.

    El TP trigger-market anterior se conserva como modo de compatibilidad mediante
    TAKE_PROFIT_ORDER_MODE=trigger_market.
    """

    BOT_ORDER_PREFIXES = ("zsl_", "ztp_")

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

        self.exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET_KEY,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })

        if USE_TESTNET:
            # Binance Futures ya no usa el antiguo sandbox/testnet de CCXT.
            # Las pruebas se realizan contra Binance Demo Trading.
            self.exchange.enable_demo_trading(True)

        self.exchange.load_markets()
        # Sincronizamos explícitamente el offset con el reloj de Binance.
        # adjustForTimeDifference=True sigue actuando como mecanismo principal,
        # pero esto fuerza una sincronización inicial.
        try:
            self.exchange.load_time_difference()
        except Exception as exc:
            print(f"Aviso: no se pudo sincronizar inicialmente el reloj con Binance: {exc}")

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
        # Migracion de estados anteriores a V7.
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
                raise ExecutionError(f"Error obteniendo balance Futures: {exc}") from exc

            self._resync_time()

            try:
                balance = self.exchange.fetch_balance()
            except Exception as retry_exc:
                raise ExecutionError(f"Error obteniendo balance Futures tras resincronizar el reloj: {retry_exc}") from retry_exc

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
            raise ExecutionError(f"Notional {notional:.4f} inferior al minimo ({min_cost}).")
        return contracts

    def _fetch_open_position(self, symbol: str) -> Optional[PositionSnapshot]:
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as exc:
            if not self._is_timestamp_error(exc):
                raise ExecutionError(f"No se pudieron consultar posiciones: {exc}") from exc
            # Un -1021 es seguro de reintentar porque fetch_positions es
            # exclusivamente una lectura: no puede duplicar operaciones.
            self._resync_time()

            try:
                positions = self.exchange.fetch_positions([symbol])
            except Exception as retry_exc:
                raise ExecutionError(f"No se pudieron consultar posiciones tras resincronizar el reloj: {retry_exc}") from retry_exc
            
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
            self._emergency_close(symbol)
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

        if position is None:
            if state.status != "flat" or state.stop_order_id or state.take_profit_order_id:
                self._journal_flat_reconciliation(symbol, state)
                try:
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
                since_ms = (int(datetime.fromisoformat(submit_time).timestamp() * 1000) - 60_000)
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
