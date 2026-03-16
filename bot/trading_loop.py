"""TradingLoop — WebSocket event handlers and strategy execution cycle."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd

from bot.config import get_settings
from bot.data_loader import CandleCache
from bot.events.bus import (
    TOPIC_PORTFOLIO_UPDATE,
    TOPIC_SIGNAL,
    TOPIC_TICKER,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)
from bot.indicators.momentum import rsi as compute_rsi
from bot.indicators.trend import adx as compute_adx, ema as compute_ema, macd as compute_macd
from bot.indicators.volatility import atr as compute_atr, bollinger_bands
from bot.indicators.volume import cmf as compute_cmf, obv as compute_obv, volume_surge_ratio
from bot.risk.fee_gate import check_fee_gate
from bot.risk.fees import get_trading_fees
from bot.risk.position_sizer import fixed_fractional_size
from bot.risk.stop_loss import check_stop_triggered, initial_stops, trail_stop
from bot.strategy.confluence import check_confluence
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS
from bot.strategy.router import Regime, StrategyRouter, detect_regime

logger = logging.getLogger(__name__)


class TradingLoop:
    """Handles WebSocket events and executes the core strategy cycle."""

    def __init__(
        self,
        candle_cache: CandleCache,
        portfolio: Any,
        risk_engine: Any,
        order_mgr: Any,
        sentiment: Any,
        router: Any,
        drawdown: Any,
        trade_analyzer: Any,
        signal_eval: Any,
        settings: Any,
        onchain: Any = None,
        orderbook: Any = None,
    ) -> None:
        self.candle_cache = candle_cache
        self.portfolio = portfolio
        self.risk_engine = risk_engine
        self.order_mgr = order_mgr
        self.sentiment = sentiment
        self.router = router
        self.drawdown = drawdown
        self.trade_analyzer = trade_analyzer
        self.signal_eval = signal_eval
        self.settings = settings
        self.onchain = onchain
        self.orderbook = orderbook

        self._pending_orders: set[str] = set()
        self._limit_mgr = None  # Will be set by main.py (Task 17)
        self._last_evaluated_hour = None

        self._last_trade_closed: Dict[str, float] = {}  # unused, kept for compat

        self._last_halt_log: float = 0  # throttle halt log messages
        self._last_portfolio_publish: float = 0  # throttle portfolio update events

    def is_symbol_pending(self, symbol: str) -> bool:
        """Check if a symbol has a pending order (used by manual close endpoint too)."""
        return symbol in self._pending_orders

    @staticmethod
    def _resample(df_5m: pd.DataFrame, freq: str) -> pd.DataFrame:
        if df_5m.empty:
            return df_5m
        return df_5m.resample(freq).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    # ── WebSocket event handlers ──────────────────────────────────────────────

    async def on_candle(self, data: Dict[str, Any]) -> None:
        self.candle_cache.cache_candle(data["symbol"], data["interval"], data)
        self.portfolio.update_price(data["symbol"], data["close"])

        # Check stop-loss / take-profit on 5m candles
        if data["interval"] == "5m":
            await self._check_stops(data["symbol"], data["close"])

            # Check limit order timeouts and price deviations
            if hasattr(self, '_limit_mgr') and self._limit_mgr is not None:
                self._limit_mgr.check_timeouts()
                self._limit_mgr.check_price_deviation(data["symbol"], data["close"])

            # 1h candle detection with hour-crossing logic
            ts = data["timestamp"]
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = None
            current_hour = ts.replace(minute=0, second=0, microsecond=0) if hasattr(ts, "replace") and ts is not None else None
            if current_hour and current_hour != self._last_evaluated_hour:
                h1_candle = self.candle_cache.build_1h_candle(
                    data["symbol"], current_hour - timedelta(hours=1)
                )
                if h1_candle is not None:
                    await self._evaluate_1h_signals(data["symbol"], h1_candle)
                self._last_evaluated_hour = current_hour

    async def on_ticker(self, data: Dict[str, Any]) -> None:
        symbol = data.get("symbol", "")
        price = data.get("price", 0.0)
        if symbol and price:
            self.portfolio.update_price(symbol, price)
            await get_bus().publish(TOPIC_TICKER, data)

            # Publish portfolio update (throttled to every 3s)
            now = time.monotonic()
            if now - self._last_portfolio_publish >= 3.0:
                self._last_portfolio_publish = now
                await get_bus().publish(TOPIC_PORTFOLIO_UPDATE, {
                    "equity_eur": self.portfolio.get_equity_eur(),
                    "positions_value_eur": self.portfolio.get_positions_value_eur(),
                })

    async def on_fill(self, data: Dict[str, Any]) -> None:
        await self.order_mgr.on_fill_notification(data)

    # ── 1h signal evaluation ─────────────────────────────────────────────────

    async def _evaluate_1h_signals(self, symbol: str, h1_candle: Dict[str, Any]) -> None:
        """Evaluate 1h signals using regime detection and strategy confluence."""
        portfolio = self.portfolio
        settings = self.settings

        # Skip if position already open or order pending
        if portfolio.has_position(symbol):
            return
        if symbol in self._pending_orders:
            return

        # Skip symbols not in adopted universe
        if hasattr(self, '_universe') and self._universe is not None:
            if not self._universe.is_adopted(symbol):
                return

        equity = portfolio.get_equity_eur()

        # Check if trading is allowed
        allowed, halt_reason = self.drawdown.is_trading_allowed(equity)
        if not allowed:
            return

        # Detect regime using 4h data
        df_5m = self.candle_cache.get_df(symbol, "5m")
        if len(df_5m) < 200:
            df_4h = None
        else:
            df_4h = self._resample(df_5m, "4h")

        if df_4h is not None and len(df_4h) >= 14:
            adx_val = float(compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).iloc[-1])
            atr_val = float(compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).iloc[-1])
            price_4h = float(df_4h["close"].iloc[-1])
            atr_pct = (atr_val / price_4h * 100.0) if price_4h > 0 else 0

            if hasattr(self, '_universe') and self._universe is not None and self._universe.is_adopted(symbol):
                rp = self._universe.get_regime_params(symbol)
                _quiet_thresh = rp["quiet_atr_threshold"]
                _regime_adx = rp["regime_adx_threshold"]
                _ranging_adx = rp.get("ranging_adx_threshold", 20)
            else:
                _quiet_thresh = CHAMPION_DEFAULTS["quiet_atr_threshold"]
                _regime_adx = CHAMPION_DEFAULTS["regime_adx_threshold"]
                _ranging_adx = CHAMPION_DEFAULTS["ranging_adx_threshold"]

            if atr_pct < _quiet_thresh:
                regime = Regime.QUIET
            elif atr_pct > 4.0:
                regime = Regime.VOLATILE
            elif adx_val > _regime_adx:
                regime = Regime.TRENDING
            elif adx_val < _ranging_adx:
                regime = Regime.RANGING
            else:
                regime = Regime.NEUTRAL
        else:
            regime = Regime.NEUTRAL

        # Skip QUIET regime
        if regime == Regime.QUIET:
            logger.debug("Skipping %s — QUIET regime detected", symbol)
            return

        # Get applicable strategies
        strategies = self.router.get_strategies(regime)

        if hasattr(self, '_universe') and self._universe is not None:
            enabled = self._universe.get_enabled_strategies(symbol)
            if enabled:
                strategies = [s for s in strategies if s.name in enabled]

        if not strategies:
            return

        # Get cached 1h dataframe for indicator computation
        df_1h = self.candle_cache.get_df(symbol, "1h")
        if len(df_1h) < 30:
            return

        # Precompute indicators from the 1h dataframe
        close_1h = df_1h["close"]
        high_1h = df_1h["high"]
        low_1h = df_1h["low"]
        volume_1h = df_1h["volume"]

        rsi_series = compute_rsi(close_1h)
        rsi_1h = float(rsi_series.iloc[-1])

        macd_df = compute_macd(close_1h)
        macd_hist = float(macd_df["histogram"].iloc[-1])
        macd_hist_prev = float(macd_df["histogram"].iloc[-2]) if len(macd_df) >= 2 else macd_hist

        bb = bollinger_bands(close_1h)
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])
        bb_mid = float(bb["mid"].iloc[-1])
        bb_bandwidth = float(bb["bandwidth"].iloc[-1])
        bb_bandwidth_prev = float(bb["bandwidth"].iloc[-2]) if len(bb) >= 2 else bb_bandwidth

        price = float(close_1h.iloc[-1])
        vol_surge_series = volume_surge_ratio(volume_1h)
        vol_surge = float(vol_surge_series.iloc[-1])

        cmf_series = compute_cmf(high_1h, low_1h, close_1h, volume_1h)
        cmf_val = float(cmf_series.iloc[-1])

        obv_series = compute_obv(close_1h, volume_1h)
        # OBV divergence: compare OBV slope vs price slope over last 5 bars
        if len(obv_series) >= 5:
            obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-5])
            price_slope = float(close_1h.iloc[-1] - close_1h.iloc[-5])
            # Divergence: OBV rising while price falling (or vice versa)
            if price_slope != 0:
                obv_divergence = -1.0 if (obv_slope > 0 and price_slope < 0) else (1.0 if (obv_slope < 0 and price_slope > 0) else 0.0)
            else:
                obv_divergence = 0.0
        else:
            obv_divergence = 0.0

        # Candle body/wick ratios from the 1h candle
        h1_open = h1_candle["open"]
        h1_high = h1_candle["high"]
        h1_low = h1_candle["low"]
        h1_close = h1_candle["close"]
        candle_range = h1_high - h1_low
        if candle_range > 0:
            body_ratio = abs(h1_close - h1_open) / candle_range
            wick_lower_ratio = (min(h1_open, h1_close) - h1_low) / candle_range
            wick_upper_ratio = (h1_high - max(h1_open, h1_close)) / candle_range
        else:
            body_ratio = 0.0
            wick_lower_ratio = 0.0
            wick_upper_ratio = 0.0

        # EMA50 slope for squeeze strategy (10-bar smoothed)
        ema50 = compute_ema(close_1h, 50)
        if len(ema50) >= 10:
            ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-10])
        elif len(ema50) >= 2:
            ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-2])
        else:
            ema50_slope = 0.0

        # EMA200 for trend filter
        ema200 = compute_ema(close_1h, 200)
        ema200_val = float(ema200.iloc[-1]) if len(ema200) > 0 else price
        ema200_dist_pct = (price - ema200_val) / ema200_val * 100.0 if ema200_val > 0 else 0.0

        # 4h RSI for funding_contrarian
        if df_4h is not None and len(df_4h) >= 14:
            rsi_4h_series = compute_rsi(df_4h["close"])
            rsi_4h = float(rsi_4h_series.iloc[-1])
            adx_4h_val = float(compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).iloc[-1])
        else:
            rsi_4h = 50.0
            adx_4h_val = 20.0

        # Collect signals from all strategies
        signals = []
        for strategy in strategies:
            try:
                if strategy.name == "orderflow":
                    direction, strength = strategy.evaluate_1h(
                        body_ratio, wick_lower_ratio, wick_upper_ratio,
                        vol_surge, rsi_1h, cmf_val, obv_divergence,
                    )
                elif strategy.name == "funding_contrarian":
                    direction, strength = strategy.evaluate_1h(
                        rsi_1h, rsi_4h, macd_hist, macd_hist_prev,
                    )
                elif strategy.name == "range":
                    direction, strength = strategy.evaluate_1h(
                        price, bb_lower, bb_upper, bb_mid, bb_bandwidth,
                        adx_4h_val, rsi_1h,
                    )
                elif strategy.name == "squeeze":
                    direction, strength = strategy.evaluate_1h(
                        bb_bandwidth_prev, bb_bandwidth, price, bb_upper,
                        bb_lower, vol_surge, ema50_slope,
                    )
                else:
                    continue

                if direction != "NEUTRAL" and strength > 0:
                    signals.append({
                        "direction": direction,
                        "strength": strength,
                        "strategy": strategy.name,
                    })
            except Exception as e:
                logger.warning("Strategy %s evaluate_1h failed: %s", strategy.name, e)

        if not signals:
            return

        # --- EMA200 trend filter: block signals against the daily trend ---
        if abs(ema200_dist_pct) > 2.0:
            blocked_dir = "LONG" if ema200_dist_pct < -2.0 else "SHORT"
            signals = [s for s in signals if s["direction"] != blocked_dir]
            if not signals:
                logger.debug("EMA200 filter blocked all signals for %s (dist=%.1f%%)", symbol, ema200_dist_pct)
                return

        # Run confluence check — relax when only 1 strategy enabled
        confluence = None  # may remain None in single-strategy path
        if hasattr(self, '_universe') and self._universe is not None:
            enabled = self._universe.get_enabled_strategies(symbol)
            if len(enabled) <= 1 and signals:
                # Single strategy — use its signal directly without confluence
                best = max(signals, key=lambda s: s["strength"])
                direction = best["direction"]
                strength = best["strength"]
            else:
                confluence = check_confluence(signals)
                if not confluence.triggered:
                    logger.debug("No confluence for %s — signals: %s", symbol, signals)
                    return
                direction = confluence.direction
                strength = confluence.strength
        else:
            confluence = check_confluence(signals)
            if not confluence.triggered:
                logger.debug("No confluence for %s — signals: %s", symbol, signals)
                return
            direction = confluence.direction
            strength = confluence.strength

        # Compute position size
        entry_price = price
        if entry_price <= 0:
            return

        # Get per-strategy params from universe
        if hasattr(self, '_universe') and self._universe is not None and signals:
            winning_strat = max(signals, key=lambda s: s["strength"])["strategy"]
            strat_params = self._universe.get_strategy_params(symbol, winning_strat)
        else:
            strat_params = CHAMPION_DEFAULTS

        # Compute stops
        stop_loss, take_profit = initial_stops(
            entry_price, df_1h, direction=direction,
            atr_multiplier=strat_params["atr_multiplier"],
            rr_ratio=strat_params["rr_ratio"],
        )
        stop_distance_pct = max(0.1, abs(entry_price - stop_loss) / entry_price * 100.0)
        tp_distance_pct = max(0.1, abs(take_profit - entry_price) / entry_price * 100.0)

        # Fee-aware gate
        atr_series_1h = compute_atr(high_1h, low_1h, close_1h)
        current_atr = float(atr_series_1h.iloc[-1]) if len(atr_series_1h) > 0 else 0.0
        atr_pct_1h = (current_atr / entry_price * 100.0) if entry_price > 0 else 1.5
        atr_pct_1h = max(atr_pct_1h, 0.1)
        atr_50 = atr_series_1h.iloc[-50:] if len(atr_series_1h) >= 50 else atr_series_1h
        median_atr = float(atr_50.median()) if len(atr_50) > 0 else current_atr
        median_atr_pct = max((median_atr / entry_price * 100.0) if entry_price > 0 else atr_pct_1h, 0.1)

        size_eur = fixed_fractional_size(
            equity=equity,
            base_risk_pct=strat_params["base_risk_pct"],
            stop_distance_pct=stop_distance_pct,
            atr_pct=atr_pct_1h,
            median_atr_pct=median_atr_pct,
        )
        size_eur *= self.drawdown.position_size_multiplier(equity)

        # Fee gate check
        _, taker_pct = get_trading_fees(symbol)
        fee_result = check_fee_gate(
            position_size=size_eur,
            tp_distance_pct=tp_distance_pct,
            taker_fee_pct=taker_pct,
            is_short=(direction == "SHORT"),
            min_profit_multiple=strat_params["min_profit_multiple"],
        )
        if not fee_result.approved:
            logger.debug("Fee gate rejected %s %s: %s", direction, symbol, fee_result.reason)
            return

        amount_base = size_eur / entry_price

        # Risk gate
        risk = stop_distance_pct
        expected_roi = risk * 2.0
        side = "buy" if direction == "LONG" else "sell"
        decision = self.risk_engine.approve(
            symbol=symbol,
            side=side,
            position_size_eur=size_eur,
            signal_confidence=strength,
            expected_roi_pct=expected_roi,
            portfolio_equity_eur=equity,
            open_position_count=portfolio.open_position_count(),
            daily_loss_eur=self.drawdown.daily_realized_loss_eur(),
            current_drawdown_pct=self.drawdown.current_drawdown_pct(equity),
            taker_fee_pct=taker_pct,
        )

        if not decision.approved:
            return

        agreeing = confluence.agreeing_strategies if confluence else []
        strategy_name = agreeing[0] if agreeing else (max(signals, key=lambda s: s["strength"])["strategy"] if signals else "unknown")

        logger.info(
            "1h Signal ACCEPTED %s %s [%s] — size EUR%.2f, strength=%.2f, confluence=%s",
            direction, symbol, strategy_name, size_eur, strength,
            agreeing,
        )

        # Place order
        self._pending_orders.add(symbol)
        try:
            journal_data = {
                "confluence": agreeing,
                "market_regime": regime.value if hasattr(regime, 'value') else str(regime),
                "strength": strength,
            }

            if direction == "SHORT":
                order_id = await self.order_mgr.submit_sell(
                    symbol, amount_base, strategy_name,
                    extra_data=journal_data,
                )
            else:
                order_id = await self.order_mgr.submit_buy(
                    symbol, amount_base, strategy_name,
                    extra_data=journal_data,
                )

            if order_id:
                await portfolio.add_position(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    entry_price=entry_price,
                    quantity=amount_base,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_order_id=order_id,
                    paper_trade=settings.paper_trading,
                    direction=direction,
                )

                # Store range levels for range strategy
                pos = portfolio.get_position(symbol)
                if pos and strategy_name == "range":
                    pos["range_mid"] = bb_mid
                    pos["range_upper"] = bb_upper
                    pos["range_lower"] = bb_lower
                    pos["tp_shifted"] = False

                    bandwidth_price = bb_upper - bb_lower
                    if direction == "LONG":
                        pos["stop_loss_price"] = bb_lower - 0.5 * bandwidth_price
                        pos["take_profit_price"] = bb_mid
                    else:
                        pos["stop_loss_price"] = bb_upper + 0.5 * bandwidth_price
                        pos["take_profit_price"] = bb_mid

                await get_bus().publish(TOPIC_TRADE_OPENED, {
                    "symbol": symbol,
                    "direction": direction,
                    "strategy_name": strategy_name,
                    "entry_price": entry_price,
                    "size_eur": size_eur,
                    "confluence": confluence.agreeing_strategies if confluence else [],
                })
        finally:
            self._pending_orders.discard(symbol)

    # ── Stop-loss monitoring ──────────────────────────────────────────────────

    async def _check_stops(self, symbol: str, current_price: float) -> None:
        pos = self.portfolio.get_position(symbol)
        if pos is None:
            return

        direction = pos.get("direction", "LONG")

        # Update trailing stop
        df_5m = self.candle_cache.get_df(symbol, "5m")
        if len(df_5m) >= 14:
            atr_val = compute_atr(df_5m["high"], df_5m["low"], df_5m["close"]).iloc[-1]
            default_stop = current_price * (1.05 if direction == "SHORT" else 0.95)
            new_stop = trail_stop(
                current_price,
                pos["highest_price"],
                pos["stop_loss_price"] or default_stop,
                atr_val,
                direction=direction,
                activation_threshold=1.5,
                entry_price=pos.get("entry_price", 0),
            )
            if direction == "SHORT":
                if new_stop < (pos["stop_loss_price"] or float("inf")):
                    pos["stop_loss_price"] = new_stop
            else:
                if new_stop > (pos["stop_loss_price"] or 0):
                    pos["stop_loss_price"] = new_stop

        # Range 72h max hold time
        if pos.get("strategy_name") == "range":
            entry_time = pos.get("entry_time")
            if entry_time is not None:
                try:
                    if isinstance(entry_time, str):
                        entry_dt = datetime.fromisoformat(entry_time)
                    else:
                        entry_dt = entry_time
                    now = datetime.now(timezone.utc)
                    if (now - entry_dt).total_seconds() >= 259200:  # 72 hours
                        await self._close_position(symbol, current_price, "range_time_exit")
                        return
                except (ValueError, TypeError):
                    pass

        # Range dynamic TP: shift from mid to opposite band
        if pos.get("strategy_name") == "range" and not pos.get("tp_shifted", False):
            crossed_mid = (
                (direction == "LONG" and current_price >= pos.get("range_mid", 0)) or
                (direction == "SHORT" and current_price <= pos.get("range_mid", float("inf")))
            )
            if crossed_mid and pos.get("range_mid", 0) > 0:
                if len(df_5m) >= 20:
                    rsi_series = compute_rsi(df_5m["close"])
                    current_rsi = rsi_series.iloc[-1]
                    prev_rsi = rsi_series.iloc[-4] if len(rsi_series) >= 4 else current_rsi

                    if direction == "LONG" and current_rsi > prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_upper"]
                        logger.info("Range TP shifted to upper band %.2f for %s", pos["range_upper"], symbol)
                    elif direction == "SHORT" and current_rsi < prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_lower"]
                        logger.info("Range TP shifted to lower band %.2f for %s", pos["range_lower"], symbol)
                    else:
                        await self._close_position(symbol, pos["range_mid"], "range_mid_exit")
                        return

        # Tight trailing stop for range positions with shifted TP
        if pos.get("strategy_name") == "range" and pos.get("tp_shifted", False):
            if direction == "LONG":
                trail_level = pos["highest_price"] * 0.99
                if trail_level > (pos["stop_loss_price"] or 0):
                    pos["stop_loss_price"] = trail_level
            elif direction == "SHORT":
                trail_level = pos["highest_price"] * 1.01
                if trail_level < (pos["stop_loss_price"] or float("inf")):
                    pos["stop_loss_price"] = trail_level

        default_tp = 0.0 if direction == "SHORT" else float("inf")
        exit_reason = check_stop_triggered(
            current_price,
            pos["stop_loss_price"] or (float("inf") if direction == "SHORT" else 0),
            pos["take_profit_price"] or default_tp,
            direction=direction,
        )

        if exit_reason:
            await self._close_position(symbol, current_price, exit_reason)

    async def _close_position(self, symbol: str, price: float, reason: str) -> None:
        portfolio = self.portfolio
        order_mgr = self.order_mgr
        sentiment = self.sentiment

        pos = portfolio.get_position(symbol)
        if pos is None:
            return

        if symbol in self._pending_orders:
            logger.warning("Skipping close for %s — order already pending", symbol)
            return

        self._pending_orders.add(symbol)
        try:
            quantity = pos["quantity"]
            direction = pos.get("direction", "LONG")

            # LONG: sell to close. SHORT: buy to close.
            if direction == "SHORT":
                order_id = await order_mgr.submit_buy(symbol, quantity, pos["strategy_name"])
            else:
                order_id = await order_mgr.submit_sell(symbol, quantity, pos["strategy_name"])

            sentiment_score = sentiment.get_score(symbol)
            result = await portfolio.close_position(symbol, price, order_id, reason, sentiment_score)

            if result:
                # Increment range bounce counter
                if pos.get("strategy_name") == "range":
                    if hasattr(self.router, '_range'):
                        self.router._range.increment_bounce(symbol)

                # Trigger learning
                analyzer = self.trade_analyzer
                evaluator = self.signal_eval
                await analyzer.process_closed_trade(
                    trade_id=result["trade_id"],
                    symbol=symbol,
                    strategy_name=pos["strategy_name"],
                    net_pnl=result["net_pnl"],
                    entry_features=pos.get("entry_features"),
                )

                if pos.get("entry_features"):
                    breakdown = pos["entry_features"].get("breakdown", {})
                    evaluator.record_trade(breakdown, direction, result["net_pnl"] > 0)

                if result["net_pnl"] < 0:
                    self.drawdown.record_realized_loss(abs(result["net_pnl"]))

                await get_bus().publish(TOPIC_TRADE_CLOSED, {
                    **result,
                    "price": price,
                    "reason": reason,
                })
        finally:
            self._pending_orders.discard(symbol)

    # ── Main strategy cycle ───────────────────────────────────────────────────

    async def run_cycle(self, tradeable_symbols: List[str], is_running: bool) -> None:
        settings = self.settings
        portfolio = self.portfolio
        risk_engine = self.risk_engine
        order_mgr = self.order_mgr
        sentiment = self.sentiment
        router = self.router
        drawdown = self.drawdown

        if not is_running:
            return

        equity = portfolio.get_equity_eur()
        drawdown.update(equity)

        for symbol in list(tradeable_symbols):
            try:
                df_5m = self.candle_cache.get_df(symbol, "5m")
                df_1h = self.candle_cache.get_df(symbol, "1h")

                if len(df_5m) < 30:
                    continue

                # Skip if position already open
                if portfolio.has_position(symbol):
                    continue

                # Skip if order is pending (prevents race condition)
                if symbol in self._pending_orders:
                    continue

                # Skip symbols not in adopted universe
                if hasattr(self, '_universe') and self._universe is not None:
                    if not self._universe.is_adopted(symbol):
                        continue

                # Check if trading is allowed
                allowed, halt_reason = drawdown.is_trading_allowed(equity)
                if not allowed:
                    import time as _t
                    now = _t.time()
                    if now - self._last_halt_log > 300:  # log once per 5 min
                        logger.warning("Trading halted: %s", halt_reason)
                        self._last_halt_log = now
                    return

                # Detect regime with per-symbol params from universe
                df_4h = self._resample(df_5m, "4h") if len(df_5m) >= 200 else None
                if df_4h is not None and len(df_4h) >= 14:
                    adx_val = float(compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).iloc[-1])
                    atr_val = float(compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).iloc[-1])
                    price_4h = float(df_4h["close"].iloc[-1])
                    atr_pct_4h = (atr_val / price_4h * 100.0) if price_4h > 0 else 0

                    if hasattr(self, '_universe') and self._universe is not None and self._universe.is_adopted(symbol):
                        rp = self._universe.get_regime_params(symbol)
                        _quiet_thresh = rp["quiet_atr_threshold"]
                        _regime_adx = rp["regime_adx_threshold"]
                        _ranging_adx = rp.get("ranging_adx_threshold", 20)
                    else:
                        _quiet_thresh = CHAMPION_DEFAULTS["quiet_atr_threshold"]
                        _regime_adx = CHAMPION_DEFAULTS["regime_adx_threshold"]
                        _ranging_adx = CHAMPION_DEFAULTS["ranging_adx_threshold"]

                    if atr_pct_4h < _quiet_thresh:
                        regime = Regime.QUIET
                    elif atr_pct_4h > 4.0:
                        regime = Regime.VOLATILE
                    elif adx_val > _regime_adx:
                        regime = Regime.TRENDING
                    elif adx_val < _ranging_adx:
                        regime = Regime.RANGING
                    else:
                        regime = Regime.NEUTRAL
                else:
                    regime = Regime.NEUTRAL

                strategies = router.get_strategies(regime)

                if hasattr(self, '_universe') and self._universe is not None:
                    enabled = self._universe.get_enabled_strategies(symbol)
                    if enabled:
                        strategies = [s for s in strategies if s.name in enabled]

                # Get sentiment
                sentiment_score = sentiment.get_score(symbol)

                # Build market context
                # Resample for MTF
                df_15m = self._resample(df_5m, "15min") if len(df_5m) >= 10 else df_5m
                df_4h_ctx = df_4h if df_4h is not None else df_1h
                df_1d = self._resample(df_5m, "1D") if len(df_5m) >= 500 else df_1h

                # Get provider scores
                onchain_score = self.onchain.score(symbol) if self.onchain and self.onchain.is_available() else 0.0
                orderbook_imb = self.orderbook.score(symbol) if self.orderbook and self.orderbook.is_available() else 0.0

                from bot.strategy.base import MarketContext
                ctx = MarketContext(
                    symbol=symbol,
                    candles_5m=df_5m,
                    candles_1h=df_1h,
                    candles_15m=df_15m,
                    candles_4h=df_4h_ctx,
                    candles_1d=df_1d,
                    current_price=df_5m["close"].iloc[-1],
                    sentiment_score=sentiment_score,
                    portfolio_equity_eur=equity,
                    open_position_count=portfolio.open_position_count(),
                    onchain_score=onchain_score,
                    orderbook_imbalance=orderbook_imb,
                    market_regime=regime,
                )

                # Run strategies and pick strongest signal
                best_signal = None
                for strategy in strategies:
                    sig = strategy.generate_signal(ctx)
                    if sig.direction == "NEUTRAL":
                        continue
                    if best_signal is None or sig.strength > best_signal.strength:
                        best_signal = sig

                if best_signal is None:
                    continue

                # Publish every signal to the live feed
                await get_bus().publish(TOPIC_SIGNAL, {
                    "symbol": symbol,
                    "direction": best_signal.direction,
                    "strength": best_signal.strength,
                    "strategy_name": best_signal.strategy_name,
                    "technical_score": best_signal.technical_score,
                    "sentiment_score": best_signal.sentiment_score,
                    "actionable": best_signal.is_actionable(settings.min_signal_confidence),
                })

                if not best_signal.is_actionable(settings.min_signal_confidence):
                    logger.info(
                        "Signal SKIPPED %s %s %s (strength=%.2f < min=%.2f)",
                        best_signal.direction, symbol, best_signal.strategy_name,
                        best_signal.strength, settings.min_signal_confidence,
                    )
                    continue

                direction = best_signal.direction
                if direction not in ("LONG", "SHORT"):
                    continue

                entry_price = ctx.current_price
                if entry_price <= 0:
                    continue

                # Get per-strategy params from universe
                if hasattr(self, '_universe') and self._universe is not None and best_signal:
                    strat_params = self._universe.get_strategy_params(symbol, best_signal.strategy_name)
                else:
                    strat_params = CHAMPION_DEFAULTS

                # Compute stops first — stop distance drives position sizing
                stop_loss, take_profit = initial_stops(
                    entry_price, df_1h, direction=direction,
                    atr_multiplier=strat_params["atr_multiplier"],
                    rr_ratio=strat_params["rr_ratio"],
                )
                stop_distance_pct = max(0.1, abs(entry_price - stop_loss) / entry_price * 100.0)
                risk = stop_distance_pct
                expected_roi = risk * 2.0  # 2:1 reward/risk

                # Compute current ATR% and median ATR% for volatility scaling
                atr_series = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"])
                current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
                atr_pct = (current_atr / entry_price * 100.0) if entry_price > 0 else 1.5
                atr_pct = max(atr_pct, 0.1)
                # Median over last 50 bars as normalisation reference
                atr_50 = atr_series.iloc[-50:] if len(atr_series) >= 50 else atr_series
                median_atr = float(atr_50.median()) if len(atr_50) > 0 else current_atr
                median_atr_pct = max((median_atr / entry_price * 100.0) if entry_price > 0 else atr_pct, 0.1)

                # Fixed fractional position size
                size_eur = fixed_fractional_size(
                    equity=equity,
                    base_risk_pct=strat_params["base_risk_pct"],
                    stop_distance_pct=stop_distance_pct,
                    atr_pct=atr_pct,
                    median_atr_pct=median_atr_pct,
                )
                size_eur *= drawdown.position_size_multiplier(equity)

                amount_base = size_eur / entry_price

                # Fee gate: reject trades where fees eat the profit
                tp_distance_pct = abs(take_profit - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
                fee_result = check_fee_gate(
                    position_size=size_eur,
                    tp_distance_pct=tp_distance_pct,
                    is_short=(direction == "SHORT"),
                    min_profit_multiple=strat_params["min_profit_multiple"],
                )
                if not fee_result.approved:
                    continue

                # Risk gate (use actual per-market taker fee)
                _, taker_pct = get_trading_fees(symbol)
                side = "buy" if direction == "LONG" else "sell"
                decision = risk_engine.approve(
                    symbol=symbol,
                    side=side,
                    position_size_eur=size_eur,
                    signal_confidence=best_signal.strength,
                    expected_roi_pct=expected_roi,
                    portfolio_equity_eur=equity,
                    open_position_count=portfolio.open_position_count(),
                    daily_loss_eur=drawdown.daily_realized_loss_eur(),
                    current_drawdown_pct=drawdown.current_drawdown_pct(equity),
                    taker_fee_pct=taker_pct,
                )

                if not decision.approved:
                    # RiskEngine already logs rejections
                    continue

                logger.info(
                    "Signal ACCEPTED %s %s %s — size EUR%.2f, confidence=%.2f",
                    direction, symbol, best_signal.strategy_name,
                    size_eur, best_signal.strength,
                )

                # Save signal
                from bot.data.database import get_session as _gs
                from bot.data.repositories import save_signal
                async with _gs() as session:
                    sig_record = await save_signal(
                        session,
                        symbol=symbol,
                        strategy_name=best_signal.strategy_name,
                        direction=direction,
                        strength=best_signal.strength,
                        technical_score=best_signal.technical_score,
                        sentiment_score=best_signal.sentiment_score,
                        indicator_snapshot=best_signal.indicator_snapshot,
                        acted_on=True,
                    )
                    signal_id = sig_record.id

                self._pending_orders.add(symbol)
                try:
                    journal_data = {
                        "signal": best_signal,
                        "market_regime": regime,
                        "candle_data": df_5m.tail(100).to_dict("records") if len(df_5m) > 0 else [],
                    }
                    # LONG: buy to open. SHORT: sell to open.
                    if direction == "SHORT":
                        order_id = await order_mgr.submit_sell(
                            symbol, amount_base, best_signal.strategy_name, signal_id,
                            extra_data=journal_data,
                        )
                    else:
                        order_id = await order_mgr.submit_buy(
                            symbol, amount_base, best_signal.strategy_name, signal_id,
                            extra_data=journal_data,
                        )

                    if order_id:
                        await portfolio.add_position(
                            symbol=symbol,
                            strategy_name=best_signal.strategy_name,
                            entry_price=entry_price,
                            quantity=amount_base,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            entry_order_id=order_id,
                            paper_trade=settings.paper_trading,
                            direction=direction,
                        )
                        # Store features on position for learning
                        pos = portfolio.get_position(symbol)
                        if pos:
                            pos["entry_features"] = best_signal.indicator_snapshot

                            # Store range levels for range strategy
                            if best_signal.strategy_name == "range":
                                snap = best_signal.indicator_snapshot
                                pos["range_mid"] = snap.get("range_mid", 0.0)
                                pos["range_upper"] = snap.get("range_upper", 0.0)
                                pos["range_lower"] = snap.get("range_lower", 0.0)
                                pos["tp_shifted"] = False

                                # Override ATR stops with range-specific stops
                                bandwidth_price = pos["range_upper"] - pos["range_lower"]
                                if direction == "LONG":
                                    pos["stop_loss_price"] = pos["range_lower"] - 0.5 * bandwidth_price
                                    pos["take_profit_price"] = pos["range_mid"]
                                else:
                                    pos["stop_loss_price"] = pos["range_upper"] + 0.5 * bandwidth_price
                                    pos["take_profit_price"] = pos["range_mid"]
                finally:
                    self._pending_orders.discard(symbol)

            except Exception as e:
                logger.error("Strategy cycle error for %s: %s", symbol, e, exc_info=True)
