"""External data pipeline: fetch, cache, and serve market data from free APIs.

Training mode: bulk-fetch history -> cache to parquet -> align to 5m index
Live mode: periodic refresh -> in-memory cache -> staleness tracking

Sources:
    - Binance Futures: funding rates
    - Alternative.me: Fear & Greed Index
    - Google Trends: search interest (pytrends)
    - yfinance: DXY, S&P 500, Gold, VIX, Treasury yields
    - BGeometrics: NVT, MVRV, SOPR, Puell, exchange flows, hashrate
    - CoinMetrics: active addresses, tx count
    - DefiLlama: stablecoin supply, DeFi TVL
    - Coinalyze: OI, liquidations
    - Binance: taker buy/sell volume, liquidation snapshots
    - Deribit: DVOL implied volatility
    - CoinGecko: BTC dominance
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Staleness thresholds per source group
STALENESS_THRESHOLDS = {
    "funding_rate": timedelta(hours=2),
    "fear_greed": timedelta(hours=48),
    "google_trends": timedelta(days=7),
    "macro": timedelta(hours=48),
    "onchain": timedelta(hours=48),
    "defi": timedelta(hours=48),
    "oi_liquidations": timedelta(hours=48),
    "dvol": timedelta(hours=48),
}

CRITICAL_STALE_RATIO = 0.30  # Disable ML signals when >30% of sources are stale


def align_to_5m(series: pd.Series, idx_5m: pd.DatetimeIndex) -> np.ndarray:
    """Align a time series to a 5m DatetimeIndex via forward-fill.

    Pre-inception periods are filled with 0.0.
    """
    if series.empty:
        return np.zeros(len(idx_5m), dtype=np.float64)

    # Ensure timezone-aware
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    if idx_5m.tz is None:
        idx_5m = idx_5m.tz_localize("UTC")

    aligned = series.reindex(idx_5m, method="ffill")
    # Pre-inception: fill NaN with 0.0
    aligned = aligned.fillna(0.0)
    return aligned.values.astype(np.float64)


def save_cache(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_cache(path: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from parquet, or None if not found."""
    if not Path(path).exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning("Failed to load cache %s: %s", path, e)
        return None


def _fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    """Fetch Fear & Greed Index from Alternative.me."""
    import requests
    try:
        url = "https://api.alternative.me/fng/"
        params = {"limit": limit, "format": "json"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc)
            rows.append({"date": ts, "value": int(d["value"])})
        df = pd.DataFrame(rows).set_index("date").sort_index()
        return df
    except Exception as e:
        logger.warning("Fear & Greed fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_funding_rates(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch funding rates from Binance Futures."""
    import requests
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        all_rows = []
        start_time = None
        for _ in range(50):  # max 50 pages
            params = {"symbol": symbol, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for d in data:
                ts = datetime.fromtimestamp(d["fundingTime"] / 1000, tz=timezone.utc)
                all_rows.append({"date": ts, "rate": float(d["fundingRate"])})
            if len(data) < limit:
                break
            start_time = data[-1]["fundingTime"] + 1
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Funding rate fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_macro_yfinance(start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Fetch macro data from yfinance: DXY, S&P 500, Gold, VIX, Treasury yields."""
    try:
        import yfinance as yf
        tickers = {
            "dxy": "DX=F",
            "sp500": "^GSPC",
            "gold": "GC=F",
            "vix": "^VIX",
            "tnx": "^TNX",  # 10Y Treasury yield
            "twoy": "2YY=F",  # 2Y Treasury yield (for 10Y-2Y spread)
        }
        result = {}
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"), progress=False)
                if not df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    result[name] = df[["Close"]].rename(columns={"Close": "value"})
            except Exception as e:
                logger.warning("yfinance %s fetch failed: %s", name, e)
        return result
    except Exception as e:
        logger.warning("yfinance import/fetch failed: %s", e)
        return {}


def _fetch_google_trends(keywords: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch Google Trends data. Optional -- frequently rate-limited."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0)
        result = {}
        for kw in keywords:
            for attempt in range(3):
                try:
                    pytrends.build_payload([kw], timeframe="today 5-y")
                    df = pytrends.interest_over_time()
                    if not df.empty and kw in df.columns:
                        result[kw] = df[[kw]].rename(columns={kw: "value"})
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        logger.warning("Google Trends '%s' failed after 3 attempts: %s", kw, e)
        return result
    except Exception as e:
        logger.warning("pytrends not available: %s", e)
        return {}


def _fetch_coinmetrics_timeseries(metric: str, asset: str = "btc") -> pd.DataFrame:
    """Fetch a single metric from CoinMetrics Community API (free tier)."""
    import requests
    try:
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        all_rows = []
        next_page = None
        for _ in range(50):  # max 50 pages
            params = {"assets": asset, "metrics": metric, "frequency": "1d", "page_size": 10000}
            if next_page:
                params["next_page_token"] = next_page
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", [])
            if not data:
                break
            for d in data:
                ts = pd.to_datetime(d["time"])
                val = float(d.get(metric, 0))
                all_rows.append({"date": ts, "value": val})
            next_page = body.get("next_page_token")
            if not next_page:
                break
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows).set_index("date").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning("CoinMetrics %s/%s fetch failed: %s", asset, metric, e)
        return pd.DataFrame()


def _fetch_onchain_mvrv() -> pd.DataFrame:
    """Fetch MVRV from CoinMetrics Community API."""
    return _fetch_coinmetrics_timeseries("CapMVRVCur", "btc")


def _fetch_onchain_nvt_proxy() -> pd.DataFrame:
    """Compute NVT proxy: MarketCap / (TxCount * Price).

    True NVT (MarketCap / TransferValueUSD) requires paid API access.
    This proxy uses tx count as a volume indicator instead.
    """
    import requests
    try:
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        params = {
            "assets": "btc",
            "metrics": "CapMrktCurUSD,TxCnt",
            "frequency": "1d",
            "page_size": 10000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = pd.to_datetime(d["time"])
            mcap_raw = d.get("CapMrktCurUSD")
            tx_raw = d.get("TxCnt")
            if mcap_raw is None or tx_raw is None:
                continue
            mcap = float(mcap_raw)
            tx_cnt = float(tx_raw)
            nvt_proxy = mcap / max(tx_cnt, 1) / 1e6  # scale down
            rows.append({"date": ts, "value": nvt_proxy})
        df = pd.DataFrame(rows).set_index("date").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning("NVT proxy computation failed: %s", e)
        return pd.DataFrame()


def _fetch_onchain_hashrate() -> pd.DataFrame:
    """Fetch BTC hashrate from CoinMetrics, fallback to Blockchain.com."""
    df = _fetch_coinmetrics_timeseries("HashRate", "btc")
    if not df.empty:
        return df
    return _fetch_blockchain_com_hashrate()


def _fetch_blockchain_com_exchange_flows() -> pd.DataFrame:
    """Fetch BTC estimated transaction volume from Blockchain.com as exchange flow proxy."""
    import requests
    try:
        url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd"
        params = {"timespan": "5years", "format": "json", "rollingAverage": "7days"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("values", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d["x"], tz=timezone.utc)
            rows.append({"date": ts, "value": float(d["y"])})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Blockchain.com tx volume fallback failed: %s", e)
        return pd.DataFrame()


def _fetch_blockchain_com_hashrate() -> pd.DataFrame:
    """Fallback: fetch BTC hashrate from Blockchain.com Charts API."""
    import requests
    try:
        url = "https://api.blockchain.info/charts/hash-rate"
        params = {"timespan": "5years", "format": "json", "rollingAverage": "7days"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("values", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d["x"], tz=timezone.utc)
            rows.append({"date": ts, "value": float(d["y"])})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Blockchain.com hashrate fallback failed: %s", e)
        return pd.DataFrame()


def _fetch_blockchain_com_puell() -> pd.DataFrame:
    """Self-calculate Puell Multiple from Blockchain.com miners-revenue.

    Puell = daily_miner_revenue / 365_day_MA(daily_miner_revenue)
    """
    import requests
    try:
        url = "https://api.blockchain.info/charts/miners-revenue"
        params = {"timespan": "5years", "format": "json"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("values", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d["x"], tz=timezone.utc)
            rows.append({"date": ts, "value": float(d["y"])})
        df = pd.DataFrame(rows).set_index("date").sort_index()
        ma365 = df["value"].rolling(365, min_periods=30).mean()
        df["value"] = df["value"] / ma365.where(ma365 > 0, 1.0)
        df = df.dropna()
        return df
    except Exception as e:
        logger.warning("Blockchain.com miners-revenue fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_binance_futures_klines(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch Binance Futures klines (full history) — extracts taker buy ratio.

    Each kline includes total volume and taker buy volume, giving us
    taker buy ratio with years of history (unlike /futures/data/* which is 30 days).
    """
    import requests
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        all_rows = []
        start_time = None
        for _ in range(500):  # up to 500 pages x 1500 candles = ~5 years of 1h
            params = {"symbol": symbol, "interval": "1h", "limit": 1500}
            if start_time:
                params["startTime"] = start_time
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for d in data:
                ts = datetime.fromtimestamp(d[0] / 1000, tz=timezone.utc)
                volume = float(d[5])        # Quote asset volume
                taker_buy = float(d[10])     # Taker buy quote volume
                all_rows.append({
                    "date": ts,
                    "volume": volume,
                    "taker_buy_vol": taker_buy,
                    "taker_buy_ratio": taker_buy / volume if volume > 0 else 0.5,
                })
            if len(data) < 1500:
                break
            start_time = data[-1][0] + 1
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows).set_index("date").sort_index().drop_duplicates()
    except Exception as e:
        logger.warning("Binance futures klines fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_bybit_open_interest(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch historical OI from Bybit (free, no auth, back to symbol launch).

    Uses daily interval, paginates with cursor. BTCUSDT launched ~late 2020.
    """
    import requests
    try:
        url = "https://api.bybit.com/v5/market/open-interest"
        all_rows = []
        cursor = None
        for _ in range(100):  # safety limit
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "1d",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            items = result.get("list", [])
            if not items:
                break
            for item in items:
                ts = datetime.fromtimestamp(int(item["timestamp"]) / 1000, tz=timezone.utc)
                all_rows.append({"date": ts, "value": float(item["openInterest"])})
            cursor = result.get("nextPageCursor", "")
            if not cursor:
                break
            time.sleep(0.15)
        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows).set_index("date").sort_index().drop_duplicates()
    except Exception as e:
        logger.warning("Bybit OI fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_coinalyze_liquidations(symbol: str = "BTCUSDT_PERP.A") -> pd.DataFrame:
    """Fetch historical liquidation data from Coinalyze (free API key, unlimited daily).

    Returns long and short liquidation volumes in USD.
    Requires COINALYZE_API_KEY env var.
    """
    import os
    import requests
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        logger.warning("COINALYZE_API_KEY not set, skipping liquidation fetch")
        return pd.DataFrame()
    try:
        url = "https://api.coinalyze.net/v1/liquidation-history"
        # Fetch from 2020 onwards
        start = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime.now(timezone.utc).timestamp())
        params = {
            "symbols": symbol,
            "interval": "daily",
            "from": start,
            "to": end,
        }
        resp = requests.get(url, params=params, headers={"api_key": api_key}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list) or not data[0].get("history"):
            return pd.DataFrame()
        rows = []
        for item in data[0]["history"]:
            ts = datetime.fromtimestamp(item["t"], tz=timezone.utc)
            rows.append({
                "date": ts,
                "long_liq": float(item.get("l", 0)),
                "short_liq": float(item.get("s", 0)),
            })
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Coinalyze liquidation fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_coinalyze_long_short_ratio(symbol: str = "BTCUSDT_PERP.A") -> pd.DataFrame:
    """Fetch historical long/short ratio from Coinalyze (free API key, unlimited daily)."""
    import os
    import requests
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        logger.warning("COINALYZE_API_KEY not set, skipping long/short ratio fetch")
        return pd.DataFrame()
    try:
        url = "https://api.coinalyze.net/v1/long-short-ratio-history"
        start = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime.now(timezone.utc).timestamp())
        params = {
            "symbols": symbol,
            "interval": "daily",
            "from": start,
            "to": end,
        }
        resp = requests.get(url, params=params, headers={"api_key": api_key}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list) or not data[0].get("history"):
            return pd.DataFrame()
        rows = []
        for item in data[0]["history"]:
            ts = datetime.fromtimestamp(item["t"], tz=timezone.utc)
            rows.append({"date": ts, "value": float(item.get("r", 0.5))})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Coinalyze long/short ratio fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_coingecko_btc_dominance_history() -> pd.DataFrame:
    """Fetch historical BTC dominance from CoinGecko market_chart endpoint.

    Computes dominance as BTC market cap / total crypto market cap.
    """
    import requests
    try:
        # BTC market cap history (max range)
        btc_resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "max", "interval": "daily"},
            timeout=30,
        )
        btc_resp.raise_for_status()
        btc_data = btc_resp.json().get("market_caps", [])
        if not btc_data:
            return pd.DataFrame()

        time.sleep(1.5)  # respect CoinGecko rate limit

        # Total crypto market cap history
        total_resp = requests.get(
            "https://api.coingecko.com/api/v3/global/market_cap_chart",
            params={"days": "max"},
            timeout=30,
        )

        if total_resp.ok:
            total_data = total_resp.json().get("market_cap_chart", {}).get("market_cap", [])
        else:
            total_data = None

        rows = []
        if total_data and len(total_data) > 100:
            # Build total mcap lookup
            total_map = {}
            for ts_ms, val in total_data:
                day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                total_map[day] = val
            for ts_ms, btc_mcap in btc_data:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                day = ts.strftime("%Y-%m-%d")
                total_mcap = total_map.get(day)
                if total_mcap and total_mcap > 0:
                    rows.append({"date": ts, "value": btc_mcap / total_mcap})
        else:
            # Fallback: estimate dominance from BTC mcap alone (change metric)
            for ts_ms, btc_mcap in btc_data:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                rows.append({"date": ts, "value": btc_mcap})

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("date").sort_index()
        # If we only got raw BTC mcap (fallback), convert to pct_change
        if total_data is None or len(total_data) <= 100:
            df["value"] = df["value"].pct_change(7).clip(-0.5, 0.5)
            df = df.dropna()
        return df
    except Exception as e:
        logger.warning("CoinGecko BTC dominance history fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_coinmetrics(asset: str, metric: str) -> pd.DataFrame:
    """Fetch metrics from CoinMetrics Community API."""
    import requests
    try:
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        params = {"assets": asset, "metrics": metric, "frequency": "1d", "page_size": 10000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = pd.to_datetime(d["time"])
            val = float(d.get(metric, 0))
            rows.append({"date": ts, "value": val})
        df = pd.DataFrame(rows).set_index("date").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning("CoinMetrics %s/%s fetch failed: %s", asset, metric, e)
        return pd.DataFrame()


def _fetch_defillama_stablecoins() -> pd.DataFrame:
    """Fetch stablecoin total supply from DefiLlama."""
    import requests
    try:
        url = "https://stablecoins.llama.fi/stablecoincharts/all"
        params = {"stablecoin": 1}  # USDT
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(int(d["date"]), tz=timezone.utc)
            circ = d.get("totalCirculatingUSD", {})
            val = float(circ.get("peggedUSD", 0)) if isinstance(circ, dict) else 0.0
            rows.append({"date": ts, "value": val})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("DefiLlama stablecoins fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_defillama_tvl() -> pd.DataFrame:
    """Fetch total DeFi TVL from DefiLlama."""
    import requests
    try:
        url = "https://api.llama.fi/v2/historicalChainTvl"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d["date"], tz=timezone.utc)
            rows.append({"date": ts, "value": float(d.get("tvl", 0))})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("DefiLlama TVL fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_deribit_dvol() -> pd.DataFrame:
    """Fetch BTC DVOL implied volatility from Deribit."""
    import requests
    try:
        url = "https://deribit.com/api/v2/public/get_volatility_index_data"
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts = end_ts - (365 * 5 * 24 * 3600 * 1000)  # ~5 years
        params = {"currency": "BTC", "start_timestamp": start_ts,
                  "end_timestamp": end_ts, "resolution": "1D"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("result", {}).get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d[0] / 1000, tz=timezone.utc)
            rows.append({"date": ts, "value": float(d[4])})  # close
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Deribit DVOL fetch failed: %s", e)
        return pd.DataFrame()


def _fetch_coingecko_btc_dominance() -> pd.DataFrame:
    """Fetch BTC dominance from CoinGecko /global endpoint."""
    import requests
    try:
        url = "https://api.coingecko.com/api/v3/global"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 0)
        ts = datetime.now(timezone.utc)
        return pd.DataFrame([{"date": ts, "value": float(btc_dom)}]).set_index("date")
    except Exception as e:
        logger.warning("CoinGecko BTC dominance fetch failed: %s", e)
        return pd.DataFrame()


class ExternalDataProvider:
    """Fetch, cache, and serve external market data for ML features.

    Training mode: call fetch_all_training() to bulk-fetch history.
    Live mode: call refresh() periodically, then build_live_features().
    """

    def __init__(self, cache_dir: str | None = "data/external_cache") -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._live_cache: dict[str, float] = {}  # source -> latest value
        self._last_fetched: dict[str, datetime] = {
            source: datetime.now(timezone.utc)
            for source in STALENESS_THRESHOLDS
        }

    def is_critically_stale(self) -> bool:
        """Check if >30% of sources exceed their staleness threshold."""
        now = datetime.now(timezone.utc)
        stale_count = 0
        total = len(STALENESS_THRESHOLDS)
        for source, threshold in STALENESS_THRESHOLDS.items():
            last = self._last_fetched.get(source, now)
            if (now - last) > threshold:
                stale_count += 1
        return (stale_count / max(total, 1)) > CRITICAL_STALE_RATIO

    def fetch_all_training(self, idx_5m: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        """Bulk-fetch all external data sources for training.

        Returns dict mapping feature names to aligned numpy arrays.
        Each array has length == len(idx_5m).
        """
        start = idx_5m[0].to_pydatetime() if len(idx_5m) > 0 else datetime(2019, 1, 1, tzinfo=timezone.utc)
        end = idx_5m[-1].to_pydatetime() if len(idx_5m) > 0 else datetime.now(timezone.utc)

        result = {}

        # Fear & Greed
        fng = self._cached_fetch("fear_greed", lambda: _fetch_fear_greed(limit=0))
        if not fng.empty:
            result["fear_greed"] = align_to_5m(fng["value"] / 100.0, idx_5m)
            # 7d momentum
            fng_7d = fng["value"].rolling(7).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 100.0 if len(x) > 1 else 0)
            result["fear_greed_mom"] = align_to_5m(fng_7d, idx_5m)
        else:
            result["fear_greed"] = np.zeros(len(idx_5m))
            result["fear_greed_mom"] = np.zeros(len(idx_5m))

        # Google Trends (optional)
        gt = _fetch_google_trends(["bitcoin", "crypto"])
        for kw in ["bitcoin", "crypto"]:
            if kw in gt and not gt[kw].empty:
                result[f"gtrends_{kw}"] = align_to_5m(gt[kw]["value"] / 100.0, idx_5m)
            else:
                result[f"gtrends_{kw}"] = np.zeros(len(idx_5m))

        # Macro (yfinance)
        macro = _fetch_macro_yfinance(start, end)
        # Shift macro data by 1 day to avoid look-ahead bias
        if "dxy" in macro and not macro["dxy"].empty:
            dxy_ret = macro["dxy"]["value"].pct_change().shift(1).clip(-0.05, 0.05) * 20
            result["dxy_return"] = align_to_5m(dxy_ret, idx_5m)
        else:
            result["dxy_return"] = np.zeros(len(idx_5m))

        if "sp500" in macro and not macro["sp500"].empty:
            sp_ret = macro["sp500"]["value"].pct_change().shift(1).clip(-0.05, 0.05) * 20
            result["sp500_return"] = align_to_5m(sp_ret, idx_5m)
        else:
            result["sp500_return"] = np.zeros(len(idx_5m))

        if "gold" in macro and not macro["gold"].empty:
            gold_ret = macro["gold"]["value"].pct_change().shift(1).clip(-0.05, 0.05) * 20
            result["gold_return"] = align_to_5m(gold_ret, idx_5m)
        else:
            result["gold_return"] = np.zeros(len(idx_5m))

        if "vix" in macro and not macro["vix"].empty:
            result["vix"] = align_to_5m((macro["vix"]["value"] / 80.0).clip(0, 1), idx_5m)
        else:
            result["vix"] = np.zeros(len(idx_5m))

        if "tnx" in macro and not macro["tnx"].empty:
            result["treasury_10y"] = align_to_5m((macro["tnx"]["value"] / 10.0).clip(0, 1), idx_5m)
            if "twoy" in macro and not macro["twoy"].empty:
                spread = (macro["tnx"]["value"] - macro["twoy"]["value"]).clip(-5, 5) / 5.0
                result["yield_spread"] = align_to_5m(spread, idx_5m)
            else:
                result["yield_spread"] = np.zeros(len(idx_5m))
        else:
            result["treasury_10y"] = np.zeros(len(idx_5m))
            result["yield_spread"] = np.zeros(len(idx_5m))

        # On-chain: NVT proxy (CoinMetrics MarketCap / TxCount)
        nvt_df = self._cached_fetch("coinmetrics_nvt_proxy", _fetch_onchain_nvt_proxy)
        if not nvt_df.empty:
            aligned_raw = align_to_5m(nvt_df["value"], idx_5m)
            result["nvt"] = np.clip(np.log1p(aligned_raw) / np.log1p(200), 0, 1)
        else:
            result["nvt"] = np.zeros(len(idx_5m))

        # On-chain: MVRV (CoinMetrics)
        mvrv_df = self._cached_fetch("coinmetrics_mvrv", _fetch_onchain_mvrv)
        if not mvrv_df.empty:
            aligned_raw = align_to_5m(mvrv_df["value"], idx_5m)
            result["mvrv"] = np.clip(aligned_raw / 5.0, 0, 1)
        else:
            result["mvrv"] = np.zeros(len(idx_5m))

        # On-chain: SOPR — no free source available, zero-fill
        result["sopr"] = np.zeros(len(idx_5m))

        # On-chain: Puell Multiple (self-calculated from Blockchain.com miners-revenue)
        puell_df = self._cached_fetch("blockchain_com_puell", _fetch_blockchain_com_puell)
        if not puell_df.empty:
            result["puell"] = np.clip(align_to_5m(puell_df["value"], idx_5m) / 4.0, 0, 1)
        else:
            result["puell"] = np.zeros(len(idx_5m))

        # On-chain: Hashrate (CoinMetrics -> Blockchain.com fallback)
        hr_df = self._cached_fetch("coinmetrics_hashrate", _fetch_onchain_hashrate)
        if not hr_df.empty:
            change = hr_df["value"].pct_change(30).clip(-1, 1)
            result["hashrate"] = align_to_5m(change, idx_5m)
        else:
            result["hashrate"] = np.zeros(len(idx_5m))

        # CoinMetrics: active addresses
        for asset, key in [("eth", "eth_active_addr")]:
            df = self._cached_fetch(
                f"coinmetrics_{asset}_addr",
                lambda a=asset: _fetch_coinmetrics(a, "AdrActCnt"),
            )
            if not df.empty:
                change = df["value"].pct_change(7).clip(-1, 1)
                result[key] = align_to_5m(change, idx_5m)
            else:
                result[key] = np.zeros(len(idx_5m))

        # DefiLlama: stablecoin supply
        stable_df = self._cached_fetch("defillama_stablecoins", _fetch_defillama_stablecoins)
        if not stable_df.empty:
            change = stable_df["value"].pct_change(7).clip(-1, 1)
            result["stable_supply_change"] = align_to_5m(change, idx_5m)
        else:
            result["stable_supply_change"] = np.zeros(len(idx_5m))

        # DefiLlama: TVL
        tvl_df = self._cached_fetch("defillama_tvl", _fetch_defillama_tvl)
        if not tvl_df.empty:
            change = tvl_df["value"].pct_change(7).clip(-1, 1)
            result["tvl_change"] = align_to_5m(change, idx_5m)
        else:
            result["tvl_change"] = np.zeros(len(idx_5m))

        # Funding rates (Binance)
        funding = self._cached_fetch("funding_rates", _fetch_funding_rates)
        if not funding.empty:
            result["funding_24h_avg"] = align_to_5m(funding["rate"].rolling(3).mean().clip(-0.01, 0.01) * 100, idx_5m)
            # Indices 55-56: current funding rate and 7d average (for training)
            result["funding_rate_current"] = align_to_5m(funding["rate"].clip(-0.01, 0.01) * 100, idx_5m)
            result["funding_7d_avg"] = align_to_5m(funding["rate"].rolling(21).mean().clip(-0.01, 0.01) * 100, idx_5m)  # 21 x 8h = ~7d
        else:
            result["funding_24h_avg"] = np.zeros(len(idx_5m))
            result["funding_rate_current"] = np.zeros(len(idx_5m))
            result["funding_7d_avg"] = np.zeros(len(idx_5m))

        # Index 57: OI change 24h (Bybit — full history back to 2020)
        oi_df = self._cached_fetch("bybit_oi", _fetch_bybit_open_interest)
        if not oi_df.empty:
            oi_aligned = align_to_5m(oi_df["value"], idx_5m)
            oi_series = pd.Series(oi_aligned, index=idx_5m)
            oi_change_24h = oi_series.pct_change(288).clip(-1, 1)  # 288 x 5m = 24h
            result["oi_change_24h"] = oi_change_24h.fillna(0).values
        else:
            result["oi_change_24h"] = np.zeros(len(idx_5m))

        # Index 58-59: Long/short liquidations (Coinalyze — unlimited daily history)
        liq_df = self._cached_fetch("coinalyze_liquidations", _fetch_coinalyze_liquidations)
        if not liq_df.empty and "long_liq" in liq_df.columns:
            long_aligned = align_to_5m(liq_df["long_liq"], idx_5m)
            short_aligned = align_to_5m(liq_df["short_liq"], idx_5m)
            # Normalize to [0, 1] range
            max_liq = max(np.nanmax(long_aligned), np.nanmax(short_aligned), 1.0)
            result["long_liq_24h"] = np.clip(long_aligned / max_liq, 0, 1)
            result["short_liq_24h"] = np.clip(short_aligned / max_liq, 0, 1)
        else:
            result["long_liq_24h"] = np.zeros(len(idx_5m))
            result["short_liq_24h"] = np.zeros(len(idx_5m))

        # Index 60: Exchange netflow (Blockchain.com tx volume as proxy)
        exflow = self._cached_fetch("blockchain_com_txvol", _fetch_blockchain_com_exchange_flows)
        if not exflow.empty:
            change = exflow["value"].pct_change(7).clip(-1, 1)
            result["exchange_netflow"] = align_to_5m(change, idx_5m)
        else:
            result["exchange_netflow"] = np.zeros(len(idx_5m))

        # Index 61: BTC active addresses change 7d (CoinMetrics)
        btc_addr = self._cached_fetch("coinmetrics_btc_addr", lambda: _fetch_coinmetrics("btc", "AdrActCnt"))
        if not btc_addr.empty:
            change = btc_addr["value"].pct_change(7).clip(-1, 1)
            result["active_addr_change_7d"] = align_to_5m(change, idx_5m)
        else:
            result["active_addr_change_7d"] = np.zeros(len(idx_5m))

        # Index 83: OI change 7d (from Bybit OI data)
        if not oi_df.empty:
            oi_change_7d = oi_series.pct_change(2016).clip(-1, 1)  # 2016 x 5m = 7d
            result["oi_change_7d"] = oi_change_7d.fillna(0).values
        else:
            result["oi_change_7d"] = np.zeros(len(idx_5m))

        # Index 84: Liquidation ratio (Coinalyze — long / (long + short))
        if not liq_df.empty and "long_liq" in liq_df.columns:
            total_liq = liq_df["long_liq"] + liq_df["short_liq"]
            ratio = (liq_df["long_liq"] / total_liq.where(total_liq > 0, 1.0)).clip(0, 1)
            result["liq_ratio"] = align_to_5m(ratio, idx_5m)
        else:
            result["liq_ratio"] = np.zeros(len(idx_5m))

        # Taker buy ratio (Binance Futures klines — full history)
        taker = self._cached_fetch("binance_futures_klines", _fetch_binance_futures_klines)
        if not taker.empty and "taker_buy_ratio" in taker.columns:
            result["taker_buy_ratio"] = align_to_5m(taker["taker_buy_ratio"], idx_5m)
        else:
            result["taker_buy_ratio"] = np.full(len(idx_5m), 0.5)

        # Deribit DVOL
        dvol = self._cached_fetch("deribit_dvol", _fetch_deribit_dvol)
        if not dvol.empty:
            result["dvol"] = align_to_5m((dvol["value"] / 200.0).clip(0, 1), idx_5m)
        else:
            result["dvol"] = np.zeros(len(idx_5m))

        # BTC dominance change (CoinGecko — full history)
        btc_dom = self._cached_fetch("coingecko_btc_dom_hist", _fetch_coingecko_btc_dominance_history)
        if not btc_dom.empty:
            # If we got actual dominance ratios (0-1), compute 7d change
            if btc_dom["value"].max() <= 1.0:
                dom_change = btc_dom["value"].diff(7).clip(-0.1, 0.1) * 10  # scale to [-1, 1]
                result["btc_dom_change"] = align_to_5m(dom_change, idx_5m)
            else:
                # Fallback: already pct_change from the fetch function
                result["btc_dom_change"] = align_to_5m(btc_dom["value"].clip(-1, 1), idx_5m)
        else:
            result["btc_dom_change"] = np.zeros(len(idx_5m))

        # Stablecoin / BTC market cap ratio
        if not stable_df.empty and "dxy" in macro:
            try:
                import yfinance as yf
                btc_hist = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                                       end=end.strftime("%Y-%m-%d"), progress=False)
                if not btc_hist.empty:
                    if isinstance(btc_hist.columns, pd.MultiIndex):
                        btc_hist.columns = btc_hist.columns.get_level_values(0)
                    btc_mcap = btc_hist["Close"] * 19_800_000
                    # Normalize index dtypes to avoid datetime64 resolution mismatch
                    stable_vals = stable_df["value"].copy()
                    if stable_vals.index.tz is None:
                        stable_vals.index = stable_vals.index.tz_localize("UTC")
                    if btc_mcap.index.tz is None:
                        btc_mcap.index = btc_mcap.index.tz_localize("UTC")
                    stable_vals.index = stable_vals.index.as_unit("ns")
                    btc_mcap.index = btc_mcap.index.as_unit("ns")
                    stable_aligned = stable_vals.reindex(btc_mcap.index, method="ffill").fillna(0)
                    ratio = (stable_aligned / btc_mcap.replace(0, np.nan)).fillna(0).clip(0, 1)
                    result["stable_btc_ratio"] = align_to_5m(ratio, idx_5m)
                else:
                    result["stable_btc_ratio"] = np.zeros(len(idx_5m))
            except Exception as e:
                logger.warning("Stablecoin/BTC ratio computation failed: %s", e)
                result["stable_btc_ratio"] = np.zeros(len(idx_5m))
        else:
            result["stable_btc_ratio"] = np.zeros(len(idx_5m))

        return result

    def build_training_features(self, idx_5m: pd.DatetimeIndex) -> np.ndarray:
        """Build 32-column external feature array aligned to 5m index.

        Layout: 7 columns for indices 55-61 + 25 columns for indices 65-89.
        This matches what _batch_extract_tabular expects.
        """
        data = self.fetch_all_training(idx_5m)
        n = len(idx_5m)
        features = np.zeros((n, 32), dtype=np.float64)

        # Columns 0-6 -> feature indices 55-61
        idx55_keys = [
            "funding_rate_current",    # 55: funding rate current
            "funding_7d_avg",          # 56: funding rate 7d average
            "oi_change_24h",           # 57: OI change 24h
            "long_liq_24h",            # 58: long liquidations 24h
            "short_liq_24h",           # 59: short liquidations 24h
            "exchange_netflow",        # 60: BTC exchange netflow
            "active_addr_change_7d",   # 61: active addresses change 7d
        ]
        for col, key in enumerate(idx55_keys):
            if key in data:
                features[:, col] = data[key]

        # Columns 7-31 -> feature indices 65-89
        idx65_keys = [
            "fear_greed",           # 65
            "fear_greed_mom",       # 66
            "gtrends_bitcoin",      # 67
            "gtrends_crypto",       # 68
            "dxy_return",           # 69
            "sp500_return",         # 70
            "gold_return",          # 71
            "vix",                  # 72
            "treasury_10y",         # 73
            "yield_spread",         # 74
            "nvt",                  # 75
            "mvrv",                 # 76
            "sopr",                 # 77
            "puell",                # 78
            "hashrate",             # 79
            "eth_active_addr",      # 80
            "stable_supply_change", # 81
            "tvl_change",           # 82
            "oi_change_7d",         # 83
            "liq_ratio",            # 84
            "taker_buy_ratio",      # 85
            "dvol",                 # 86
            "funding_24h_avg",      # 87
            "btc_dom_change",       # 88
            "stable_btc_ratio",     # 89
        ]
        for col, key in enumerate(idx65_keys):
            if key in data:
                features[:, 7 + col] = data[key]

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def build_live_features(self) -> np.ndarray:
        """Build 25-element feature vector from live cached data.

        Returns zeros for any source that hasn't been fetched yet.
        Feature keys map to indices 65-89 (same order as build_training_features).
        """
        feature_keys = [
            "fear_greed", "fear_greed_mom", "gtrends_bitcoin", "gtrends_crypto",
            "dxy_return", "sp500_return", "gold_return", "vix",
            "treasury_10y", "yield_spread", "nvt", "mvrv", "sopr", "puell",
            "hashrate", "eth_active_addr", "stable_supply_change", "tvl_change",
            "oi_change_7d", "liq_ratio", "taker_buy_ratio", "dvol",
            "funding_24h_avg", "btc_dom_change", "stable_btc_ratio",
        ]
        features = np.zeros(25, dtype=np.float32)
        for i, key in enumerate(feature_keys):
            features[i] = float(self._live_cache.get(key, 0.0))
        return features

    def refresh_live(self) -> None:
        """Refresh live data cache (called periodically by trading loop).

        Each source is fetched with error handling. On success, the cache
        and last-fetched timestamp are updated. On failure, stale values
        are retained and a warning is logged.
        """
        now = datetime.now(timezone.utc)

        # Fear & Greed (daily source)
        try:
            fng = _fetch_fear_greed(limit=8)
            if not fng.empty:
                val = float(fng["value"].iloc[-1]) / 100.0
                self._live_cache["fear_greed"] = val
                if len(fng) >= 7:
                    self._live_cache["fear_greed_mom"] = (float(fng["value"].iloc[-1]) - float(fng["value"].iloc[-7])) / 100.0
                elif len(fng) >= 2:
                    self._live_cache["fear_greed_mom"] = (float(fng["value"].iloc[-1]) - float(fng["value"].iloc[0])) / 100.0
                self._last_fetched["fear_greed"] = now
        except Exception as e:
            logger.warning("Live refresh fear_greed failed: %s", e)

        # Macro data (yfinance)
        try:
            end = now
            start = now - timedelta(days=7)
            macro = _fetch_macro_yfinance(start, end)
            for key, ticker_key, transform in [
                ("dxy_return", "dxy", lambda s: float(s.pct_change().iloc[-1]) * 20 if len(s) > 1 else 0.0),
                ("sp500_return", "sp500", lambda s: float(s.pct_change().iloc[-1]) * 20 if len(s) > 1 else 0.0),
                ("gold_return", "gold", lambda s: float(s.pct_change().iloc[-1]) * 20 if len(s) > 1 else 0.0),
                ("vix", "vix", lambda s: np.clip(float(s.iloc[-1]) / 80.0, 0, 1) if len(s) > 0 else 0.0),
                ("treasury_10y", "tnx", lambda s: np.clip(float(s.iloc[-1]) / 10.0, 0, 1) if len(s) > 0 else 0.0),
            ]:
                if ticker_key in macro and not macro[ticker_key].empty:
                    self._live_cache[key] = transform(macro[ticker_key]["value"])
            self._last_fetched["macro"] = now
        except Exception as e:
            logger.warning("Live refresh macro failed: %s", e)

        # Funding rates
        try:
            funding = _fetch_funding_rates(limit=10)
            if not funding.empty:
                avg = float(funding["rate"].tail(3).mean()) * 100
                self._live_cache["funding_24h_avg"] = np.clip(avg, -1, 1)
                self._last_fetched["funding_rate"] = now
        except Exception as e:
            logger.warning("Live refresh funding failed: %s", e)

        # On-chain (CoinMetrics)
        try:
            mvrv_df = _fetch_coinmetrics_timeseries("CapMVRVCur", "btc")
            if not mvrv_df.empty:
                self._live_cache["mvrv"] = float(np.clip(mvrv_df["value"].iloc[-1] / 5.0, 0, 1))
            nvt_df = _fetch_onchain_nvt_proxy()
            if not nvt_df.empty:
                self._live_cache["nvt"] = float(np.clip(np.log1p(nvt_df["value"].iloc[-1]) / np.log1p(200), 0, 1))
            self._last_fetched["onchain"] = now
        except Exception as e:
            logger.warning("Live refresh onchain failed: %s", e)

        # DeFi (DefiLlama)
        try:
            tvl = _fetch_defillama_tvl()
            if not tvl.empty and len(tvl) >= 8:
                change = (float(tvl["value"].iloc[-1]) - float(tvl["value"].iloc[-8])) / max(float(tvl["value"].iloc[-8]), 1)
                self._live_cache["tvl_change"] = np.clip(change, -1, 1)
            self._last_fetched["defi"] = now
        except Exception as e:
            logger.warning("Live refresh defi failed: %s", e)

        # Deribit DVOL
        try:
            dvol = _fetch_deribit_dvol()
            if not dvol.empty:
                self._live_cache["dvol"] = np.clip(float(dvol["value"].iloc[-1]) / 200.0, 0, 1)
                self._last_fetched["dvol"] = now
        except Exception as e:
            logger.warning("Live refresh dvol failed: %s", e)

        # Binance Futures: OI and taker buy ratio (live)
        try:
            import requests
            # Open Interest (current snapshot)
            resp = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                                params={"symbol": "BTCUSDT"}, timeout=10)
            if resp.ok:
                oi_val = float(resp.json().get("openInterest", 0))
                prev_oi = self._live_cache.get("_oi_prev", oi_val)
                self._live_cache["oi_change_24h"] = np.clip((oi_val - prev_oi) / max(prev_oi, 1) * 10, -1, 1)
                self._live_cache["_oi_prev"] = oi_val

            # Taker buy ratio from recent klines
            resp2 = requests.get("https://fapi.binance.com/fapi/v1/klines",
                                 params={"symbol": "BTCUSDT", "interval": "1h", "limit": 10},
                                 timeout=10)
            if resp2.ok:
                data = resp2.json()
                if data:
                    last = data[-1]
                    vol = float(last[5])
                    taker_buy = float(last[10])
                    self._live_cache["taker_buy_ratio"] = taker_buy / vol if vol > 0 else 0.5

            self._last_fetched["oi_liquidations"] = now
        except Exception as e:
            logger.warning("Live refresh Binance derivatives failed: %s", e)

        # Coinalyze: liquidations and long/short ratio (live)
        try:
            liq = _fetch_coinalyze_liquidations()
            if not liq.empty and "long_liq" in liq.columns:
                last = liq.iloc[-1]
                total = max(last["long_liq"] + last["short_liq"], 1.0)
                self._live_cache["long_liq_24h"] = np.clip(last["long_liq"] / total, 0, 1)
                self._live_cache["short_liq_24h"] = np.clip(last["short_liq"] / total, 0, 1)
                self._live_cache["liq_ratio"] = np.clip(last["long_liq"] / total, 0, 1)
        except Exception as e:
            logger.warning("Live refresh Coinalyze liquidations failed: %s", e)

        # Puell Multiple (Blockchain.com)
        try:
            puell_df = _fetch_blockchain_com_puell()
            if not puell_df.empty:
                self._live_cache["puell"] = float(np.clip(puell_df["value"].iloc[-1] / 4.0, 0, 1))
        except Exception as e:
            logger.warning("Live refresh puell failed: %s", e)

        # Hashrate (CoinMetrics -> Blockchain.com fallback)
        try:
            hr = _fetch_onchain_hashrate()
            if not hr.empty and len(hr) >= 31:
                change = (float(hr["value"].iloc[-1]) - float(hr["value"].iloc[-31])) / max(float(hr["value"].iloc[-31]), 1)
                self._live_cache["hashrate"] = np.clip(change, -1, 1)
        except Exception as e:
            logger.warning("Live refresh hashrate failed: %s", e)

        # ETH active addresses (CoinMetrics)
        try:
            eth_addr = _fetch_coinmetrics("eth", "AdrActCnt")
            if not eth_addr.empty and len(eth_addr) >= 8:
                change = (float(eth_addr["value"].iloc[-1]) - float(eth_addr["value"].iloc[-8])) / max(float(eth_addr["value"].iloc[-8]), 1)
                self._live_cache["eth_active_addr"] = np.clip(change, -1, 1)
        except Exception as e:
            logger.warning("Live refresh eth_active_addr failed: %s", e)

        # Stablecoin supply change (DefiLlama)
        try:
            stable = _fetch_defillama_stablecoins()
            if not stable.empty and len(stable) >= 8:
                change = (float(stable["value"].iloc[-1]) - float(stable["value"].iloc[-8])) / max(float(stable["value"].iloc[-8]), 1)
                self._live_cache["stable_supply_change"] = np.clip(change, -1, 1)
        except Exception as e:
            logger.warning("Live refresh stable_supply_change failed: %s", e)

        # Yield spread (compute from macro if available)
        try:
            end_ys = now
            start_ys = now - timedelta(days=7)
            macro_ys = _fetch_macro_yfinance(start_ys, end_ys)
            if "tnx" in macro_ys and "twoy" in macro_ys:
                tnx_val = float(macro_ys["tnx"]["value"].iloc[-1]) if not macro_ys["tnx"].empty else 0
                twoy_val = float(macro_ys["twoy"]["value"].iloc[-1]) if not macro_ys["twoy"].empty else 0
                self._live_cache["yield_spread"] = np.clip((tnx_val - twoy_val) / 5.0, -1, 1)
        except Exception as e:
            logger.warning("Live refresh yield_spread failed: %s", e)

        # BTC dominance change (CoinGecko -- live only)
        try:
            btc_dom = _fetch_coingecko_btc_dominance()
            if not btc_dom.empty:
                current = float(btc_dom["value"].iloc[-1]) / 100.0
                prev = self._live_cache.get("_btc_dom_prev", current)
                self._live_cache["btc_dom_change"] = np.clip((current - prev) * 10, -1, 1)
                self._live_cache["_btc_dom_prev"] = current
        except Exception as e:
            logger.warning("Live refresh btc_dom failed: %s", e)

        # Google Trends -- skipped in live mode (weekly data, rate-limited)

        # stable_btc_ratio -- simplified in live mode
        try:
            if "stable_supply_change" in self._live_cache:
                self._live_cache["stable_btc_ratio"] = 0.0
        except Exception:
            pass

    def _cached_fetch(self, name: str, fetch_fn) -> pd.DataFrame:
        """Fetch data with parquet cache."""
        if self._cache_dir:
            cache_path = str(self._cache_dir / f"{name}.parquet")
            cached = load_cache(cache_path)
            if cached is not None:
                logger.info("Using cached data for %s (%d rows)", name, len(cached))
                return cached

        logger.info("Fetching %s from API...", name)
        try:
            df = fetch_fn()
            if self._cache_dir and not df.empty:
                save_cache(df, str(self._cache_dir / f"{name}.parquet"))
                logger.info("Cached %s (%d rows)", name, len(df))
            return df
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)
            return pd.DataFrame()
