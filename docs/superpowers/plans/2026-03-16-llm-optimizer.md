# LLM-Based Walk-Forward Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Optuna with a local Ollama + Qwen 3 8B agent that reasons about market conditions to propose better trading parameters in fewer trials.

**Architecture:** Two new modules (`llm_optimizer.py` for Ollama client/prompt building, `market_summarizer.py` for candle-to-text conversion) replace `_run_optuna_window()` in `walk_forward.py`. The async/sync boundary is preserved — LLM calls happen in sync context via `run_in_executor`, with a sentinel pattern for error propagation to the async caller.

**Tech Stack:** Ollama REST API via `requests`, Qwen 3 8B (Q4_K_M), existing `bot/indicators/` for market stats.

**Spec:** `docs/superpowers/specs/2026-03-16-llm-optimizer-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `bot/learning/llm_optimizer.py` | Create | Ollama HTTP client, prompt builder, JSON response parser, parameter clamping |
| `bot/learning/market_summarizer.py` | Create | Convert raw candle data into text summary (stats + daily CSV + 4h CSV) |
| `bot/learning/walk_forward.py` | Modify | Replace `_run_optuna_window` with `_run_llm_window`, expand `_run_single_backtest` return, add sentinel handling |
| `bot/main.py` | Modify | Add Ollama health check on startup, pass event_bus to WalkForwardOptimizer |
| `docker-compose.yml` | Modify | Add `extra_hosts` and `OLLAMA_URL` env var to bot service |
| `requirements.txt` | Modify | Remove `optuna`, add `requests` |
| `tests/test_llm_optimizer.py` | Create | Unit tests for Ollama client, JSON parsing, clamping, fallback |
| `tests/test_market_summarizer.py` | Create | Unit tests for candle summarization, all three sections |
| `tests/test_llm_walk_forward.py` | Create | Unit tests for `_run_llm_window` round flow, scoring, sentinel |

---

## Chunk 1: Core LLM Optimizer Module

### Task 1: Ollama Health Check and Error Types

**Files:**
- Create: `bot/learning/llm_optimizer.py`
- Create: `tests/test_llm_optimizer.py`

- [ ] **Step 1: Write failing tests for health check**

```python
# tests/test_llm_optimizer.py
"""Tests for LLM optimizer — Ollama client, prompt builder, response parser."""
from __future__ import annotations
import json
import pytest
from unittest.mock import patch, MagicMock

from bot.learning.llm_optimizer import (
    OllamaUnavailableError,
    check_ollama,
    OLLAMA_URL,
    OLLAMA_MODEL,
)


class TestCheckOllama:
    def test_check_ollama_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        with patch("bot.learning.llm_optimizer.requests.get", return_value=mock_resp):
            check_ollama()  # Should not raise

    def test_check_ollama_connection_error(self):
        with patch("bot.learning.llm_optimizer.requests.get", side_effect=Exception("Connection refused")):
            with pytest.raises(OllamaUnavailableError):
                check_ollama()

    def test_check_ollama_bad_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        with patch("bot.learning.llm_optimizer.requests.get", return_value=mock_resp):
            with pytest.raises(OllamaUnavailableError):
                check_ollama()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot.learning.llm_optimizer'`

- [ ] **Step 3: Implement health check and error types**

```python
# bot/learning/llm_optimizer.py
"""LLM-based parameter optimizer — Ollama client, prompt builder, response parser."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = 120  # seconds per call
TEMPERATURE = 0.7
NUM_PREDICT = 1024

# Parameter ranges for validation/clamping
PARAM_RANGES = {
    "atr_multiplier":       (2.5, 5.0, 0.5),
    "rr_ratio":             (2.0, 4.0, 0.5),
    "base_risk_pct":        (2.0, 5.0, 0.5),
    "min_profit_multiple":  (2.0, 4.0, 0.5),
    "max_hold_hours":       (48, 240, 24),
    "quiet_atr_threshold":  (0.8, 1.5, 0.1),
    "regime_adx_threshold": (20, 30, 2),
    "ranging_adx_threshold":(15, 25, 2),
}


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not reachable or model not found."""


def check_ollama() -> None:
    """Health check — GET /api/tags. Raises OllamaUnavailableError if unreachable or model missing."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Check model is available (match with or without tag suffix)
        model_base = OLLAMA_MODEL.split(":")[0]
        if not any(model_base in m for m in models):
            raise OllamaUnavailableError(
                f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                f"Available: {models}. Run: ollama pull {OLLAMA_MODEL}"
            )
    except OllamaUnavailableError:
        raise
    except Exception as exc:
        raise OllamaUnavailableError(f"Ollama not reachable at {OLLAMA_URL}: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_optimizer.py::TestCheckOllama -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add bot/learning/llm_optimizer.py tests/test_llm_optimizer.py
git commit -m "feat: add Ollama health check and OllamaUnavailableError"
```

---

### Task 2: JSON Response Parsing and Parameter Clamping

**Files:**
- Modify: `bot/learning/llm_optimizer.py`
- Modify: `tests/test_llm_optimizer.py`

- [ ] **Step 1: Write failing tests for parsing and clamping**

```python
# Append to tests/test_llm_optimizer.py
from bot.learning.llm_optimizer import _parse_response, _clamp_params, PARAM_RANGES


class TestParseResponse:
    def test_parse_clean_json(self):
        raw = '[{"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}]'
        result = _parse_response(raw)
        assert len(result) == 1
        assert result[0]["atr_multiplier"] == 3.5

    def test_parse_markdown_fenced_json(self):
        raw = '```json\n[{"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}]\n```'
        result = _parse_response(raw)
        assert len(result) == 1

    def test_parse_invalid_json_returns_none(self):
        result = _parse_response("This is not JSON at all")
        assert result is None

    def test_parse_missing_keys_skips_entry(self):
        raw = '[{"atr_multiplier": 3.5}]'
        result = _parse_response(raw)
        assert result == []  # Skipped because missing 7 required keys

    def test_parse_multiple_entries(self):
        entry = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        raw = json.dumps([entry, entry, entry])
        result = _parse_response(raw)
        assert len(result) == 3


class TestClampParams:
    def test_clamp_within_range_unchanged(self):
        params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        result = _clamp_params(params)
        assert result == params

    def test_clamp_out_of_range_values(self):
        params = {"atr_multiplier": 10.0, "rr_ratio": 0.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 500, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        result = _clamp_params(params)
        assert result["atr_multiplier"] == 5.0
        assert result["rr_ratio"] == 2.0
        assert result["max_hold_hours"] == 240

    def test_clamp_rounds_to_step(self):
        params = {"atr_multiplier": 3.17, "rr_ratio": 2.73, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 100, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        result = _clamp_params(params)
        assert result["atr_multiplier"] == 3.0  # rounded to nearest 0.5
        assert result["rr_ratio"] == 2.5  # rounded to nearest 0.5
        assert result["max_hold_hours"] == 96  # rounded to nearest 24
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_optimizer.py::TestParseResponse tests/test_llm_optimizer.py::TestClampParams -v`
Expected: FAIL — `ImportError: cannot import name '_parse_response'`

- [ ] **Step 3: Implement parsing and clamping**

Append to `bot/learning/llm_optimizer.py`:

```python
def _clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp values to valid ranges and round to nearest step."""
    clamped = {}
    for key, (lo, hi, step) in PARAM_RANGES.items():
        val = float(params.get(key, lo))
        val = max(lo, min(hi, val))
        # Round to nearest step
        val = round(round((val - lo) / step) * step + lo, 4)
        # Ensure still in range after rounding
        val = max(lo, min(hi, val))
        int_keys = {"max_hold_hours", "regime_adx_threshold", "ranging_adx_threshold"}
        clamped[key] = int(val) if key in int_keys else round(val, 2)
    return clamped


def _parse_response(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Extract and validate JSON array of parameter sets from LLM response.

    Returns list of valid param dicts, None if no JSON array found at all.
    """
    # Strip markdown code fences
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None

    try:
        arr = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(arr, list):
        return None

    # Validate and clamp each entry
    required_keys = set(PARAM_RANGES.keys())
    valid = []
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        if not required_keys.issubset(entry.keys()):
            logger.warning("LLM response missing keys: %s", required_keys - set(entry.keys()))
            continue
        valid.append(_clamp_params(entry))

    return valid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_optimizer.py::TestParseResponse tests/test_llm_optimizer.py::TestClampParams -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add bot/learning/llm_optimizer.py tests/test_llm_optimizer.py
git commit -m "feat: add JSON response parsing and parameter clamping for LLM optimizer"
```

---

### Task 3: Ollama API Call and Prompt Functions

**Files:**
- Modify: `bot/learning/llm_optimizer.py`
- Modify: `tests/test_llm_optimizer.py`

- [ ] **Step 1: Write failing tests for propose and refine**

```python
# Append to tests/test_llm_optimizer.py
from bot.learning.llm_optimizer import propose_params, refine_params, _call_ollama, STRATEGY_DESCRIPTIONS


class TestCallOllama:
    def test_call_ollama_returns_text(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "hello"}
        with patch("bot.learning.llm_optimizer.requests.post", return_value=mock_resp):
            result = _call_ollama("test prompt")
            assert result == "hello"

    def test_call_ollama_raises_on_failure(self):
        with patch("bot.learning.llm_optimizer.requests.post", side_effect=Exception("timeout")):
            with pytest.raises(OllamaUnavailableError):
                _call_ollama("test prompt")


VALID_DEFAULTS = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}


class TestProposeParams:
    def test_propose_returns_param_list(self):
        entry = json.dumps([VALID_DEFAULTS] * 6)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": entry}
        with patch("bot.learning.llm_optimizer.requests.post", return_value=mock_resp):
            result = propose_params("market summary", VALID_DEFAULTS, "orderflow", None)
            assert len(result) == 6

    def test_propose_fallback_on_bad_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "I cannot help with that"}
        with patch("bot.learning.llm_optimizer.requests.post", return_value=mock_resp):
            result = propose_params("market summary", VALID_DEFAULTS, "orderflow", None)
            assert len(result) == 1
            assert result[0] == VALID_DEFAULTS

    def test_propose_raises_on_ollama_down(self):
        with patch("bot.learning.llm_optimizer.requests.post", side_effect=Exception("refused")):
            with pytest.raises(OllamaUnavailableError):
                propose_params("market summary", VALID_DEFAULTS, "orderflow", None)


class TestRefineParams:
    def test_refine_returns_param_list(self):
        entry = json.dumps([VALID_DEFAULTS] * 6)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": entry}
        with patch("bot.learning.llm_optimizer.requests.post", return_value=mock_resp):
            result = refine_params("market summary", [{"params": VALID_DEFAULTS, "pnl": 100}], VALID_DEFAULTS, "orderflow")
            assert len(result) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_optimizer.py::TestCallOllama tests/test_llm_optimizer.py::TestProposeParams tests/test_llm_optimizer.py::TestRefineParams -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement Ollama API call and prompt functions**

Append to `bot/learning/llm_optimizer.py`:

```python
STRATEGY_DESCRIPTIONS = {
    "orderflow": "Detects absorption patterns — high volume candles with small bodies and long wicks. LONG on buying absorption (lower wicks), SHORT on selling absorption (upper wicks). Confirmed by CMF and OBV divergence.",
    "funding_contrarian": "Contrarian strategy — goes LONG when RSI is oversold (<28) with MACD recovering, SHORT when RSI is overbought (>72) with MACD fading. Fades crowd extremes.",
    "range": "Range-bound trading — buys near lower Bollinger Band when RSI <35, sells near upper BB when RSI >65. Only active when ADX <22 (non-trending). Profits from oscillation within bands.",
    "squeeze": "Volatility squeeze breakout — detects BB compression then expansion with volume surge. Goes LONG on upward breakout (EMA50 slope positive), SHORT on downward breakout (EMA50 slope negative).",
}

_PARAM_RANGES_TEXT = """- atr_multiplier: 2.5-5.0 (stop-loss distance, ATR * multiplier)
- rr_ratio: 2.0-4.0 (take-profit distance, risk * ratio)
- base_risk_pct: 2.0-5.0 (position size as % of equity risked)
- min_profit_multiple: 2.0-4.0 (reject trades where profit < N * fees)
- max_hold_hours: 48-240 (maximum trade duration)
- quiet_atr_threshold: 0.8-1.5 (ATR% below this = skip, too quiet)
- regime_adx_threshold: 20-30 (ADX above this = trending market)
- ranging_adx_threshold: 15-25 (ADX below this = ranging market)"""


def _call_ollama(prompt: str) -> str:
    """Send prompt to Ollama, return raw response text. Raises OllamaUnavailableError on failure."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "/no_think\n" + prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")
        logger.debug("Ollama prompt:\n%s", prompt[:500])
        logger.debug("Ollama response:\n%s", response_text[:500])
        return response_text
    except Exception as exc:
        raise OllamaUnavailableError(f"Ollama call failed: {exc}") from exc


def _build_propose_prompt(
    market_summary: str,
    champion_defaults: Dict,
    target_strategy: str | None,
    prev_windows: List[Dict] | None,
) -> str:
    """Build Round 1 (propose) prompt."""
    strategy_name = target_strategy or "all"
    strategy_desc = STRATEGY_DESCRIPTIONS.get(target_strategy, "Multi-strategy trading system.")

    prev_section = ""
    if prev_windows:
        lines = []
        for w in prev_windows:
            lines.append(f"Window {w['window']}: pnl=€{w['pnl']:.2f}, sharpe={w['sharpe']:.2f}, pf={w['profit_factor']:.2f}")
        prev_section = "\nPREVIOUS WINDOW RESULTS:\n" + "\n".join(lines)

    return f"""You are a quantitative trading parameter optimizer for cryptocurrency markets.

STRATEGY: {strategy_name}
{strategy_desc}

MARKET CONDITIONS:
{market_summary}

{prev_section}

PARAMETER RANGES:
{_PARAM_RANGES_TEXT}

CURRENT DEFAULTS: {json.dumps(champion_defaults)}

Based on the market conditions, suggest 6 parameter sets optimized for this market environment. Consider:
- In downtrends, wider stops and longer hold times often help
- In ranging markets, tighter stops and shorter holds work better
- High volatility needs wider ATR multipliers
- Low volatility benefits from lower quiet_atr_threshold

Respond ONLY with a JSON array of 6 objects. No explanation."""


def _build_refine_prompt(
    market_summary: str,
    results: List[Dict],
    champion_defaults: Dict,
    target_strategy: str | None,
    round_number: int = 2,
) -> str:
    """Build Round 2+ (refine) prompt."""
    strategy_name = target_strategy or "all"
    strategy_desc = STRATEGY_DESCRIPTIONS.get(target_strategy, "Multi-strategy trading system.")

    # Build results table — group by round label if present
    header = "#  atr  rr  risk  hold  pnl      sharpe  pf    trades  W/L    avg_win%  avg_loss%  avg_hold"

    def _format_row(i, r):
        p = r["params"]
        return (
            f"{i}  {p['atr_multiplier']:.1f}  {p['rr_ratio']:.1f}  {p['base_risk_pct']:.1f}  "
            f"{p['max_hold_hours']}h  €{r['pnl']:.2f}  {r['sharpe']:.2f}  {r['pf']:.2f}  "
            f"{r.get('trades', 0)}  {r.get('wins', 0)}/{r.get('losses', 0)}  "
            f"{r.get('avg_win_pct', 0):.2f}  {r.get('avg_loss_pct', 0):.2f}  "
            f"{r.get('avg_hold_hours', 0):.0f}h"
        )

    # Check if results have round labels (Round 3 combined format)
    has_rounds = any("round" in r for r in results)
    if has_rounds:
        sections = []
        for label in ("ROUND 1", "ROUND 2"):
            group = [r for r in results if r.get("round") == label]
            if group:
                rows = [f"{label}:", header]
                for i, r in enumerate(group, 1):
                    rows.append(_format_row(i, r))
                sections.append("\n".join(rows))
        results_table = "\n\n".join(sections)
    else:
        rows = [header]
        for i, r in enumerate(results, 1):
            rows.append(_format_row(i, r))
        results_table = "\n".join(rows)

    return f"""You are refining trading parameters based on backtest results.

STRATEGY: {strategy_name}
{strategy_desc}

MARKET CONDITIONS:
{market_summary}

PARAMETER RANGES:
{_PARAM_RANGES_TEXT}

CURRENT DEFAULTS: {json.dumps(champion_defaults)}

BACKTEST RESULTS FROM ROUND {round_number - 1}:
{results_table}

Suggest 6 improved parameter sets. Keep what worked, adjust what didn't.
Respond ONLY with a JSON array of 6 objects. No explanation."""


def propose_params(
    market_summary: str,
    champion_defaults: Dict,
    target_strategy: str | None,
    prev_windows: List[Dict] | None,
) -> List[Dict[str, Any]]:
    """Round 1: propose 6 parameter sets based on market context.

    Raises OllamaUnavailableError if Ollama is not reachable.
    Falls back to [champion_defaults] if JSON parsing fails after retry.
    """
    prompt = _build_propose_prompt(market_summary, champion_defaults, target_strategy, prev_windows)
    raw = _call_ollama(prompt)

    parsed = _parse_response(raw)
    if parsed is None or len(parsed) == 0:
        logger.warning("LLM returned unparseable response, retrying with JSON-only instruction")
        raw = _call_ollama(prompt + "\n\nRespond with valid JSON only.")
        parsed = _parse_response(raw)

    if parsed is None or len(parsed) == 0:
        logger.warning("LLM retry also failed — falling back to champion defaults")
        return [dict(champion_defaults)]

    return parsed


def refine_params(
    market_summary: str,
    results: List[Dict],
    champion_defaults: Dict,
    target_strategy: str | None,
    round_number: int = 2,
) -> List[Dict[str, Any]]:
    """Round 2+: refine params based on backtest results.

    Raises OllamaUnavailableError if Ollama is not reachable.
    Falls back to [champion_defaults] if JSON parsing fails after retry.
    """
    prompt = _build_refine_prompt(market_summary, results, champion_defaults, target_strategy, round_number)
    raw = _call_ollama(prompt)

    parsed = _parse_response(raw)
    if parsed is None or len(parsed) == 0:
        logger.warning("LLM refine returned unparseable response, retrying")
        raw = _call_ollama(prompt + "\n\nRespond with valid JSON only.")
        parsed = _parse_response(raw)

    if parsed is None or len(parsed) == 0:
        logger.warning("LLM refine retry failed — falling back to champion defaults")
        return [dict(champion_defaults)]

    return parsed
```

- [ ] **Step 4: Run all llm_optimizer tests**

Run: `python -m pytest tests/test_llm_optimizer.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add bot/learning/llm_optimizer.py tests/test_llm_optimizer.py
git commit -m "feat: add Ollama API calls, prompt builder, propose/refine functions"
```

---

## Chunk 2: Market Summarizer

### Task 4: Market Summarizer Module

**Files:**
- Create: `bot/learning/market_summarizer.py`
- Create: `tests/test_market_summarizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_market_summarizer.py
"""Tests for market summarizer — candle data to text conversion."""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.learning.market_summarizer import summarize


def _make_candles(n_5m=25920, base_price=60000.0):
    """Generate n_5m synthetic 5-minute candles (~90 days = 25920)."""
    candles = []
    price = base_price
    start = datetime(2025, 10, 21)
    for i in range(n_5m):
        ts = start + timedelta(minutes=5 * i)
        noise = np.random.uniform(-0.003, 0.003)
        trend = -0.00001 * i  # slight downtrend
        price *= (1 + noise + trend)
        o = price
        h = o * np.random.uniform(1.001, 1.015)
        l = o * np.random.uniform(0.985, 0.999)
        c = np.random.uniform(l, h)
        vol = np.random.uniform(500000, 2000000)
        candles.append({
            "timestamp": int(ts.timestamp() * 1000),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": round(vol, 2),
        })
    return candles


class TestSummarize:
    def test_returns_string(self):
        candles = _make_candles(n_5m=5000)
        result = summarize(candles)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_computed_stats(self):
        candles = _make_candles(n_5m=5000)
        result = summarize(candles)
        assert "Period:" in result
        assert "Trend:" in result
        assert "Volatility:" in result

    def test_contains_daily_candle_csv(self):
        candles = _make_candles(n_5m=5000)
        result = summarize(candles)
        assert "date,open,high,low,close,volume,atr_pct" in result

    def test_contains_4h_candle_csv(self):
        candles = _make_candles(n_5m=5000)
        result = summarize(candles)
        assert "date,open,high,low,close,volume,atr_pct,rsi,adx" in result

    def test_empty_candles_returns_minimal(self):
        result = summarize([])
        assert isinstance(result, str)

    def test_with_strategy_name(self):
        candles = _make_candles(n_5m=5000)
        result = summarize(candles, target_strategy="squeeze")
        assert isinstance(result, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_market_summarizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement market summarizer**

```python
# bot/learning/market_summarizer.py
"""Market summarizer — convert raw candle data into text for LLM prompts."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bot.indicators.trend import adx as compute_adx, ema as compute_ema
from bot.indicators.momentum import rsi as compute_rsi
from bot.indicators.volatility import atr as compute_atr

logger = logging.getLogger(__name__)


def _candles_to_df(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw candle dicts to DataFrame with datetime index."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    return df


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV data to a larger timeframe."""
    return df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def _compute_stats(daily: pd.DataFrame) -> str:
    """Compute summary statistics from daily candles."""
    if daily.empty:
        return "No data available."

    n_days = len(daily)
    start_date = daily.index[0].strftime("%b %d")
    end_date = daily.index[-1].strftime("%b %d")
    first_close = float(daily["close"].iloc[0])
    last_close = float(daily["close"].iloc[-1])
    trend_pct = (last_close - first_close) / first_close * 100

    # ATR%
    atr_series = compute_atr(daily["high"], daily["low"], daily["close"], period=14)
    atr_pct = (atr_series / daily["close"] * 100).dropna()
    avg_atr_pct = float(atr_pct.mean()) if len(atr_pct) > 0 else 0.0
    min_atr_pct = float(atr_pct.min()) if len(atr_pct) > 0 else 0.0
    max_atr_pct = float(atr_pct.max()) if len(atr_pct) > 0 else 0.0

    # Price stats
    high_price = float(daily["high"].max())
    low_price = float(daily["low"].min())
    pct_from_high = (last_close - high_price) / high_price * 100

    # EMA200
    ema200 = compute_ema(daily["close"], 200) if n_days >= 200 else compute_ema(daily["close"], min(n_days, 50))
    ema200_val = float(ema200.iloc[-1]) if len(ema200) > 0 else last_close
    ema200_status = "above" if last_close > ema200_val else "below"

    # Volume trend
    vol_first_half = float(daily["volume"].iloc[:n_days // 2].mean())
    vol_second_half = float(daily["volume"].iloc[n_days // 2:].mean())
    vol_trend_pct = (vol_second_half - vol_first_half) / vol_first_half * 100 if vol_first_half > 0 else 0

    # Regime distribution via ADX
    adx_series = compute_adx(daily["high"], daily["low"], daily["close"], period=14)
    if len(adx_series) > 0:
        adx_vals = adx_series.dropna()
        trending = (adx_vals > 25).sum() / len(adx_vals) * 100 if len(adx_vals) > 0 else 0
        ranging = (adx_vals < 20).sum() / len(adx_vals) * 100 if len(adx_vals) > 0 else 0
        neutral = 100 - trending - ranging
    else:
        trending = ranging = neutral = 33.3

    trend_dir = "gain" if trend_pct > 0 else "decline"
    lines = [
        f"Period: {n_days} days ({start_date} - {end_date})",
        f"Trend: {trend_pct:+.1f}% {trend_dir}, price {ema200_status} EMA200",
        f"Volatility: avg ATR% {avg_atr_pct:.1f} (range {min_atr_pct:.1f}-{max_atr_pct:.1f})",
        f"Regimes: TRENDING {trending:.0f}%, RANGING {ranging:.0f}%, NEUTRAL {neutral:.0f}%",
        f"Price: high {high_price:.0f} / low {low_price:.0f} / current {last_close:.0f} ({pct_from_high:+.0f}% from high)",
        f"Volume: {vol_trend_pct:+.0f}% trend over period",
    ]
    return "\n".join(lines)


def _daily_csv(daily: pd.DataFrame) -> str:
    """Format daily candles as CSV with ATR%."""
    if daily.empty:
        return ""
    atr_series = compute_atr(daily["high"], daily["low"], daily["close"], period=14)
    atr_pct = (atr_series / daily["close"] * 100).fillna(0)

    lines = ["date,open,high,low,close,volume,atr_pct"]
    for i, (idx, row) in enumerate(daily.iterrows()):
        lines.append(
            f"{idx.strftime('%Y-%m-%d')},"
            f"{row['open']:.0f},{row['high']:.0f},{row['low']:.0f},{row['close']:.0f},"
            f"{row['volume']:.0f},{atr_pct.iloc[i]:.1f}"
        )
    return "\n".join(lines)


def _recent_4h_csv(df_4h: pd.DataFrame, last_n_days: int = 7) -> str:
    """Format recent 4h candles with RSI and ADX."""
    if df_4h.empty:
        return ""

    cutoff = df_4h.index[-1] - pd.Timedelta(days=last_n_days)
    recent = df_4h[df_4h.index >= cutoff]
    if recent.empty:
        return ""

    rsi_series = compute_rsi(df_4h["close"], period=14)
    adx_series = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], period=14)
    atr_series = compute_atr(df_4h["high"], df_4h["low"], df_4h["close"], period=14)
    atr_pct = (atr_series / df_4h["close"] * 100).fillna(0)

    lines = ["date,open,high,low,close,volume,atr_pct,rsi,adx"]
    for idx, row in recent.iterrows():
        loc = df_4h.index.get_loc(idx)
        rsi_val = float(rsi_series.iloc[loc]) if loc < len(rsi_series) else 50.0
        adx_val = float(adx_series.iloc[loc]) if loc < len(adx_series) else 20.0
        atr_val = float(atr_pct.iloc[loc]) if loc < len(atr_pct) else 0.0
        lines.append(
            f"{idx.strftime('%Y-%m-%d %H:%M')},"
            f"{row['open']:.0f},{row['high']:.0f},{row['low']:.0f},{row['close']:.0f},"
            f"{row['volume']:.0f},{atr_val:.1f},{rsi_val:.0f},{adx_val:.0f}"
        )
    return "\n".join(lines)


def summarize(candles: List, target_strategy: str | None = None) -> str:
    """Convert raw 5m candle data into a text summary for LLM prompts.

    Returns three sections: computed stats, daily candle CSV, recent 4h CSV.
    """
    if not candles:
        return "No candle data available."

    df = _candles_to_df(candles)
    if df.empty:
        return "No candle data available."

    daily = _resample(df, "1D")
    df_4h = _resample(df, "4h")

    stats = _compute_stats(daily)
    daily_csv = _daily_csv(daily)
    recent_4h = _recent_4h_csv(df_4h)

    sections = [stats]
    if daily_csv:
        sections.append(f"DAILY CANDLES:\n{daily_csv}")
    if recent_4h:
        sections.append(f"RECENT 4H CANDLES (last 7 days):\n{recent_4h}")

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_market_summarizer.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add bot/learning/market_summarizer.py tests/test_market_summarizer.py
git commit -m "feat: add market summarizer for LLM optimizer prompts"
```

---

## Chunk 3: Walk-Forward Integration

### Task 5: Expand `_run_single_backtest` Return Value

**Files:**
- Modify: `bot/learning/walk_forward.py:21-35`
- Create: `tests/test_llm_walk_forward.py`

- [ ] **Step 1: Write failing test for expanded return**

```python
# tests/test_llm_walk_forward.py
"""Tests for LLM-integrated walk-forward optimization."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from bot.learning.walk_forward import _run_single_backtest


class TestExpandedBacktestReturn:
    def test_returns_10_element_tuple(self):
        """_run_single_backtest must return 10-element tuple with trade stats."""
        mock_result = MagicMock()
        mock_result.total_pnl = 100.0
        mock_result.sharpe_ratio = 0.8
        mock_result.profit_factor = 1.3
        mock_result.profit_per_fee = 2.0
        mock_result.winning_trades = 5
        mock_result.losing_trades = 3
        mock_result.avg_win_pct = 1.2
        mock_result.avg_loss_pct = -0.8
        mock_result.trade_log = []

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result

        with patch("bot.learning.walk_forward.BacktestEngine", return_value=mock_engine):
            result = _run_single_backtest([], {"atr_multiplier": 3.5})
            assert len(result) == 10
            params, pnl, sharpe, pf, ppf, wins, losses, avg_win, avg_loss, avg_hold = result
            assert pnl == 100.0
            assert wins == 5
            assert losses == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_walk_forward.py::TestExpandedBacktestReturn -v`
Expected: FAIL — tuple has 5 elements, not 10

- [ ] **Step 3: Expand `_run_single_backtest` in `walk_forward.py`**

Replace lines 21-35 of `bot/learning/walk_forward.py`:

```python
def _run_single_backtest(candles, params, target_strategy=None):
    """Top-level function so ProcessPoolExecutor can pickle it.

    Returns (params, pnl, sharpe, pf, ppf, winning_trades, losing_trades,
             avg_win_pct, avg_loss_pct, avg_hold_hours).
    """
    import time
    from datetime import datetime as dt

    t0 = time.perf_counter()
    from bot.backtest.engine import BacktestEngine
    engine = BacktestEngine(candles, strategy_params=params, slippage_pct=0.001, target_strategy=target_strategy)
    result = engine.run()
    elapsed = time.perf_counter() - t0

    pf = result.profit_factor if hasattr(result, 'profit_factor') else 1.0
    ppf = result.profit_per_fee if hasattr(result, 'profit_per_fee') else 0.0

    # Compute avg hold hours from trade_log
    avg_hold_hours = 0.0
    if result.trade_log:
        hold_hours = []
        for t in result.trade_log:
            try:
                entry = dt.fromisoformat(t.entry_time)
                exit_ = dt.fromisoformat(t.exit_time)
                hold_hours.append((exit_ - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
        avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    logging.getLogger(__name__).info(
        "Backtest done: %d candles, pnl=€%.2f, sharpe=%.2f, pf=%.2f, ppf=%.2f, %.1fs",
        len(candles), result.total_pnl, result.sharpe_ratio, pf, ppf, elapsed
    )
    return (params, result.total_pnl, result.sharpe_ratio, pf, ppf,
            result.winning_trades, result.losing_trades,
            result.avg_win_pct, result.avg_loss_pct,
            round(avg_hold_hours, 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_walk_forward.py::TestExpandedBacktestReturn -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/walk_forward.py tests/test_llm_walk_forward.py
git commit -m "feat: expand _run_single_backtest to return trade stats for LLM"
```

---

### Task 6: Add Scoring Function and `_run_llm_window`

**Files:**
- Modify: `bot/learning/walk_forward.py`
- Modify: `tests/test_llm_walk_forward.py`

- [ ] **Step 1: Write failing tests for scoring and LLM window**

```python
# Append to tests/test_llm_walk_forward.py
from bot.learning.walk_forward import _score_result


class TestScoreResult:
    def test_dead_zone_penalty(self):
        assert _score_result(0.0, 0.0, 0.0, 0.0) == -10.0

    def test_positive_score(self):
        score = _score_result(500.0, 1.0, 1.5, 2.0)
        assert score > 0

    def test_pnl_clamped(self):
        s1 = _score_result(5000.0, 1.0, 1.5, 2.0)
        s2 = _score_result(50000.0, 1.0, 1.5, 2.0)
        assert s1 == s2  # Both clamped at 3.0


class TestRunLlmWindow:
    def test_returns_best_params_and_result(self):
        """_run_llm_window returns (best_params, BacktestResult)."""
        from bot.learning.walk_forward import _run_llm_window

        valid_params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}

        mock_result = MagicMock()
        mock_result.total_pnl = 100.0
        mock_result.sharpe_ratio = 0.8
        mock_result.profit_factor = 1.3
        mock_result.profit_per_fee = 2.0
        mock_result.winning_trades = 5
        mock_result.losing_trades = 3
        mock_result.avg_win_pct = 1.2
        mock_result.avg_loss_pct = 0.8
        mock_result.trade_log = []

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result

        with patch("bot.learning.walk_forward.BacktestEngine", return_value=mock_engine), \
             patch("bot.learning.walk_forward.propose_params", return_value=[valid_params] * 6), \
             patch("bot.learning.walk_forward.refine_params", return_value=[valid_params] * 6), \
             patch("bot.learning.walk_forward.summarize", return_value="market summary"):
            best_params, test_result = _run_llm_window([], [], 4, "orderflow")
            assert best_params is not None
            assert test_result is not None

    def test_returns_sentinel_on_ollama_down(self):
        from bot.learning.walk_forward import _run_llm_window
        from bot.learning.llm_optimizer import OllamaUnavailableError

        with patch("bot.learning.walk_forward.propose_params", side_effect=OllamaUnavailableError("down")), \
             patch("bot.learning.walk_forward.summarize", return_value="summary"):
            best_params, test_result = _run_llm_window([], [], 4, "orderflow")
            assert best_params is None
            assert test_result is None

    def test_round_3_triggers_when_round_2_worse(self):
        """Round 3 should fire when round 2 best score < round 1 best score."""
        from bot.learning.walk_forward import _run_llm_window

        valid_params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}

        good_result = MagicMock()
        good_result.total_pnl = 200.0
        good_result.sharpe_ratio = 1.2
        good_result.profit_factor = 1.8
        good_result.profit_per_fee = 3.0
        good_result.winning_trades = 8
        good_result.losing_trades = 2
        good_result.avg_win_pct = 1.5
        good_result.avg_loss_pct = 0.6
        good_result.trade_log = []

        bad_result = MagicMock()
        bad_result.total_pnl = -50.0
        bad_result.sharpe_ratio = -0.3
        bad_result.profit_factor = 0.7
        bad_result.profit_per_fee = 0.5
        bad_result.winning_trades = 2
        bad_result.losing_trades = 5
        bad_result.avg_win_pct = 0.8
        bad_result.avg_loss_pct = 1.2
        bad_result.trade_log = []

        call_count = {"refine": 0}
        def mock_refine(*args, **kwargs):
            call_count["refine"] += 1
            return [valid_params] * 6

        # Round 1 gets good results, Round 2 gets bad results → Round 3 should trigger
        results_cycle = [good_result, bad_result, good_result]
        result_idx = {"i": 0}
        def engine_factory(*args, **kwargs):
            mock_eng = MagicMock()
            r = results_cycle[result_idx["i"] % len(results_cycle)]
            result_idx["i"] += 1
            mock_eng.run.return_value = r
            return mock_eng

        with patch("bot.learning.walk_forward.BacktestEngine", side_effect=engine_factory), \
             patch("bot.learning.walk_forward.propose_params", return_value=[valid_params] * 6), \
             patch("bot.learning.walk_forward.refine_params", side_effect=mock_refine), \
             patch("bot.learning.walk_forward.summarize", return_value="market summary"):
            _run_llm_window([], [], 1, "orderflow")
            # refine called twice: round 2 + round 3
            assert call_count["refine"] == 2


class TestFormatResultsForLlm:
    def test_formats_correctly(self):
        from bot.learning.walk_forward import _format_results_for_llm
        params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        bt = (params, 100.0, 0.8, 1.3, 2.0, 5, 3, 1.2, 0.8, 18.5)
        result = _format_results_for_llm([(params, 1.5, bt)])
        assert len(result) == 1
        assert result[0]["pnl"] == 100.0
        assert result[0]["wins"] == 5
        assert result[0]["losses"] == 3
        assert result[0]["avg_hold_hours"] == 18.5

    def test_with_round_label(self):
        from bot.learning.walk_forward import _format_results_for_llm
        params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}
        bt = (params, 100.0, 0.8, 1.3, 2.0, 5, 3, 1.2, 0.8, 18.5)
        result = _format_results_for_llm([(params, 1.5, bt)], "ROUND 1")
        assert result[0]["round"] == "ROUND 1"


class TestBacktestCandidatesCrash:
    def test_skips_crashed_candidates(self):
        from bot.learning.walk_forward import _backtest_candidates

        valid_params = {"atr_multiplier": 3.5, "rr_ratio": 2.5, "base_risk_pct": 3.0, "min_profit_multiple": 3.0, "max_hold_hours": 120, "quiet_atr_threshold": 1.0, "regime_adx_threshold": 24, "ranging_adx_threshold": 20}

        def mock_backtest(candles, params, target_strategy=None):
            if params.get("atr_multiplier") == 5.0:
                raise RuntimeError("Backtest crash")
            return (params, 100.0, 0.8, 1.3, 2.0, 5, 3, 1.2, 0.8, 18.5)

        crash_params = dict(valid_params, atr_multiplier=5.0)
        with patch("bot.learning.walk_forward._run_single_backtest", side_effect=mock_backtest):
            results = _backtest_candidates([], [valid_params, crash_params], 1, None)
            assert len(results) == 1  # crashed candidate skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_walk_forward.py::TestScoreResult tests/test_llm_walk_forward.py::TestRunLlmWindow -v`
Expected: FAIL

- [ ] **Step 3: Implement scoring function and `_run_llm_window`**

Add to `bot/learning/walk_forward.py` after the `_run_single_backtest` function, replacing `_run_optuna_window`:

```python
from bot.learning.llm_optimizer import (
    OllamaUnavailableError, propose_params, refine_params,
)
from bot.learning.market_summarizer import summarize


def _score_result(pnl: float, sharpe: float, pf: float, ppf: float) -> float:
    """Score a backtest result. Higher is better."""
    if pnl == 0.0 and sharpe == 0.0:
        return -10.0
    pnl_norm = max(min(pnl / 1000.0, 3.0), -3.0)
    return sharpe * 0.30 + pnl_norm * 0.30 + pf * 0.20 + ppf * 0.20


def _run_llm_window(train_candles, test_candles, max_workers,
                     target_strategy=None, prev_windows=None):
    """LLM-driven parameter optimization for a single train/test window.

    Returns (best_params, BacktestResult) on success.
    Returns (None, None) sentinel if Ollama is unavailable.
    """
    from concurrent.futures import ProcessPoolExecutor
    from bot.backtest.engine import BacktestEngine

    try:
        market_summary = summarize(train_candles, target_strategy)
        champion = dict(CHAMPION_DEFAULTS)

        all_results = []  # (params, score, backtest_tuple) across all rounds

        # Round 1: Propose
        candidates = propose_params(market_summary, champion, target_strategy, prev_windows)
        round_1_results = _backtest_candidates(train_candles, candidates, max_workers, target_strategy)
        all_results.extend(round_1_results)
        best_r1_score = max(r[1] for r in round_1_results) if round_1_results else -999

        # Round 2: Refine
        results_for_llm = _format_results_for_llm(round_1_results)
        candidates_r2 = refine_params(market_summary, results_for_llm, champion, target_strategy, round_number=2)
        round_2_results = _backtest_candidates(train_candles, candidates_r2, max_workers, target_strategy)
        all_results.extend(round_2_results)
        best_r2_score = max(r[1] for r in round_2_results) if round_2_results else -999

        # Round 3: Conditional — only if round 2 regressed
        if best_r2_score < best_r1_score:
            logger.info("LLM round 2 regressed (%.2f < %.2f) — running round 3", best_r2_score, best_r1_score)
            combined_results = _format_results_for_llm(round_1_results, "ROUND 1") + _format_results_for_llm(round_2_results, "ROUND 2")
            candidates_r3 = refine_params(market_summary, combined_results, champion, target_strategy, round_number=3)
            round_3_results = _backtest_candidates(train_candles, candidates_r3, max_workers, target_strategy)
            all_results.extend(round_3_results)

        # Pick overall best
        if not all_results:
            best_params = champion
        else:
            best_entry = max(all_results, key=lambda r: r[1])
            best_params = best_entry[0]

        logger.info("LLM optimization done: %d total candidates, best score=%.2f",
                     len(all_results), max(r[1] for r in all_results) if all_results else 0)

        # Test on out-of-sample data
        test_engine = BacktestEngine(test_candles, strategy_params=best_params, slippage_pct=0.001, target_strategy=target_strategy)
        test_result = test_engine.run()
        return best_params, test_result

    except OllamaUnavailableError as exc:
        logger.error("Ollama unavailable: %s", exc)
        return None, None


def _backtest_candidates(candles, candidates, max_workers, target_strategy):
    """Run backtests for a list of candidate param sets. Returns [(params, score, bt_tuple)]."""
    from concurrent.futures import ProcessPoolExecutor

    if not candidates:
        return []

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single_backtest, candles, params, target_strategy): params
            for params in candidates
        }
        for future in futures:
            try:
                bt = future.result()
                params, pnl, sharpe, pf, ppf = bt[0], bt[1], bt[2], bt[3], bt[4]
                score = _score_result(pnl, sharpe, pf, ppf)
                results.append((params, score, bt))
            except Exception as exc:
                logger.warning("Backtest candidate failed: %s", exc)

    return results


def _format_results_for_llm(results, round_label=None):
    """Format backtest results into dicts for the refine prompt."""
    formatted = []
    for params, score, bt in results:
        entry = {
            "params": params,
            "pnl": bt[1],
            "sharpe": bt[2],
            "pf": bt[3],
            "trades": bt[5] + bt[6],
            "wins": bt[5],
            "losses": bt[6],
            "avg_win_pct": bt[7],
            "avg_loss_pct": bt[8],
            "avg_hold_hours": bt[9],
        }
        if round_label:
            entry["round"] = round_label
        formatted.append(entry)
    return formatted
```

- [ ] **Step 4: Delete `_run_optuna_window` function**

Remove `_run_optuna_window` (lines 39-124 of current walk_forward.py) and the `OPTUNA_TRIALS` constant (line 18). Also remove `import optuna` from `_run_optuna_window` body.

- [ ] **Step 5: Update `_execute` and `_execute_from_candles` to call `_run_llm_window`**

In `_execute` method, replace the `_run_optuna_window` call with:
```python
prev_windows_data = [
    {"window": i + 1, "best_params": w.params, "pnl": w.pnl,
     "sharpe": w.sharpe, "profit_factor": w.profit_per_fee}
    for i, w in enumerate(wf_windows)
] or None
best_params, test_result = await loop.run_in_executor(
    None, _run_llm_window, train_candles, test_candles, max_workers, None, prev_windows_data
)
if best_params is None:
    if self._event_bus:
        await self._event_bus.publish("risk.halt", {
            "reason": "Ollama unavailable — walk-forward skipped",
            "symbol": symbol,
        })
    self._ollama_down = True
    return WFResult()
```

In `_execute_from_candles` method, replace the `_run_optuna_window` call with:
```python
prev_windows_data = [
    {"window": i + 1, "best_params": w.params, "pnl": w.pnl,
     "sharpe": w.sharpe, "profit_factor": w.profit_per_fee}
    for i, w in enumerate(wf_windows)
] or None
best_params, test_result = await loop.run_in_executor(
    None, _run_llm_window, train_candles, test_candles, max_workers, target_strategy, prev_windows_data
)
if best_params is None:
    if self._event_bus:
        await self._event_bus.publish("risk.halt", {
            "reason": "Ollama unavailable — walk-forward skipped",
            "symbol": symbol,
        })
    self._ollama_down = True
    return WFResult()
```

Note: `_execute_from_candles` already receives `target_strategy` as a parameter — pass it through. `_execute` does not use `target_strategy` (it is a legacy entry point used only by `run` and `run_multi`; the main code path is `run_multi_per_strategy` → `_execute_from_candles`).

- [ ] **Step 6: Add `_event_bus` and `_ollama_down` to `WalkForwardOptimizer.__init__`**

```python
def __init__(self, event_bus=None) -> None:
    self._latest_result: Optional[WFResult] = None
    self._results_by_symbol: Dict[str, WFResult] = {}
    self._running = False
    self._event_bus = event_bus
    self._ollama_down = False
    self._last_ollama_warning: float = 0
```

- [ ] **Step 7: Update log messages from "Optuna" to "LLM"**

Replace all `"Optuna"` strings in walk_forward.py log messages with `"LLM"`, and remove `OPTUNA_TRIALS` from log format strings.

- [ ] **Step 8: Run all tests**

Run: `python -m pytest tests/test_llm_walk_forward.py tests/test_walk_forward.py -v`
Expected: All passed

- [ ] **Step 9: Commit**

```bash
git add bot/learning/walk_forward.py tests/test_llm_walk_forward.py
git commit -m "feat: replace Optuna with LLM-driven _run_llm_window"
```

---

## Chunk 4: Integration and Cleanup

### Task 7: Wire Ollama Health Check in `main.py`

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Add Ollama health check after WalkForwardOptimizer initialization**

Find where `_walk_forward` is created in `main.py` and add:

```python
from bot.learning.llm_optimizer import check_ollama, OllamaUnavailableError
from bot.events.bus import get_bus

# Pass event bus to walk-forward optimizer
walk_forward = WalkForwardOptimizer(event_bus=get_bus())

# Ollama health check
try:
    check_ollama()
    logger.info("Ollama health check passed — LLM optimizer ready")
except OllamaUnavailableError as exc:
    logger.warning("Ollama not available — walk-forward optimization will be skipped until Ollama is running: %s", exc)
    walk_forward._ollama_down = True
```

- [ ] **Step 2: Add Ollama retry in walk-forward scheduler callback**

In `_run_walk_forward()`, add at the top:

```python
async def _run_walk_forward():
    # Retry Ollama if previously down
    if walk_forward._ollama_down:
        try:
            check_ollama()
            walk_forward._ollama_down = False
            logger.info("Ollama recovered — resuming walk-forward optimization")
        except OllamaUnavailableError:
            import time
            now = time.time()
            if now - walk_forward._last_ollama_warning > 1800:  # 30 min
                walk_forward._last_ollama_warning = now
                logger.warning("Ollama still unavailable — skipping walk-forward")
                bus = get_bus()
                await bus.publish("risk.halt", {"reason": "Ollama unavailable — walk-forward skipped"})
            return

    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    # ... rest unchanged
```

- [ ] **Step 3: Commit**

```bash
git add bot/main.py
git commit -m "feat: wire Ollama health check and retry logic in main.py"
```

---

### Task 8: Update Docker and Dependencies

**Files:**
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add `extra_hosts` and `OLLAMA_URL` to docker-compose.yml**

In the `bot` service, add:

```yaml
bot:
  build:
    context: .
    dockerfile: Dockerfile
  restart: unless-stopped
  depends_on:
    db:
      condition: service_healthy
  env_file: .env
  environment:
    DATABASE_URL: postgresql+asyncpg://hellacash:hellacash@db:5432/hellacash
    OLLAMA_URL: http://host.docker.internal:11434
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ports:
    - "8000:8000"
  volumes:
    - ./models:/app/models
```

- [ ] **Step 2: Remove `optuna` and add `requests` to `requirements.txt`**

Delete the line `optuna>=3.6.0` from `requirements.txt`. Add `requests>=2.31.0` (needed for sync Ollama HTTP calls in `llm_optimizer.py`).

- [ ] **Step 3: Verify no remaining optuna imports**

Run: `grep -r "optuna" bot/ --include="*.py"`
Expected: No output (all optuna references removed)

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml requirements.txt
git commit -m "feat: add Docker Ollama networking, remove optuna dependency"
```

---

### Task 9: Run Full Test Suite and Verify

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass. Any tests that imported `optuna` directly will need updating.

- [ ] **Step 2: Fix any broken tests**

If `tests/test_wf_integration.py` or others reference `_run_optuna_window`, update them to use `_run_llm_window` or mock the new functions.

- [ ] **Step 3: Run linting check**

Run: `python -m py_compile bot/learning/llm_optimizer.py && python -m py_compile bot/learning/market_summarizer.py && python -m py_compile bot/learning/walk_forward.py`
Expected: No syntax errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: update remaining tests for LLM optimizer migration"
```
