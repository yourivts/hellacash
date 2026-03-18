# ML Signal Generator Improvements — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ML signal generator accuracy by adding 25 external data features, replacing LSTM with Transformer encoder, adding class-weighted training with threshold tuning and Optuna HPO, and implementing API failsafe mechanisms.

**Architecture:** External data pipeline fetches from 12+ free APIs and caches to parquet. Transformer encoder replaces LSTM with same interface (96×7 → 16-dim embedding). Feature count expands 65→90. Training uses Optuna HPO for both Transformer and XGBoost, with class weighting and threshold search. Failsafe auto-disables ML signals when >30% of data sources are stale.

**Tech Stack:** Python, PyTorch (Transformer), XGBoost, Optuna, yfinance, pytrends, pandas/numpy, aiohttp (API fetching), pyarrow (parquet caching)

**Spec:** `docs/superpowers/specs/2026-03-18-ml-signal-improvements-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | MODIFY | Add optuna, pytrends, yfinance, pyarrow |
| `bot/learning/transformer_embedder.py` | CREATE | Transformer encoder with same interface as LSTMEmbedder |
| `bot/data/external_features.py` | CREATE | External data fetching, caching, staleness tracking |
| `bot/learning/ml_features.py` | MODIFY | N_TABULAR 65→90, accept external_data dict |
| `bot/learning/ml_signal_generator.py` | MODIFY | Load Transformer, external data integration, failsafe |
| `bot/trading_loop.py` | MODIFY | Fetch external data, pass to ML signal generator |
| `bot/main.py` | MODIFY | Initialize external data provider |
| `scripts/train_ml_signals.py` | MODIFY | Full pipeline: Transformer, external data, HPO, class weights |
| `tests/test_transformer_embedder.py` | CREATE | Transformer model tests |
| `tests/test_external_features.py` | CREATE | External data pipeline tests |
| `tests/test_ml_features.py` | MODIFY | Update for N_TABULAR=90 |
| `tests/test_ml_signal_generator.py` | MODIFY | Update for Transformer + external data |
| `tests/test_train_ml_signals.py` | MODIFY | Update for new training pipeline |

---

## Chunk 1: Foundation

### Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt:35-45`

- [ ] **Step 1: Add new dependencies**

Add to the `# ── Machine learning ──` section in `requirements.txt`:

```
optuna>=3.0
pytrends>=4.9
yfinance>=0.2
pyarrow>=14.0
```

- [ ] **Step 2: Install and verify**

Run: `pip install optuna pytrends yfinance pyarrow`
Expected: All packages install without error

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add optuna, pytrends, yfinance, pyarrow dependencies"
```

---

### Task 2: Transformer Embedder Module

**Files:**
- Create: `bot/learning/transformer_embedder.py`
- Create: `tests/test_transformer_embedder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transformer_embedder.py`:

```python
"""Tests for Transformer embedder module."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TestTransformerEmbedder:
    """Tests for TransformerEmbedder model."""

    def test_embed_output_shape(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        x = torch.randn(4, 96, 7)
        out = model.embed(x)
        assert out.shape == (4, 16), f"Expected (4, 16), got {out.shape}"

    def test_predict_next_output_shape(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        x = torch.randn(4, 96, 7)
        out = model.predict_next(x)
        assert out.shape == (4, 60), f"Expected (4, 60), got {out.shape}"

    def test_embed_numpy_single_sequence(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        seq = np.random.randn(96, 7).astype(np.float32)
        out = model.embed_numpy(seq)
        assert out.shape == (16,), f"Expected (16,), got {out.shape}"
        assert out.dtype == np.float32

    def test_embed_values_finite(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        x = torch.randn(8, 96, 7)
        out = model.embed(x)
        assert torch.all(torch.isfinite(out)), "Non-finite values in embedding"

    def test_save_and_load_roundtrip(self, tmp_path):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        x = torch.randn(2, 96, 7)
        emb_before = model.embed(x).detach()

        path = str(tmp_path / "transformer.pt")
        model.save(path)

        model2 = TransformerEmbedder()
        model2.load(path)
        emb_after = model2.embed(x).detach()

        assert torch.allclose(emb_before, emb_after, atol=1e-5), "Embeddings differ after save/load"

    def test_different_inputs_different_embeddings(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        x1 = torch.randn(1, 96, 7)
        x2 = torch.randn(1, 96, 7) + 5.0
        e1 = model.embed(x1).detach()
        e2 = model.embed(x2).detach()
        assert not torch.allclose(e1, e2, atol=1e-3), "Different inputs should produce different embeddings"

    def test_embed_deterministic_in_eval_mode(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder()
        model.eval()
        x = torch.randn(2, 96, 7)
        e1 = model.embed(x).detach()
        e2 = model.embed(x).detach()
        assert torch.allclose(e1, e2), "eval() mode should be deterministic"


class TestTransformerEmbedderCustomConfig:
    """Tests for non-default Transformer configurations (for Optuna HPO)."""

    def test_custom_d_model_and_heads(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder(d_model=128, nhead=8, num_layers=2, dropout=0.05)
        x = torch.randn(2, 96, 7)
        out = model.embed(x)
        assert out.shape == (2, 16)

    def test_custom_small_config(self):
        from bot.learning.transformer_embedder import TransformerEmbedder
        model = TransformerEmbedder(d_model=32, nhead=2, num_layers=2, dropout=0.1)
        x = torch.randn(2, 96, 7)
        out = model.embed(x)
        assert out.shape == (2, 16)


class TestTrainTransformer:
    """Tests for Transformer training function."""

    def test_train_returns_loss_history(self):
        from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
        model = TransformerEmbedder(d_model=32, nhead=2, num_layers=1)
        sequences = np.random.randn(64, 96, 7).astype(np.float32)
        targets = np.random.randn(64, 60).astype(np.float32)
        losses = train_transformer(model, sequences, targets, epochs=2, batch_size=32, lr=1e-3)
        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)

    def test_train_loss_decreases(self):
        from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
        import torch
        torch.manual_seed(42)
        np.random.seed(42)
        model = TransformerEmbedder(d_model=32, nhead=2, num_layers=1)
        sequences = np.random.randn(128, 96, 7).astype(np.float32)
        targets = np.random.randn(128, 60).astype(np.float32)
        losses = train_transformer(model, sequences, targets, epochs=10, batch_size=64, lr=1e-3)
        assert losses[-1] < losses[0], f"Loss should decrease: first={losses[0]}, last={losses[-1]}"

    def test_train_with_validation(self):
        from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
        model = TransformerEmbedder(d_model=32, nhead=2, num_layers=1)
        sequences = np.random.randn(64, 96, 7).astype(np.float32)
        targets = np.random.randn(64, 60).astype(np.float32)
        val_seqs = np.random.randn(16, 96, 7).astype(np.float32)
        val_targets = np.random.randn(16, 60).astype(np.float32)
        losses = train_transformer(
            model, sequences, targets,
            val_seqs=val_seqs, val_targets=val_targets,
            epochs=2, batch_size=32, lr=1e-3,
        )
        assert len(losses) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transformer_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot.learning.transformer_embedder'`

- [ ] **Step 3: Implement the Transformer Embedder**

Create `bot/learning/transformer_embedder.py`:

```python
"""Transformer embedder: encode price sequences into fixed-dim embeddings.

Architecture:
    Input: (batch, 96, 7)
    → Linear(7 → d_model) projection
    → + SinusoidalPositionalEncoding(96, d_model)
    → TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
    → MeanPooling(dim=1) → (batch, d_model)       [h_pooled]
    ┌─→ Linear(d_model → 16) + ReLU → embedding   [embed branch]
    └─→ Linear(d_model → 60)                       [predict branch, training only]

Self-supervised training objective:
    Predict next 12 bars' normalized OHLCV (5 channels × 12 = 60 outputs).
    Prediction head discarded after training.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SEQ_LEN = 96
INPUT_DIM = 7
EMBED_DIM = 16
PREDICT_BARS = 12
PREDICT_CHANNELS = 5  # OHLCV
PREDICT_DIM = PREDICT_BARS * PREDICT_CHANNELS  # 60

# Default architecture params (can be overridden for Optuna HPO)
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 4
DEFAULT_NUM_LAYERS = 4
DEFAULT_DIM_FF = 128
DEFAULT_DROPOUT = 0.1


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learned params)."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEmbedder(nn.Module):
    """Transformer sequence encoder with self-supervised prediction head."""

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dim_feedforward: int | None = None,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        self.d_model = d_model

        # Input projection: 7 → d_model
        self.input_proj = nn.Linear(INPUT_DIM, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(SEQ_LEN, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Embedding projection: d_model → 16
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, EMBED_DIM),
            nn.ReLU(),
        )

        # Self-supervised prediction head (discarded after training)
        self.predict_head = nn.Linear(d_model, PREDICT_DIM)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run Transformer, return mean-pooled hidden state. Shape: (batch, d_model)."""
        # x: (batch, seq_len, 7)
        h = self.input_proj(x)           # (batch, seq_len, d_model)
        h = self.pos_enc(h)              # add positional encoding
        h = self.transformer(h)          # (batch, seq_len, d_model)
        h_pooled = h.mean(dim=1)         # (batch, d_model)
        return h_pooled

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get 16-dim embedding. Input: (batch, 96, 7) → Output: (batch, 16)."""
        h = self._encode(x)
        return self.embed_proj(h)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next 12 bars OHLCV. Input: (batch, 96, 7) → Output: (batch, 60)."""
        h = self._encode(x)
        return self.predict_head(h)

    def embed_numpy(self, seq: np.ndarray) -> np.ndarray:
        """Convenience: single numpy sequence (96, 7) → embedding (16,)."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(seq).unsqueeze(0).float().to(device)
        with torch.no_grad():
            emb = self.embed(x)
        return emb.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights (CPU state for portability, then restore device)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        device = next(self.parameters()).device
        torch.save(self.cpu().state_dict(), path)
        self.to(device)  # restore to original device
        logger.info("Transformer embedder saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("Transformer embedder loaded from %s", path)


def train_transformer(
    model: TransformerEmbedder,
    sequences: np.ndarray,
    targets: np.ndarray,
    val_seqs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[float]:
    """Train the Transformer via self-supervised next-bar prediction.

    Args:
        sequences: (N, 96, 7) input sequences
        targets: (N, 60) normalized OHLCV of next 12 bars
        val_seqs/val_targets: optional validation set
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate

    Returns:
        List of per-epoch average training loss values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    logger.info("Transformer training on %s", device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X = torch.from_numpy(sequences).float().to(device)
    Y = torch.from_numpy(targets).float().to(device)
    n = len(X)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = X[idx]
            y_batch = Y[idx]

            optimizer.zero_grad()
            pred = model.predict_next(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(batches, 1)
        loss_history.append(avg_loss)

        # Validation
        if val_seqs is not None and val_targets is not None:
            model.eval()
            with torch.no_grad():
                vX = torch.from_numpy(val_seqs).float().to(device)
                vY = torch.from_numpy(val_targets).float().to(device)
                val_pred = model.predict_next(vX)
                val_loss = criterion(val_pred, vY).item()
            logger.info(
                "Transformer epoch %d/%d — train_loss=%.6f, val_loss=%.6f, lr=%.2e",
                epoch + 1, epochs, avg_loss, val_loss, scheduler.get_last_lr()[0],
            )
        else:
            logger.info("Transformer epoch %d/%d — train_loss=%.6f", epoch + 1, epochs, avg_loss)

    return loss_history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transformer_embedder.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/transformer_embedder.py tests/test_transformer_embedder.py
git commit -m "feat: add Transformer embedder with same interface as LSTM"
```

---

### Task 3: External Data Pipeline — Fetcher Classes

**Files:**
- Create: `bot/data/external_features.py`
- Create: `tests/test_external_features.py`

This task creates the external data provider module with individual fetcher functions for each API source, parquet caching, and data alignment. The module is large (~500 lines) but has a single responsibility: fetch, cache, and serve external market data.

- [ ] **Step 1: Write tests for the external data pipeline**

Create `tests/test_external_features.py`:

```python
"""Tests for external data pipeline — fetching, caching, staleness."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _make_5m_index(days: int = 30) -> pd.DatetimeIndex:
    """Create a 5-minute DatetimeIndex for testing."""
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=days)
    return pd.date_range(start, end, freq="5min", tz=timezone.utc)


class TestExternalDataProvider:
    """Tests for the ExternalDataProvider class.

    All tests mock fetch_all_training() to avoid real HTTP requests.
    This ensures tests are fast, deterministic, and work offline/in CI.
    """

    def _make_mock_training_data(self, idx):
        """Create a synthetic training data dict (same keys as fetch_all_training)."""
        n = len(idx)
        return {
            "fear_greed": np.random.rand(n) * 0.5 + 0.25,
            "fear_greed_mom": np.random.randn(n) * 0.1,
            "gtrends_bitcoin": np.zeros(n),
            "gtrends_crypto": np.zeros(n),
            "dxy_return": np.random.randn(n) * 0.1,
            "sp500_return": np.random.randn(n) * 0.1,
            "gold_return": np.random.randn(n) * 0.1,
            "vix": np.random.rand(n) * 0.5,
            "treasury_10y": np.random.rand(n) * 0.5,
            "yield_spread": np.random.randn(n) * 0.2,
            "nvt": np.random.rand(n) * 0.5,
            "mvrv": np.random.rand(n) * 0.5,
            "sopr": np.random.randn(n) * 0.2,
            "puell": np.random.rand(n) * 0.5,
            "hashrate": np.random.randn(n) * 0.1,
            "eth_active_addr": np.random.randn(n) * 0.1,
            "stable_supply_change": np.random.randn(n) * 0.05,
            "tvl_change": np.random.randn(n) * 0.05,
            "oi_change_7d": np.zeros(n),
            "liq_ratio": np.zeros(n),
            "taker_buy_ratio": np.random.rand(n) * 0.3 + 0.35,
            "dvol": np.random.rand(n) * 0.5,
            "funding_24h_avg": np.random.randn(n) * 0.05,
            "btc_dom_change": np.zeros(n),
            "stable_btc_ratio": np.zeros(n),
            "funding_rate_current": np.random.randn(n) * 0.05,
            "funding_7d_avg": np.random.randn(n) * 0.03,
            "oi_change_24h": np.zeros(n),
            "long_liq_24h": np.zeros(n),
            "short_liq_24h": np.zeros(n),
            "exchange_netflow": np.random.randn(n) * 0.1,
            "active_addr_change_7d": np.random.randn(n) * 0.1,
        }

    def test_build_training_features_returns_correct_shape(self):
        """Training features should have 32 columns (7 for 55-61 + 25 for 65-89)."""
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(30)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(idx), 32), f"Expected (N, 32), got {result.shape}"

    def test_build_training_features_dtype_float32(self):
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert result.dtype == np.float32

    def test_build_training_features_all_finite(self):
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert np.all(np.isfinite(result)), "Non-finite values in training features"

    def test_build_training_features_empty_returns_zeros(self):
        """When fetch_all_training returns empty dict, all features should be zero."""
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        with patch.object(provider, "fetch_all_training", return_value={}):
            result = provider.build_training_features(idx)
        assert np.all(result == 0.0)

    def test_build_live_features_returns_correct_shape(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert isinstance(result, np.ndarray)
        assert result.shape == (25,), "Live features cover indices 65-89 (25 features)"
        assert result.dtype == np.float32

    def test_build_live_features_all_finite(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert np.all(np.isfinite(result))


class TestStalenessTracking:
    """Tests for data source staleness tracking."""

    def test_initial_staleness_all_fresh(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        # Before any data is fetched, staleness check should indicate fresh
        # (no data = zero-filled = acceptable)
        assert not provider.is_critically_stale()

    def test_mark_source_stale(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        # Manually set all sources to stale timestamps
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        for source in provider._last_fetched:
            provider._last_fetched[source] = stale_time
        # >30% stale should trigger critical
        assert provider.is_critically_stale()

    def test_staleness_recovers(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        # Set all stale
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        for source in provider._last_fetched:
            provider._last_fetched[source] = stale_time
        assert provider.is_critically_stale()
        # Update all to fresh
        fresh_time = datetime.now(timezone.utc)
        for source in provider._last_fetched:
            provider._last_fetched[source] = fresh_time
        assert not provider.is_critically_stale()

    def test_partial_staleness_below_threshold(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        fresh_time = datetime.now(timezone.utc)
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        sources = list(provider._last_fetched.keys())
        # Set all fresh first
        for s in sources:
            provider._last_fetched[s] = fresh_time
        # Make just 1 source stale (< 30%)
        if sources:
            provider._last_fetched[sources[0]] = stale_time
        assert not provider.is_critically_stale()


class TestDataAlignment:
    """Tests for data alignment to 5m index."""

    def test_align_daily_to_5m_forward_fills(self):
        from bot.data.external_features import align_to_5m
        idx_5m = _make_5m_index(3)
        daily_idx = pd.date_range("2024-05-29", "2024-06-01", freq="D", tz=timezone.utc)
        daily_data = pd.Series([10.0, 20.0, 30.0, 40.0], index=daily_idx)
        aligned = align_to_5m(daily_data, idx_5m)
        assert len(aligned) == len(idx_5m)
        assert np.all(np.isfinite(aligned))

    def test_align_fills_zeros_before_inception(self):
        from bot.data.external_features import align_to_5m
        idx_5m = _make_5m_index(30)
        # Data only covers last 5 days
        short_idx = pd.date_range("2024-05-27", "2024-06-01", freq="D", tz=timezone.utc)
        short_data = pd.Series(range(len(short_idx)), index=short_idx, dtype=float)
        aligned = align_to_5m(short_data, idx_5m)
        assert len(aligned) == len(idx_5m)
        # Early values should be 0 (pre-inception)
        assert aligned[0] == 0.0


class TestParquetCache:
    """Tests for parquet file caching."""

    def test_save_and_load_cache(self, tmp_path):
        from bot.data.external_features import save_cache, load_cache
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
        cache_path = tmp_path / "test_cache.parquet"
        save_cache(df, str(cache_path))
        loaded = load_cache(str(cache_path))
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ["value"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        from bot.data.external_features import load_cache
        result = load_cache(str(tmp_path / "nonexistent.parquet"))
        assert result is None


class TestBuildLiveFeatures:
    """Tests for live feature building from cache."""

    def test_live_features_reflect_cache(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        provider._live_cache["fear_greed"] = 0.75
        provider._live_cache["vix"] = 0.3
        result = provider.build_live_features()
        assert result[0] == pytest.approx(0.75)  # fear_greed is col 0
        assert result[7] == pytest.approx(0.3)   # vix is col 7

    def test_live_features_default_zero_for_missing(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert np.all(result == 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_external_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the external data provider**

Create `bot/data/external_features.py`:

```python
"""External data pipeline: fetch, cache, and serve market data from free APIs.

Training mode: bulk-fetch history → cache to parquet → align to 5m index
Live mode: periodic refresh → in-memory cache → staleness tracking

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
import os
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
            "dxy": "DX-Y.NYB",
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
    """Fetch Google Trends data. Optional — frequently rate-limited."""
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


def _fetch_bgeometrics(metric: str) -> pd.DataFrame:
    """Fetch on-chain metrics from BGeometrics Charts API.

    Falls back to Blockchain.com Charts API for hashrate if BGeometrics fails.
    """
    import requests
    try:
        url = f"https://charts.bgeometrics.com/api/{metric}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Empty response")
        rows = []
        for d in data:
            ts = pd.to_datetime(d.get("date") or d.get("t"))
            val = float(d.get("value") or d.get("v") or 0)
            rows.append({"date": ts, "value": val})
        df = pd.DataFrame(rows).set_index("date").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning("BGeometrics %s fetch failed: %s", metric, e)
        # Fallback for hashrate: try Blockchain.com Charts API
        if metric == "hashrate":
            return _fetch_blockchain_com_hashrate()
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


def _fetch_coinmetrics(asset: str, metric: str) -> pd.DataFrame:
    """Fetch metrics from CoinMetrics Community API."""
    import requests
    try:
        url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
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
            ts = datetime.fromtimestamp(d["date"], tz=timezone.utc)
            val = float(d.get("totalCirculatingUSD", {}).get("peggedUSD", 0))
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


def _fetch_taker_buy_ratio(symbol: str = "BTCUSDT", interval: str = "1h",
                           limit: int = 1500) -> pd.DataFrame:
    """Fetch taker buy/sell volume ratio from Binance Futures klines.

    Uses 1h granularity (practical compromise — 5m over 5y = too many requests).
    Forward-filled to 5m resolution via align_to_5m().
    """
    import requests
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for d in data:
            ts = datetime.fromtimestamp(d[0] / 1000, tz=timezone.utc)
            total_vol = float(d[5])
            taker_buy_vol = float(d[9])
            ratio = taker_buy_vol / total_vol if total_vol > 0 else 0.5
            rows.append({"date": ts, "value": ratio})
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        logger.warning("Taker buy ratio fetch failed: %s", e)
        return pd.DataFrame()


class ExternalDataProvider:
    """Fetch, cache, and serve external market data for ML features.

    Training mode: call fetch_all_training() to bulk-fetch history.
    Live mode: call refresh() periodically, then build_live_features().
    """

    def __init__(self, cache_dir: str | None = "data/external_cache") -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._live_cache: dict[str, float] = {}  # source → latest value
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
        # (markets close ~21:00 UTC, data available next day)
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

        # On-chain (BGeometrics)
        for metric, key, scale_fn in [
            ("nvt", "nvt", lambda s: np.clip(np.log1p(s) / np.log1p(200), 0, 1)),
            ("mvrv", "mvrv", lambda s: np.clip(s / 5.0, 0, 1)),
            ("sopr", "sopr", lambda s: np.clip((s - 1.0) * 5.0, -1, 1)),
            ("puell-multiple", "puell", lambda s: np.clip(s / 4.0, 0, 1)),
            ("hashrate", "hashrate", None),
        ]:
            df = self._cached_fetch(f"bgeometrics_{metric}", lambda m=metric: _fetch_bgeometrics(m))
            if not df.empty:
                vals = df["value"]
                if key == "hashrate":
                    # 30d change
                    change = vals.pct_change(30).clip(-1, 1)
                    result[key] = align_to_5m(change, idx_5m)
                elif scale_fn is not None:
                    aligned_raw = align_to_5m(vals, idx_5m)
                    result[key] = scale_fn(aligned_raw)
            else:
                result[key] = np.zeros(len(idx_5m))

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
            result["funding_7d_avg"] = align_to_5m(funding["rate"].rolling(21).mean().clip(-0.01, 0.01) * 100, idx_5m)  # 21 × 8h = ~7d
        else:
            result["funding_24h_avg"] = np.zeros(len(idx_5m))
            result["funding_rate_current"] = np.zeros(len(idx_5m))
            result["funding_7d_avg"] = np.zeros(len(idx_5m))

        # Index 57: OI change 24h (Coinalyze — optional, needs free signup API key)
        # Index 58-59: Long/short liquidations (Binance CSV — manual download)
        # These are zero-filled when API key or data not available.
        # The model learns to ignore zero-valued features via feature importance.
        result["oi_change_24h"] = np.zeros(len(idx_5m))
        result["long_liq_24h"] = np.zeros(len(idx_5m))
        result["short_liq_24h"] = np.zeros(len(idx_5m))
        logger.info("OI/liquidation features zero-filled (optional data sources)")

        # Index 60: Exchange netflow (BGeometrics exchange-flows)
        exflow = self._cached_fetch("bgeometrics_exchange-flows", lambda: _fetch_bgeometrics("exchange-flows"))
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

        # Index 83: OI change 7d (zero-filled without Coinalyze API key)
        result["oi_change_7d"] = np.zeros(len(idx_5m))

        # Index 84: Liquidation ratio (zero-filled without Binance CSV data)
        result["liq_ratio"] = np.zeros(len(idx_5m))

        # Taker buy ratio
        taker = self._cached_fetch("taker_buy_ratio", _fetch_taker_buy_ratio)
        if not taker.empty:
            result["taker_buy_ratio"] = align_to_5m(taker["value"], idx_5m)
        else:
            result["taker_buy_ratio"] = np.full(len(idx_5m), 0.5)

        # Deribit DVOL
        dvol = self._cached_fetch("deribit_dvol", _fetch_deribit_dvol)
        if not dvol.empty:
            result["dvol"] = align_to_5m((dvol["value"] / 200.0).clip(0, 1), idx_5m)
        else:
            result["dvol"] = np.zeros(len(idx_5m))

        # BTC dominance change (CoinGecko — limited history for training)
        # CoinGecko /global only returns current value, not historical.
        # For training, derive from yfinance BTC-USD market cap.
        # Zero-filled for early periods where data unavailable.
        result["btc_dom_change"] = np.zeros(len(idx_5m))

        # Stablecoin / BTC market cap ratio
        # Use stablecoin supply (already fetched) / BTC close * ~19.8M supply
        if not stable_df.empty and "dxy" in macro:
            try:
                import yfinance as yf
                btc_hist = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                                       end=end.strftime("%Y-%m-%d"), progress=False)
                if not btc_hist.empty:
                    if isinstance(btc_hist.columns, pd.MultiIndex):
                        btc_hist.columns = btc_hist.columns.get_level_values(0)
                    btc_mcap = btc_hist["Close"] * 19_800_000  # approx circulating supply
                    stable_aligned = stable_df["value"].reindex(btc_mcap.index, method="ffill").fillna(0)
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

        # Columns 0-6 → feature indices 55-61
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

        # Columns 7-31 → feature indices 65-89
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
            fng = _fetch_fear_greed(limit=8)  # 7 days + 1 to compute 7d momentum
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

        # Macro data (yfinance — daily, shift by 1 day for look-ahead bias)
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

        # On-chain (BGeometrics)
        try:
            for metric, key, transform in [
                ("nvt", "nvt", lambda v: np.clip(np.log1p(v) / np.log1p(200), 0, 1)),
                ("mvrv", "mvrv", lambda v: np.clip(v / 5.0, 0, 1)),
                ("sopr", "sopr", lambda v: np.clip((v - 1.0) * 5.0, -1, 1)),
                ("puell-multiple", "puell", lambda v: np.clip(v / 4.0, 0, 1)),
            ]:
                df = _fetch_bgeometrics(metric)
                if not df.empty:
                    self._live_cache[key] = float(transform(df["value"].iloc[-1]))
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

        # Taker buy ratio
        try:
            taker = _fetch_taker_buy_ratio(limit=10)
            if not taker.empty:
                self._live_cache["taker_buy_ratio"] = float(taker["value"].iloc[-1])
                self._last_fetched["oi_liquidations"] = now
        except Exception as e:
            logger.warning("Live refresh taker_buy failed: %s", e)

        # Yield spread (from macro data already fetched above)
        # treasury_10y and yield_spread are populated in the macro block above

        # Hashrate (BGeometrics)
        try:
            hr = _fetch_bgeometrics("hashrate")
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

        # BTC dominance change (CoinGecko — live only)
        try:
            btc_dom = _fetch_coingecko_btc_dominance()
            if not btc_dom.empty:
                # Store current value; change computed as delta from last cached value
                current = float(btc_dom["value"].iloc[-1]) / 100.0
                prev = self._live_cache.get("_btc_dom_prev", current)
                self._live_cache["btc_dom_change"] = np.clip((current - prev) * 10, -1, 1)
                self._live_cache["_btc_dom_prev"] = current
        except Exception as e:
            logger.warning("Live refresh btc_dom failed: %s", e)

        # Google Trends — skipped in live mode (weekly data, rate-limited)
        # gtrends_bitcoin and gtrends_crypto remain at 0.0 (acceptable per spec)

        # stable_btc_ratio — requires BTC price + stablecoin supply; approximate from cached values
        # This is a best-effort live approximation
        try:
            if "stable_supply_change" in self._live_cache:
                # Use a simplified proxy: stable_supply_change as directional signal
                self._live_cache["stable_btc_ratio"] = 0.0  # Zero-fill in live (training provides historical)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_external_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/data/external_features.py tests/test_external_features.py
git commit -m "feat: add external data pipeline with 12+ API sources and caching"
```

---

## Chunk 2: Feature Expansion and Signal Generator

### Task 4: Expand Feature Count to 90

**Files:**
- Modify: `bot/learning/ml_features.py:37` (N_TABULAR), `:99-421` (extract_tabular_features), `:685-831` (_batch_extract_tabular)
- Modify: `tests/test_ml_features.py`

- [ ] **Step 1: Update test expectations for N_TABULAR=90**

Edit `tests/test_ml_features.py` — update all assertions from 65 to 90:

```python
# In TestExtractTabularFeatures:
def test_output_shape_is_90(self):
    from bot.learning.ml_features import extract_tabular_features, N_TABULAR
    assert N_TABULAR == 90
    df = _make_5m_candles(2000)
    result = extract_tabular_features(df, "ETHUSDT", None, external_data={})
    assert result.shape == (90,), f"Expected (90,), got {result.shape}"

# Update ALL calls to extract_tabular_features to use new signature:
# Old: extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
# New: extract_tabular_features(df, "ETHUSDT", None, external_data={})
# (Apply to ALL test methods in TestExtractTabularFeatures)

# In TestExtractAllFeatures:
def test_tabular_shape_n_90(self):
    from bot.learning.ml_features import extract_all_features
    df = _make_5m_candles(2000)
    tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
    assert len(tabular) > 0, "Should produce at least one sample"
    assert tabular.shape[1] == 90, f"Expected 90 tabular features, got {tabular.shape[1]}"
```

**Bulk find-and-replace across ALL test methods in the file:**

1. Replace every call matching `extract_tabular_features(df, sym, btc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)` with `extract_tabular_features(df, sym, btc, external_data={})`.
2. Replace every call matching `extract_tabular_features(df, sym, btc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N, M)` (where N and M are regime params) with `extract_tabular_features(df, sym, btc, external_data={}, regime_id=N, regime_hours=M)`.
3. Change all `65` shape assertions to `90` (e.g., `assert result.shape == (65,)` → `assert result.shape == (90,)`, `assert N_TABULAR == 65` → `assert N_TABULAR == 90`, etc.).
4. This applies to ALL 9 test methods in `TestExtractTabularFeatures` and all methods in `TestExtractAllFeatures`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_features.py -v`
Expected: FAIL — shape mismatch (still 65) and signature mismatch

- [ ] **Step 3: Update ml_features.py**

**3a.** Change `N_TABULAR` constant (line 37):

```python
N_TABULAR: int = 90
```

**3b.** Change `extract_tabular_features()` signature (line 99). Replace the 7 individual float params with a single `external_data` dict, keeping `regime_id` and `regime_hours` as separate parameters:

```python
def extract_tabular_features(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    external_data: dict | None = None,
    regime_id: int = 0,
    regime_hours: float = 0.0,
) -> np.ndarray:
```

**3c.** Inside `extract_tabular_features()`, replace the block that sets features 55-61 from individual params. Instead, read from the `external_data` dict:

```python
    # ---- 55-61: External data (live) ----
    ext = external_data or {}
    features[55] = np.clip(ext.get("funding_rate", 0.0), -1.0, 1.0)
    features[56] = np.clip(ext.get("funding_7d_avg", 0.0), -1.0, 1.0)
    features[57] = np.clip(ext.get("oi_change_24h", 0.0), -1.0, 1.0)
    features[58] = np.clip(ext.get("long_liq_24h", 0.0), -1.0, 1.0)
    features[59] = np.clip(ext.get("short_liq_24h", 0.0), -1.0, 1.0)
    features[60] = np.clip(ext.get("exchange_netflow", 0.0), -1.0, 1.0)
    features[61] = np.clip(ext.get("active_addr_change", 0.0), -1.0, 1.0)
```

**3d.** Add new feature slots 65-89 at the end:

```python
    # ---- 65-89: Extended external features ----
    features[65] = np.clip(ext.get("fear_greed", 0.0), 0.0, 1.0)
    features[66] = np.clip(ext.get("fear_greed_mom", 0.0), -1.0, 1.0)
    features[67] = np.clip(ext.get("gtrends_bitcoin", 0.0), 0.0, 1.0)
    features[68] = np.clip(ext.get("gtrends_crypto", 0.0), 0.0, 1.0)
    features[69] = np.clip(ext.get("dxy_return", 0.0), -1.0, 1.0)
    features[70] = np.clip(ext.get("sp500_return", 0.0), -1.0, 1.0)
    features[71] = np.clip(ext.get("gold_return", 0.0), -1.0, 1.0)
    features[72] = np.clip(ext.get("vix", 0.0), 0.0, 1.0)
    features[73] = np.clip(ext.get("treasury_10y", 0.0), 0.0, 1.0)
    features[74] = np.clip(ext.get("yield_spread", 0.0), -1.0, 1.0)
    features[75] = np.clip(ext.get("nvt", 0.0), 0.0, 1.0)
    features[76] = np.clip(ext.get("mvrv", 0.0), 0.0, 1.0)
    features[77] = np.clip(ext.get("sopr", 0.0), -1.0, 1.0)
    features[78] = np.clip(ext.get("puell", 0.0), 0.0, 1.0)
    features[79] = np.clip(ext.get("hashrate", 0.0), -1.0, 1.0)
    features[80] = np.clip(ext.get("eth_active_addr", 0.0), -1.0, 1.0)
    features[81] = np.clip(ext.get("stable_supply_change", 0.0), -1.0, 1.0)
    features[82] = np.clip(ext.get("tvl_change", 0.0), -1.0, 1.0)
    features[83] = np.clip(ext.get("oi_change_7d", 0.0), -1.0, 1.0)
    features[84] = np.clip(ext.get("liq_ratio", 0.0), 0.0, 1.0)
    features[85] = np.clip(ext.get("taker_buy_ratio", 0.0), 0.0, 1.0)
    features[86] = np.clip(ext.get("dvol", 0.0), 0.0, 1.0)
    features[87] = np.clip(ext.get("funding_24h_avg", 0.0), -1.0, 1.0)
    features[88] = np.clip(ext.get("btc_dom_change", 0.0), -1.0, 1.0)
    features[89] = np.clip(ext.get("stable_btc_ratio", 0.0), 0.0, 1.0)
```

**3e.** Update `_batch_extract_tabular()` to accept and merge external training data:

Change signature:

```python
def _batch_extract_tabular(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    sample_indices: np.ndarray,
    external_features: np.ndarray | None = None,
) -> np.ndarray:
```

Change the features array size from 65 to 90 (it already uses `N_TABULAR`), and add after the cross-asset block:

```python
    # ---- 55-61: External data (populated from external_features during training) ----
    # ---- 65-89: Extended external features ----
    if external_features is not None:
        # external_features shape: (len(df_5m), 32) — 7 cols for indices 55-61 + 25 cols for 65-89
        ext_at_samples = external_features[idx]  # (ns, 32)
        features[:, 55:62] = ext_at_samples[:, :7]   # funding, OI, liq, netflow, active_addr
        features[:, 65:90] = ext_at_samples[:, 7:32]  # 25 extended features
```

**3f.** Update `extract_all_features()` signature to accept external features:

```python
def extract_all_features(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    external_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List]:
```

And pass it through to `_batch_extract_tabular()`:

```python
    tabular = _batch_extract_tabular(df_5m, symbol, btc_df_5m, sample_indices, external_features)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml_features.py -v`
Expected: All tests PASS with updated assertions

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_features.py tests/test_ml_features.py
git commit -m "feat: expand N_TABULAR to 90, accept external_data dict"
```

---

### Task 5: Update ML Signal Generator

**Files:**
- Modify: `bot/learning/ml_signal_generator.py`
- Modify: `tests/test_ml_signal_generator.py`

- [ ] **Step 1: Update tests for Transformer + external data + failsafe**

Edit `tests/test_ml_signal_generator.py`. Update `_make_generator` to use Transformer, update feature count from 81 to 106, add failsafe test:

```python
"""Tests for ML signal generator prediction interface."""
import json
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=timestamps)


class TestMLSignalGenerator:
    def test_predict_returns_correct_structure(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert "probabilities" in result
        assert "direction" in result
        assert "net_up" in result
        assert "net_down" in result
        assert len(result["probabilities"]) == 12

    def test_predict_probabilities_bounded(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        for p in result["probabilities"]:
            assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds"

    def test_predict_direction_valid(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert result["direction"] in ("LONG", "SHORT", None)

    def test_predict_insufficient_data(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(50)
        result = gen.predict(df, symbol="BTC-EUR")
        assert result["direction"] is None
        assert all(p == 0.5 for p in result["probabilities"])

    def test_models_not_found_raises(self):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        with pytest.raises(FileNotFoundError):
            MLSignalGenerator(model_dir="/nonexistent/path")

    def test_falls_back_to_lstm(self, tmp_path):
        """When transformer.pt doesn't exist, should load lstm.pt."""
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path, use_lstm=True)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert "probabilities" in result

    def test_disabled_returns_none_direction(self, tmp_path):
        """When ML signals are disabled, predict() returns no signal."""
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        gen._disabled = True
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert result["direction"] is None

    def _make_generator(self, tmp_path, use_lstm=False):
        """Create a generator with dummy models for testing."""
        from bot.learning.ml_signal_generator import MLSignalGenerator
        import xgboost as xgb

        model_dir = tmp_path / "ml_signals"
        model_dir.mkdir()

        if use_lstm:
            from bot.learning.lstm_embedder import LSTMEmbedder
            lstm = LSTMEmbedder()
            lstm.save(str(model_dir / "lstm.pt"))
        else:
            from bot.learning.transformer_embedder import TransformerEmbedder
            transformer = TransformerEmbedder()
            transformer.save(str(model_dir / "transformer.pt"))

        # Save 12 dummy XGBoost models — 106 features (90 tabular + 16 embed)
        n_features = 106
        horizons = ["30m", "1h", "4h", "12h", "24h", "72h"]
        directions = ["up", "down"]
        for h in horizons:
            for d in directions:
                X = np.random.randn(100, n_features).astype(np.float32)
                y = np.random.randint(0, 2, 100)
                model = xgb.XGBClassifier(
                    n_estimators=5, max_depth=2, use_label_encoder=False,
                    eval_metric="logloss",
                )
                model.fit(X, y)
                model.save_model(str(model_dir / f"xgb_{h}_{d}.json"))

        # Save feature config
        config = {"n_tabular": 90, "n_embed": 16, "horizons": horizons}
        (model_dir / "feature_config.json").write_text(json.dumps(config))

        return MLSignalGenerator(model_dir=str(model_dir))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_signal_generator.py -v`
Expected: FAIL — import errors, signature mismatches

- [ ] **Step 3: Update ml_signal_generator.py**

Replace `bot/learning/ml_signal_generator.py` with:

```python
"""ML Signal Generator: load trained models and predict directional probabilities.

Loads:
    - Transformer embedder (transformer.pt, fallback to lstm.pt)
    - 12 XGBoost classifiers (xgb_{horizon}_{direction}.json)
    - Feature config (feature_config.json)

Exposes:
    predict(df_5m, symbol) → dict with probabilities, direction, net_up, net_down
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from bot.learning.ml_features import extract_tabular_features, build_lstm_sequence

logger = logging.getLogger(__name__)

HORIZONS = ["30m", "1h", "4h", "12h", "24h", "72h"]
DIRECTIONS = ["up", "down"]
MIN_SIGNAL_PROB = 0.4  # minimum average probability to generate a direction signal


class MLSignalGenerator:
    """Load trained embedder + XGBoost models and generate trading signals."""

    def __init__(self, model_dir: str = "models/ml_signals") -> None:
        self._model_dir = Path(model_dir)
        self._disabled = False  # Set by failsafe system

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"ML model directory not found: {model_dir}. "
                f"Run scripts/train_ml_signals.py first."
            )

        # Load embedder: prefer Transformer, fall back to LSTM
        transformer_path = self._model_dir / "transformer.pt"
        lstm_path = self._model_dir / "lstm.pt"

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if transformer_path.exists():
            from bot.learning.transformer_embedder import TransformerEmbedder
            self._embedder = TransformerEmbedder()
            self._embedder.load(str(transformer_path))
            self._embedder.to(device)
            self._embedder.eval()
            logger.info("Loaded Transformer embedder on %s from %s", device, transformer_path)
        elif lstm_path.exists():
            from bot.learning.lstm_embedder import LSTMEmbedder
            self._embedder = LSTMEmbedder()
            self._embedder.load(str(lstm_path))
            self._embedder.to(device)
            self._embedder.eval()
            logger.info("Loaded LSTM embedder (fallback) on %s from %s", device, lstm_path)
        else:
            raise FileNotFoundError(
                f"No embedder model found. Expected transformer.pt or lstm.pt in {model_dir}"
            )

        # Load XGBoost models (GPU inference if available)
        self._xgb_models: Dict[str, xgb.XGBClassifier] = {}
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                path = self._model_dir / f"xgb_{key}.json"
                if not path.exists():
                    raise FileNotFoundError(f"XGBoost model not found: {path}")
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                # Enable GPU prediction if model was trained on GPU
                try:
                    model.set_params(device="cuda")
                except Exception:
                    pass  # Fall back to CPU prediction if GPU not available
                self._xgb_models[key] = model

        # Load feature config (accept both n_embed and n_lstm_embed for backward compat)
        config_path = self._model_dir / "feature_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"n_tabular": 90, "n_embed": 16, "horizons": HORIZONS}

        # Accept both key names
        if "n_embed" not in self._config and "n_lstm_embed" in self._config:
            self._config["n_embed"] = self._config["n_lstm_embed"]

        logger.info(
            "MLSignalGenerator loaded: embedder + %d XGBoost models from %s",
            len(self._xgb_models), model_dir,
        )

    def predict(
        self,
        df_5m: "pd.DataFrame",
        symbol: str = "BTC-EUR",
        btc_df_5m: Optional["pd.DataFrame"] = None,
        external_data: dict | None = None,
        **live_features,
    ) -> Dict[str, Any]:
        """Generate predictions for the current market state.

        Args:
            df_5m: Recent 5m candles (at least MIN_BARS_5M bars)
            symbol: Trading pair symbol
            btc_df_5m: BTC candles for cross-asset features
            external_data: Dict of external feature values (from ExternalDataProvider)
            **live_features: Additional live features (regime_id, regime_hours)

        Returns:
            dict with keys:
                probabilities: list of 12 floats [up_30m, down_30m, up_1h, down_1h, ...]
                direction: "LONG", "SHORT", or None (no signal)
                net_up: float (mean of up probabilities across horizons)
                net_down: float (mean of down probabilities across horizons)
        """
        default = {
            "probabilities": [0.5] * 12,
            "direction": None,
            "net_up": 0.5,
            "net_down": 0.5,
        }

        # Failsafe: if ML signals are disabled, return no signal
        if self._disabled:
            return default

        # Build external_data dict from explicit parameter
        ext = dict(external_data or {})
        # Note: legacy kwargs (funding_score, ob_imbalance, etc.) are NOT mapped
        # because the feature indices they targeted (55-61) now have different
        # semantics. Callers should pass data via external_data dict instead.

        regime_id = int(live_features.get("regime_id", 0))
        regime_hours = live_features.get("regime_hours", 0.0)

        # Extract features
        tabular = extract_tabular_features(
            df_5m, symbol, btc_df_5m,
            external_data=ext,
            regime_id=regime_id,
            regime_hours=regime_hours,
        )
        seq = build_lstm_sequence(df_5m)

        # Check for insufficient data: sequence is all zeros when < LSTM_MIN_BARS,
        # tabular is all zeros when < MIN_BARS_5M. Either condition means no signal.
        if np.all(tabular == 0) or np.all(seq == 0):
            return default

        # Get embedding (works for both Transformer and LSTM — same interface)
        embedding = self._embedder.embed_numpy(seq)  # (16,)

        # Concatenate: 90 tabular + 16 embedding = 106 features
        combined = np.concatenate([tabular, embedding]).reshape(1, -1)

        # Run all 12 XGBoost models
        probabilities = []
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                model = self._xgb_models[key]
                prob = float(model.predict_proba(combined)[0, 1])
                probabilities.append(prob)

        # Compute net direction
        up_probs = [probabilities[i] for i in range(0, 12, 2)]
        down_probs = [probabilities[i] for i in range(1, 12, 2)]
        net_up = float(np.mean(up_probs))
        net_down = float(np.mean(down_probs))

        # Direction decision
        direction = None
        if max(net_up, net_down) >= MIN_SIGNAL_PROB:
            direction = "LONG" if net_up > net_down else "SHORT"

        return {
            "probabilities": probabilities,
            "direction": direction,
            "net_up": net_up,
            "net_down": net_down,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml_signal_generator.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_signal_generator.py tests/test_ml_signal_generator.py
git commit -m "feat: update signal generator for Transformer, external data, failsafe"
```

---

## Chunk 3: Training Pipeline

### Task 6: Update Training Script — Transformer + External Data

**Files:**
- Modify: `scripts/train_ml_signals.py`

This task updates the training script to: (1) use Transformer instead of LSTM, (2) fetch external data and merge into features, (3) use the 90→106 feature pipeline.

- [ ] **Step 1: Update imports and constants**

Replace the LSTM imports and params with Transformer equivalents:

```python
"""Train ML signal generator models: Transformer + 12 XGBoost classifiers.

Pipeline:
    1. Fetch 5 years of 5m candles per pair
    2. Fetch external data from 12+ APIs
    3. Compute labels (horizon-scaled thresholds)
    4. Optuna HPO: Transformer architecture search (20 trials)
    5. Train Transformer (self-supervised next-bar prediction)
    6. Generate embeddings
    7. Threshold search per horizon (maximize F1)
    8. Optuna HPO: XGBoost hyperparameters (30 shared + 10 per-model)
    9. Train 12 XGBoost models (up/down × 6 horizons) with class weights
    10. Report F1, precision, recall, AUC-ROC per model
    11. Save all models to models/ml_signals/
"""
import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from bot.exchange.bitvavo_client import BitvavoClient
from bot.data.external_features import ExternalDataProvider
from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
from bot.learning.ml_features import (
    extract_all_features, build_lstm_sequence, N_TABULAR, LSTM_WINDOW, LSTM_CHANNELS,
    LSTM_MIN_BARS,
)

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
YEARS = 5
MODEL_DIR = "models/ml_signals"

HORIZONS = [
    ("30m",  6,    0.15),
    ("1h",   12,   0.3),
    ("4h",   48,   0.8),
    ("12h",  144,  1.5),
    ("24h",  288,  2.5),
    ("72h",  864,  4.0),
]

# Transformer training params
TRANSFORMER_EPOCHS = 50
TRANSFORMER_BATCH = 256
TRANSFORMER_LR = 1e-3
TRANSFORMER_PREDICT_BARS = 12
LSTM_PREDICT_BARS = TRANSFORMER_PREDICT_BARS  # alias — build_lstm_targets() uses this name

# XGBoost base params (HPO will override most of these)
XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 50,
}
```

**Note:** The existing `compute_labels()` and `build_lstm_targets()` functions in the current training script are kept unchanged. They remain as local functions in the file.

- [ ] **Step 2: Add Optuna Transformer HPO function**

```python
def optuna_transformer_hpo(train_seqs, train_targets, val_seqs, val_targets, n_trials=20):
    """Search for best Transformer architecture using Optuna."""
    import optuna

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        # Constraint: nhead must divide d_model
        if d_model % nhead != 0:
            raise optuna.TrialPruned()
        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4, 6])
        dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.3])
        lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])

        # Reduce batch size for larger models (VRAM management)
        batch_size = TRANSFORMER_BATCH
        if d_model >= 128:
            batch_size = batch_size // 2

        model = TransformerEmbedder(d_model=d_model, nhead=nhead, num_layers=n_layers, dropout=dropout)

        try:
            import torch
            losses = train_transformer(
                model, train_seqs, train_targets,
                val_seqs=val_seqs, val_targets=val_targets,
                epochs=15, batch_size=batch_size, lr=lr,
            )
            # Compute validation MSE
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                vX = torch.from_numpy(val_seqs).float().to(device)
                vY = torch.from_numpy(val_targets).float().to(device)
                val_pred = model.predict_next(vX)
                val_mse = float(torch.nn.functional.mse_loss(val_pred, vY).item())
            return val_mse
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"  Best Transformer params: {study.best_params}")
    print(f"  Best val MSE: {study.best_value:.6f}")
    return study.best_params
```

- [ ] **Step 3: Add threshold search function**

```python
def search_thresholds(X, all_dfs, processed_pairs, train_mask, val_mask, horizons):
    """Search for optimal thresholds per horizon that maximize validation F1.

    Computes labels per-pair on full-resolution close prices (5m bars) to avoid
    subsampling artifacts and cross-pair contamination. Then subsamples to match
    the feature matrix X.

    Args:
        X: feature matrix (N, n_features)
        all_dfs: dict of pair → DataFrame with full 5m candles
        processed_pairs: list of pair names that were successfully processed in X
        train_mask: boolean mask for training samples
        val_mask: boolean mask for validation samples
        horizons: list of (name, bars_ahead, base_threshold_pct) tuples

    Returns:
        dict mapping horizon name to optimal threshold percentage.
    """
    from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY

    optimal = {}
    for h_idx, (h_name, bars, base_threshold) in enumerate(horizons):
        best_f1 = -1
        best_thresh = base_threshold
        for factor in np.linspace(0.5, 1.5, 10):
            thresh_pct = base_threshold * factor
            thresh_frac = thresh_pct / 100.0

            # Recompute labels per-pair on full-resolution 5m close prices
            all_y_up = []
            for pair in processed_pairs:
                df = all_dfs[pair]
                close = df["close"].values.astype(np.float64)
                n = len(close)
                y_up_full = np.full(n, np.nan, dtype=np.float32)
                valid = n - bars
                if valid > 0:
                    future_close = close[bars:bars + valid]
                    current_close = close[:valid]
                    future_return = (future_close - current_close) / (current_close + 1e-12)
                    y_up_full[:valid] = (future_return > thresh_frac).astype(np.float32)
                # Subsample with i-1 offset to match extract_all_features timestamps
                sample_indices = np.arange(MIN_BARS_5M, n, SIGNAL_EVERY)
                all_y_up.append(y_up_full[sample_indices - 1])

            y_up = np.concatenate(all_y_up)

            train_valid = train_mask & ~np.isnan(y_up)
            val_valid = val_mask & ~np.isnan(y_up)

            if train_valid.sum() < 100 or val_valid.sum() < 100:
                continue

            X_t, y_t = X[train_valid], y_up[train_valid]
            X_v, y_v = X[val_valid], y_up[val_valid]

            quick_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                tree_method="hist", device="cuda", eval_metric="logloss",
                early_stopping_rounds=20,
            )
            quick_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            preds = quick_model.predict(X_v)
            f1 = f1_score(y_v, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh_pct

        optimal[h_name] = best_thresh
        print(f"    {h_name}: best_threshold={best_thresh:.3f}% (F1={best_f1:.3f})")

    return optimal
```

- [ ] **Step 4: Add Optuna XGBoost HPO function**

```python
def optuna_xgboost_hpo(X_train, y_train, X_val, y_val, n_trials=30,
                       scale_pos_weight=1.0, seed_params=None):
    """Search for best XGBoost hyperparameters using Optuna.

    Args:
        seed_params: If provided, used as initial trial (enqueue) so the search
                     starts from a known-good point (e.g. shared baseline params).
    """
    import optuna

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    study = optuna.create_study(direction="maximize")

    # Seed the study with known-good params so fine-tuning starts from baseline
    if seed_params:
        study.enqueue_trial({
            k: seed_params[k] for k in (
                "max_depth", "learning_rate", "n_estimators",
                "min_child_weight", "subsample", "colsample_bytree",
                "reg_alpha", "reg_lambda",
            ) if k in seed_params
        })

    study.optimize(objective, n_trials=n_trials)

    print(f"  Best XGBoost params: max_depth={study.best_params['max_depth']}, "
          f"lr={study.best_params['learning_rate']:.4f}, "
          f"n_est={study.best_params['n_estimators']}")
    print(f"  Best val F1: {study.best_value:.4f}")
    return study.best_params
```

- [ ] **Step 5: Rewrite main() function**

The main function follows the spec's training pipeline order:

```python
def main():
    print("=" * 80)
    print("  ML SIGNAL GENERATOR TRAINING PIPELINE v2")
    print("  Transformer + External Data + Optuna HPO + Class Weights")
    print("=" * 80)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch candles ──
    all_dfs = {}
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=YEARS * 365)

    for pair in PAIRS:
        print(f"\n  Fetching {pair}...", end="", flush=True)
        candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
        if not candles or len(candles) < 10000:
            print(f" skipped (insufficient data: {len(candles) if candles else 0})")
            continue
        df = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        } for c in candles], index=[c.timestamp for c in candles])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        print(f" {len(df):,} candles")
        all_dfs[pair] = df
        time.sleep(1)

    if not all_dfs:
        print("  ERROR: No data fetched. Exiting.")
        sys.exit(1)

    # ── Step 2: Fetch external data ──
    print("\n  Fetching external data from 12+ APIs...")
    ext_provider = ExternalDataProvider(cache_dir="data/external_cache")
    # Use BTC-EUR's index as reference for alignment
    btc_df = all_dfs.get("BTC-EUR")
    if btc_df is not None:
        idx_5m = btc_df.index
        if idx_5m.tz is None:
            idx_5m = idx_5m.tz_localize("UTC")
        ext_features_full = ext_provider.build_training_features(idx_5m)
        print(f"  External features: {ext_features_full.shape}")
    else:
        ext_features_full = None

    # ── Step 3: Build Transformer training data ──
    print("\n  Building Transformer training data...")
    from bot.learning.ml_features import build_lstm_sequence, LSTM_MIN_BARS
    all_seqs = []
    all_transformer_targets = []
    STRIDE = 6
    WINDOW_PAD = 300
    for pair, df in all_dfs.items():
        targets = build_lstm_targets(df)
        n_pair = len(df) - TRANSFORMER_PREDICT_BARS
        count_before = len(all_seqs)
        for i in range(LSTM_MIN_BARS, n_pair, STRIDE):
            window_start = max(0, i + 1 - WINDOW_PAD)
            seq = build_lstm_sequence(df.iloc[window_start:i + 1])
            if not np.all(seq == 0):
                all_seqs.append(seq)
                all_transformer_targets.append(targets[i])
        print(f"    {pair}: {len(all_seqs) - count_before:,} sequences ({len(all_seqs):,} total)")

    seqs_arr = np.stack(all_seqs)
    targets_arr = np.stack(all_transformer_targets)
    print(f"  Total Transformer samples: {len(seqs_arr):,}")

    val_cutoff = int(len(seqs_arr) * 0.88)
    train_seqs, val_seqs = seqs_arr[:val_cutoff], seqs_arr[val_cutoff:]
    train_targets, val_targets = targets_arr[:val_cutoff], targets_arr[val_cutoff:]
    print(f"  Transformer train: {len(train_seqs):,}, val: {len(val_seqs):,}")

    # ── Step 4: Optuna Transformer HPO ──
    print("\n  Running Optuna Transformer HPO (20 trials)...")
    t0 = time.time()
    best_transformer_params = optuna_transformer_hpo(
        train_seqs, train_targets, val_seqs, val_targets, n_trials=20,
    )
    print(f"  HPO completed in {time.time() - t0:.0f}s")

    # ── Step 5: Train best Transformer ──
    # Clean up VRAM from HPO trials before full training
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    print("\n  Training Transformer with best params...")
    transformer = TransformerEmbedder(
        d_model=best_transformer_params["d_model"],
        nhead=best_transformer_params["nhead"],
        num_layers=best_transformer_params["n_layers"],
        dropout=best_transformer_params["dropout"],
    )
    t0 = time.time()
    # Reduce batch size for large d_model (VRAM management on RTX 2060 SUPER)
    final_batch = TRANSFORMER_BATCH
    if best_transformer_params["d_model"] >= 128:
        final_batch = TRANSFORMER_BATCH // 2

    train_transformer(
        transformer, train_seqs, train_targets,
        val_seqs=val_seqs, val_targets=val_targets,
        epochs=TRANSFORMER_EPOCHS, batch_size=final_batch,
        lr=best_transformer_params["lr"],
    )
    print(f"  Transformer trained in {time.time() - t0:.0f}s")
    transformer.save(str(model_dir / "transformer.pt"))

    # ── Step 6: Extract features + embeddings for XGBoost ──
    print("\n  Extracting tabular features + Transformer embeddings...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    transformer.eval()

    all_tabular = []
    all_labels = []
    all_embeddings = []
    all_timestamps_flat = []
    processed_pairs = []  # Track which pairs had valid features (for Steps 8/8b)

    for pair, df in all_dfs.items():
        print(f"    {pair}: extracting features...", end="", flush=True)
        t0 = time.time()

        # Build per-pair external features if available
        pair_ext = None
        if ext_features_full is not None and pair == "BTC-EUR":
            pair_ext = ext_features_full
        elif ext_features_full is not None:
            # Re-align to this pair's index
            pair_idx = df.index
            if pair_idx.tz is None:
                pair_idx = pair_idx.tz_localize("UTC")
            pair_ext = ext_provider.build_training_features(pair_idx)

        tabular, sequences, timestamps = extract_all_features(
            df, symbol=pair,
            btc_df_5m=btc_df if pair != "BTC-EUR" else None,
            external_features=pair_ext,
        )
        if len(tabular) == 0:
            print(" skipped (no features)")
            continue

        processed_pairs.append(pair)

        labels = compute_labels(df)
        ts_indices = [df.index.get_loc(ts) for ts in timestamps]
        label_rows = labels[ts_indices]

        # Batch embedding extraction in chunks (GPU memory limited)
        EMB_BATCH = 4096
        emb_parts = []
        with torch.no_grad():
            for eb_start in range(0, len(sequences), EMB_BATCH):
                chunk = torch.from_numpy(sequences[eb_start:eb_start + EMB_BATCH]).float().to(device)
                emb_parts.append(transformer.embed(chunk).cpu().numpy())
        embeddings = np.vstack(emb_parts)

        all_tabular.append(tabular)
        all_labels.append(label_rows)
        all_embeddings.append(embeddings)
        all_timestamps_flat.extend(timestamps)
        print(f" {len(tabular):,} samples ({time.time() - t0:.0f}s)")

    X_tab = np.vstack(all_tabular)
    y_all = np.vstack(all_labels)
    X_emb = np.vstack(all_embeddings)
    X = np.hstack([X_tab, X_emb])
    print(f"  Total XGBoost samples: {len(X):,}, features: {X.shape[1]}")

    # ── Step 7: Walk-forward split ──
    all_ts = pd.DatetimeIndex(all_timestamps_flat)
    total_span = (all_ts.max() - all_ts.min()).days
    train_cutoff = all_ts.min() + timedelta(days=int(total_span * 0.6))
    val_cutoff_dt = all_ts.min() + timedelta(days=int(total_span * 0.8))

    train_mask = all_ts < train_cutoff
    val_mask = (all_ts >= train_cutoff) & (all_ts < val_cutoff_dt)
    test_mask = all_ts >= val_cutoff_dt

    print(f"  Walk-forward split: train={train_mask.sum():,}, "
          f"val={val_mask.sum():,}, test={test_mask.sum():,}")

    # ── Step 8: Threshold search (needs embeddings for F1 evaluation) ──
    # search_thresholds computes labels per-pair on full-resolution 5m data
    # to avoid subsampling artifacts and cross-pair contamination.
    print("\n  Searching optimal thresholds per horizon...")
    optimal_thresholds = search_thresholds(X, all_dfs, processed_pairs, train_mask, val_mask, HORIZONS)

    # ── Step 8b: Recompute labels with optimal thresholds ──
    # CRITICAL: models must train on labels computed with the same thresholds
    # that will be used for live inference (saved in feature_config.json).
    print("  Recomputing labels with optimal thresholds...")
    OPT_HORIZONS = [
        (name, bars, optimal_thresholds.get(name, thresh))
        for name, bars, thresh in HORIZONS
    ]
    # Rebuild y_all with optimal thresholds (only processed pairs, matching Step 6)
    from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY
    all_labels = []
    for pair in processed_pairs:
        df = all_dfs[pair]
        n = len(df)
        close = df["close"].values.astype(np.float64)
        labels = np.full((n, 12), np.nan, dtype=np.float32)
        for h_idx, (name, bars, threshold) in enumerate(OPT_HORIZONS):
            threshold_frac = threshold / 100.0
            valid = n - bars
            if valid <= 0:
                continue
            future_close = close[bars:bars + valid]
            current_close = close[:valid]
            future_return = (future_close - current_close) / (current_close + 1e-12)
            labels[:valid, h_idx * 2] = (future_return > threshold_frac).astype(np.float32)
            labels[:valid, h_idx * 2 + 1] = (future_return < -threshold_frac).astype(np.float32)

        sample_indices = np.arange(MIN_BARS_5M, n, SIGNAL_EVERY)
        # Use i-1 offset to match extract_all_features() timestamp convention
        all_labels.append(labels[sample_indices - 1])

    y_all = np.vstack(all_labels)
    print(f"  Labels recomputed with optimal thresholds for {len(y_all):,} samples")

    # ── Step 9: Optuna XGBoost HPO (shared baseline on 4h_up) ──
    print("\n  Running Optuna XGBoost HPO (30 shared trials on 4h_up)...")
    h_idx_4h = 2  # 4h is index 2 in HORIZONS
    col_4h_up = h_idx_4h * 2
    y_4h = y_all[:, col_4h_up]
    tv = train_mask & ~np.isnan(y_4h)
    vv = val_mask & ~np.isnan(y_4h)
    n_pos = y_4h[tv].sum()
    n_neg = tv.sum() - n_pos
    spw = n_neg / max(n_pos, 1)

    shared_xgb_params = optuna_xgboost_hpo(
        X[tv], y_4h[tv], X[vv], y_4h[vv],
        n_trials=30, scale_pos_weight=spw,
    )

    # ── Step 10: Train 12 XGBoost models with class weights + per-model fine-tune ──
    print("\n  Training 12 XGBoost models (shared params + 10 fine-tune trials each)...")
    importances = {}
    for h_idx, (h_name, bars, threshold) in enumerate(HORIZONS):
        for d_idx, d_name in enumerate(["up", "down"]):
            col = h_idx * 2 + d_idx
            y = y_all[:, col]

            train_valid = train_mask & ~np.isnan(y)
            val_valid = val_mask & ~np.isnan(y)
            test_valid = test_mask & ~np.isnan(y)

            X_train, y_train = X[train_valid], y[train_valid]
            X_val, y_val = X[val_valid], y[val_valid]
            X_test, y_test = X[test_valid], y[test_valid]

            # Class weighting
            n_pos = float(y_train.sum())
            n_neg = float(len(y_train) - n_pos)
            scale_pos_weight = n_neg / max(n_pos, 1)

            # Per-model fine-tune: 10 Optuna trials seeded from shared baseline
            fine_tuned = optuna_xgboost_hpo(
                X_train, y_train, X_val, y_val,
                n_trials=10, scale_pos_weight=scale_pos_weight,
                seed_params=shared_xgb_params,
            )

            # Merge fine-tuned params with base
            model_params = {
                **XGB_BASE_PARAMS,
                **{k: fine_tuned[k] for k in fine_tuned
                   if k in ("max_depth", "learning_rate", "n_estimators",
                            "min_child_weight", "subsample", "colsample_bytree",
                            "reg_alpha", "reg_lambda")},
                "scale_pos_weight": scale_pos_weight,
            }

            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate on test set
            test_pred = model.predict(X_test)
            test_prob = model.predict_proba(X_test)[:, 1]

            acc = float(np.mean(test_pred == y_test))
            pos_rate = float(y_test.mean())
            f1 = f1_score(y_test, test_pred, zero_division=0)
            prec = precision_score(y_test, test_pred, zero_division=0)
            rec = recall_score(y_test, test_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, test_prob)
            except ValueError:
                auc = 0.0

            fi = model.feature_importances_
            top_feat = int(fi.argmax())
            max_fi = float(fi.max())
            importances[f"{h_name}_{d_name}"] = fi.tolist()

            print(f"    xgb_{h_name}_{d_name}: F1={f1:.3f}, prec={prec:.3f}, "
                  f"rec={rec:.3f}, acc={acc:.3f}, AUC={auc:.3f}, "
                  f"pos_rate={pos_rate:.3f}, spw={scale_pos_weight:.1f}, "
                  f"top_feat={top_feat}({max_fi:.1%})")

            model.save_model(str(model_dir / f"xgb_{h_name}_{d_name}.json"))

    (model_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    # ── Step 11: Save config ──
    config = {
        "n_tabular": N_TABULAR,
        "n_embed": 16,
        "horizons": [h[0] for h in HORIZONS],
        "thresholds": {h[0]: optimal_thresholds.get(h[0], h[2]) for h in HORIZONS},
        "pairs_trained": list(all_dfs.keys()),
        "transformer_params": best_transformer_params,
        "xgb_shared_params": shared_xgb_params,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n  All models saved to {MODEL_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run training script tests**

Run: `pytest tests/test_train_ml_signals.py -v`
Expected: All tests PASS (compute_labels and build_lstm_targets unchanged)

- [ ] **Step 7: Commit**

```bash
git add scripts/train_ml_signals.py
git commit -m "feat: rewrite training pipeline with Transformer, external data, Optuna HPO, class weights"
```

---

## Chunk 4: Integration and Final Wiring

### Task 7: Update Trading Loop and Main.py

**Files:**
- Modify: `bot/trading_loop.py:60-67,782-785`
- Modify: `bot/main.py:290-326`

- [ ] **Step 1: Update main.py to initialize ExternalDataProvider and check for transformer.pt**

In `_get_ml_signal_generator()` (line 290), update to check for `transformer.pt` first:

```python
def _get_ml_signal_generator():
    """Load ML signal generator. Raises if models not found (per spec: no fallback)."""
    model_dir = "models/ml_signals"
    model_path = Path(model_dir)
    has_transformer = (model_path / "transformer.pt").exists()
    has_lstm = (model_path / "lstm.pt").exists()

    if model_path.exists() and (has_transformer or has_lstm):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        ml_gen = MLSignalGenerator(model_dir=model_dir)
        logger.info("ML Signal Generator loaded from %s", model_dir)
        return ml_gen
    else:
        raise FileNotFoundError(
            f"ML models not found at {model_dir}. "
            f"Run 'python scripts/train_ml_signals.py' first. "
            f"The bot requires trained ML models to start."
        )
```

Add a new function to initialize the external data provider:

```python
_ext_data_provider = None

def _get_ext_data_provider():
    """Initialize external data provider for live features."""
    global _ext_data_provider
    if _ext_data_provider is None:
        from bot.data.external_features import ExternalDataProvider
        _ext_data_provider = ExternalDataProvider(cache_dir="data/external_cache")
        logger.info("External data provider initialized")
    return _ext_data_provider
```

Pass it into the trading loop (line ~322):

```python
        ml_signal_generator=_get_ml_signal_generator(),
        ext_data_provider=_get_ext_data_provider(),
```

Add `ext_data_provider=_get_ext_data_provider()` as a new kwarg to the existing `TradingLoop(...)` constructor call at line 322. The other parameters remain unchanged.

- [ ] **Step 2: Update TradingLoop to accept and use external data provider**

In `bot/trading_loop.py`, add `ext_data_provider` parameter to `__init__`:

```python
def __init__(self, ..., ml_signal_generator=None, ext_data_provider=None):
    # ... existing assignments ...
    self._ml_signal_generator = ml_signal_generator
    self._ext_data_provider = ext_data_provider
```

Add a periodic refresh call at the start of each evaluation cycle (before ML signal path). This ensures `refresh_live()` is called once per evaluation cycle (typically every 1h):

```python
                # Refresh external data (once per evaluation cycle)
                if self._ext_data_provider is not None:
                    try:
                        self._ext_data_provider.refresh_live()
                    except Exception as e:
                        logger.warning("External data refresh failed: %s", e)
```

Update the ML signal path (around line 782) to fetch and pass external data:

```python
                # --- ML signal path ---
                if self._ml_signal_generator is not None:
                    # Check failsafe: is external data critically stale?
                    if self._ext_data_provider is not None:
                        if self._ext_data_provider.is_critically_stale():
                            self._ml_signal_generator._disabled = True
                            logger.warning("ML signals disabled: external data critically stale")
                        else:
                            self._ml_signal_generator._disabled = False

                    ml_window_start = max(0, len(df_5m) - 2000)
                    ml_df = df_5m.iloc[ml_window_start:]

                    # Build external data dict for live inference
                    ext_data = {}
                    if self._ext_data_provider is not None:
                        live_features = self._ext_data_provider.build_live_features()
                        # Map 25-element array to dict keys matching extract_tabular_features
                        ext_keys = [
                            "fear_greed", "fear_greed_mom", "gtrends_bitcoin", "gtrends_crypto",
                            "dxy_return", "sp500_return", "gold_return", "vix",
                            "treasury_10y", "yield_spread", "nvt", "mvrv", "sopr", "puell",
                            "hashrate", "eth_active_addr", "stable_supply_change", "tvl_change",
                            "oi_change_7d", "liq_ratio", "taker_buy_ratio", "dvol",
                            "funding_24h_avg", "btc_dom_change", "stable_btc_ratio",
                        ]
                        for i, key in enumerate(ext_keys):
                            ext_data[key] = float(live_features[i])

                    ml_result = self._ml_signal_generator.predict(
                        ml_df, symbol=symbol, external_data=ext_data,
                    )
```

- [ ] **Step 3: Run all ML tests**

Run: `pytest tests/test_ml_features.py tests/test_ml_signal_generator.py tests/test_transformer_embedder.py tests/test_external_features.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add bot/trading_loop.py bot/main.py
git commit -m "feat: wire external data provider into trading loop and main.py"
```

---

### Task 8: Update Remaining Tests

**Files:**
- Modify: `tests/test_train_ml_signals.py`
- Modify: `tests/test_engine_ml_signals.py` (if it references N_TABULAR or feature count)

- [ ] **Step 1: Update test_train_ml_signals.py**

The `compute_labels()` and `build_lstm_targets()` functions haven't changed their interface, so existing tests should still pass. Add a test for the new threshold search:

```python
class TestThresholdSearch:
    """Tests for threshold optimization."""

    def _make_synthetic_pairs(self, n_bars=3000):
        """Create synthetic pair DataFrames for threshold search tests."""
        np.random.seed(42)
        timestamps = pd.date_range("2020-01-01", periods=n_bars, freq="5min")
        base = 50000.0
        close = base * np.exp(np.cumsum(np.random.randn(n_bars) * 0.001))
        return {"BTC-EUR": pd.DataFrame({
            "open": close * 0.999, "high": close * 1.002,
            "low": close * 0.998, "close": close,
            "volume": np.random.exponential(100, n_bars),
        }, index=timestamps)}

    def test_search_returns_dict_with_horizon_keys(self):
        from scripts.train_ml_signals import search_thresholds, HORIZONS
        from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY
        np.random.seed(42)
        all_dfs = self._make_synthetic_pairs()
        n_bars = len(all_dfs["BTC-EUR"])
        n_samples = len(np.arange(MIN_BARS_5M, n_bars, SIGNAL_EVERY))
        X = np.random.randn(n_samples, 106).astype(np.float32)
        train_mask = np.arange(n_samples) < int(n_samples * 0.6)
        val_mask = (np.arange(n_samples) >= int(n_samples * 0.6)) & (np.arange(n_samples) < int(n_samples * 0.8))
        result = search_thresholds(X, all_dfs, ["BTC-EUR"], train_mask, val_mask, HORIZONS)
        assert isinstance(result, dict)
        for h_name, _, _ in HORIZONS:
            assert h_name in result

    def test_search_returns_different_thresholds(self):
        """Optimal thresholds should vary across horizons."""
        from scripts.train_ml_signals import search_thresholds, HORIZONS
        from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY
        np.random.seed(42)
        all_dfs = self._make_synthetic_pairs()
        n_bars = len(all_dfs["BTC-EUR"])
        n_samples = len(np.arange(MIN_BARS_5M, n_bars, SIGNAL_EVERY))
        X = np.random.randn(n_samples, 106).astype(np.float32)
        train_mask = np.arange(n_samples) < int(n_samples * 0.6)
        val_mask = (np.arange(n_samples) >= int(n_samples * 0.6)) & (np.arange(n_samples) < int(n_samples * 0.8))
        result = search_thresholds(X, all_dfs, ["BTC-EUR"], train_mask, val_mask, HORIZONS)
        values = list(result.values())
        # At least some should differ from their base values
        assert len(set(f"{v:.3f}" for v in values)) >= 2, "All thresholds are identical"
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v -k "ml"`
Expected: All ML-related tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_ml_signals.py
git commit -m "test: update training script tests for threshold search"
```

---

### Task 9: Run Full Test Suite and Final Verification

- [ ] **Step 1: Run entire test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (except known pre-existing failures unrelated to ML)

- [ ] **Step 2: Verify model loading with both Transformer and LSTM fallback**

```python
# Quick smoke test (run in Python REPL or as a script):
from bot.learning.transformer_embedder import TransformerEmbedder
t = TransformerEmbedder()
import torch
x = torch.randn(1, 96, 7)
print(t.embed(x).shape)  # Should be (1, 16)
print(t.predict_next(x).shape)  # Should be (1, 60)
```

- [ ] **Step 3: Verify external data provider initialization**

```python
from bot.data.external_features import ExternalDataProvider
p = ExternalDataProvider(cache_dir=None)
print(p.build_live_features().shape)  # Should be (25,)
print(p.is_critically_stale())  # Should be False
```

- [ ] **Step 4: Final commit if needed**

```bash
git add -A
git status  # Verify no unintended files
```
