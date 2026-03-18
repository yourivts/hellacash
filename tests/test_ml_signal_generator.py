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

    def _make_generator(self, tmp_path):
        """Create a generator with dummy models for testing."""
        from bot.learning.ml_signal_generator import MLSignalGenerator
        from bot.learning.lstm_embedder import LSTMEmbedder
        import xgboost as xgb

        model_dir = tmp_path / "ml_signals"
        model_dir.mkdir()

        # Save dummy LSTM
        lstm = LSTMEmbedder()
        lstm.save(str(model_dir / "lstm.pt"))

        # Save 12 dummy XGBoost models
        horizons = ["30m", "1h", "4h", "12h", "24h", "72h"]
        directions = ["up", "down"]
        for h in horizons:
            for d in directions:
                X = np.random.randn(100, 81).astype(np.float32)
                y = np.random.randint(0, 2, 100)
                model = xgb.XGBClassifier(
                    n_estimators=5, max_depth=2, use_label_encoder=False,
                    eval_metric="logloss",
                )
                model.fit(X, y)
                model.save_model(str(model_dir / f"xgb_{h}_{d}.json"))

        # Save feature config
        config = {"n_tabular": 65, "n_lstm_embed": 16, "horizons": horizons}
        (model_dir / "feature_config.json").write_text(json.dumps(config))

        return MLSignalGenerator(model_dir=str(model_dir))
