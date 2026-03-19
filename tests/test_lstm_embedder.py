"""Tests for LSTM embedder model."""
import numpy as np
import pytest
import torch


class TestLSTMEmbedder:
    def test_embedding_shape(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(4, 96, 7)  # batch=4, seq=96, channels=7
        emb = model.embed(x)
        assert emb.shape == (4, 16), f"Expected (4, 16), got {emb.shape}"

    def test_prediction_shape(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(4, 96, 7)
        pred = model.predict_next(x)
        assert pred.shape == (4, 60), f"Expected (4, 60), got {pred.shape}"

    def test_single_sample(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(1, 96, 7)
        emb = model.embed(x)
        assert emb.shape == (1, 16)

    def test_embed_deterministic(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            e1 = model.embed(x)
            e2 = model.embed(x)
        assert torch.allclose(e1, e2)

    def test_save_load_roundtrip(self, tmp_path):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            emb_before = model.embed(x)

        path = str(tmp_path / "lstm.pt")
        model.save(path)

        model2 = LSTMEmbedder()
        model2.load(path)
        model2.eval()
        with torch.no_grad():
            emb_after = model2.embed(x)

        assert torch.allclose(emb_before, emb_after, atol=1e-6)


class TestLSTMEmbedderNumpy:
    def test_embed_numpy(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        arr = np.random.randn(96, 7).astype(np.float32)
        emb = model.embed_numpy(arr)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (16,)
        assert emb.dtype == np.float32


class TestLSTMTraining:
    def _make_training_data(self, n_samples=200):
        """Create synthetic (sequence, target) pairs."""
        np.random.seed(42)
        sequences = np.random.randn(n_samples, 96, 7).astype(np.float32)
        # Targets: next 12 bars OHLCV (60 values)
        targets = np.random.randn(n_samples, 60).astype(np.float32) * 0.01
        return sequences, targets

    def test_train_reduces_loss(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(200)
        history = train_lstm(model, seqs, targets, epochs=5, batch_size=32, lr=1e-3)
        assert history[-1] < history[0], "Loss should decrease over training"

    def test_train_returns_loss_history(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(100)
        history = train_lstm(model, seqs, targets, epochs=3, batch_size=32, lr=1e-3)
        assert len(history) == 3
        assert all(isinstance(v, float) for v in history)

    def test_train_with_validation(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(200)
        history = train_lstm(
            model, seqs[:160], targets[:160],
            val_seqs=seqs[160:], val_targets=targets[160:],
            epochs=3, batch_size=32, lr=1e-3,
        )
        assert len(history) == 3
