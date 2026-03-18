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
        model.eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            emb_before = model.embed(x)

        path = str(tmp_path / "transformer.pt")
        model.save(path)

        model2 = TransformerEmbedder()
        model2.load(path)
        model2.eval()
        with torch.no_grad():
            emb_after = model2.embed(x)

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
