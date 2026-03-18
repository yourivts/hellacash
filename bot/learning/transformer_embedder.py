"""Transformer embedder: encode price sequences into fixed-dim embeddings.

Architecture:
    Input: (batch, 96, 7)
    -> Linear(7 -> d_model) projection
    -> + SinusoidalPositionalEncoding(96, d_model)
    -> TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
    -> MeanPooling(dim=1) -> (batch, d_model)       [h_pooled]
    +-> Linear(d_model -> 16) + ReLU -> embedding   [embed branch]
    +-> Linear(d_model -> 60)                       [predict branch, training only]

Self-supervised training objective:
    Predict next 12 bars' normalized OHLCV (5 channels x 12 = 60 outputs).
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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

        # Input projection: 7 -> d_model
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Embedding projection: d_model -> 16
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, EMBED_DIM),
            nn.ReLU(),
        )

        # Self-supervised prediction head (discarded after training)
        self.predict_head = nn.Linear(d_model, PREDICT_DIM)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run Transformer, return mean-pooled hidden state. Shape: (batch, d_model)."""
        # x: (batch, seq_len, 7)
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)  # add positional encoding
        h = self.transformer(h)  # (batch, seq_len, d_model)
        h_pooled = h.mean(dim=1)  # (batch, d_model)
        return h_pooled

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get 16-dim embedding. Input: (batch, 96, 7) -> Output: (batch, 16)."""
        h = self._encode(x)
        return self.embed_proj(h)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next 12 bars OHLCV. Input: (batch, 96, 7) -> Output: (batch, 60)."""
        h = self._encode(x)
        return self.predict_head(h)

    def embed_numpy(self, seq: np.ndarray) -> np.ndarray:
        """Convenience: single numpy sequence (96, 7) -> embedding (16,)."""
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

    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
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
                epoch + 1,
                epochs,
                avg_loss,
                val_loss,
                scheduler.get_last_lr()[0],
            )
        else:
            logger.info(
                "Transformer epoch %d/%d — train_loss=%.6f",
                epoch + 1,
                epochs,
                avg_loss,
            )

    return loss_history
