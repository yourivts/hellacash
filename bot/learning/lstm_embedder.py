"""LSTM embedder: encode price sequences into fixed-dim embeddings.

Architecture:
    Input: (batch, 96, 7)
    → LSTM(7 → 32, 2 layers, dropout=0.2)
    → last hidden → Linear(32 → 16) + ReLU
    → embedding (batch, 16)

Self-supervised training objective:
    Predict next 12 bars' normalized OHLCV (5 channels × 12 = 60 outputs).
    Prediction head: Linear(32 → 60), discarded after training.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SEQ_LEN = 96
INPUT_DIM = 7
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2
EMBED_DIM = 16
PREDICT_BARS = 12
PREDICT_CHANNELS = 5  # OHLCV
PREDICT_DIM = PREDICT_BARS * PREDICT_CHANNELS  # 60


class LSTMEmbedder(nn.Module):
    """LSTM sequence encoder with self-supervised prediction head."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
            nn.ReLU(),
        )
        # Self-supervised prediction head (discarded after training)
        self.predict_head = nn.Linear(HIDDEN_DIM, PREDICT_DIM)

    def _lstm_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Run LSTM, return last hidden state. Shape: (batch, HIDDEN_DIM)."""
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        return h_n[-1]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get 16-dim embedding. Input: (batch, 96, 7) → Output: (batch, 16)."""
        h = self._lstm_hidden(x)
        return self.embed_proj(h)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next 12 bars OHLCV. Input: (batch, 96, 7) → Output: (batch, 60)."""
        h = self._lstm_hidden(x)
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
        """Save model weights (always saved as CPU state for portability)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cpu().state_dict(), path)
        logger.info("LSTM embedder saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("LSTM embedder loaded from %s", path)


def train_lstm(
    model: LSTMEmbedder,
    sequences: np.ndarray,
    targets: np.ndarray,
    val_seqs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[float]:
    """Train the LSTM via self-supervised next-bar prediction.

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
    logger.info("LSTM training on %s", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
                "LSTM epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch + 1, epochs, avg_loss, val_loss,
            )
        else:
            logger.info("LSTM epoch %d/%d — train_loss=%.6f", epoch + 1, epochs, avg_loss)

    return loss_history
