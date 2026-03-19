"""Per-signal RL evaluator with online learning (PPO micro-updates).

Each trading signal is evaluated individually by a small Actor-Critic network.
The model sees market context + signal info + portfolio state and outputs:
  - confidence (0-1): gate for taking the trade (> CONF_THRESHOLD = take)
  - atr_multiplier adjustment (0.5-4.0): stop-loss distance
  - rr_ratio adjustment (1.0-5.0): reward-to-risk ratio

After each closed trade, the P&L is used as reward for a PPO micro-update,
so the model keeps learning in real time.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

OBS_DIM = 30  # 12 ML predictions + 4 cross-horizon + 10 portfolio + 4 coin
ACT_DIM = 3
CONF_THRESHOLD = 0.3
HIDDEN = 64
BUFFER_SIZE = 32  # trigger PPO update every N closed trades
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 3e-4
PPO_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

# ---------------------------------------------------------------------------
# Coin profile data (static features the model can use to distinguish coins)
# ---------------------------------------------------------------------------
# market_cap_tier: 0=mega (>100B), 1=large (10-100B), 2=mid (1-10B), 3=small (<1B)
# volatility_class: 0=low, 1=medium, 2=high (historical avg daily vol)
# coin_age_years: approx years since mainnet launch (as of 2025)
# is_btc: 1 if BTC (market leader), 0 otherwise
_COIN_PROFILES_BY_BASE: Dict[str, Dict[str, float]] = {
    "BTC":   {"cap_tier": 0, "vol_class": 1, "age": 16, "is_btc": 1},
    "ETH":   {"cap_tier": 0, "vol_class": 1, "age": 10, "is_btc": 0},
    "XRP":   {"cap_tier": 1, "vol_class": 2, "age": 13, "is_btc": 0},
    "SOL":   {"cap_tier": 1, "vol_class": 2, "age": 5,  "is_btc": 0},
    "ADA":   {"cap_tier": 1, "vol_class": 2, "age": 8,  "is_btc": 0},
    "DOGE":  {"cap_tier": 1, "vol_class": 2, "age": 11, "is_btc": 0},
    "AVAX":  {"cap_tier": 1, "vol_class": 2, "age": 4,  "is_btc": 0},
    "DOT":   {"cap_tier": 1, "vol_class": 2, "age": 5,  "is_btc": 0},
    "LINK":  {"cap_tier": 1, "vol_class": 2, "age": 8,  "is_btc": 0},
    "MATIC": {"cap_tier": 2, "vol_class": 2, "age": 5,  "is_btc": 0},
    "UNI":   {"cap_tier": 2, "vol_class": 2, "age": 5,  "is_btc": 0},
    "LTC":   {"cap_tier": 2, "vol_class": 1, "age": 13, "is_btc": 0},
    "ATOM":  {"cap_tier": 2, "vol_class": 2, "age": 6,  "is_btc": 0},
    "NEAR":  {"cap_tier": 2, "vol_class": 2, "age": 4,  "is_btc": 0},
    "BNB":   {"cap_tier": 1, "vol_class": 2, "age": 8,  "is_btc": 0},
}
_DEFAULT_COIN_PROFILE = {"cap_tier": 2, "vol_class": 2, "age": 5, "is_btc": 0}


def _base_symbol(symbol: str) -> str:
    """Strip quote asset suffixes to get the base symbol."""
    s = symbol.upper()
    for suffix in ("-EUR", "-USDT", "-BTC", "-ETH", "USDT", "EUR"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def get_coin_profile(symbol: str) -> Dict[str, float]:
    """Look up static coin profile. Handles any pair format (BTC-EUR, BTCUSDT, etc)."""
    return _COIN_PROFILES_BY_BASE.get(_base_symbol(symbol), _DEFAULT_COIN_PROFILE)


@dataclass
class EvalResult:
    """Output of a signal evaluation — stored with the open position."""
    confidence: float = 0.0
    atr_multiplier: float = 2.0
    rr_ratio: float = 2.0
    obs: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    log_prob: float = 0.0
    take_trade: bool = False


# LEGACY: will be removed once ML signal path is fully wired
def build_signal_obs(
    # Market context (from pre-computed arrays in the engine)
    rsi_1h: float = 50.0,
    macd_hist_1h: float = 0.0,
    bb_pct_b: float = 0.5,
    bb_bandwidth: float = 0.05,
    adx_4h: float = 25.0,
    atr_pct: float = 2.0,
    ema50_slope_4h: float = 0.0,
    ema200_dist_pct: float = 0.0,
    volume_surge: float = 1.0,
    cmf: float = 0.0,
    regime_id: int = 4,  # 0-4 for the 5 regimes
    cci_1h: float = 0.0,
    # Signal info
    direction_sign: float = 1.0,  # +1 LONG, -1 SHORT
    strength: float = 0.5,
    strategy_name: str = "orderflow",
    tf_score: float = 1.0,
    consecutive_confirms: int = 1,
    # Portfolio state
    equity_ratio: float = 1.0,  # current_equity / initial_capital
    drawdown_pct: float = 0.0,
    open_pos_ratio: float = 0.0,  # len(positions) / max_open
    win_rate_recent: float = 0.5,
    avg_pnl_recent: float = 0.0,
    bars_since_trade: int = 100,
    balance_ratio: float = 1.0,  # balance / initial_capital
    recent_loss_streak: int = 0,
    total_trades: int = 0,
    recent_sharpe: float = 0.0,
    # Coin markers
    symbol: str = "BTC-EUR",
) -> np.ndarray:
    """Build a 32-dim observation vector from engine state."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Market context (12 features, indices 0-11)
    obs[0] = np.clip(rsi_1h / 100.0, 0.0, 1.0)
    obs[1] = np.clip(np.sign(macd_hist_1h), -1.0, 1.0)
    obs[2] = np.clip(bb_pct_b, 0.0, 1.0)
    obs[3] = np.clip(bb_bandwidth / 0.2, 0.0, 1.0)
    obs[4] = np.clip(adx_4h / 50.0, 0.0, 1.0)
    obs[5] = np.clip(atr_pct / 10.0, 0.0, 1.0)
    obs[6] = np.clip(ema50_slope_4h, -1.0, 1.0)
    obs[7] = np.clip(ema200_dist_pct / 20.0, -1.0, 1.0)
    obs[8] = np.clip(volume_surge / 5.0, 0.0, 1.0)
    obs[9] = np.clip(cmf, -1.0, 1.0)
    obs[10] = regime_id / 4.0
    obs[11] = np.clip(cci_1h / 200.0, -1.0, 1.0)

    # Signal info (6 features, indices 12-17)
    obs[12] = direction_sign
    obs[13] = np.clip(strength, 0.0, 1.0)
    _strategy_ids = {
        "orderflow": 0, "funding_contrarian": 1, "range": 2, "squeeze": 3,
        "breakout": 4, "trend_following": 5, "momentum": 6, "mean_reversion": 7,
    }
    obs[14] = _strategy_ids.get(strategy_name, 0) / 7.0
    obs[15] = np.clip(tf_score / 3.0, 0.0, 1.0)
    obs[16] = np.clip(consecutive_confirms / 5.0, 0.0, 1.0)
    obs[17] = strength  # duplicate raw for the network to use freely

    # Portfolio state (10 features, indices 18-27)
    obs[18] = np.clip(equity_ratio, 0.0, 3.0) / 3.0
    obs[19] = np.clip(drawdown_pct / 20.0, 0.0, 1.0)
    obs[20] = np.clip(open_pos_ratio, 0.0, 1.0)
    obs[21] = np.clip(win_rate_recent, 0.0, 1.0)
    obs[22] = np.clip(avg_pnl_recent / 5.0, -1.0, 1.0)
    obs[23] = np.clip(bars_since_trade / 500.0, 0.0, 1.0)
    obs[24] = np.clip(balance_ratio, 0.0, 3.0) / 3.0
    obs[25] = np.clip(recent_loss_streak / 10.0, 0.0, 1.0)
    obs[26] = np.clip(total_trades / 500.0, 0.0, 1.0)
    obs[27] = np.clip(recent_sharpe / 3.0, -1.0, 1.0)

    # Coin markers (4 features, indices 28-31)
    coin = get_coin_profile(symbol)
    obs[28] = coin["cap_tier"] / 3.0          # market cap tier (0-1)
    obs[29] = coin["vol_class"] / 2.0         # volatility class (0-1)
    obs[30] = np.clip(coin["age"] / 16.0, 0.0, 1.0)  # coin maturity
    obs[31] = coin["is_btc"]                  # BTC flag (0 or 1)

    return obs


def build_ml_signal_obs(
    # ML predictions (12 probabilities)
    probabilities: list[float] | None = None,
    # Portfolio state (10 features)
    equity_ratio: float = 1.0,
    drawdown_pct: float = 0.0,
    open_pos_ratio: float = 0.0,
    win_rate_recent: float = 0.5,
    avg_pnl_recent: float = 0.0,
    bars_since_trade: int = 100,
    balance_ratio: float = 1.0,
    recent_loss_streak: int = 0,
    total_trades: int = 0,
    recent_sharpe: float = 0.0,
    # Coin markers
    symbol: str = "BTC-EUR",
) -> np.ndarray:
    """Build a 30-dim observation vector from ML predictions + portfolio state.

    Layout:
        0-11:  ML predictions (prob_up/down × 6 horizons)
        12-15: Cross-horizon features (max_prob, min_prob, horizon_agreement, trend_alignment)
        16-25: Portfolio state
        26-29: Coin markers
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    probs = probabilities if probabilities is not None else [0.5] * 12

    # ML predictions (12 features, indices 0-11)
    for i in range(min(12, len(probs))):
        obs[i] = np.clip(probs[i], 0.0, 1.0)

    # Cross-horizon features (4 features, indices 12-15)
    obs[12] = max(probs)                              # max_prob
    obs[13] = min(probs)                              # min_prob
    up_probs = [probs[i] for i in range(0, 12, 2)]
    down_probs = [probs[i] for i in range(1, 12, 2)]
    up_votes = sum(1 for p in up_probs if p > 0.5)
    down_votes = sum(1 for p in down_probs if p > 0.5)
    obs[14] = max(up_votes, down_votes) / 6.0         # horizon_agreement (0-1)
    net_up = np.mean(up_probs)
    net_down = np.mean(down_probs)
    obs[15] = np.clip(net_up - net_down, -1.0, 1.0)   # trend_alignment

    # Portfolio state (10 features, indices 16-25)
    obs[16] = np.clip(equity_ratio, 0.0, 3.0) / 3.0
    obs[17] = np.clip(drawdown_pct / 20.0, 0.0, 1.0)
    obs[18] = np.clip(open_pos_ratio, 0.0, 1.0)
    obs[19] = np.clip(win_rate_recent, 0.0, 1.0)
    obs[20] = np.clip(avg_pnl_recent / 5.0, -1.0, 1.0)
    obs[21] = np.clip(bars_since_trade / 500.0, 0.0, 1.0)
    obs[22] = np.clip(balance_ratio, 0.0, 3.0) / 3.0
    obs[23] = np.clip(recent_loss_streak / 10.0, 0.0, 1.0)
    obs[24] = np.clip(total_trades / 500.0, 0.0, 1.0)
    obs[25] = np.clip(recent_sharpe / 3.0, -1.0, 1.0)

    # Coin markers (4 features, indices 26-29)
    coin = get_coin_profile(symbol)
    obs[26] = coin["cap_tier"] / 3.0
    obs[27] = coin["vol_class"] / 2.0
    obs[28] = np.clip(coin["age"] / 16.0, 0.0, 1.0)
    obs[29] = coin["is_btc"]

    return obs


class RLSignalEvaluator:
    """Actor-Critic network for per-signal evaluation with online PPO learning.

    Uses numpy-only implementation (no PyTorch dependency in the hot path).
    Weights are stored as plain numpy arrays for fast inference.
    """

    def __init__(self, lr: float = LR, device: str = "cpu") -> None:
        self._lr = lr
        # Actor: obs -> hidden -> hidden -> 3 outputs (confidence, atr_mult, rr_ratio)
        # Critic: obs -> hidden -> hidden -> 1 value
        self._init_weights()
        # Experience replay buffer
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[np.ndarray] = []
        self._logp_buf: List[float] = []
        self._rew_buf: List[float] = []
        self._val_buf: List[float] = []
        # Stats
        self._updates = 0
        self._total_rewards: List[float] = []

    def _init_weights(self) -> None:
        """Initialize Actor-Critic weights with Xavier init."""
        scale1 = math.sqrt(2.0 / (OBS_DIM + HIDDEN))
        scale2 = math.sqrt(2.0 / (HIDDEN + HIDDEN))
        scale3_a = math.sqrt(2.0 / (HIDDEN + ACT_DIM))
        scale3_v = math.sqrt(2.0 / (HIDDEN + 1))

        rng = np.random.default_rng(42)
        # Shared layer 1
        self.w1 = (rng.standard_normal((OBS_DIM, HIDDEN)) * scale1).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        # Shared layer 2
        self.w2 = (rng.standard_normal((HIDDEN, HIDDEN)) * scale2).astype(np.float32)
        self.b2 = np.zeros(HIDDEN, dtype=np.float32)
        # Actor head
        self.w_act = (rng.standard_normal((HIDDEN, ACT_DIM)) * scale3_a).astype(np.float32)
        self.b_act = np.zeros(ACT_DIM, dtype=np.float32)
        # Actor log_std (learnable)
        self.log_std = np.full(ACT_DIM, -0.5, dtype=np.float32)
        # Critic head
        self.w_val = (rng.standard_normal((HIDDEN, 1)) * scale3_v).astype(np.float32)
        self.b_val = np.zeros(1, dtype=np.float32)

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward pass: returns (action_mean, action_std, value).

        Stores intermediate activations for backprop during PPO update.
        """
        # Layer 1: ReLU
        z1 = obs @ self.w1 + self.b1
        h1 = np.maximum(z1, 0)
        # Layer 2: ReLU
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0)
        # Actor: mean of Gaussian policy
        act_mean = h2 @ self.w_act + self.b_act
        act_std = np.exp(self.log_std)
        # Critic: state value
        value = float((h2 @ self.w_val + self.b_val)[0])

        return act_mean, act_std, value

    def evaluate(self, obs: np.ndarray, deterministic: bool = False) -> EvalResult:
        """Evaluate a signal: returns confidence, atr_multiplier, rr_ratio."""
        act_mean, act_std, value = self._forward(obs)

        if deterministic:
            raw_action = act_mean
            log_prob = 0.0
        else:
            # Sample from Gaussian
            noise = np.random.randn(ACT_DIM).astype(np.float32)
            raw_action = act_mean + act_std * noise
            # Log probability of the action
            log_prob = float(-0.5 * np.sum(
                ((raw_action - act_mean) / (act_std + 1e-8)) ** 2
                + 2 * self.log_std + math.log(2 * math.pi)
            ))

        # Transform actions to useful ranges
        confidence = float(_sigmoid(raw_action[0]))
        atr_multiplier = float(0.5 + 3.5 * _sigmoid(raw_action[1]))  # [0.5, 4.0]
        rr_ratio = float(1.0 + 4.0 * _sigmoid(raw_action[2]))  # [1.0, 5.0]

        take = confidence >= CONF_THRESHOLD

        return EvalResult(
            confidence=confidence,
            atr_multiplier=atr_multiplier,
            rr_ratio=rr_ratio,
            obs=obs.copy(),
            action=raw_action.copy(),
            log_prob=log_prob,
            take_trade=take,
        )

    def record_outcome(self, eval_result: EvalResult, reward: float) -> bool:
        """Record a closed trade outcome. Returns True if a PPO update was triggered."""
        if eval_result.obs is None or eval_result.action is None:
            return False

        _, _, value = self._forward(eval_result.obs)

        self._obs_buf.append(eval_result.obs)
        self._act_buf.append(eval_result.action)
        self._logp_buf.append(eval_result.log_prob)
        self._rew_buf.append(reward)
        self._val_buf.append(value)
        self._total_rewards.append(reward)

        if len(self._obs_buf) >= BUFFER_SIZE:
            self._ppo_update()
            return True
        return False

    def flush(self) -> bool:
        """Force a PPO update with whatever is in the buffer."""
        if len(self._obs_buf) >= 4:  # need at least a few samples
            self._ppo_update()
            return True
        return False

    def _ppo_update(self) -> None:
        """Run PPO clipped update on buffered experiences."""
        n = len(self._obs_buf)
        if n < 2:
            return

        obs_arr = np.stack(self._obs_buf)
        act_arr = np.stack(self._act_buf)
        old_logp = np.array(self._logp_buf, dtype=np.float32)
        rewards = np.array(self._rew_buf, dtype=np.float32)
        old_values = np.array(self._val_buf, dtype=np.float32)

        # Compute advantages (GAE-lambda simplified: just reward - value baseline)
        advantages = rewards - old_values
        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        returns = rewards  # Single-step, no bootstrapping needed

        for epoch in range(PPO_EPOCHS):
            for i in range(n):
                obs_i = obs_arr[i]
                act_i = act_arr[i]
                adv_i = advantages[i]
                ret_i = returns[i]
                old_lp_i = old_logp[i]

                # Forward pass with gradient tracking
                z1 = obs_i @ self.w1 + self.b1
                h1 = np.maximum(z1, 0)
                z2 = h1 @ self.w2 + self.b2
                h2 = np.maximum(z2, 0)

                act_mean = h2 @ self.w_act + self.b_act
                act_std = np.exp(self.log_std)
                value = float((h2 @ self.w_val + self.b_val)[0])

                # New log prob
                diff = act_i - act_mean
                new_lp = float(-0.5 * np.sum(
                    (diff / (act_std + 1e-8)) ** 2
                    + 2 * self.log_std + math.log(2 * math.pi)
                ))

                # PPO ratio
                ratio = math.exp(min(new_lp - old_lp_i, 20.0))
                clipped = np.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                policy_loss = -min(ratio * adv_i, clipped * adv_i)

                # Value loss
                value_loss = 0.5 * (value - ret_i) ** 2

                # Entropy bonus
                entropy = float(0.5 * np.sum(1.0 + 2 * self.log_std + math.log(2 * math.pi)))

                # Total loss
                total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                # ---- Manual backprop ----
                # Gradient of value loss w.r.t. value output
                dval = VALUE_COEF * (value - ret_i)

                # Gradient of policy loss w.r.t. act_mean
                if ratio * adv_i < clipped * adv_i:
                    dpolicy_ratio = -adv_i
                else:
                    dpolicy_ratio = 0.0  # clipped, no gradient

                # d(log_prob)/d(act_mean) = (act - mean) / std^2
                dlogp_dmean = diff / (act_std ** 2 + 1e-8)
                dact_mean = dpolicy_ratio * ratio * dlogp_dmean

                # d(log_prob)/d(log_std) = ((act-mean)^2/std^2 - 1)
                dlogp_dlogstd = (diff ** 2 / (act_std ** 2 + 1e-8)) - 1.0
                dlog_std = dpolicy_ratio * ratio * dlogp_dlogstd - ENTROPY_COEF * np.ones(ACT_DIM)

                # Backprop through actor head
                dw_act = np.outer(h2, dact_mean)
                db_act = dact_mean

                # Backprop through critic head
                dw_val = h2.reshape(-1, 1) * dval
                db_val = np.array([dval], dtype=np.float32)

                # Combine gradients flowing back through h2
                dh2 = dact_mean @ self.w_act.T + dval * self.w_val.squeeze()

                # ReLU backward for layer 2
                dz2 = dh2 * (z2 > 0).astype(np.float32)
                dw2 = np.outer(h1, dz2)
                db2 = dz2

                # Backprop to layer 1
                dh1 = dz2 @ self.w2.T
                dz1 = dh1 * (z1 > 0).astype(np.float32)
                dw1 = np.outer(obs_i, dz1)
                db1 = dz1

                # Gradient clipping
                max_norm = 1.0
                for g in [dw1, db1, dw2, db2, dw_act, db_act, dw_val, db_val, dlog_std]:
                    gnorm = np.linalg.norm(g)
                    if gnorm > max_norm:
                        g *= max_norm / gnorm

                # Apply updates
                self.w1 -= self._lr * dw1
                self.b1 -= self._lr * db1
                self.w2 -= self._lr * dw2
                self.b2 -= self._lr * db2
                self.w_act -= self._lr * dw_act
                self.b_act -= self._lr * db_act
                self.w_val -= self._lr * dw_val.astype(np.float32)
                self.b_val -= self._lr * db_val
                self.log_std -= self._lr * dlog_std.astype(np.float32)

        self._updates += 1
        avg_reward = float(rewards.mean())
        logger.info(
            "PPO update #%d: %d samples, avg_reward=%.4f, total_trades_seen=%d",
            self._updates, n, avg_reward, len(self._total_rewards),
        )

        # Clear buffer
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._rew_buf.clear()
        self._val_buf.clear()

    def save(self, path: str) -> None:
        """Save model weights to a .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w_act=self.w_act, b_act=self.b_act,
            w_val=self.w_val, b_val=self.b_val,
            log_std=self.log_std,
        )
        logger.info("RLSignalEvaluator saved to %s", path)

    def load(self, path: str) -> bool:
        """Load model weights from a .npz file. Returns False if file not found."""
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w_act = data["w_act"]
        self.b_act = data["b_act"]
        self.w_val = data["w_val"]
        self.b_val = data["b_val"]
        self.log_std = data["log_std"]
        logger.info("RLSignalEvaluator loaded from %s", path)
        return True

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "updates": self._updates,
            "total_trades_seen": len(self._total_rewards),
            "buffer_size": len(self._obs_buf),
            "avg_reward_last_100": (
                float(np.mean(self._total_rewards[-100:]))
                if self._total_rewards else 0.0
            ),
        }


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)
