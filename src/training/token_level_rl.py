"""
Token-Level Reinforcement Learning
===================================
Per-token reward assignment, temporal credit, and token-level policy gradient training.

Pure PyTorch only — no external RL libraries required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Token Reward Assigner
# ---------------------------------------------------------------------------


class TokenRewardAssigner:
    """Converts a scalar sequence reward into per-token reward signals.

    Parameters
    ----------
    method : str
        One of "dense", "sparse", "gamma_decay", "credit_propagation".
    gamma : float
        Decay factor used by "gamma_decay" and "credit_propagation".
    """

    VALID_METHODS = ("dense", "sparse", "gamma_decay", "credit_propagation")

    def __init__(self, method: str = "dense", gamma: float = 0.99) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.VALID_METHODS}.")
        self.method = method
        self.gamma = gamma

    # ------------------------------------------------------------------
    def assign(self, sequence_reward: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Assign per-token rewards from a scalar sequence reward.

        Parameters
        ----------
        sequence_reward : Tensor of shape [B]
            One scalar reward per sequence in the batch.
        seq_len : int
            Number of tokens T.

        Returns
        -------
        Tensor of shape [B, T]
        """
        B = sequence_reward.shape[0]
        device = sequence_reward.device
        T = seq_len

        if self.method == "dense":
            return self._dense(sequence_reward, B, T, device)
        elif self.method == "sparse":
            return self._sparse(sequence_reward, B, T, device)
        elif self.method == "gamma_decay":
            return self._gamma_decay(sequence_reward, B, T, device)
        else:  # credit_propagation
            return self._credit_propagation(sequence_reward, B, T, device)

    # ------------------------------------------------------------------
    def _dense(self, reward: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Broadcast reward equally to all tokens."""
        token_r = reward.unsqueeze(1).expand(B, T)
        return token_r.clone()

    def _sparse(self, reward: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Reward only the last token; zeros elsewhere."""
        token_r = torch.zeros(B, T, device=device, dtype=reward.dtype)
        token_r[:, -1] = reward
        return token_r

    def _gamma_decay(
        self, reward: torch.Tensor, B: int, T: int, device: torch.device
    ) -> torch.Tensor:
        """r_t = γ^(T-1-t) * R  (last token gets full reward, earlier tokens decay)."""
        # exponents: T-1, T-2, ..., 1, 0  (index 0 gets highest exponent)
        exponents = torch.arange(T - 1, -1, -1, device=device, dtype=torch.float32)
        decay = self.gamma**exponents  # shape [T]
        token_r = reward.float().unsqueeze(1) * decay.unsqueeze(0)  # [B, T]
        return token_r

    def _credit_propagation(
        self, reward: torch.Tensor, B: int, T: int, device: torch.device
    ) -> torch.Tensor:
        """Soft distance-weighted credit propagation from the last token."""
        # Weight each position by how close it is to the last token.
        # w_t = exp(-distance / T) normalized so they sum to 1.
        distances = torch.arange(T - 1, -1, -1, device=device, dtype=torch.float32)
        # distance of token t from the last token = T-1-t
        weights = torch.exp(-distances / max(T, 1))  # [T]
        weights = weights / weights.sum()  # normalize → softmax-like
        token_r = reward.float().unsqueeze(1) * weights.unsqueeze(0)  # [B, T]
        return token_r


# ---------------------------------------------------------------------------
# Token-Level Value Head
# ---------------------------------------------------------------------------


class TokenLevelValueHead(nn.Module):
    """Projects hidden states to scalar value estimates per token.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the policy model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [B, T, d_model]

        Returns
        -------
        values : [B, T]
        """
        return self.linear(hidden_states).squeeze(-1)


# ---------------------------------------------------------------------------
# Token Advantage Estimator
# ---------------------------------------------------------------------------


class TokenAdvantageEstimator:
    """Computes GAE advantages and discounted returns over token sequences.

    Parameters
    ----------
    gamma : float
        Discount factor.
    lam : float
        GAE lambda (bias-variance trade-off).
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam

    # ------------------------------------------------------------------
    def gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Generalized Advantage Estimation.

        δ_t = r_t + γ·V_{t+1}·(1 - done_t) - V_t
        A_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

        Parameters
        ----------
        rewards : [B, T]
        values  : [B, T]   — V(s_t) predictions
        dones   : [B, T]   — 1.0 at terminal steps, 0.0 otherwise

        Returns
        -------
        advantages : [B, T]
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae_val = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(T)):
            next_value = (
                values[:, t + 1]
                if t + 1 < T
                else torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
            )
            not_done = 1.0 - dones[:, t]
            delta = rewards[:, t] + self.gamma * next_value * not_done - values[:, t]
            gae_val = delta + self.gamma * self.lam * not_done * gae_val
            advantages[:, t] = gae_val

        return advantages

    # ------------------------------------------------------------------
    def returns(self, rewards: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
        """Compute discounted returns G_t = r_t + γ·G_{t+1}.

        Parameters
        ----------
        rewards : [B, T]
        gamma   : float, overrides self.gamma if provided

        Returns
        -------
        returns_ : [B, T]
        """
        if gamma is None:
            gamma = self.gamma
        B, T = rewards.shape
        returns_ = torch.zeros_like(rewards)
        g = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            g = rewards[:, t] + gamma * g
            returns_[:, t] = g
        return returns_


# ---------------------------------------------------------------------------
# Token-Level PPO
# ---------------------------------------------------------------------------


class TokenPPO(nn.Module):
    """Token-level Proximal Policy Optimization.

    Parameters
    ----------
    policy     : nn.Module
        Must expose a ``forward(input_ids)`` that returns a dict containing
        ``"logits"`` of shape [B, T, V] and ``"hidden_states"`` of shape
        [B, T, d_model].
    value_head : TokenLevelValueHead
    lr         : float
    clip_eps   : float  PPO clip epsilon
    vf_coef    : float  value loss coefficient
    ent_coef   : float  entropy bonus coefficient
    """

    def __init__(
        self,
        policy: nn.Module,
        value_head: TokenLevelValueHead,
        lr: float,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.value_head = value_head
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.optimizer = Adam(list(policy.parameters()) + list(value_head.parameters()), lr=lr)

    # ------------------------------------------------------------------
    def policy_loss(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped PPO surrogate objective.

        L = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]
        where r = exp(log_prob_new - log_prob_old)

        Parameters
        ----------
        log_probs_new : [B, T]
        log_probs_old : [B, T]
        advantages    : [B, T]

        Returns
        -------
        loss : scalar
        """
        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate = torch.min(ratio * advantages, clipped * advantages)
        return -surrogate.mean()

    # ------------------------------------------------------------------
    def value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """MSE value loss.

        Parameters
        ----------
        values  : [B, T]
        returns : [B, T]

        Returns
        -------
        loss : scalar
        """
        return F.mse_loss(values, returns)

    # ------------------------------------------------------------------
    def entropy_bonus(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Mean token-level entropy from log-probability distributions.

        Parameters
        ----------
        log_probs : [B, T, V]  — log-softmax over vocabulary

        Returns
        -------
        entropy : scalar  (≥ 0)
        """
        # H = -sum_v p_v * log_p_v
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
        return entropy.mean()

    # ------------------------------------------------------------------
    def total_loss(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined PPO loss.

        Parameters
        ----------
        log_probs_new  : [B, T]   log-prob of taken action (new policy)
        log_probs_old  : [B, T]   log-prob of taken action (old policy)
        values         : [B, T]   value estimates
        returns        : [B, T]   discounted returns
        advantages     : [B, T]   GAE advantages
        log_probs_dist : [B, T, V] full log-prob distribution (for entropy)

        Returns
        -------
        (total_loss, policy_loss, value_loss, entropy_loss) — all scalars
        """
        policy_l = self.policy_loss(log_probs_new, log_probs_old, advantages)
        value_l = self.value_loss(values, returns)
        entropy_l = self.entropy_bonus(log_probs_dist)
        loss = policy_l + self.vf_coef * value_l - self.ent_coef * entropy_l
        return loss, policy_l, value_l, entropy_l

    # ------------------------------------------------------------------
    def update_step(self, input_ids: torch.Tensor, rewards: torch.Tensor) -> dict[str, float]:
        """Single PPO update step.

        Parameters
        ----------
        input_ids : [B, T]   token indices
        rewards   : [B]      scalar sequence rewards

        Returns
        -------
        dict with keys: "loss", "policy_loss", "value_loss", "entropy_loss"
        """
        B, T = input_ids.shape

        # --- forward pass (new policy) ---
        output = self.policy(input_ids)
        logits: torch.Tensor = output["logits"]  # [B, T, V]
        hidden: torch.Tensor = output["hidden_states"]  # [B, T, d_model]

        log_probs_dist = F.log_softmax(logits, dim=-1)  # [B, T, V]
        # Gather log-prob of each chosen token (teacher-forced)
        log_probs_new = log_probs_dist.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T]

        # --- old policy (detached reference) ---
        with torch.no_grad():
            log_probs_old = log_probs_new.detach()

        # --- value estimates ---
        values = self.value_head(hidden)  # [B, T]

        # --- reward assignment & returns ---
        assigner = TokenRewardAssigner(method="gamma_decay")
        token_rewards = assigner.assign(rewards, T)  # [B, T]

        estimator = TokenAdvantageEstimator()
        returns_ = estimator.returns(token_rewards)
        dones = torch.zeros_like(token_rewards)
        dones[:, -1] = 1.0
        advantages = estimator.gae(token_rewards, values.detach(), dones)

        # --- PPO loss ---
        loss, policy_l, value_l, entropy_l = self.total_loss(
            log_probs_new,
            log_probs_old,
            values,
            returns_,
            advantages,
            log_probs_dist,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_l.item(),
            "value_loss": value_l.item(),
            "entropy_loss": entropy_l.item(),
        }


# ---------------------------------------------------------------------------
# Token Credit Model
# ---------------------------------------------------------------------------


class TokenCreditModel(nn.Module):
    """Learns to predict per-token importance weights for credit assignment.

    A small transformer-free network that takes token embeddings + a scalar
    reward signal and outputs a probability distribution over T tokens.

    Parameters
    ----------
    d_model    : int
    vocab_size : int
    n_layers   : int  number of hidden MLP layers (≥1)
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Build MLP: (d_model + 1) → d_model → … → 1
        layers: list[nn.Module] = []
        in_dim = d_model + 1  # +1 for the scalar reward
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            in_dim = d_model
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor, sequence_reward: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids       : [B, T]
        sequence_reward : [B]

        Returns
        -------
        token_credits : [B, T]  (softmax over T — sums to 1 per sequence)
        """
        B, T = input_ids.shape
        emb = self.embedding(input_ids)  # [B, T, d_model]
        # Broadcast scalar reward to each token position
        r = sequence_reward.float().unsqueeze(1).unsqueeze(2).expand(B, T, 1)
        x = torch.cat([emb, r], dim=-1)  # [B, T, d_model + 1]
        logits = self.mlp(x).squeeze(-1)  # [B, T]
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenRLConfig:
    """Hyperparameter configuration for token-level RL training."""

    # Temporal credit / advantage
    gamma: float = 0.99
    lam: float = 0.95

    # PPO coefficients
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 1e-4

    # Reward assignment
    method: str = "gamma_decay"

    # Model dimensions (used by value head and credit model)
    d_model: int = 32
    vocab_size: int = 64
