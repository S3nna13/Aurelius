"""
Soft Actor-Critic (SAC) adapted for LLM decoding.

Maximum-entropy RL framework that balances reward maximization with entropy
regularization, applied to token-level decoding decisions.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Policy Head
# ---------------------------------------------------------------------------


class SACPolicyHead(nn.Module):
    """
    Stochastic policy that maps hidden states to token distributions.

    Maintains a learnable log-temperature (log_alpha) for entropy regularization.
    """

    def __init__(self, d_model: int, vocab_size: int, alpha_init: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.linear = nn.Linear(d_model, vocab_size)
        # log_alpha is auto-tuned; initialise so exp(log_alpha) == alpha_init
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))

    @property
    def alpha(self) -> Tensor:
        """Temperature coefficient (always positive)."""
        return self.log_alpha.exp()

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden: [B, d_model]

        Returns:
            logits:    [B, vocab_size]
            log_probs: [B, vocab_size]  (log-softmax of logits)
        """
        logits = self.linear(hidden)  # [B, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, V]
        return logits, log_probs

    def sample_action(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Sample a token from the policy distribution.

        Args:
            hidden: [B, d_model]

        Returns:
            action:   [B]  integer token indices
            log_prob: [B]  log probability of the sampled token
        """
        logits, log_probs = self.forward(hidden)  # [B, V]
        probs = torch.softmax(logits, dim=-1)  # [B, V]
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()  # [B]
        # Gather log_prob of the chosen action
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]
        return action, log_prob


# ---------------------------------------------------------------------------
# Q-Network (Double-Q)
# ---------------------------------------------------------------------------


def _build_mlp(d_model: int, vocab_size: int, n_layers: int) -> nn.Sequential:
    """Build a simple MLP: d_model -> ... -> vocab_size."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    layers: list = []
    in_dim = d_model
    hidden_dim = max(d_model, 64)
    for i in range(n_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, vocab_size))
    return nn.Sequential(*layers)


class SACQNetwork(nn.Module):
    """
    Double-Q network: two independent MLPs that each produce per-token Q-values.

    Using two separate networks and taking the minimum at training time reduces
    overestimation bias (Hasselt et al., 2016; Haarnoja et al., 2018).
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.q1 = _build_mlp(d_model, vocab_size, n_layers)
        self.q2 = _build_mlp(d_model, vocab_size, n_layers)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            state: [B, d_model]

        Returns:
            q1: [B, vocab_size]
            q2: [B, vocab_size]
        """
        return self.q1(state), self.q2(state)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class SACReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (s, a, r, s', done) transitions.

    Tensors are pre-allocated on CPU; samples are returned as CPU tensors.
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self._pos = 0
        self._size = 0

        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

    def push(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
        done: bool,
    ) -> None:
        """Store a single transition (overwrites oldest if full)."""
        i = self._pos
        self.states[i] = state.detach().cpu().float()
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state.detach().cpu().float()
        self.dones[i] = bool(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Uniformly sample a mini-batch of transitions.

        Returns:
            states:      [batch_size, state_dim]
            actions:     [batch_size]
            rewards:     [batch_size]
            next_states: [batch_size, state_dim]
            dones:       [batch_size]
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has {self._size} entries."
            )
        idx = torch.randint(0, self._size, (batch_size,))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# SAC Trainer
# ---------------------------------------------------------------------------


class SACTrainer:
    """
    Orchestrates SAC updates for a (policy, q-network) pair.

    Implements:
      - Critic (TD) update with soft target networks
      - Actor update maximising Q - alpha * log_pi
      - Automatic entropy-temperature (alpha) tuning
    """

    def __init__(
        self,
        policy: SACPolicyHead,
        q_net: SACQNetwork,
        lr: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
    ) -> None:
        self.policy = policy
        self.q_net = q_net
        self.gamma = gamma
        self.tau = tau

        # Target network — soft-updated copy, no gradients
        self.target_q_net: SACQNetwork = copy.deepcopy(q_net)
        for p in self.target_q_net.parameters():
            p.requires_grad_(False)

        # Target entropy: default to maximum-entropy baseline -log(1/V) = log(V)
        if target_entropy is None:
            self.target_entropy = math.log(policy.vocab_size)
        else:
            self.target_entropy = target_entropy

        self.q_optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([policy.log_alpha], lr=lr)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def critic_loss(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        TD-error loss for both Q-networks.

        Target: y = r + γ * (1 - done) * (min(Q1',Q2')(s',a') - α * log π(a'|s'))
        Loss:   MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
        """
        states, actions, rewards, next_states, dones = batch
        alpha = self.policy.alpha.detach()

        with torch.no_grad():
            # Next-state distribution
            _, next_log_probs = self.policy(next_states)  # [B, V]
            next_probs = next_log_probs.exp()  # [B, V]

            # Soft target value (expectation over actions)
            tq1, tq2 = self.target_q_net(next_states)  # [B, V]
            min_tq = torch.min(tq1, tq2)  # [B, V]
            soft_v_next = (next_probs * (min_tq - alpha * next_log_probs)).sum(dim=-1)  # [B]

            target = rewards + self.gamma * (~dones).float() * soft_v_next  # [B]

        q1, q2 = self.q_net(states)  # [B, V]
        # Extract Q-value for the taken action
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)
        return loss

    def actor_loss(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Policy loss: maximise E[min(Q1,Q2)(s,a) - α * log π(a|s)].

        We use the reparameterised (softmax-weighted) expectation so gradients
        flow through the distribution parameters.
        """
        states = batch[0]
        alpha = self.policy.alpha.detach()

        _, log_probs = self.policy(states)  # [B, V]
        probs = log_probs.exp()  # [B, V]

        q1, q2 = self.q_net(states)  # [B, V]
        min_q = torch.min(q1, q2)  # [B, V]

        # E_a[min_Q - alpha * log_pi]  —  gradient w.r.t. policy params
        loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()
        return loss

    def alpha_loss(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Temperature loss: push H(π) towards target_entropy.

        Loss = -α * (log π(a|s) + target_entropy)
        """
        states = batch[0]

        with torch.no_grad():
            _, log_probs = self.policy(states)  # [B, V]
            probs = log_probs.exp()  # [B, V]
            # Expected entropy under current policy
            ent = -(probs * log_probs).sum(dim=-1)  # [B]

        # Gradient w.r.t. log_alpha only
        loss = -(self.policy.log_alpha * (ent - self.target_entropy).detach()).mean()
        return loss

    def update(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> dict[str, float]:
        """
        Perform one full SAC update step.

        Returns:
            dict with keys: critic_loss, actor_loss, alpha_loss, alpha
        """
        # --- Critic ---
        c_loss = self.critic_loss(batch)
        self.q_optimizer.zero_grad()
        c_loss.backward()
        self.q_optimizer.step()

        # --- Actor ---
        a_loss = self.actor_loss(batch)
        self.policy_optimizer.zero_grad()
        a_loss.backward()
        self.policy_optimizer.step()

        # --- Alpha ---
        al_loss = self.alpha_loss(batch)
        self.alpha_optimizer.zero_grad()
        al_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft target update ---
        self.soft_update_target()

        return {
            "critic_loss": c_loss.item(),
            "actor_loss": a_loss.item(),
            "alpha_loss": al_loss.item(),
            "alpha": self.policy.alpha.item(),
        }

    def soft_update_target(self) -> None:
        """Polyak-average the target Q-network towards the main Q-network."""
        tau = self.tau
        for p_target, p_main in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            p_target.data.mul_(1.0 - tau).add_(tau * p_main.data)


# ---------------------------------------------------------------------------
# Maximum-Entropy Decoder
# ---------------------------------------------------------------------------


class SACDecoder:
    """
    Greedy-free decoder that uses a SAC policy head for token selection.

    The policy's tempered distribution replaces argmax / beam search,
    encouraging diversity via entropy regularisation.
    """

    def __init__(self, lm: nn.Module, policy: SACPolicyHead, max_len: int = 32) -> None:
        self.lm = lm
        self.policy = policy
        self.max_len = max_len

    @torch.no_grad()
    def max_entropy_decode(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate up to max_len tokens using the policy's tempered distribution.

        Args:
            input_ids:   [B, T]  context token ids
            temperature: float   extra temperature scaling on top of policy alpha

        Returns:
            generated: [B, max_len]  newly generated token ids
        """
        B = input_ids.size(0)
        device = input_ids.device

        generated = torch.zeros(B, self.max_len, dtype=torch.long, device=device)
        current_ids = input_ids

        for step in range(self.max_len):
            # Get hidden states from the LM backbone
            # Supports models that return (logits, hidden) or a dict with
            # 'last_hidden_state', or raw logits — we try to extract a
            # [B, d_model]-shaped vector from the last token position.
            out = self.lm(current_ids)

            if isinstance(out, dict):
                hidden = out.get("last_hidden_state", out.get("logits"))
                if hidden is None:
                    raise KeyError("LM output dict has no 'last_hidden_state' or 'logits' key.")
            elif isinstance(out, (tuple, list)):
                # Convention: first element is logits/hidden states
                hidden = out[0]
            else:
                hidden = out

            # Take last token's hidden state: [B, d_model]
            hidden = hidden[:, -1, :]

            # Apply extra temperature scaling to logits via a scaled forward pass
            logits = self.policy.linear(hidden) / max(temperature, 1e-6)
            log_probs_t = F.log_softmax(logits, dim=-1)
            probs_t = log_probs_t.exp()

            dist = torch.distributions.Categorical(probs=probs_t)
            token = dist.sample()  # [B]
            generated[:, step] = token

            # Append sampled token to context
            current_ids = torch.cat([current_ids, token.unsqueeze(1)], dim=1)

        return generated


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    """Default hyperparameters for a small SAC-LLM experiment."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_capacity: int = 1000
    alpha_init: float = 1.0
