"""
Multi-Objective RLHF: Pareto optimization balancing multiple reward signals
(helpfulness, safety, harmlessness).

Pure PyTorch only — no external alignment libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MultiObjectiveReward
# ---------------------------------------------------------------------------


class MultiObjectiveReward(nn.Module):
    """Wraps a list of reward models, one per objective.

    Each reward model is expected to accept ``input_ids`` of shape ``[B, T]``
    and return a scalar reward per batch element (shape ``[B]``).  If a model
    returns a 2-D tensor ``[B, 1]`` it will be squeezed automatically.
    """

    def __init__(
        self,
        reward_models: list[nn.Module],
        objective_names: list[str],
    ) -> None:
        super().__init__()
        if len(reward_models) != len(objective_names):
            raise ValueError("reward_models and objective_names must have the same length")
        self.reward_models = nn.ModuleList(reward_models)
        self.objective_names = list(objective_names)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score *input_ids* with every reward model.

        Args:
            input_ids: ``[B, T]`` integer token ids.

        Returns:
            dict mapping objective name → reward tensor of shape ``[B]``.
        """
        rewards: dict[str, torch.Tensor] = {}
        for name, model in zip(self.objective_names, self.reward_models):
            r = model(input_ids)
            if r.dim() == 2:
                r = r.squeeze(-1)
            rewards[name] = r
        return rewards

    # ------------------------------------------------------------------
    def scalarize(
        self,
        rewards_dict: dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of objective rewards.

        Args:
            rewards_dict: output of :meth:`forward`.
            weights: ``[n_obj]`` weight vector (need not sum to 1).

        Returns:
            Tensor of shape ``[B]``.
        """
        reward_tensors = [rewards_dict[n] for n in self.objective_names]
        # Stack → [n_obj, B], then weighted sum → [B]
        stacked = torch.stack(reward_tensors, dim=0)  # [n_obj, B]
        w = weights.to(stacked.dtype).to(stacked.device)
        return (w.unsqueeze(-1) * stacked).sum(dim=0)  # [B]


# ---------------------------------------------------------------------------
# ParetoFront
# ---------------------------------------------------------------------------


class ParetoFront:
    """Maintains a set of Pareto-optimal solutions discovered during training."""

    def __init__(self) -> None:
        self.solutions: list[dict] = []
        # Each entry: {"weights": Tensor[n_obj], "rewards": Tensor[n_obj], "loss": float}

    # ------------------------------------------------------------------
    def update(
        self,
        weights: torch.Tensor,
        rewards: torch.Tensor,
        loss: float,
    ) -> None:
        """Add a candidate solution; rebuild the non-dominated set."""
        candidate = {
            "weights": weights.detach().clone(),
            "rewards": rewards.detach().clone(),
            "loss": float(loss),
        }
        self.solutions.append(candidate)
        # Recompute the full Pareto front
        self.solutions = self.get_pareto_front()

    # ------------------------------------------------------------------
    @staticmethod
    def is_pareto_dominant(r_new: torch.Tensor, r_old: torch.Tensor) -> bool:
        """Return True if *r_new* Pareto-dominates *r_old*.

        Dominance: r_new >= r_old in all objectives AND r_new > r_old in at
        least one objective.
        """
        all_ge = torch.all(r_new >= r_old).item()
        any_gt = torch.any(r_new > r_old).item()
        return bool(all_ge and any_gt)

    # ------------------------------------------------------------------
    def get_pareto_front(self) -> list[dict]:
        """Return only the non-dominated solutions from the current set."""
        n = len(self.solutions)
        if n == 0:
            return []

        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.is_pareto_dominant(
                    self.solutions[j]["rewards"],
                    self.solutions[i]["rewards"],
                ):
                    dominated[i] = True
                    break

        return [s for s, d in zip(self.solutions, dominated) if not d]

    # ------------------------------------------------------------------
    def hypervolume_indicator(self, reference_point: torch.Tensor) -> float:
        """Compute the 2-D hypervolume indicator.

        For n_obj == 2 this is exact (sort by first objective, sweep).
        For n_obj > 2 we fall back to a Monte-Carlo approximation.

        Args:
            reference_point: worst-case reference, shape ``[n_obj]``.

        Returns:
            Non-negative hypervolume as a Python float.
        """
        front = self.get_pareto_front()
        if not front:
            return 0.0

        n_obj = reference_point.shape[0]

        if n_obj == 2:
            return self._hypervolume_2d(front, reference_point)
        else:
            return self._hypervolume_nd_mc(front, reference_point, n_samples=10_000)

    # --- helpers ---

    @staticmethod
    def _hypervolume_2d(
        front: list[dict],
        ref: torch.Tensor,
    ) -> float:
        """Exact 2-D hypervolume via sweep."""
        # Extract (f1, f2) pairs; keep only points that dominate reference
        pts = []
        for sol in front:
            r = sol["rewards"]
            f1, f2 = float(r[0]), float(r[1])
            if f1 > float(ref[0]) and f2 > float(ref[1]):
                pts.append((f1, f2))

        if not pts:
            return 0.0

        # Sort descending by f1
        pts.sort(key=lambda x: x[0], reverse=True)

        hv = 0.0
        float(ref[0])
        float(ref[1])

        # Sweep from right to left in f1 dimension
        # We need to sort ascending in f1 and track maximum f2 seen
        sorted(pts, key=lambda x: x[0])
        float(ref[0])
        float(ref[1])

        # Standard sweep for 2-D: sort by f1 descending, track running max of f2
        pts_desc = sorted(pts, key=lambda x: x[0], reverse=True)
        running_max_f2 = float(ref[1])
        sweep_pts: list[tuple[float, float]] = []
        for f1, f2 in pts_desc:
            if f2 > running_max_f2:
                sweep_pts.append((f1, f2))
                running_max_f2 = f2

        if not sweep_pts:
            return 0.0

        # sweep_pts is sorted by f1 descending; compute area
        hv = 0.0
        ref_f1 = float(ref[0])
        ref_f2 = float(ref[1])

        for idx, (f1, f2) in enumerate(sweep_pts):
            width = f1 - ref_f1
            if idx + 1 < len(sweep_pts):
                next_f2 = sweep_pts[idx + 1][1]
            else:
                next_f2 = ref_f2
            height = f2 - next_f2
            hv += width * height
            ref_f1 = f1  # not used further after this; recalculate properly

        # Redo correctly: rectangles between consecutive Pareto points
        # Sort by f1 ascending
        sorted_pts = sorted(sweep_pts, key=lambda x: x[0])
        ref_f2_val = float(ref[1])
        ref_f1_val = float(ref[0])

        hv = 0.0
        for idx, (f1, f2) in enumerate(sorted_pts):
            if idx + 1 < len(sorted_pts):
                sorted_pts[idx + 1][0]
            else:
                pass  # last point; width contribution handled below
            width = f1 - ref_f1_val
            # height = f2 above the ref
            # We use the standard sweep: each point contributes a rectangle
            # whose width extends to the next point's f1 value
            # Rewrite using the canonical sweep:
            pass

        # Canonical 2-D hypervolume sweep (sort by f1 ascending):
        # For each point p_i with f1_i, f2_i (sorted by f1 asc, f2 desc):
        # contrib = (f1_i - f1_{i-1}) * (f2_i - ref_f2)
        # where f1_0 = ref_f1

        # Filter non-dominated in 2-D and sort
        pts2 = sorted(sweep_pts, key=lambda x: x[0])  # already non-dominated in f2
        hv = 0.0
        prev = ref_f1_val
        for f1, f2 in pts2:
            hv += (f1 - prev) * (f2 - ref_f2_val)
            prev = f1
        return max(hv, 0.0)

    @staticmethod
    def _hypervolume_nd_mc(
        front: list[dict],
        ref: torch.Tensor,
        n_samples: int = 10_000,
    ) -> float:
        """Monte-Carlo estimate of hypervolume for n_obj > 2."""
        if not front:
            return 0.0

        n_obj = ref.shape[0]
        # Determine bounding box
        all_r = torch.stack([s["rewards"] for s in front], dim=0)  # [M, n_obj]
        upper = all_r.max(dim=0).values  # [n_obj]

        # If any upper bound <= reference, hypervolume is 0
        if torch.any(upper <= ref).item():
            return 0.0

        # Volume of the bounding box
        box_vol = torch.prod(upper - ref).item()
        if box_vol <= 0:
            return 0.0

        # Sample uniformly
        samples = torch.rand(n_samples, n_obj) * (upper - ref).unsqueeze(0) + ref.unsqueeze(0)

        # A sample is dominated by the front if ∃ p in front s.t. p >= sample
        dominated_count = 0
        for s in range(n_samples):
            pt = samples[s]
            for sol in front:
                if torch.all(sol["rewards"] >= pt).item():
                    dominated_count += 1
                    break

        return box_vol * (dominated_count / n_samples)


# ---------------------------------------------------------------------------
# WeightSampler
# ---------------------------------------------------------------------------


class WeightSampler:
    """Utilities for sampling scalarization weight vectors."""

    # ------------------------------------------------------------------
    @staticmethod
    def uniform_simplex(n_obj: int, n_samples: int) -> torch.Tensor:
        """Sample uniformly from the probability simplex.

        Uses Dirichlet(1, 1, ...) which is equivalent to normalising
        i.i.d. Exponential(1) samples.

        Returns:
            ``[n_samples, n_obj]`` tensor with rows summing to 1.
        """
        # Exponential samples = -log(U)
        u = torch.rand(n_samples, n_obj).clamp(min=1e-10)
        exp_samples = -torch.log(u)
        return exp_samples / exp_samples.sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    @staticmethod
    def linear_scalarization_weights(n_obj: int, n_steps: int) -> torch.Tensor:
        """Evenly-spaced weight vectors over the simplex.

        For n_obj == 2 this is an exact grid.  For n_obj > 2 we use a
        recursive construction; if the exact number of grid points differs
        from ``n_steps`` we sample ``n_steps`` rows without replacement
        (with repetition if needed).

        Returns:
            ``[n_steps, n_obj]`` tensor with rows summing to 1.
        """
        if n_obj == 1:
            return torch.ones(n_steps, 1)

        if n_obj == 2:
            t = torch.linspace(0.0, 1.0, n_steps)
            return torch.stack([t, 1.0 - t], dim=-1)

        # General case: enumerate all (n_steps−1) grid points on the simplex
        # Use recursive enumeration with step size 1/(n_steps-1)
        grid_pts = WeightSampler._simplex_grid(n_obj, n_steps - 1)
        grid = torch.tensor(grid_pts, dtype=torch.float32)

        if grid.shape[0] == n_steps:
            return grid
        elif grid.shape[0] > n_steps:
            idx = torch.randperm(grid.shape[0])[:n_steps]
            return grid[idx]
        else:
            # Repeat with wrap-around
            repeats = (n_steps + grid.shape[0] - 1) // grid.shape[0]
            grid = grid.repeat(repeats, 1)[:n_steps]
            return grid

    @staticmethod
    def _simplex_grid(n_obj: int, divisions: int) -> list[list[float]]:
        """Recursively enumerate grid points on the simplex."""
        if n_obj == 1:
            return [[1.0]]
        result = []
        for k in range(divisions + 1):
            w1 = k / divisions
            sub = WeightSampler._simplex_grid(n_obj - 1, divisions - k)
            for rest in sub:
                result.append([w1] + [r * (1.0 - w1) for r in rest])
        # Normalize to handle floating point drift
        normed = []
        for row in result:
            s = sum(row)
            normed.append([v / s for v in row])
        return normed

    # ------------------------------------------------------------------
    @staticmethod
    def chebyshev_weights(
        ideal_point: torch.Tensor,
        utopia_gap: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive weights based on distance from the ideal (utopia) point.

        Chebyshev scalarization weights are set inversely proportional to
        the gap between the current solution and the ideal point on each
        objective, so that objectives further from ideal receive more weight.

        Args:
            ideal_point: ``[n_obj]`` best achievable value per objective.
            utopia_gap: ``[n_obj]`` current gap (ideal - current); non-negative.

        Returns:
            ``[n_obj]`` weight vector summing to 1.
        """
        # Avoid division by zero
        gap = utopia_gap.clamp(min=1e-8)
        # Inverse-gap weighting: larger gap → more weight
        w = 1.0 / gap
        return w / w.sum()


# ---------------------------------------------------------------------------
# MOPPOTrainer
# ---------------------------------------------------------------------------


class MOPPOTrainer:
    """Multi-Objective PPO-style trainer using REINFORCE with scalarized reward.

    The policy is assumed to be a language-model-like ``nn.Module`` that
    accepts ``input_ids: [B, T]`` and returns *logits* of shape ``[B, T, V]``.
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_model: MultiObjectiveReward,
        lr: float = 1e-4,
        n_obj: int = 3,
    ) -> None:
        self.policy = policy
        self.reward_model = reward_model
        self.n_obj = n_obj
        self.current_weights: torch.Tensor = torch.ones(n_obj) / n_obj
        self.pareto_front = ParetoFront()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self._weight_sampler = WeightSampler()

    # ------------------------------------------------------------------
    def policy_step(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Single REINFORCE step with scalarized reward.

        Args:
            input_ids: ``[B, T]``.
            weights: ``[n_obj]`` scalarization weights.

        Returns:
            ``(loss, rewards_dict)`` where loss is a scalar tensor.
        """
        # Forward through policy to get log-probs of the input sequence
        logits = self.policy(input_ids)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # Gather log-prob of each token
        # Shift: predict token t+1 from context up to t
        # Use tokens 1..T as targets, log-probs 0..T-1 as predictions
        T = input_ids.shape[1]
        if T > 1:
            target_ids = input_ids[:, 1:]  # [B, T-1]
            pred_log_probs = log_probs[:, :-1, :]  # [B, T-1, V]
        else:
            target_ids = input_ids  # [B, T]
            pred_log_probs = log_probs  # [B, T, V]

        # Gather: [B, T-1]
        gathered = pred_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        # Sum log-probs over time → sequence log-prob [B]
        seq_log_prob = gathered.sum(dim=-1)

        # Compute rewards (detached from policy graph)
        with torch.no_grad():
            rewards_dict = self.reward_model(input_ids)
            scalarized = self.reward_model.scalarize(rewards_dict, weights)

        # REINFORCE: loss = -E[log π(a|s) * R]
        loss = -(seq_log_prob * scalarized.detach()).mean()
        return loss, rewards_dict

    # ------------------------------------------------------------------
    def multi_objective_step(
        self,
        input_ids: torch.Tensor,
        n_weight_samples: int = 4,
    ) -> dict:
        """Sample multiple weight vectors, compute and average loss, update.

        Args:
            input_ids: ``[B, T]``.
            n_weight_samples: number of weight vectors to sample.

        Returns:
            dict with keys "total_loss", "pareto_size", "weight_samples".
        """
        weight_samples = self._weight_sampler.uniform_simplex(
            self.n_obj, n_weight_samples
        )  # [n_weight_samples, n_obj]

        total_loss = torch.tensor(0.0, requires_grad=True)
        all_rewards: list[torch.Tensor] = []

        for i in range(n_weight_samples):
            w = weight_samples[i]
            loss, rewards_dict = self.policy_step(input_ids, w)
            total_loss = total_loss + loss

            # Mean reward per objective across batch
            mean_r = torch.stack(
                [rewards_dict[name].mean() for name in self.reward_model.objective_names]
            )
            all_rewards.append(mean_r.detach())

            # Update Pareto front with this weight/reward combination
            self.pareto_front.update(w, mean_r, float(loss.item()))

        total_loss = total_loss / n_weight_samples

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update current weights to the last sampled weight vector
        self.current_weights = weight_samples[-1].detach()

        return {
            "total_loss": float(total_loss.item()),
            "pareto_size": len(self.pareto_front.solutions),
            "weight_samples": weight_samples,
        }

    # ------------------------------------------------------------------
    def get_best_weights(self, preference: torch.Tensor) -> torch.Tensor:
        """Return the Pareto-front solution whose reward vector best aligns
        with the given preference direction.

        Uses cosine similarity between normalized reward vectors and the
        preference direction.

        Args:
            preference: ``[n_obj]`` non-negative preference vector.

        Returns:
            ``[n_obj]`` weight tensor from the best-matching Pareto solution.
        """
        front = self.pareto_front.get_pareto_front()
        if not front:
            return torch.ones(self.n_obj) / self.n_obj

        pref_norm = F.normalize(preference.float().unsqueeze(0), dim=-1)  # [1, n_obj]
        best_sim = -float("inf")
        best_weights = front[0]["weights"]

        for sol in front:
            r = sol["rewards"].float().unsqueeze(0)  # [1, n_obj]
            r_norm = F.normalize(r, dim=-1)
            sim = (pref_norm * r_norm).sum().item()
            if sim > best_sim:
                best_sim = sim
                best_weights = sol["weights"]

        return best_weights


# ---------------------------------------------------------------------------
# RewardWeightScheduler
# ---------------------------------------------------------------------------


class RewardWeightScheduler:
    """Produce scalarization weight vectors according to a named schedule."""

    VALID_SCHEDULES = ("uniform", "alternating", "random")

    def __init__(self, n_obj: int, schedule: str = "uniform") -> None:
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(f"schedule must be one of {self.VALID_SCHEDULES}, got {schedule!r}")
        self.n_obj = n_obj
        self.schedule = schedule
        self._sampler = WeightSampler()

    # ------------------------------------------------------------------
    def step(self, t: int) -> torch.Tensor:
        """Return the weight vector at step *t*.

        Args:
            t: current training step (0-indexed).

        Returns:
            ``[n_obj]`` weight tensor summing to 1.
        """
        if self.schedule == "uniform":
            return torch.ones(self.n_obj) / self.n_obj

        elif self.schedule == "alternating":
            # Cycle: step t focuses entirely on objective (t % n_obj)
            w = torch.zeros(self.n_obj)
            w[t % self.n_obj] = 1.0
            return w

        else:  # "random"
            return self._sampler.uniform_simplex(self.n_obj, 1).squeeze(0)


# ---------------------------------------------------------------------------
# MOConfig
# ---------------------------------------------------------------------------


@dataclass
class MOConfig:
    """Configuration for multi-objective RLHF training."""

    n_objectives: int = 3
    objective_names: list[str] = field(
        default_factory=lambda: ["helpfulness", "safety", "harmlessness"]
    )
    lr: float = 1e-4
    n_weight_samples: int = 4
    schedule: str = "uniform"
