import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
        chosen_reward = self.compute_reward(chosen_logps, ref_chosen_logps, beta)
        rejected_reward = self.compute_reward(rejected_logps, ref_rejected_logps, beta)
        logits = chosen_reward - rejected_reward
        loss = -F.logsigmoid(logits).mean()
        accuracy = (chosen_reward > rejected_reward).float().mean().detach()
        return loss, accuracy

    def compute_reward(self, logps, ref_logps, beta=0.1):
        return beta * (logps - ref_logps)


class CritiqueHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class ConstitutionalUpdateRule:
    def generate_preference_pair(self, original, revised):
        if isinstance(original, torch.Tensor) and isinstance(revised, torch.Tensor):
            return torch.arange(len(revised)), torch.arange(len(original))
        return (0, 1)


class ConstitutionalClassifier(nn.Module):
    def __init__(self, d_model, n_principles=16):
        super().__init__()
        self.n_principles = n_principles
        self.principle_embeddings = nn.Parameter(torch.randn(n_principles, d_model))
        self.critique_head = CritiqueHead(d_model)
        self.update_rule = ConstitutionalUpdateRule()

    def forward(self, text_embedding):
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        text_norm = F.normalize(text_embedding, dim=-1)
        principle_norm = F.normalize(self.principle_embeddings, dim=-1)
        violation_scores = torch.mm(text_norm, principle_norm.t())
        violation_types = []
        for i in range(text_embedding.size(0)):
            violated = torch.where(violation_scores[i] > 0.5)[0]
            violation_types.append(violated.tolist())
        return violation_scores, violation_types

    def critique(self, text_embedding):
        violations, _ = self.forward(text_embedding)
        suggestions = self.critique_head(text_embedding)
        return violations, suggestions


class EvidenceRetriever(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def forward(self, query_embedding, evidence_keys):
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        q = F.normalize(self.query_proj(query_embedding), dim=-1)
        k = F.normalize(self.key_proj(evidence_keys), dim=-1)
        scores = torch.mm(q, k.t())
        return scores


class ClaimVerifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, claim_embedding, evidence_embedding):
        if claim_embedding.dim() == 1:
            claim_embedding = claim_embedding.unsqueeze(0)
        if evidence_embedding.dim() == 2:
            evidence_embedding = evidence_embedding.unsqueeze(0)
        n_ev = evidence_embedding.size(1)
        claim_expanded = claim_embedding.unsqueeze(1).expand(-1, n_ev, -1)
        evidence_embedding = evidence_embedding.expand(claim_embedding.shape[0], -1, -1)
        pair = torch.cat([claim_expanded, evidence_embedding], dim=-1)
        scores = self.net(pair).squeeze(-1)
        return scores


class RevisionGenerator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, claim_embedding, evidence_embedding):
        combined = torch.cat([claim_embedding, evidence_embedding], dim=-1)
        return self.net(combined)


class RARRRetriever(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.evidence_retriever = EvidenceRetriever(d_model)
        self.claim_verifier = ClaimVerifier(d_model)
        self.revision_generator = RevisionGenerator(d_model)
        self.evidence_store = []

    def add_evidence(self, key, value, source=""):
        self.evidence_store.append((key, value, source))

    def forward(self, claim_embedding):
        if not self.evidence_store:
            return [], torch.tensor(0.0, device=claim_embedding.device)
        keys = torch.stack([e[0] for e in self.evidence_store])
        values = torch.stack([e[1] for e in self.evidence_store])
        scores = self.evidence_retriever(claim_embedding, keys)
        top_idx = scores.argsort(descending=True)[0, :min(5, len(self.evidence_store))]
        retrieved_values = values[top_idx]
        consistency = self.claim_verifier(claim_embedding, retrieved_values)
        confidence = consistency.max().detach()
        sources = [self.evidence_store[i.item()][2] for i in top_idx]
        evidence = list(zip(retrieved_values, scores[0, top_idx].detach(), consistency.detach(), sources))
        return evidence, confidence

    def verify_and_revise(self, original_embedding, query_embedding):
        evidence, confidence = self.forward(original_embedding)
        changes_made = confidence < 0.5
        if changes_made and evidence:
            best_evidence = evidence[0][0]
            revised = self.revision_generator(original_embedding, best_evidence)
            return revised, True
        return original_embedding, False


class StepEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, bidirectional=True, batch_first=True)

    def forward(self, step_hiddens):
        outputs, _ = self.gru(step_hiddens)
        return outputs


class RewardHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ProcessRewardModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.step_encoder = StepEncoder(d_model)
        self.reward_head = RewardHead(d_model)

    def forward(self, step_hiddens):
        encoded = self.step_encoder(step_hiddens)
        rewards = self.reward_head(encoded)
        return rewards

    def compute_outcome_reward(self, step_rewards, discount=0.95):
        steps = step_rewards.size(-1)
        discounts = torch.pow(
            discount,
            torch.arange(steps - 1, -1, -1, device=step_rewards.device).float(),
        )
        return (step_rewards * discounts).sum(dim=-1)

    def compute_advantage(self, step_rewards, values, gamma=1.0, lam=0.95):
        batch, steps = step_rewards.shape
        next_values = torch.cat(
            [values[:, 1:], torch.zeros(batch, 1, device=values.device)], dim=1
        )
        deltas = step_rewards + gamma * next_values - values
        advantages = torch.zeros_like(deltas)
        running = torch.zeros(batch, device=deltas.device)
        for t in reversed(range(steps)):
            running = deltas[:, t] + gamma * lam * running
            advantages[:, t] = running
        return advantages
