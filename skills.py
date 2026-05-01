import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SkillEmbedding(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128, max_skills: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.skill_dim = skill_dim
        self.max_skills = max_skills
        self.embeddings = nn.Parameter(torch.randn(max_skills, skill_dim) * 0.02)
        self.name_proj = nn.Linear(64, skill_dim)
        self.activation = nn.GELU()

    def forward(self, skill_ids: torch.Tensor | None = None) -> torch.Tensor:
        if skill_ids is not None:
            if skill_ids.numel() > 0 and (skill_ids.max() >= self.max_skills or skill_ids.min() < 0):
                raise ValueError(f"skill_ids out of range [0, {self.max_skills}), got max={skill_ids.max().item()}, min={skill_ids.min().item()}")
            return self.embeddings[skill_ids]
        return self.embeddings

    def get_named_skills(self, skill_names: torch.Tensor) -> torch.Tensor:
        return self.name_proj(skill_names)

    def add_skill(self, skill_embed: torch.Tensor, idx: int | None = None) -> int:
        with torch.no_grad():
            if idx is None:
                unused = (self.embeddings.abs().sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
                if len(unused) > 0:
                    idx = unused[0].item()
                else:
                    idx = torch.randint(0, self.max_skills, (1,)).item()
            self.embeddings[idx] = skill_embed.detach().squeeze()
        return idx


class SkillRetriever(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128, n_top_k: int = 8):
        super().__init__()
        self.n_top_k = n_top_k
        self.skill_dim = skill_dim
        self.query_proj = nn.Linear(d_model, skill_dim)
        self.key_proj = nn.Linear(skill_dim, skill_dim)

    def forward(self, h: torch.Tensor, skill_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(h[:, -1])
        keys = self.key_proj(skill_embeds)
        scores = query @ keys.transpose(-2, -1) / math.sqrt(self.skill_dim)
        top_scores, top_idx = scores.topk(self.n_top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)
        retrieved = (weights.unsqueeze(-1) * skill_embeds[top_idx]).sum(dim=-2)
        return retrieved, top_idx


class SkillController(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128):
        super().__init__()
        self.controller = nn.Linear(d_model + skill_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        skill_expanded = skill.unsqueeze(1).expand(-1, t, -1)
        control = self.controller(torch.cat([h, skill_expanded], dim=-1))
        return h + torch.tanh(self.gate) * control


class SkillExecutionAdapter(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128, n_heads: int = 16):
        super().__init__()
        self.skill_to_kv = nn.Linear(skill_dim, 2 * d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, h: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        kv = self.skill_to_kv(skill)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1).expand(-1, t, -1).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(1).expand(-1, t, -1).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class SkillAcquisition(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, skill_dim * 2),
            nn.ReLU(),
            nn.Linear(skill_dim * 2, skill_dim),
        )
        self.momentum = 0.99

    def extract_skill(self, trajectories: torch.Tensor) -> torch.Tensor:
        pooled = trajectories.mean(dim=1)
        return self.encoder(pooled)

    def update_embedding(self, old_embed: torch.Tensor, new_skill: torch.Tensor) -> torch.Tensor:
        return self.momentum * old_embed + (1 - self.momentum) * new_skill


class SkillRegistry(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128, max_skills: int = 4096,
                 n_top_k: int = 8):
        super().__init__()
        self.skill_dim = skill_dim
        self.max_skills = max_skills
        self.n_top_k = n_top_k

        self.embedding = SkillEmbedding(d_model, skill_dim, max_skills)
        self.retriever = SkillRetriever(d_model, skill_dim, n_top_k)
        self.controller = SkillController(d_model, skill_dim)
        self.adapter = SkillExecutionAdapter(d_model, skill_dim)
        self.acquisition = SkillAcquisition(d_model, skill_dim)

        self.register_buffer('skill_success_rate', torch.ones(max_skills) * 0.5)
        self.register_buffer('skill_usage_count', torch.zeros(max_skills, dtype=torch.long))

    def forward(self, h: torch.Tensor, skill_ids: torch.Tensor | None = None,
                learn: bool = False) -> tuple:
        top_idx = None
        if skill_ids is not None:
            skill = self.embedding(skill_ids)
        else:
            skill, top_idx = self.retriever(h, self.embedding())
            self.skill_usage_count[top_idx] += 1

        h_controlled = self.controller(h, skill)
        h_executed = self.adapter(h_controlled, skill)

        if learn:
            return h_executed, skill, top_idx
        return h_executed, skill

    def learn_skill(self, trajectories: torch.Tensor, success: float) -> int:
        new_skill = self.acquisition.extract_skill(trajectories)
        idx = self.embedding.add_skill(new_skill)
        self.skill_success_rate[idx] = success
        self.skill_usage_count[idx] = 1
        return idx

    def update_skill(self, idx: int, trajectory: torch.Tensor, success: float):
        new = self.acquisition.extract_skill(trajectory)
        updated = self.acquisition.update_embedding(self.embedding.embeddings[idx], new)
        with torch.no_grad():
            self.embedding.embeddings.data[idx] = updated.detach()
        self.skill_success_rate[idx] = (
            self.skill_success_rate[idx] * 0.9 + success * 0.1
        )
        self.skill_usage_count[idx] += 1

    def get_top_skills(self, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        success_weighted = self.skill_success_rate * torch.log(
            self.skill_usage_count.float() + 1
        )
        values, indices = success_weighted.topk(k)
        return indices, values


class SkillLibrary(nn.Module):
    def __init__(self, d_model: int, skill_dim: int = 128, max_skills: int = 4096):
        super().__init__()
        self.registry = SkillRegistry(d_model, skill_dim, max_skills)
        self.task_embedder = nn.Linear(d_model, skill_dim)
        self.composition_head = nn.Linear(skill_dim * 2, skill_dim)

    def compose_skills(self, skill_a: torch.Tensor, skill_b: torch.Tensor) -> torch.Tensor:
        return self.composition_head(torch.cat([skill_a, skill_b], dim=-1))

    def forward(self, h: torch.Tensor, task_desc: torch.Tensor | None = None) -> tuple:
        if task_desc is not None:
            task_skill = self.task_embedder(task_desc[:, -1])
            matched, idx = self.registry.retriever(
                h, self.registry.embedding()
            )
            composed = self.compose_skills(matched, task_skill)
            h_out, _ = self.registry.controller(h, composed)
            h_out = self.registry.adapter(h_out, composed)
            return h_out, idx
        return self.registry(h)
