import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Any, Callable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        return self.pe[:, :seq_len, :].to(device=device)


class SelfTaughtReasoner(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        reasoning_max_len: int = 128,
        answer_max_len: int = 64,
        filter_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.reasoning_max_len = reasoning_max_len
        self.answer_max_len = answer_max_len
        self.filter_threshold = filter_threshold

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, n_layers)

        self.rationale_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)

        self.rationale_filter = nn.Linear(d_model, 1)

        self.hint_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.hint_proj = nn.Linear(d_model, d_model)
        self.hint_norm = nn.LayerNorm(d_model)

        self.frozen_backbone: Optional[nn.TransformerEncoder] = None

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        tok = self.token_embedding(input_ids)
        pos = self.pos_encoding(seq_len, input_ids.device)
        return tok + pos

    def _generate_autoregressive(
        self, prefix: torch.Tensor, head: nn.Module, max_len: int
    ) -> torch.Tensor:
        batch_size = prefix.shape[0]
        device = prefix.device
        generated = []

        for _ in range(max_len):
            current = torch.cat([prefix] + generated, dim=1) if generated else prefix
            embeds = self._embed(current)
            seq_len = current.shape[1]
            mask = self._causal_mask(seq_len).to(device)
            hidden = self.backbone(embeds, mask=mask, is_causal=True)
            logits = head(hidden[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if (next_token == self.eos_token_id).all():
                break

        return torch.cat(generated, dim=1) if generated else torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embeds = self._embed(input_ids)
        seq_len = input_ids.shape[1]
        mask = self._causal_mask(seq_len).to(input_ids.device)
        hidden = self.backbone(embeds, mask=mask, is_causal=True)
        logits = self.rationale_head(hidden)

        result: Dict[str, torch.Tensor] = {'logits': logits, 'hidden_states': hidden}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=self.pad_token_id,
            )
            result['loss'] = loss
        return result

    def generate_reasoning(self, question: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            reasoning = self._generate_autoregressive(question, self.rationale_head, self.reasoning_max_len)
            combined = torch.cat([question, reasoning], dim=1)
            answer = self._generate_autoregressive(combined, self.answer_head, self.answer_max_len)
        self.train()
        return reasoning, answer

    def filter_correct(
        self,
        reasoning_traces: torch.Tensor,
        answers: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            combined = reasoning_traces
            if combined.dim() == 1:
                combined = combined.unsqueeze(0)
            embeds = self._embed(combined)
            seq_len = combined.shape[1]
            mask = self._causal_mask(seq_len).to(combined.device)
            hidden = self.backbone(embeds, mask=mask, is_causal=True)
            last_hidden = hidden[:, -1, :]
            scores = torch.sigmoid(self.rationale_filter(last_hidden)).squeeze(-1)

        answer_correct = (answers == ground_truth).all(dim=1) if answers.dim() > 1 else (answers == ground_truth).all()
        filter_mask = (scores > self.filter_threshold) & answer_correct
        return filter_mask

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(
            input_ids=batch['input_ids'],
            labels=batch.get('labels'),
        )

    def self_train_iterations(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        n_rounds: int = 3,
    ) -> List[float]:
        avg_losses = []

        if self.frozen_backbone is None:
            frozen_layer = nn.TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff, self.dropout, batch_first=True
            )
            self.frozen_backbone = nn.TransformerEncoder(frozen_layer, self.n_layers)
            for p in self.frozen_backbone.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        for round_idx in range(n_rounds):
            self.frozen_backbone.load_state_dict(self.backbone.state_dict())
            round_losses = []

            for question_tokens, answer_tokens in dataset:
                if question_tokens.dim() == 1:
                    question_tokens = question_tokens.unsqueeze(0)
                if answer_tokens.dim() == 1:
                    answer_tokens = answer_tokens.unsqueeze(0)

                reasoning, answer = self.generate_reasoning(question_tokens)

                correct_mask = self.filter_correct(reasoning, answer, answer_tokens)

                if not correct_mask.any():
                    continue

                full_input = torch.cat([question_tokens, reasoning], dim=1)
                reasoning_labels = torch.cat([
                    torch.full_like(question_tokens, -100),
                    reasoning,
                ], dim=1)

                with torch.no_grad():
                    base_embeds = self._embed(full_input)
                    base_seq_len = full_input.shape[1]
                    base_mask = self._causal_mask(base_seq_len).to(full_input.device)
                    base_hidden = self.frozen_backbone(base_embeds, mask=base_mask, is_causal=True)
                    base_logits = self.rationale_head(base_hidden)
                    base_log_probs = F.log_softmax(base_logits, dim=-1)
                    base_ans_log_probs = base_log_probs.gather(-1, reasoning.unsqueeze(-1)).squeeze(-1)
                    base_score = base_ans_log_probs.sum(dim=-1)

                cur_embeds = self._embed(full_input)
                cur_hidden = self.backbone(cur_embeds, mask=base_mask, is_causal=True)
                cur_logits = self.rationale_head(cur_hidden)

                cur_log_probs = F.log_softmax(cur_logits, dim=-1)
                cur_ans_log_probs = cur_log_probs.gather(-1, reasoning.unsqueeze(-1)).squeeze(-1)
                cur_score = cur_ans_log_probs.sum(dim=-1)

                utility_weight = torch.exp(cur_score - base_score)
                utility_weight = torch.clamp(utility_weight, 0.1, 5.0)

                valid_indices = correct_mask.nonzero(as_tuple=True)[0]
                for idx in valid_indices:
                    loss = F.cross_entropy(
                        cur_logits[idx:idx+1].view(-1, self.vocab_size),
                        reasoning_labels[idx:idx+1].view(-1),
                        ignore_index=-100,
                    )
                    loss = loss * utility_weight[idx]
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    round_losses.append(loss.item())

            avg_loss = sum(round_losses) / max(len(round_losses), 1)
            avg_losses.append(avg_loss)

        return avg_losses


class TreeOfThoughtSearcher(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 10000,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        n_encoder_layers: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        self.value_heuristic = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.proposal_generator = nn.Linear(d_model, vocab_size)

        self.backtrack_trigger = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        tok = self.token_embedding(input_ids)
        pos = self.pos_encoding(seq_len, input_ids.device)
        return tok + pos

    def _evaluate_state(self, state_ids: torch.Tensor) -> torch.Tensor:
        embeds = self._embed(state_ids)
        pooled = embeds.mean(dim=1)
        return self.value_heuristic(pooled).squeeze(-1)

    def _generate_candidates(
        self, state_ids: torch.Tensor, n_candidates: int
    ) -> torch.Tensor:
        device = state_ids.device
        embeds = self._embed(state_ids)
        with torch.no_grad():
            hidden = self.backbone(embeds)
        last_hidden = hidden[:, -1, :]
        logits = self.proposal_generator(last_hidden)

        tokens = []
        for _ in range(n_candidates):
            probs = F.softmax(logits / 0.8, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            tokens.append(token)
        return torch.cat(tokens, dim=1)

    def forward(self, problem: torch.Tensor) -> Dict[str, Any]:
        path, scores = self.bfs(problem)
        return {'best_path': path, 'value_scores': scores}

    def bfs(
        self,
        problem: torch.Tensor,
        n_branches: int = 5,
        depth: int = 3,
    ) -> Tuple[torch.Tensor, List[float]]:
        if problem.dim() == 1:
            problem = problem.unsqueeze(0)

        paths: List[Tuple[torch.Tensor, float]] = [(problem.clone(), 0.0)]

        for d in range(depth):
            new_paths: List[Tuple[torch.Tensor, float]] = []

            for state_ids, parent_score in paths:
                candidate_tokens = self._generate_candidates(state_ids, n_branches)

                for k in range(n_branches):
                    new_token = candidate_tokens[:, k:k+1]
                    new_state = torch.cat([state_ids, new_token], dim=1)
                    value = self._evaluate_state(new_state).mean().item()
                    new_paths.append((new_state, value))

            if not new_paths:
                break
            new_paths.sort(key=lambda x: x[1], reverse=True)
            paths = new_paths

        best_path = paths[0][0] if paths else problem

        path_values: List[float] = []
        for i in range(1, best_path.shape[1] + 1):
            segment = best_path[:, :i]
            val = self._evaluate_state(segment).item()
            path_values.append(val)

        return best_path, path_values

    def dfs(
        self,
        problem: torch.Tensor,
        max_depth: int = 5,
    ) -> Tuple[torch.Tensor, List[float]]:
        if problem.dim() == 1:
            problem = problem.unsqueeze(0)

        best_path = problem.clone()
        best_score = torch.tensor(float('-inf'))

        def backtrack(state_ids: torch.Tensor, depth_so_far: int):
            nonlocal best_path, best_score

            if depth_so_far >= max_depth:
                current_score = self._evaluate_state(state_ids)
                if current_score > best_score:
                    best_score = current_score
                    best_path = state_ids.clone()
                return

            if depth_so_far > 0:
                embeds = self._embed(state_ids)
                pooled = embeds.mean(dim=1)
                backtrack_logit = self.backtrack_trigger(pooled)
                backtrack_prob = torch.sigmoid(backtrack_logit).mean().item()
                if backtrack_prob > 0.7 and depth_so_far > 1:
                    return

            current_value = self._evaluate_state(state_ids).mean().item()
            if current_value < -2.0 and depth_so_far > 1:
                return

            candidate_tokens = self._generate_candidates(state_ids, 3)

            candidates: List[Tuple[torch.Tensor, float]] = []
            for k in range(3):
                new_token = candidate_tokens[:, k:k+1]
                new_state = torch.cat([state_ids, new_token], dim=1)
                new_value = self._evaluate_state(new_state).mean().item()
                candidates.append((new_state, new_value))

            candidates.sort(key=lambda x: x[1], reverse=True)

            for new_state, _ in candidates:
                backtrack(new_state, depth_so_far + 1)

        backtrack(problem, 0)

        path_values: List[float] = []
        for i in range(1, best_path.shape[1] + 1):
            segment = best_path[:, :i]
            val = self._evaluate_state(segment).item()
            path_values.append(val)

        return best_path, path_values


class QuietStarEngine(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_layers_main: int = 6,
        n_heads_main: int = 8,
        n_layers_thought: int = 4,
        d_ff_thought: int = 2048,
        d_ff_main: int = 2048,
        max_seq_len: int = 1024,
        max_thought_steps: int = 64,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_thought_steps = max_thought_steps

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.start_thought = nn.Parameter(torch.randn(d_model) * 0.02)
        self.end_thought = nn.Parameter(torch.randn(d_model) * 0.02)

        self.thought_gate = nn.Linear(d_model, 2)

        self.think_pos_enc = nn.Embedding(max_thought_steps, d_model)

        main_layer = nn.TransformerEncoderLayer(
            d_model, n_heads_main, d_ff_main, dropout, batch_first=True
        )
        self.main_backbone = nn.TransformerEncoder(main_layer, n_layers_main)

        self.lm_head = nn.Linear(d_model, vocab_size)

        thought_layer = nn.TransformerEncoderLayer(
            d_model, n_heads_main, d_ff_thought, dropout, batch_first=True
        )
        self.thought_generator = nn.TransformerEncoder(thought_layer, n_layers_thought)

        self.thought_value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        tok = self.token_embedding(input_ids)
        pos = self.pos_encoding(seq_len, input_ids.device)
        return tok + pos

    def _generate_thoughts(
        self, context_embed: torch.Tensor, n_think_tokens: int
    ) -> torch.Tensor:
        batch_size = context_embed.shape[0]
        device = context_embed.device

        thought_embeds = context_embed.expand(-1, n_think_tokens, -1).contiguous()
        think_pos = self.think_pos_enc(
            torch.arange(n_think_tokens, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        thought_embeds = thought_embeds + think_pos
        return self.thought_generator(thought_embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_think_tokens: int = 4,
        return_thought_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        base_embeds = self._embed(input_ids)

        thought_decisions = []
        thought_generated = []

        for pos in range(seq_len):
            pos_embed = base_embeds[:, pos:pos+1, :]
            gate_logits = self.thought_gate(pos_embed.mean(dim=1))
            gate_probs = F.gumbel_softmax(gate_logits, tau=1.0, hard=False)
            should_think = (gate_probs[:, 1] > 0.5)
            thought_decisions.append(should_think)

            if should_think.any():
                think_embeds = self._generate_thoughts(pos_embed, n_think_tokens)
                thought_generated.append(think_embeds)
            else:
                thought_generated.append(None)

        final_embeds_list: List[torch.Tensor] = []
        thought_positions: List[int] = []

        insert_pos = 0
        for pos in range(seq_len):
            final_embeds_list.append(base_embeds[:, pos:pos+1, :])
            insert_pos += 1

            should_think = thought_decisions[pos]
            if should_think.any() and thought_generated[pos] is not None:
                start_emb = self.start_thought.view(1, 1, -1).expand(batch_size, -1, -1)
                final_embeds_list.append(start_emb)
                thought_positions.append(insert_pos)
                insert_pos += 1

                think_embeds = thought_generated[pos]
                tpos = self.think_pos_enc(
                    torch.arange(n_think_tokens, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                final_embeds_list.append(think_embeds + tpos)
                for _ in range(n_think_tokens):
                    thought_positions.append(insert_pos)
                    insert_pos += 1

                end_emb = self.end_thought.view(1, 1, -1).expand(batch_size, -1, -1)
                final_embeds_list.append(end_emb)
                thought_positions.append(insert_pos)
                insert_pos += 1

        final_embeds = torch.cat(final_embeds_list, dim=1)
        final_seq_len = final_embeds.shape[1]

        mask = self._causal_mask(final_seq_len).to(device)
        hidden = self.main_backbone(final_embeds, mask=mask, is_causal=True)
        logits = self.lm_head(hidden)

        result: Dict[str, torch.Tensor] = {'logits': logits}

        if return_thought_info:
            thought_mask = torch.zeros(final_seq_len, dtype=torch.bool, device=device)
            for tp in thought_positions:
                if tp < final_seq_len:
                    thought_mask[tp] = True
            result['thought_mask'] = thought_mask

        return result

    def generate_with_thought(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 64,
        n_think_tokens: int = 4,
    ) -> torch.Tensor:
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        batch_size = prompt.shape[0]
        device = prompt.device
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            embeds = self._embed(generated)
            seq_len = generated.shape[1]

            last_embed = embeds[:, -1:, :]
            gate_logits = self.thought_gate(last_embed.mean(dim=1))
            gate_probs = F.softmax(gate_logits, dim=-1)
            should_think = gate_probs[:, 1] > 0.5

            if should_think.any():
                think_embeds = self._generate_thoughts(last_embed, n_think_tokens)
                start_emb = self.start_thought.view(1, 1, -1).expand(batch_size, -1, -1)
                end_emb = self.end_thought.view(1, 1, -1).expand(batch_size, -1, -1)
                tpos = self.think_pos_enc(
                    torch.arange(n_think_tokens, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                full_embeds = torch.cat([
                    embeds, start_emb, think_embeds + tpos, end_emb
                ], dim=1)
            else:
                full_embeds = embeds

            full_seq_len = full_embeds.shape[1]
            mask = self._causal_mask(full_seq_len).to(device)
            hidden = self.main_backbone(full_embeds, mask=mask, is_causal=True)
            logits = self.lm_head(hidden[:, -1, :])

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == 1).all():
                break

        return generated


class SkillLibraryVoyager(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_skill: int = 256,
        max_skills: int = 1000,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_skill = d_skill
        self.max_skills = max_skills

        self.skill_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True),
            n_encoder_layers,
        )

        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True),
            n_encoder_layers,
        )

        self.skill_proj = nn.Linear(d_model, d_skill)
        self.obs_proj = nn.Linear(d_model, d_skill)

        self.skill_discovery_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_skill),
        )

        self.self_verification_head = nn.Sequential(
            nn.Linear(d_skill + d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.skill_to_model = nn.Linear(d_skill, d_model)

        action_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.action_decoder = nn.TransformerDecoder(action_layer, n_encoder_layers)
        self.action_head = nn.Linear(d_model, d_model)

        self.skill_names: List[str] = []
        self.skill_programs: Dict[str, str] = {}
        self.skill_embeddings: List[torch.Tensor] = []
        self.verification_fns: Dict[str, Callable] = {}

        self.register_buffer('skill_matrix', torch.zeros(max_skills, d_skill))
        self.n_skills: int = 0

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def encode_skill(self, program_tokens: torch.Tensor) -> torch.Tensor:
        embeds = self.skill_encoder(program_tokens)
        pooled = embeds.mean(dim=1)
        return self.skill_proj(pooled)

    def add_skill(
        self,
        name: str,
        program: str,
        embedding: torch.Tensor,
        verification_fn: Optional[Callable] = None,
    ) -> None:
        if self.n_skills >= self.max_skills:
            return

        emb = embedding.detach().cpu().flatten()
        if emb.shape[0] > self.d_skill:
            emb = emb[:self.d_skill]
        elif emb.shape[0] < self.d_skill:
            emb = torch.cat([emb, torch.zeros(self.d_skill - emb.shape[0])])

        self.skill_names.append(name)
        self.skill_programs[name] = program
        self.skill_embeddings.append(emb.clone())
        if verification_fn is not None:
            self.verification_fns[name] = verification_fn

        self.skill_matrix[self.n_skills] = emb
        self.n_skills += 1

    def retrieve(
        self,
        task_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        if self.n_skills == 0:
            return []

        valid_matrix = self.skill_matrix[:self.n_skills].to(task_embedding.device)

        te = task_embedding.flatten()
        if te.shape[0] > self.d_skill:
            te = te[:self.d_skill]

        similarities = F.cosine_similarity(
            te.unsqueeze(0).unsqueeze(0),
            valid_matrix.unsqueeze(0),
            dim=-1,
        ).squeeze(0)

        topk = min(top_k, self.n_skills)
        top_scores, top_indices = similarities.topk(topk, dim=-1)

        results = []
        for score, idx in zip(top_scores, top_indices):
            idx = idx.item()
            results.append((
                self.skill_names[idx],
                self.skill_embeddings[idx].to(task_embedding.device),
                score.item(),
            ))

        return results

    def execute(self, skill_name: str, context: torch.Tensor) -> torch.Tensor:
        if skill_name not in self.skill_names:
            raise ValueError(f"Skill '{skill_name}' not found")

        idx = self.skill_names.index(skill_name)
        skill_embed = self.skill_embeddings[idx].to(context.device)

        skill_embed_proj = self.skill_to_model(skill_embed)
        skill_embed_expanded = skill_embed_proj.view(1, 1, -1).expand(
            context.shape[0], 1, -1
        )

        context_seq = context.unsqueeze(1) if context.dim() == 2 else context
        tgt_mask = self._causal_mask(context_seq.shape[1]).to(context.device)

        output = self.action_decoder(context_seq, skill_embed_expanded, tgt_mask=tgt_mask)
        action = self.action_head(output)

        if skill_name in self.verification_fns:
            verified = self.verification_fns[skill_name](context, action)
            if isinstance(verified, torch.Tensor):
                verified = verified.item()
            if not verified:
                action = action * 0.0

        return action

    def discover_skill(self, trajectory_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = trajectory_embeddings.mean(dim=1)
        return self.skill_discovery_head(pooled)

    def verify_skill_application(
        self,
        skill_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = skill_embedding.shape[0]
        se = skill_embedding.view(batch_size, -1)
        ce = context_embedding.view(batch_size, -1)

        if se.shape[-1] != self.d_skill:
            se = F.adaptive_avg_pool1d(se.unsqueeze(0), self.d_skill).squeeze(0)

        combined = torch.cat([se, ce], dim=-1)
        logit = self.self_verification_head(combined)
        return torch.sigmoid(logit).squeeze(-1)

    def forward(
        self,
        task_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        return self.retrieve(task_embedding, top_k)
