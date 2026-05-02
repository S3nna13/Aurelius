import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import Counter
import logging
logger = logging.getLogger(__name__)


def _forward_model(model, input_ids, device="cpu"):
    input_ids = input_ids.to(device)
    try:
        result = model(input_ids, use_brain=False, return_agent_state=False)
        if isinstance(result, dict):
            return result.get("logits", result)
        return result
    except TypeError:
        pass
    try:
        result = model(input_ids, return_mem_state=False)
        if isinstance(result, tuple):
            return result[0]
        return result
    except TypeError:
        pass
    result = model(input_ids)
    if isinstance(result, dict):
        return result.get("logits", result)
    if isinstance(result, tuple):
        return result[0]
    return result


class CrossSessionRecallBenchmark:
    def __init__(self, model_config=None, device="cpu"):
        self.model_config = model_config or {}
        self.device = device

    def run(self, model, n_trials=5):
        model.eval()
        d_mem = model.blocks[0].memory.d_mem
        d_model = model.blocks[0].memory.d_model

        n_kv_pairs = 16
        n_sessions_list = [0, 1, 2, 4, 8]

        all_cosine = {n: [] for n in n_sessions_list}
        all_accuracy = {n: [] for n in n_sessions_list}

        with torch.no_grad():
            for trial in range(n_trials):
                original_mems = {
                    i: block.memory.lts.mem.data.clone()
                    for i, block in enumerate(model.blocks)
                }

                keys = torch.randn(1, n_kv_pairs, d_model, device=self.device)
                values = torch.randn(1, n_kv_pairs, d_model, device=self.device)
                importance = torch.ones(1, n_kv_pairs, 1, device=self.device) * 0.9

                target_v = model.blocks[0].memory.v_proj(values)

                for block in model.blocks:
                    k = block.memory.k_proj(keys)
                    v = block.memory.v_proj(values)
                    for _ in range(100):
                        block.memory.lts.write(k, v, importance)

                injected_mems = {
                    i: block.memory.lts.mem.data.clone()
                    for i, block in enumerate(model.blocks)
                }

                for n_sessions in n_sessions_list:
                    for i, block in enumerate(model.blocks):
                        block.memory.lts.mem.data = injected_mems[i].clone()

                    for _ in range(n_sessions):
                        for block in model.blocks:
                            rand_h = F.layer_norm(
                                torch.randn(1, 8, d_model, device=self.device),
                                [d_model],
                            )
                            k = block.memory.k_proj(rand_h)
                            v = block.memory.v_proj(rand_h)
                            surprise, lam = block.memory.surprise(rand_h)
                            block.memory.lts.write(k, v, lam)

                    query = model.blocks[0].memory.q_proj(keys)
                    readout = model.blocks[0].memory.lts.read(query)

                    cos_sim = F.cosine_similarity(
                        readout.flatten().unsqueeze(0),
                        target_v.flatten().unsqueeze(0),
                    ).item()
                    all_cosine[n_sessions].append(cos_sim)

                    read_norm = F.normalize(readout, dim=-1)
                    tgt_norm = F.normalize(target_v, dim=-1)
                    sim = (read_norm * tgt_norm).sum(dim=-1)
                    accuracy = (sim > 0.3).float().mean().item()
                    all_accuracy[n_sessions].append(accuracy)

                for i, block in enumerate(model.blocks):
                    block.memory.lts.mem.data = original_mems[i]

        metrics = {}
        for n in n_sessions_list:
            c = all_cosine[n]
            a = all_accuracy[n]
            metrics[f"cosine_sim_{n}_sessions"] = sum(c) / len(c) if c else 0.0
            metrics[f"recall_accuracy_{n}_sessions"] = (
                sum(a) / len(a) if a else 0.0
            )

        avg_cosines = [
            sum(all_cosine[n]) / len(all_cosine[n]) if all_cosine[n] else 0.0
            for n in n_sessions_list
        ]
        metrics["mean_recall_accuracy"] = sum(avg_cosines) / len(avg_cosines)
        metrics["recall_decay_rate"] = avg_cosines[0] - avg_cosines[-1]

        return metrics


class SurprisePrioritizationBenchmark:
    def __init__(self, model_config=None, device="cpu"):
        self.model_config = model_config or {}
        self.device = device

    def run(self, model, n_trials=5):
        model.eval()
        d_model = model.blocks[0].memory.d_model

        all_novel_surprise = []
        all_familiar_surprise = []
        all_novel_lambda = []
        all_familiar_lambda = []

        with torch.no_grad():
            for trial in range(n_trials):
                n_familiar = 32
                n_novel = 32

                familiar_base = torch.randn(1, 1, d_model, device=self.device)
                familiar = familiar_base.expand(1, n_familiar, d_model).clone()
                familiar = familiar + torch.randn_like(familiar) * 0.05

                novel = torch.randn(1, n_novel, d_model, device=self.device)

                combined = torch.cat([familiar, novel], dim=1)
                perm = torch.randperm(combined.shape[1])
                combined = combined[:, perm]
                labels = torch.cat(
                    [torch.zeros(n_familiar), torch.ones(n_novel)]
                )[perm]

                surprise_scores, lambda_weights = model.blocks[0].memory.surprise(
                    combined
                )

                if surprise_scores.dim() == 3:
                    s_per = surprise_scores.mean(dim=-1).squeeze(0)
                else:
                    s_per = surprise_scores.squeeze()

                if lambda_weights.dim() == 3:
                    l_per = lambda_weights.mean(dim=-1).squeeze(0)
                elif lambda_weights.dim() == 2:
                    l_per = lambda_weights.squeeze(-1)
                else:
                    l_per = lambda_weights.squeeze()

                if s_per.dim() == 2:
                    s_per = s_per.squeeze(0)
                if l_per.dim() == 2:
                    l_per = l_per.squeeze(0)

                n_mask = labels == 1
                f_mask = labels == 0

                all_novel_surprise.extend(s_per[n_mask].cpu().tolist())
                all_familiar_surprise.extend(s_per[f_mask].cpu().tolist())
                all_novel_lambda.extend(l_per[n_mask].cpu().tolist())
                all_familiar_lambda.extend(l_per[f_mask].cpu().tolist())

        surprise_auc = self._compute_auc(all_familiar_surprise, all_novel_surprise)
        lambda_auc = self._compute_auc(all_familiar_lambda, all_novel_lambda)

        ns = all_novel_surprise
        fs = all_familiar_surprise
        nl = all_novel_lambda
        fl = all_familiar_lambda

        metrics = {
            "surprise_roc_auc": surprise_auc,
            "lambda_roc_auc": lambda_auc,
            "novel_surprise_mean": sum(ns) / len(ns) if ns else 0.0,
            "familiar_surprise_mean": sum(fs) / len(fs) if fs else 0.0,
            "surprise_separation": (
                (sum(ns) / len(ns) - sum(fs) / len(fs))
                if ns and fs
                else 0.0
            ),
            "lambda_novel_mean": sum(nl) / len(nl) if nl else 0.0,
            "lambda_familiar_mean": sum(fl) / len(fl) if fl else 0.0,
            "lambda_separation": (
                (sum(nl) / len(nl) - sum(fl) / len(fl))
                if nl and fl
                else 0.0
            ),
        }

        return metrics

    @staticmethod
    def _compute_auc(neg_scores, pos_scores):
        if not neg_scores or not pos_scores:
            return 0.5

        pairs = [(s, 0) for s in neg_scores] + [(s, 1) for s in pos_scores]
        pairs.sort(key=lambda x: x[0], reverse=True)

        total_pos = sum(1 for _, l in pairs if l == 1)
        total_neg = len(pairs) - total_pos

        if total_pos == 0 or total_neg == 0:
            return 0.5

        tp = 0
        fp = 0
        auc = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0

        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / total_pos
            fpr = fp / total_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr = tpr
            prev_fpr = fpr

        return auc


class RelationalGraphBenchmark:
    def __init__(self, model_config=None, device="cpu"):
        self.model_config = model_config or {}
        self.device = device

    def run(self, model, n_trials=5):
        model.eval()

        d_mem = model.blocks[0].memory.d_mem
        graph = model.blocks[0].memory.graph
        threshold = graph.threshold

        n_clusters = 5
        points_per_cluster = 20

        all_ari = []
        all_purity = []
        all_nmi = []

        for trial in range(n_trials):
            torch.manual_seed(trial * 42 + 7)
            random.seed(trial * 42 + 7)

            centers = torch.randn(n_clusters, d_mem, device=self.device)
            centers = F.normalize(centers, dim=-1) * 3.0

            embeddings = []
            true_labels = []

            for c_idx in range(n_clusters):
                noise = (
                    torch.randn(points_per_cluster, d_mem, device=self.device) * 0.5
                )
                cluster_pts = centers[c_idx].unsqueeze(0) + noise
                embeddings.append(cluster_pts)
                true_labels.extend([c_idx] * points_per_cluster)

            slot_input = torch.cat(embeddings, dim=0).unsqueeze(0)
            true_t = torch.tensor(true_labels, device=self.device)

            with torch.no_grad():
                consolidated = graph(slot_input)

            pred_labels = self._kmeans_cluster(
                consolidated.squeeze(0), n_clusters
            )

            ari = self._adjusted_rand_index(true_t, pred_labels)
            purity = self._cluster_purity(true_t, pred_labels, n_clusters)
            nmi = self._normalized_mutual_info(true_t, pred_labels)

            all_ari.append(ari)
            all_purity.append(purity)
            all_nmi.append(nmi)

        metrics = {
            "adjusted_rand_index": sum(all_ari) / len(all_ari),
            "cluster_purity": sum(all_purity) / len(all_purity),
            "normalized_mutual_info": sum(all_nmi) / len(all_nmi),
            "graph_threshold": threshold,
            "n_clusters": n_clusters,
        }

        return metrics

    @staticmethod
    def _kmeans_cluster(embeddings, n_clusters, n_iters=20):
        n = embeddings.shape[0]
        if n <= n_clusters:
            return torch.arange(n, device=embeddings.device)

        indices = torch.randperm(n)[:n_clusters]
        centers = embeddings[indices].clone()

        for _ in range(n_iters):
            sims = embeddings @ centers.t()
            assignments = sims.argmax(dim=-1)

            new_centers = torch.zeros_like(centers)
            for i in range(n_clusters):
                mask = assignments == i
                if mask.any():
                    new_centers[i] = embeddings[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]

            new_centers = F.normalize(new_centers, dim=-1)
            if (new_centers - centers).abs().max() < 1e-5:
                break
            centers = new_centers

        return assignments

    @staticmethod
    def _adjusted_rand_index(true_labels, pred_labels):
        true_list = true_labels.cpu().tolist()
        pred_list = pred_labels.cpu().tolist()
        n = len(true_list)

        contingency = {}
        for i in range(n):
            key = (true_list[i], pred_list[i])
            contingency[key] = contingency.get(key, 0) + 1

        row_sums = {}
        col_sums = {}
        for (t, p), count in contingency.items():
            row_sums[t] = row_sums.get(t, 0) + count
            col_sums[p] = col_sums.get(p, 0) + count

        comb = lambda x: x * (x - 1) / 2

        index = sum(comb(v) for v in contingency.values())
        sum_a = sum(comb(v) for v in row_sums.values())
        sum_b = sum(comb(v) for v in col_sums.values())

        n_choose_2 = comb(n)
        if n_choose_2 == 0:
            return 0.0

        expected = sum_a * sum_b / n_choose_2
        max_index = (sum_a + sum_b) / 2.0

        if max_index == expected:
            return 1.0

        ari = (index - expected) / (max_index - expected)
        return max(ari, -1.0)

    @staticmethod
    def _cluster_purity(true_labels, pred_labels, n_clusters):
        true_list = true_labels.cpu().tolist()
        pred_list = pred_labels.cpu().tolist()
        n = len(true_list)

        if n == 0:
            return 0.0

        total_correct = 0
        unique_preds = set(pred_list)
        for p in unique_preds:
            indices = [i for i in range(n) if pred_list[i] == p]
            cluster_true = [true_list[i] for i in indices]
            if cluster_true:
                counts = Counter(cluster_true)
                total_correct += max(counts.values())

        return total_correct / n

    @staticmethod
    def _normalized_mutual_info(true_labels, pred_labels):
        true_list = true_labels.cpu().tolist()
        pred_list = pred_labels.cpu().tolist()
        n = len(true_list)

        if n == 0:
            return 0.0

        true_counts = Counter(true_list)
        pred_counts = Counter(pred_list)
        joint_counts = Counter(zip(true_list, pred_list))

        h_true = -sum(
            (c / n) * math.log(c / n + 1e-10) for c in true_counts.values()
        )
        h_pred = -sum(
            (c / n) * math.log(c / n + 1e-10) for c in pred_counts.values()
        )

        mi = 0.0
        for (t, p), c in joint_counts.items():
            p_t = true_counts[t] / n
            p_p = pred_counts[p] / n
            p_tp = c / n
            if p_t * p_p > 0:
                mi += p_tp * math.log(p_tp / (p_t * p_p) + 1e-10)

        denom = (h_true + h_pred) / 2.0 if (h_true + h_pred) > 0 else 1.0
        nmi = mi / denom

        return max(0.0, min(1.0, nmi))


class ForgetGateBenchmark:
    def __init__(self, model_config=None, device="cpu"):
        self.model_config = model_config or {}
        self.device = device

    def run(self, model, n_trials=5):
        model.eval()
        d_model = model.blocks[0].memory.d_model

        importance_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_memories_per_level = 8
        n_consolidation_cycles = 50

        all_precision = []
        all_recall = []
        all_f1 = []
        retention_by_level = {level: [] for level in importance_levels}

        with torch.no_grad():
            for trial in range(n_trials):
                probe = model.blocks[0]
                original_mem = probe.memory.lts.mem.data.clone()

                stored = {}
                for imp_val in importance_levels:
                    keys = torch.randn(1, n_memories_per_level, d_model, device=self.device)
                    values = torch.randn(1, n_memories_per_level, d_model, device=self.device)
                    importance = (
                        torch.ones(1, n_memories_per_level, 1, device=self.device)
                        * imp_val
                    )

                    k_proj = probe.memory.k_proj(keys)
                    v_proj = probe.memory.v_proj(values)

                    for _ in range(80):
                        probe.memory.lts.write(k_proj, v_proj, importance)

                    query = probe.memory.q_proj(keys)
                    readout = probe.memory.lts.read(query)

                    per_pair_sim = F.cosine_similarity(
                        readout, v_proj, dim=-1
                    ).squeeze(0)

                    stored[imp_val] = {
                        "keys": keys,
                        "values": values,
                        "importance": importance,
                        "k_proj": k_proj,
                        "v_proj": v_proj,
                        "initial_sims": per_pair_sim,
                    }

                for cycle in range(n_consolidation_cycles):
                    rand_h = F.layer_norm(
                        torch.randn(1, 4, d_model, device=self.device), [d_model]
                    )
                    for blk in model.blocks:
                        k = blk.memory.k_proj(rand_h)
                        v = blk.memory.v_proj(rand_h)
                        surprise, lam = blk.memory.surprise(rand_h)
                        blk.memory.lts.write(k, v, lam)

                is_high_importance = []
                retained_flags = []

                for imp_val in importance_levels:
                    info = stored[imp_val]
                    query = probe.memory.q_proj(info["keys"])
                    readout = probe.memory.lts.read(query)

                    per_pair_sim = F.cosine_similarity(
                        readout, info["v_proj"], dim=-1
                    ).squeeze(0)
                    initial_sims = info["initial_sims"]

                    retention_ratios = per_pair_sim / (initial_sims + 1e-8)
                    retention_ratios = retention_ratios.clamp(min=0.0).tolist()

                    retention_by_level[imp_val].extend(retention_ratios)

                    is_high = imp_val >= 0.5
                    for ratio in retention_ratios:
                        is_high_importance.append(is_high)
                        retained_flags.append(ratio > 0.5)

                n_total = len(retained_flags)
                if n_total == 0:
                    probe.memory.lts.mem.data = original_mem
                    continue

                tp = sum(
                    1
                    for i in range(n_total)
                    if is_high_importance[i] and retained_flags[i]
                )
                fp = sum(
                    1
                    for i in range(n_total)
                    if not is_high_importance[i] and retained_flags[i]
                )
                fn = sum(
                    1
                    for i in range(n_total)
                    if is_high_importance[i] and not retained_flags[i]
                )

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

                probe.memory.lts.mem.data = original_mem

        metrics = {
            "precision": (
                sum(all_precision) / len(all_precision) if all_precision else 0.0
            ),
            "recall": sum(all_recall) / len(all_recall) if all_recall else 0.0,
            "f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        }

        for level in importance_levels:
            r = retention_by_level[level]
            metrics[f"retention_importance_{level:.1f}"] = (
                sum(r) / len(r) if r else 0.0
            )

        return metrics


class LongRangeCoherenceBenchmark:
    def __init__(self, model_config=None, device="cpu"):
        self.model_config = model_config or {}
        self.device = device

    def _create_synthetic_document(
        self, vocab_size, seq_len, n_motifs, motif_len=8
    ):
        doc = torch.randint(0, vocab_size // 2, (1, seq_len), device=self.device)

        motif_tokens = []
        motif_positions = {}

        for m in range(n_motifs):
            motif = torch.randint(
                vocab_size // 2, vocab_size, (motif_len,), device=self.device
            )
            motif_tokens.append(motif)
            motif_positions[m] = []

        spacing = max(seq_len // (n_motifs * 3 + 1), motif_len * 4)
        pos = spacing

        for m_idx in range(n_motifs):
            for repeat in range(3):
                end_pos = min(pos + motif_len, seq_len)
                actual_len = end_pos - pos
                if actual_len < motif_len:
                    break
                doc[0, pos:end_pos] = motif_tokens[m_idx][:actual_len]
                motif_positions[m_idx].append(pos)
                pos += spacing
                if pos + motif_len >= seq_len:
                    break
            if pos + motif_len >= seq_len:
                break

        return doc, motif_tokens, motif_positions

    def _forward_through_memory(self, model, input_ids):
        d_model = model.blocks[0].memory.d_model
        embed = model.token_embedding(input_ids)

        cum_log_probs = torch.zeros(input_ids.shape[0], input_ids.shape[1], device=input_ids.device)

        mem_reads = []
        for block in model.blocks:
            h = F.layer_norm(embed, [d_model])
            _, mem_state = block.memory(h, return_mem_state=True)
            mem_reads.append(mem_state)

        pseudo_logits = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            model.config.get("vocab_size", 50257),
            device=input_ids.device,
        )
        return pseudo_logits, mem_reads

    def run(self, model, n_trials=5):
        model.eval()
        model.to(self.device)

        vocab_size = model.config.get("vocab_size", 50257)
        seq_len = min(model.config.get("max_seq_len", 512), 512)
        n_motifs = 3
        motif_len = 8

        all_motif_awareness = []
        all_ppl_consistency = []
        all_read_variance = []

        use_model_forward = True
        with torch.no_grad():
            test_ids = torch.randint(0, vocab_size, (1, 16), device=self.device)
            try:
                _forward_model(model, test_ids, self.device)
            except Exception:
                use_model_forward = False

        with torch.no_grad():
            for trial in range(n_trials):
                torch.manual_seed(trial * 100 + 7)

                doc, motif_tokens, motif_positions = (
                    self._create_synthetic_document(
                        vocab_size, seq_len, n_motifs, motif_len
                    )
                )

                original_step_counters = []
                for block in model.blocks:
                    original_step_counters.append(block.memory.step_counter.item())

                if use_model_forward:
                    logits = _forward_model(model, doc, self.device)
                else:
                    logits, _ = self._forward_through_memory(model, doc)

                log_probs = F.log_softmax(logits, dim=-1)
                target_lp = log_probs.gather(2, doc.unsqueeze(-1)).squeeze(-1)

                if use_model_forward:
                    window_size = 32
                    position_pplxs = []
                    for start in range(0, seq_len - window_size, window_size):
                        window_lp = target_lp[0, start : start + window_size].mean().item()
                        position_pplxs.append(math.exp(min(-window_lp, 20.0)))

                    if len(position_pplxs) > 1:
                        ppl_var = torch.tensor(
                            position_pplxs, dtype=torch.float
                        ).var().item()
                        ppl_consistency = 1.0 / (1.0 + ppl_var)
                    else:
                        ppl_consistency = 0.0
                    all_ppl_consistency.append(ppl_consistency)

                    motif_scores = []
                    for m_idx in range(n_motifs):
                        positions = motif_positions.get(m_idx, [])
                        if len(positions) < 2:
                            continue

                        late_pos = positions[-1]
                        if late_pos + motif_len >= seq_len:
                            continue

                        late_lp = target_lp[0, late_pos : late_pos + motif_len].mean().item()

                        neighbors = list(
                            range(max(0, late_pos - motif_len), late_pos)
                        ) + list(
                            range(
                                late_pos + motif_len,
                                min(seq_len, late_pos + 2 * motif_len),
                            )
                        )
                        neighbors = [
                            p for p in neighbors if 0 <= p < seq_len
                        ]

                        if neighbors:
                            neighbor_lp = (
                                target_lp[0, neighbors].mean().item()
                            )
                            awareness = late_lp - neighbor_lp
                        else:
                            awareness = 0.0

                        motif_scores.append(awareness)

                    avg_awareness = (
                        sum(motif_scores) / len(motif_scores)
                        if motif_scores
                        else 0.0
                    )
                    all_motif_awareness.append(avg_awareness)
                else:
                    all_ppl_consistency.append(0.0)
                    all_motif_awareness.append(0.0)

                for block in model.blocks:
                    mem_read = block.memory.lts.read(
                        block.memory.q_proj(
                            torch.randn(1, 1, block.memory.d_model, device=self.device)
                        )
                    )
                    block.memory.step_counter.zero_()

                read_patterns = []
                for block in model.blocks:
                    mem_read = block.memory.lts.read(
                        block.memory.q_proj(
                            torch.randn(1, seq_len // 4, block.memory.d_model, device=self.device)
                        )
                    )
                    read_patterns.append(mem_read.norm().item())

                if len(read_patterns) > 1:
                    read_var = torch.tensor(
                        read_patterns, dtype=torch.float
                    ).var().item()
                else:
                    read_var = 0.0
                all_read_variance.append(read_var)

                for i, block in enumerate(model.blocks):
                    if i < len(original_step_counters):
                        block.memory.step_counter.fill_(original_step_counters[i])

        metrics = {
            "motif_awareness": (
                sum(all_motif_awareness) / len(all_motif_awareness)
                if all_motif_awareness
                else 0.0
            ),
            "perplexity_consistency": (
                sum(all_ppl_consistency) / len(all_ppl_consistency)
                if all_ppl_consistency
                else 0.0
            ),
            "read_pattern_variance": (
                sum(all_read_variance) / len(all_read_variance)
                if all_read_variance
                else 0.0
            ),
            "seq_len": seq_len,
            "n_motifs": n_motifs,
            "motif_len": motif_len,
        }

        return metrics