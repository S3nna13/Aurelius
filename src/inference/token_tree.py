from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class TreeNode:
    token_id: int
    prob: float
    children: list["TreeNode"] = field(default_factory=list)
    depth: int = 0
    path: list[int] = field(default_factory=list)


class TokenTree:
    """Token tree for multi-step speculative decoding (SpecTree / Medusa)."""

    def __init__(self, branching_factor: int = 2, max_depth: int = 4) -> None:
        if branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.branching_factor = branching_factor
        self.max_depth = max_depth

    def build(self, logits_per_step: list[torch.Tensor]) -> TreeNode:
        root = TreeNode(token_id=-1, prob=1.0, depth=0, path=[])
        steps = logits_per_step[: self.max_depth]
        self._expand(root, steps, depth=0)
        return root

    def _expand(
        self, node: TreeNode, logits_per_step: list[torch.Tensor], depth: int
    ) -> None:
        if depth >= len(logits_per_step):
            return
        logits = logits_per_step[depth]
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(self.branching_factor, probs.size(-1)))
        for i in range(top_ids.size(0)):
            tok = int(top_ids[i].item())
            p = float(top_probs[i].item())
            child = TreeNode(
                token_id=tok,
                prob=p,
                depth=depth + 1,
                path=node.path + [tok],
            )
            node.children.append(child)
            self._expand(child, logits_per_step, depth + 1)

    def get_paths(self, root: TreeNode) -> list[list[int]]:
        paths: list[list[int]] = []
        self._dfs(root, [], paths)
        return paths

    def _dfs(self, node: TreeNode, current: list[int], paths: list[list[int]]) -> None:
        if node.token_id != -1:
            current = current + [node.token_id]
        if not node.children:
            if current:
                paths.append(current)
            return
        for child in node.children:
            self._dfs(child, current, paths)

    def best_path(self, root: TreeNode) -> list[int]:
        best: list[int] = []
        best_score: list[float] = [-1.0]

        def search(node: TreeNode, score: float, path: list[int]) -> None:
            if node.token_id != -1:
                score = score * node.prob
                path = path + [node.token_id]
            if not node.children:
                if score > best_score[0]:
                    best_score[0] = score
                    best.clear()
                    best.extend(path)
                return
            for child in node.children:
                search(child, score, path)

        search(root, 1.0, [])
        return best

    def prune(self, root: TreeNode, threshold: float = 0.01) -> TreeNode:
        root.children = [
            self.prune(child, threshold)
            for child in root.children
            if child.prob >= threshold
        ]
        return root

    def n_nodes(self, root: TreeNode) -> int:
        count = 1
        for child in root.children:
            count += self.n_nodes(child)
        return count

    def depth(self, root: TreeNode) -> int:
        if not root.children:
            return 0
        return 1 + max(self.depth(child) for child in root.children)
