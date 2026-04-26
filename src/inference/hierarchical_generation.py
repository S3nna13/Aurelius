"""Hierarchical document generation: outline → sections → full document."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical document generation."""

    n_sections: int = 3
    section_tokens: int = 64
    outline_tokens: int = 32
    coherence_weight: float = 0.5
    temperature: float = 0.8


@dataclass
class OutlineNode:
    """A node in the outline tree (section or subsection)."""

    title: str
    level: int = 1  # 1 = section, 2 = subsection
    children: list[OutlineNode] = field(default_factory=list)
    content: str = ""

    def to_flat_list(self) -> list[OutlineNode]:
        """Flatten the outline tree using DFS traversal."""
        result: list[OutlineNode] = [self]
        for child in self.children:
            result.extend(child.to_flat_list())
        return result


def _greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
) -> torch.Tensor:
    """Simple greedy/sampling generation using the model's forward pass."""
    input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
    )
    return output


def generate_outline(
    model: torch.nn.Module,
    prompt: str,
    n_sections: int,
    tokenizer_encode,
    tokenizer_decode,
    max_new_tokens: int = 32,
) -> list[OutlineNode]:
    """Generate a document outline with section titles.

    Prompts the model with a numbered outline format and parses numbered lines.
    Falls back to ["Section 1", "Section 2", ...] if parsing fails or yields
    too few sections.
    """
    outline_prompt = f"Write an outline with {n_sections} sections for: {prompt}\n1."
    prompt_ids = tokenizer_encode(outline_prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = _greedy_generate(model, input_ids, max_new_tokens=max_new_tokens)

    new_token_ids = output[0, len(prompt_ids) :].tolist()
    generated_text = "1." + tokenizer_decode(new_token_ids)

    # Parse numbered lines like "1. Title" or "2. Title"
    pattern = re.compile(r"^\s*\d+\.\s*(.+)", re.MULTILINE)
    matches = pattern.findall(generated_text)

    nodes: list[OutlineNode] = []
    for title in matches[:n_sections]:
        title = title.strip()
        if title:
            nodes.append(OutlineNode(title=title, level=1))

    # Fallback: fill remaining slots with default section names
    if len(nodes) < n_sections:
        fallback_nodes = [OutlineNode(title=f"Section {i + 1}", level=1) for i in range(n_sections)]
        return fallback_nodes

    return nodes


def compute_coherence_score(prev_text: str, next_text: str) -> float:
    """Measure how well next_text follows prev_text using character trigram overlap.

    Returns a score in [0, 1] where 1 means identical trigram sets.
    """
    if not prev_text or not next_text:
        return 0.0

    def get_trigrams(text: str) -> set[str]:
        text = text.lower()
        return {text[i : i + 3] for i in range(len(text) - 2)}

    prev_trigrams = get_trigrams(prev_text)
    next_trigrams = get_trigrams(next_text)

    if not prev_trigrams or not next_trigrams:
        return 0.0

    intersection = prev_trigrams & next_trigrams
    union = prev_trigrams | next_trigrams

    if not union:
        return 0.0

    score = len(intersection) / len(union)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


class HierarchicalGenerator:
    """Generate full structured documents using a hierarchical pipeline."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: HierarchicalConfig,
        tokenizer_encode,
        tokenizer_decode,
    ) -> None:
        self.model = model
        self.config = config
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode

    def generate_section(
        self,
        outline_node: OutlineNode,
        context: str,
        max_new_tokens: int,
    ) -> str:
        """Generate content for one section given its title and preceding context.

        Args:
            outline_node: The section to generate content for.
            context: Text from preceding sections (empty string if first section).
            max_new_tokens: Maximum tokens to generate for this section.

        Returns:
            Generated section content as a string.
        """
        if context:
            section_prompt = f"{context}\n\nSection: {outline_node.title}\nContent:"
        else:
            section_prompt = f"Section: {outline_node.title}\nContent:"

        prompt_ids = self.encode(section_prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            output = _greedy_generate(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
            )

        new_token_ids = output[0, len(prompt_ids) :].tolist()
        return self.decode(new_token_ids)

    def generate_document(self, prompt: str) -> dict:
        """Full hierarchical document generation pipeline.

        Steps:
            1. Generate outline from the prompt.
            2. Generate content for each section, passing prior sections as context.
            3. Assemble the full document.
            4. Compute coherence scores between adjacent sections.

        Returns:
            dict with keys:
                "outline"          — list of section title strings
                "sections"         — list of generated section content strings
                "document"         — full assembled document string
                "coherence_scores" — list of float scores (length = n_sections - 1)
        """
        # Step 1: Generate outline
        outline_nodes = generate_outline(
            self.model,
            prompt,
            self.config.n_sections,
            self.encode,
            self.decode,
            max_new_tokens=self.config.outline_tokens,
        )

        outline_titles = [node.title for node in outline_nodes]

        # Step 2: Generate content for each section
        sections: list[str] = []
        context = ""

        for node in outline_nodes:
            section_content = self.generate_section(
                node,
                context=context,
                max_new_tokens=self.config.section_tokens,
            )
            sections.append(section_content)
            # Accumulate context from previous sections
            context = "\n\n".join(sections)

        # Step 3: Assemble full document
        doc_parts: list[str] = []
        for title, content in zip(outline_titles, sections):
            doc_parts.append(f"## {title}\n\n{content}")
        document = "\n\n".join(doc_parts)

        # Step 4: Compute coherence scores between adjacent sections
        coherence_scores: list[float] = []
        for i in range(len(sections) - 1):
            score = compute_coherence_score(sections[i], sections[i + 1])
            coherence_scores.append(score)

        return {
            "outline": outline_titles,
            "sections": sections,
            "document": document,
            "coherence_scores": coherence_scores,
        }

    def refine_section(self, section: str, outline_node: OutlineNode) -> str:
        """Regenerate a section conditioned on its title.

        Args:
            section: The original section content (used as additional context).
            outline_node: The outline node for this section.

        Returns:
            Revised section content as a string.
        """
        refine_prompt = (
            f"Section title: {outline_node.title}\nOriginal content: {section}\nRevised content:"
        )

        prompt_ids = self.encode(refine_prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            output = _greedy_generate(
                self.model,
                input_ids,
                max_new_tokens=self.config.section_tokens,
                temperature=self.config.temperature,
            )

        new_token_ids = output[0, len(prompt_ids) :].tolist()
        return self.decode(new_token_ids)
