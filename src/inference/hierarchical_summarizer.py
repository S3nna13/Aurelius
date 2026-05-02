"""Hierarchical summarization for long documents using AureliusTransformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SummarizerConfig:
    """Configuration for hierarchical summarization."""

    chunk_size: int = 256
    overlap: int = 32
    max_summary_tokens: int = 64
    n_levels: int = 2
    compression_ratio: float = 0.3


@dataclass
class TextChunk:
    """A chunk of text with its token representation."""

    text: str
    token_ids: list
    start_pos: int
    level: int = 0


def chunk_document(token_ids, chunk_size, overlap):
    """Split a token sequence into overlapping chunks.

    Args:
        token_ids: Full list of token IDs.
        chunk_size: Number of tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.

    Returns:
        List of TextChunk objects with correct start_pos values.
    """
    if not token_ids:
        return []

    stride = chunk_size - overlap
    chunks = []
    i = 0
    while i < len(token_ids):
        chunk = token_ids[i : i + chunk_size]
        chunks.append(
            TextChunk(
                text="",
                token_ids=chunk,
                start_pos=i,
                level=0,
            )
        )
        if i + chunk_size >= len(token_ids):
            break
        i += stride
    return chunks


def encode_chunk(model, chunk):
    """Encode a chunk by mean-pooling last-layer hidden states.

    Uses a forward hook on the last transformer layer to capture hidden states,
    then mean-pools across the sequence dimension.

    Args:
        model: AureliusTransformer instance.
        chunk: TextChunk containing token_ids to encode.

    Returns:
        Tensor of shape (d_model,).
    """
    captured = []

    def hook_fn(module, inp, out):
        # out may be (hidden, kv) tuple -- capture hidden states
        if isinstance(out, tuple):
            captured.append(out[0].detach())
        else:
            captured.append(out.detach())

    last_layer = model.layers[-1]
    handle = last_layer.register_forward_hook(hook_fn)

    try:
        ids = torch.tensor(chunk.token_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    hidden = captured[0]  # (1, S, d_model)
    # Mean pool over sequence dimension
    return hidden.squeeze(0).mean(dim=0)  # (d_model,)


def greedy_summarize(model, input_ids, max_new_tokens, tokenizer_decode):
    """Greedily generate summary tokens appended to input, then decode.

    Args:
        model: AureliusTransformer instance.
        input_ids: Context token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        tokenizer_decode: Callable mapping list[int] -> str.

    Returns:
        Decoded string of the generated tokens.
    """
    generated = list(input_ids)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
            # Truncate to max_seq_len if needed
            max_len = model.config.max_seq_len
            if ids.shape[1] > max_len:
                ids = ids[:, -max_len:]
            _, logits, _ = model(ids)
            next_token = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_token)

    new_tokens = generated[len(input_ids) :]
    return tokenizer_decode(new_tokens)


class HierarchicalSummarizer:
    """Hierarchically summarize long documents using a transformer model."""

    def __init__(self, model, config, tokenizer_encode, tokenizer_decode):
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def summarize_chunk(self, chunk):
        """Generate a short summary string for a single chunk."""
        return greedy_summarize(
            self.model,
            chunk.token_ids,
            self.config.max_summary_tokens,
            self.tokenizer_decode,
        )

    def summarize_level(self, chunks):
        """Summarize each chunk, returning new TextChunks at level+1."""
        new_chunks = []
        pos = 0
        for chunk in chunks:
            summary_text = self.summarize_chunk(chunk)
            summary_ids = self.tokenizer_encode(summary_text)
            new_chunks.append(
                TextChunk(
                    text=summary_text,
                    token_ids=summary_ids,
                    start_pos=pos,
                    level=chunk.level + 1,
                )
            )
            pos += len(summary_ids)
        return new_chunks

    def summarize(self, text):
        """Hierarchically summarize a document.

        Returns:
            Dict with keys: summary, n_chunks_level0, n_levels, compression.
        """
        input_ids = self.tokenizer_encode(text)
        input_len = max(len(input_ids), 1)

        # Level-0 chunks
        chunks = chunk_document(input_ids, self.config.chunk_size, self.config.overlap)
        if not chunks:
            chunks = [TextChunk(text=text, token_ids=input_ids or [0], start_pos=0, level=0)]
        n_chunks_level0 = len(chunks)

        levels_applied = 0
        for _ in range(self.config.n_levels - 1):
            chunks = self.summarize_level(chunks)
            levels_applied += 1
            # Re-chunk summaries for next level
            all_ids = []
            for c in chunks:
                all_ids.extend(c.token_ids)
            if len(all_ids) <= self.config.chunk_size:
                break
            chunks = chunk_document(all_ids, self.config.chunk_size, self.config.overlap)

        # Final: concatenate remaining chunk ids and produce one summary
        all_ids = []
        for c in chunks:
            all_ids.extend(c.token_ids)

        # Truncate to fit max_seq_len
        max_ctx = self.model.config.max_seq_len - self.config.max_summary_tokens
        if max_ctx < 1:
            max_ctx = 1
        all_ids = all_ids[:max_ctx]

        final_summary = greedy_summarize(
            self.model,
            all_ids,
            self.config.max_summary_tokens,
            self.tokenizer_decode,
        )
        levels_applied += 1

        summary_len = max(len(self.tokenizer_encode(final_summary)), 1)
        compression = input_len / summary_len

        return {
            "summary": final_summary,
            "n_chunks_level0": n_chunks_level0,
            "n_levels": levels_applied,
            "compression": compression,
        }

    def extractive_summary(self, text, n_sentences=3):
        """Select top-n most central sentences by cosine similarity to document mean."""
        raw_sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
        sentences = [s for s in raw_sentences if s]
        if not sentences:
            return text

        embeddings = []
        for sent in sentences:
            ids = self.tokenizer_encode(sent)
            if not ids:
                ids = [0]
            chunk = TextChunk(text=sent, token_ids=ids, start_pos=0, level=0)
            emb = encode_chunk(self.model, chunk)
            embeddings.append(emb)

        emb_matrix = torch.stack(embeddings, dim=0)  # (N, d_model)
        doc_mean = emb_matrix.mean(dim=0)  # (d_model,)

        emb_norm = F.normalize(emb_matrix, dim=-1)
        doc_norm = F.normalize(doc_mean.unsqueeze(0), dim=-1)
        scores = (emb_norm @ doc_norm.T).squeeze(-1)  # (N,)

        k = min(n_sentences, len(sentences))
        top_indices = torch.topk(scores, k).indices.tolist()
        top_indices_sorted = sorted(top_indices)

        selected = [sentences[i] for i in top_indices_sorted]
        return ". ".join(selected) + "."
