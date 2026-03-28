"""
Knowledge retriever: embedding-based semantic search + BM25 keyword search.

Embedding retrieval (sentence-transformers) captures semantic similarity.
BM25 retrieval (rank-bm25) captures keyword overlap — useful when the query
shares exact terminology with stored entries.  The two can be combined via
hybrid retrieval with Reciprocal Rank Fusion (RRF).

Text representation used per entry type:

    MemoryEntry
      - embedding: content
      - BM25:      content + evidence

    SkillEntry
      - embedding: title + steps (joined)
      - BM25:      title + steps (joined) + evidence
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np

from context.store import MemoryEntry, SkillEntry


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase split on non-alphanumeric chars — simple but effective for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class KnowledgeRetriever:
    """Retrieve memory / skill entries via embedding, BM25, or hybrid search."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for KnowledgeRetriever. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(embedding_model, device="cpu")

    # ── Text representations ─────────────────────────────────────────

    @staticmethod
    def _embedding_text(entry: Union[MemoryEntry, SkillEntry]) -> str:
        """Text used for dense embedding (semantic meaning)."""
        if isinstance(entry, MemoryEntry):
            return entry.content
        return f"{entry.title}. {' '.join(entry.steps)}"

    @staticmethod
    def _bm25_text(entry: Union[MemoryEntry, SkillEntry]) -> str:
        """Text used for BM25 (keyword matching).

        Includes evidence on top of the embedding text so that domain-specific
        terms mentioned in the human feedback can be matched.
        """
        if isinstance(entry, MemoryEntry):
            return f"{entry.content} {entry.evidence}"
        return f"{entry.title}. {' '.join(entry.steps)} {entry.evidence}"

    # ── Embedding helpers ────────────────────────────────────────────

    def embed_entries(self, entries: list[Union[MemoryEntry, SkillEntry]]):
        """Compute and store embeddings on each entry (in-place)."""
        texts = [self._embedding_text(e) for e in entries]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        for entry, emb in zip(entries, embeddings):
            entry.embedding = emb.tolist()

    # ── Embedding retrieval ──────────────────────────────────────────

    def _retrieve_embedding(
        self,
        query: str,
        entries: list,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple]:
        if not entries:
            return []

        needs_embed = [e for e in entries if e.embedding is None]
        if needs_embed:
            self.embed_entries(needs_embed)

        query_emb = self.model.encode(query, convert_to_numpy=True)
        matrix = np.array([e.embedding for e in entries])

        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(m_norms, q_norm)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            (entries[i], float(sims[i]))
            for i in top_indices
            if sims[i] >= min_similarity
        ]

    # ── BM25 retrieval ───────────────────────────────────────────────

    def _retrieve_bm25(
        self,
        query: str,
        entries: list,
        top_k: int,
    ) -> list[tuple]:
        if not entries:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )

        corpus = [_tokenize_for_bm25(self._bm25_text(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        query_tokens = _tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (entries[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    # ── Hybrid retrieval (RRF) ───────────────────────────────────────

    def _retrieve_hybrid(
        self,
        query: str,
        entries: list,
        top_k: int,
        min_similarity: float,
        rrf_k: int = 60,
    ) -> list[tuple]:
        """Reciprocal Rank Fusion of embedding and BM25 results."""
        emb_results = self._retrieve_embedding(query, entries, top_k=len(entries), min_similarity=0.0)
        bm25_results = self._retrieve_bm25(query, entries, top_k=len(entries))

        rrf_scores: dict[int, float] = {}
        for rank, (entry, _) in enumerate(emb_results):
            idx = id(entry)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, (entry, _) in enumerate(bm25_results):
            idx = id(entry)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        entry_map = {id(e): e for e in entries}
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(entry_map[idx], score) for idx, score in ranked[:top_k]]

    # ── Public API ───────────────────────────────────────────────────

    def retrieve_memories(
        self,
        query: str,
        memories: list[MemoryEntry],
        top_k: int = 5,
        min_similarity: float = 0.5,
        method: str = "embedding",
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve memories. method: 'embedding', 'bm25', or 'hybrid'."""
        if method == "bm25":
            return self._retrieve_bm25(query, memories, top_k)
        if method == "hybrid":
            return self._retrieve_hybrid(query, memories, top_k, min_similarity)
        return self._retrieve_embedding(query, memories, top_k, min_similarity)

    def retrieve_skills(
        self,
        query: str,
        skills: list[SkillEntry],
        top_k: int = 3,
        min_similarity: float = 0.5,
        method: str = "embedding",
    ) -> list[tuple[SkillEntry, float]]:
        """Retrieve skills. method: 'embedding', 'bm25', or 'hybrid'."""
        if method == "bm25":
            return self._retrieve_bm25(query, skills, top_k)
        if method == "hybrid":
            return self._retrieve_hybrid(query, skills, top_k, min_similarity)
        return self._retrieve_embedding(query, skills, top_k, min_similarity)

    # ── Prompt formatting ───────────────────────────────────────────

    @staticmethod
    def format_for_prompt(
        memories: list[tuple[MemoryEntry, float]],
        skills: list[tuple[SkillEntry, float]],
    ) -> str:
        """Format retrieved entries for injection into an agent's system prompt."""
        parts: list[str] = []

        if memories:
            parts.append("## Relevant Memories (from past interactions)\n")
            for idx, (m, score) in enumerate(memories, 1):
                parts.append(f"{idx}. **[{m.type}]** {m.content}")
            parts.append("")

        if skills:
            parts.append("## Relevant Skills (from past workflows)\n")
            for idx, (s, score) in enumerate(skills, 1):
                steps_str = "\n".join(f"   {i+1}. {st}" for i, st in enumerate(s.steps))
                parts.append(f"### Skill {idx}: {s.title}\n{steps_str}\n")

        if parts:
            parts.append(
                "Use these memories and skills as guidance when applicable. "
                "Prioritize the current task requirements."
            )

        return "\n".join(parts)
