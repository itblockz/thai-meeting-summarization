"""
Retrieval rankers shared by the E0 harness and Tier-A experiments.
Each ranker returns a full ranked list of para_ids, best first.
"""
import numpy as np


def tokenize_th(text: str) -> list:
    """Thai word tokenization (pythainlp newmm) — shared by BM25 callers."""
    from pythainlp.tokenize import word_tokenize
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def rank_dense(query_emb: np.ndarray, para_embs: np.ndarray, para_ids: list) -> list:
    """Dense cosine ranking. Embeddings must be L2-normalized (dot == cosine)."""
    if len(para_ids) == 0:
        return []
    scores = para_embs @ query_emb
    return [para_ids[i] for i in np.argsort(-scores)]


def rank_bm25(query_tokens: list, bm25, para_ids: list) -> list:
    """BM25 lexical ranking over Thai-tokenized paragraphs."""
    if len(para_ids) == 0:
        return []
    scores = np.asarray(bm25.get_scores(query_tokens))
    return [para_ids[i] for i in np.argsort(-scores)]


def rrf_fuse(ranked_lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion of several ranked para_id lists."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda pid: -scores[pid])


def rank_rerank(scored_pool: list) -> list:
    """Rank a cross-encoder-scored pool. scored_pool: list of (para_id, score)."""
    return [pid for pid, _ in sorted(scored_pool, key=lambda x: -x[1])]
