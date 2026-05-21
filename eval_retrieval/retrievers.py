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


def colbert_score(q_vecs: np.ndarray, p_vecs: np.ndarray) -> float:
    """MaxSim: for each query token, take max similarity over paragraph tokens, sum.

    Assumes q_vecs and p_vecs are L2-normalized (bge-m3 returns them so).
    """
    if q_vecs.size == 0 or p_vecs.size == 0:
        return 0.0
    sim = q_vecs.astype(np.float32) @ p_vecs.astype(np.float32).T
    return float(sim.max(axis=1).sum())


def rank_colbert(q_vecs: np.ndarray, para_vecs_list: list, para_ids: list) -> list:
    """Late-interaction (ColBERT-style) ranking from per-token embeddings."""
    if not para_ids:
        return []
    scores = np.array([colbert_score(q_vecs, p) for p in para_vecs_list])
    return [para_ids[i] for i in np.argsort(-scores)]
