"""Hybrid retrieval: BM25 over article tokens, dense search over article embeddings, fused with Reciprocal Rank Fusion.

Dense retrieval understands paraphrase but misses exact tokens — "SWIFT", "MitID",
"e-Boks" — because a 384-dimensional average has nowhere to put a rare literal.
BM25 catches exactly those and misses everything phrased differently. Fusing them
covers both failure modes, and RRF does it on ranks alone, so it needs no tuning
and no score normalisation between two scales that are not comparable.
"""

import re

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.retrieval.corpus import FaqArticle, load_faq_corpus
from src.retrieval.index import embed_query, load_collection

# PLAN §7's constant, from the original RRF paper. It damps the contribution of
# top ranks so one system cannot dominate the fusion on a single confident hit.
RRF_K = 60


class RetrievedArticle(BaseModel):
    """One result, carrying both retrievers' opinions so disagreement stays visible."""

    doc_id: str = Field(description="Corpus identifier, e.g. 'doc_028'.")
    title: str = Field(description="Article title, also used as the citation label.")
    content: str = Field(description="Article body, the text a reply must be grounded in.")
    dense_score: float = Field(description="Cosine similarity to the query, in [-1, 1].")
    dense_rank: int = Field(description="1-based rank under dense retrieval alone.")
    bm25_score: float = Field(description="Okapi BM25 score. Unbounded; 0.0 means no term overlap.")
    bm25_rank: int | None = Field(
        description="1-based rank under BM25 alone, or None when BM25 did not match this article."
    )
    rrf_score: float = Field(description="Fused Reciprocal Rank Fusion score.")
    fused_rank: int = Field(description="1-based rank after fusion. The ordering callers should use.")


def tokenise(text: str) -> list[str]:
    """Lowercase word characters. Deliberately crude: no stemming, no stop-word list.

    BM25's job here is to catch literals like "SWIFT" and "MitID", and stemming
    those does nothing while adding a dependency and a behaviour to explain.
    """
    return re.findall(r"\w+", text.lower())


_bm25: BM25Okapi | None = None
_bm25_ids: list[str] = []


def sparse_ranking(query: str) -> list[tuple[str, float]]:
    """Articles BM25 actually matched, best first. Built once and reused.

    Zero-scoring articles are dropped rather than ranked. A BM25 score of 0.0 means
    the article shares no term with the query, so it is not a weak result — it is
    not a result. Ranking them anyway hands RRF a long tail of arbitrary ordering
    that it cannot tell apart from evidence.
    """
    global _bm25, _bm25_ids
    if _bm25 is None:
        articles = load_faq_corpus()
        _bm25_ids = [article.doc_id for article in articles]
        _bm25 = BM25Okapi([tokenise(article.embedding_text()) for article in articles])

    scores = _bm25.get_scores(tokenise(query))
    matched = [(doc_id, float(score)) for doc_id, score in zip(_bm25_ids, scores) if score > 0.0]
    return sorted(matched, key=lambda pair: pair[1], reverse=True)


def dense_ranking(query: str, limit: int) -> list[tuple[str, float]]:
    """Every article scored by cosine similarity to the query, best first."""
    collection = load_collection()
    response = collection.query(
        query_embeddings=[embed_query(query)], n_results=limit, include=["distances"]
    )
    # Chroma reports cosine *distance*; the collection is built with
    # hnsw:space=cosine, so similarity is 1 - distance.
    return [
        (doc_id, 1.0 - distance)
        for doc_id, distance in zip(response["ids"][0], response["distances"][0])
    ]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Sum 1/(k + rank) across rankings. Ranks only — the two score scales never meet."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(query: str, top_k: int = 5) -> list[RetrievedArticle]:
    """Retrieve and fuse. Returns ranked articles and nothing else.

    Deciding whether these are good enough to answer from is the abstention gate's
    job, not retrieval's. This function never abstains and never judges.
    """
    articles: dict[str, FaqArticle] = {a.doc_id: a for a in load_faq_corpus()}

    # Dense ranks the whole corpus, since cosine gives every article a real score.
    # BM25 contributes only what it matched. Fusing lists of different lengths is
    # what RRF is for: an article missing from one list simply scores nothing there.
    dense = dense_ranking(query, len(articles))
    sparse = sparse_ranking(query)
    dense_scores, sparse_scores = dict(dense), dict(sparse)
    dense_positions = {doc_id: i + 1 for i, (doc_id, _) in enumerate(dense)}
    sparse_positions = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sparse)}

    fused = reciprocal_rank_fusion([[d for d, _ in dense], [d for d, _ in sparse]])
    ordered = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)

    return [
        RetrievedArticle(
            doc_id=doc_id,
            title=articles[doc_id].title,
            content=articles[doc_id].content,
            dense_score=dense_scores[doc_id],
            dense_rank=dense_positions[doc_id],
            bm25_score=sparse_scores.get(doc_id, 0.0),
            bm25_rank=sparse_positions.get(doc_id),
            rrf_score=fused[doc_id],
            fused_rank=position + 1,
        )
        for position, doc_id in enumerate(ordered[:top_k])
    ]
