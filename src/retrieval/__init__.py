"""Retrieval over the 30-article FAQ corpus: whole-document indexing, then hybrid BM25 + dense search fused with Reciprocal Rank Fusion.

Re-exported here so callers write `from src.retrieval import load_faq_corpus`
regardless of which module inside the package happens to own it.
"""

from src.retrieval.corpus import FaqArticle, load_faq_corpus
from src.retrieval.hybrid import RetrievedArticle, search
from src.retrieval.index import build_index, embed_query, load_collection

__all__ = [
    "FaqArticle",
    "RetrievedArticle",
    "build_index",
    "embed_query",
    "load_collection",
    "load_faq_corpus",
    "search",
]
