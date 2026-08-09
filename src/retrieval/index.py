"""Embeds the 30 FAQ articles as whole documents and persists them to a Chroma collection on disk, so the model runs once at build time rather than once per query.

Deliberately unchunked. The articles are 141–387 characters — already smaller than
a typical chunk — so chunking would split a single coherent answer across
fragments and destroy the context it needs to be useful. The corpus granularity
already matches the query granularity, so whole documents are the retrieval unit.
"""

from pathlib import Path

import chromadb
import yaml
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.retrieval.corpus import load_faq_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "faq_articles"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["embeddings"]
MODEL_NAME: str = _settings["model"]
QUERY_PREFIX: str = _settings["query_prefix"]

# Loaded on first use, then reused: the CLI would otherwise pay the model load
# twice, once to build and once to query.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """The one embedding model, shared by retrieval and by eval/coverage_map.py.

    Both must use it, or article vectors and intent centroids stop being
    comparable and every similarity threshold tuned on one is wrong for the other.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a customer message, with the model's query instruction attached.

    bge is trained asymmetrically: the instruction goes on the query side only and
    passages are indexed bare. Putting it on both sides, or on the wrong side,
    lowers every score without raising an error — see DECISIONS.md.
    """
    vector = get_model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed articles for indexing. No prefix, by design — see `embed_query`."""
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def build_index() -> Collection:
    """Embed all 30 articles as whole documents and write them to disk.

    Drops any existing collection first, so a rebuild is a rebuild and never a
    half-updated index silently mixing vectors from two different models.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    articles = load_faq_corpus()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        # Cosine, to match the coverage map. Chroma's default is L2, which would
        # rank differently and make the tuned thresholds meaningless.
        # The model and prefix are stamped in so a config change is detectable.
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "query_prefix": QUERY_PREFIX,
        },
        # Chroma otherwise embeds with its own bundled model, which is not the one
        # the rest of this project uses. Passing None keeps us the only embedder.
        embedding_function=None,
    )
    collection.add(
        ids=[article.doc_id for article in articles],
        embeddings=embed_passages([article.embedding_text() for article in articles]),
        documents=[article.content for article in articles],
        metadatas=[{"title": article.title} for article in articles],
    )
    return collection


def load_collection() -> Collection:
    """Return the persisted collection, rebuilding it if it is missing or stale.

    Stale means the settings that produced it no longer match the settings in
    force now. Silently querying an index built by a different model is exactly
    the kind of failure that shows up as slightly-worse answers and nothing else.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        return build_index()

    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    stamped = collection.metadata or {}
    if (stamped.get("embedding_model"), stamped.get("query_prefix")) != (MODEL_NAME, QUERY_PREFIX):
        print(f"index was built with {stamped.get('embedding_model')!r}; rebuilding for {MODEL_NAME!r}")
        return build_index()
    if collection.count() != len(load_faq_corpus()):
        print(f"index holds {collection.count()} articles, corpus has {len(load_faq_corpus())}; rebuilding")
        return build_index()
    return collection
