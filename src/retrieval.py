"""FAQ corpus retrieval: embeds and persists the 30 articles in ChromaDB as whole documents (deliberately unchunked), then serves hybrid BM25 + dense queries fused with Reciprocal Rank Fusion."""
