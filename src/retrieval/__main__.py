"""Query CLI for eyeballing retrieval before anything is built on top of it.

    python -m src.retrieval --build
    python -m src.retrieval "how long does an international transfer take?"

Shows dense score, BM25 rank and fused rank side by side, because the point of a
hybrid retriever is the cases where its two halves disagree.
"""

import argparse

from src.retrieval.hybrid import search
from src.retrieval.index import MODEL_NAME, build_index


def format_results(query: str, results: list) -> str:
    """One row per article, with both retrievers' opinions and a disagreement marker."""
    lines = [
        f'query: "{query}"',
        f"model: {MODEL_NAME}",
        "",
        f"{'fused':>5}  {'dense':>5}  {'cosine':>7}  {'bm25':>4}  {'score':>6}  {'':2} {'doc':8} title",
        "-" * 100,
    ]
    for r in results:
        # Flag the rows the fusion actually exists to handle: where the two
        # retrievers disagree by more than two places, or where BM25 has no
        # opinion at all because the query shares no term with the article.
        if r.bm25_rank is None:
            rank, score, marker = "-", "-", "**"
        else:
            rank, score, marker = str(r.bm25_rank), f"{r.bm25_score:.2f}", (
                "<>" if abs(r.dense_rank - r.bm25_rank) > 2 else "  "
            )
        lines.append(
            f"{r.fused_rank:>5}  {r.dense_rank:>5}  {r.dense_score:>7.4f}  "
            f"{rank:>4}  {score:>6}  {marker} {r.doc_id:8} {r.title}"
        )
    lines += [
        "",
        "<> = dense and BM25 disagree by more than two ranks",
        "** = no BM25 match (query shares no term with the article); dense-only result",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the FAQ retriever.")
    parser.add_argument("query", nargs="?", help="Customer message to retrieve for.")
    parser.add_argument("--build", action="store_true", help="Rebuild the index and exit.")
    parser.add_argument("--top-k", type=int, default=5, help="How many results to show.")
    args = parser.parse_args()

    if args.build:
        collection = build_index()
        print(f"indexed {collection.count()} articles with {MODEL_NAME}")
        return
    if not args.query:
        parser.error("give a query, or --build to index")

    print(format_results(args.query, search(args.query, top_k=args.top_k)))


if __name__ == "__main__":
    main()
