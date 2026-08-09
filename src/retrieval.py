"""FAQ corpus retrieval: embeds and persists the 30 articles in ChromaDB as whole documents (deliberately unchunked), then serves hybrid BM25 + dense queries fused with Reciprocal Rank Fusion."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
FAQ_CSV = REPO_ROOT / "data" / "banking_faq_corpus.csv"


class FaqArticle(BaseModel):
    """One FAQ article. The whole article is the retrieval unit; we never chunk."""

    doc_id: str = Field(description="Corpus identifier, e.g. 'doc_028'.")
    title: str = Field(description="Article title, also used as the citation label.")
    content: str = Field(description="Article body. Multi-line, with CRLF normalised to LF.")

    def embedding_text(self) -> str:
        """Title and body together, which is what gets embedded and BM25-indexed."""
        return f"{self.title}\n{self.content}"


def load_faq_corpus() -> list[FaqArticle]:
    """Read the 30-article corpus from CSV.

    The file is UTF-8 with a BOM, uses CRLF line endings, and has multi-line
    quoted fields — so it must be parsed as CSV, never split on newlines.
    """
    frame = pd.read_csv(FAQ_CSV, encoding="utf-8-sig")
    return [
        FaqArticle(
            doc_id=row.ID,
            title=row.Title.strip(),
            content=row.Content.replace("\r\n", "\n").strip(),
        )
        for row in frame.itertuples()
    ]
