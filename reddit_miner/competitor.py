from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)


# Capitalized tokens 3+ chars, optionally with a tool-suffix domain (.ai, .io, .app, etc.)
TOOL_TOKEN_RE = re.compile(
    r"\b("
    r"[A-Z][a-zA-Z0-9_-]{2,}"
    r"(?:\.(?:ai|io|app|com|net|dev|so|co))?"
    r")\b"
)

# Capitalized words that aren't tools. Conservative — only the highest-frequency offenders.
CAP_DENYLIST: frozenset[str] = frozenset(
    {
        "The", "This", "That", "These", "Those", "Then", "There",
        "They", "Their", "Them",
        "You", "Your", "Yours",
        "We", "Our", "Ours", "Us",
        "He", "She", "His", "Her", "Hers",
        "It", "Its",
        "I'm", "I've", "I'd", "I'll",
        "Yes", "No", "Not", "Maybe", "OK", "Okay",
        "And", "But", "Or", "So", "Just", "Now", "Why", "How", "What",
        "When", "Where", "Who", "Whom", "Whose", "Which",
        "Reddit", "Subreddit",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "June", "July",
        "August", "September", "October", "November", "December",
        "English", "American", "European", "Asian",
        "TLDR", "DM", "Edit", "Update", "Edited", "Updated", "Source", "PSA",
    }
)  # fmt: skip


CONTEXT_RADIUS = 80


@lru_cache(maxsize=1)
def _get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def extract_tool_mentions(text: str) -> list[tuple[str, str]]:
    """Return (tool_name, context_window) pairs. Deduped by tool within `text`."""
    if not text:
        return []
    out: dict[str, str] = {}
    for m in TOOL_TOKEN_RE.finditer(text):
        tool = m.group(1)
        if tool in CAP_DENYLIST or len(tool) < 3:
            continue
        # Skip ALL-CAPS short acronyms (often noise) but keep brand-style mixed-case
        if tool.isupper() and len(tool) <= 4:
            continue
        if tool not in out:
            start = max(0, m.start() - CONTEXT_RADIUS)
            end = min(len(text), m.end() + CONTEXT_RADIUS)
            out[tool] = text[start:end].strip()
    return list(out.items())


def classify_sentiment(context: str) -> str:
    """VADER compound score thresholds (positive/negative/neutral)."""
    scores = _get_sentiment_analyzer().polarity_scores(context)
    compound = float(scores["compound"])
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _latest_run_id(conn: Any) -> int | None:
    row = conn.execute("SELECT MAX(id) AS id FROM cluster_runs").fetchone()
    return int(row["id"]) if row and row["id"] is not None else None


def sweep_competitor_mentions(settings: Settings) -> int:
    """Extract tool mentions from every document in the latest cluster run's clusters.
    Idempotent: deletes prior mentions for the same run before inserting."""
    with connect(settings.db_path) as conn:
        run_id = _latest_run_id(conn)
        if run_id is None:
            logger.warning("No cluster_runs found — run `cluster` first.")
            return 0
        docs = conn.execute(
            "SELECT d.id, d.title, d.body, d.cluster_id "
            "FROM documents d JOIN clusters c ON d.cluster_id = c.id "
            "WHERE c.run_id = ?",
            (run_id,),
        ).fetchall()

    inserts: list[tuple[int, str, str, str, str]] = []
    for d in docs:
        text = f"{d['title'] or ''} {d['body'] or ''}".strip()
        if not text:
            continue
        for tool_name, context in extract_tool_mentions(text):
            sentiment = classify_sentiment(context)
            inserts.append((int(d["cluster_id"]), tool_name, sentiment, context, str(d["id"])))

    with connect(settings.db_path) as conn, transaction(conn):
        conn.execute(
            "DELETE FROM competitor_mentions WHERE cluster_id IN "
            "(SELECT id FROM clusters WHERE run_id = ?)",
            (run_id,),
        )
        if inserts:
            conn.executemany(
                "INSERT INTO competitor_mentions "
                "(cluster_id, tool_name, sentiment, context, source_document_id) "
                "VALUES (?, ?, ?, ?, ?)",
                inserts,
            )

    logger.info(
        "Extracted %d competitor mentions across %d documents (run_id=%d)",
        len(inserts),
        len(docs),
        run_id,
    )
    return len(inserts)
