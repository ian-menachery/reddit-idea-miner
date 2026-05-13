from __future__ import annotations

import time
from typing import TYPE_CHECKING

from reddit_miner.competitor import (
    classify_sentiment,
    extract_tool_mentions,
    sweep_competitor_mentions,
)
from reddit_miner.config import Settings
from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    from pathlib import Path


# ---------- extract_tool_mentions ----------


def test_extract_tool_mentions_picks_capitalized_brand() -> None:
    mentions = extract_tool_mentions("I tried Notion but it's too slow for me.")
    tools = {t for t, _ in mentions}
    assert "Notion" in tools


def test_extract_tool_mentions_keeps_domain_suffix() -> None:
    mentions = extract_tool_mentions("Switched from Linear.app to Height.app last month.")
    tools = {t for t, _ in mentions}
    assert "Linear.app" in tools
    assert "Height.app" in tools


def test_extract_tool_mentions_dedupes_within_doc() -> None:
    text = "Asana is fine. Asana works. Asana again."
    mentions = extract_tool_mentions(text)
    assert sum(1 for t, _ in mentions if t == "Asana") == 1


def test_extract_tool_mentions_skips_pronouns_and_articles() -> None:
    text = "The team They We You I This That."
    assert extract_tool_mentions(text) == []


def test_extract_tool_mentions_skips_short_all_caps() -> None:
    # 'TLDR' is in denylist; 'CEO' should also be skipped as short all-caps
    mentions = extract_tool_mentions("TLDR: CEO loves Trello.")
    tools = {t for t, _ in mentions}
    assert "TLDR" not in tools
    assert "CEO" not in tools
    assert "Trello" in tools


def test_extract_tool_mentions_returns_context_window() -> None:
    text = "x" * 200 + " I love Figma for design " + "y" * 200
    mentions = dict(extract_tool_mentions(text))
    assert "Figma" in mentions
    assert "Figma" in mentions["Figma"]
    assert len(mentions["Figma"]) <= 200  # context bounded by radius


def test_extract_tool_mentions_empty_input() -> None:
    assert extract_tool_mentions("") == []


# ---------- classify_sentiment ----------


def test_classify_sentiment_positive() -> None:
    assert classify_sentiment("I absolutely love this tool, it's amazing!") == "positive"


def test_classify_sentiment_negative() -> None:
    assert classify_sentiment("I hate this. It's terrible, useless garbage.") == "negative"


def test_classify_sentiment_neutral() -> None:
    assert classify_sentiment("This is a feature.") == "neutral"


# ---------- sweep_competitor_mentions ----------


def _seed_cluster_with_docs(db_path: Path, docs: list[tuple[str, str, str]]) -> int:
    """Seed one cluster_run with one cluster containing the given (id, title, body) docs.
    Returns the inserted cluster id."""
    now = time.time()
    with connect(db_path) as conn, transaction(conn):
        cur = conn.execute(
            "INSERT INTO umap_models (fitted_at, n_documents, params_json) VALUES (?, 1, '{}')",
            (now,),
        )
        umap_version = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO cluster_runs (run_at, n_documents, n_clusters, umap_version) "
            "VALUES (?, ?, 1, ?)",
            (now, len(docs), umap_version),
        )
        run_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO clusters (run_id, label_tfidf, size, avg_pain_score, "
            "subreddit_spread, signal_score, centroid, created_at) "
            "VALUES (?, 'test', ?, 1.0, 1, 1.0, X'00', ?)",
            (run_id, len(docs), now),
        )
        cluster_id = cur.lastrowid
        for doc_id, title, body in docs:
            conn.execute(
                "INSERT INTO documents (id, type, parent_id, subreddit, title, body, "
                "score, num_comments, created_utc, permalink, pain_score, cluster_id, "
                "ingested_at) "
                "VALUES (?, 'post', NULL, 'r/test', ?, ?, 10, 0, ?, '/r/test/x', 1.0, ?, ?)",
                (doc_id, title, body, now, cluster_id, now),
            )
    assert cluster_id is not None
    return int(cluster_id)


def test_sweep_populates_mentions(tmp_db: Path) -> None:
    cluster_id = _seed_cluster_with_docs(
        tmp_db,
        [
            ("d1", "Notion vs Asana", "I tried Notion but Asana feels lighter."),
            ("d2", "Trello sucks", "Trello is awful and broken for power users."),
        ],
    )
    s = Settings(db_path=tmp_db)
    n = sweep_competitor_mentions(s)
    assert n >= 3  # at least Notion, Asana, Trello

    with connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT tool_name, sentiment FROM competitor_mentions WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchall()
    tools = {r["tool_name"] for r in rows}
    assert {"Notion", "Asana", "Trello"} <= tools

    trello_rows = [r for r in rows if r["tool_name"] == "Trello"]
    assert trello_rows[0]["sentiment"] == "negative"


def test_sweep_is_idempotent(tmp_db: Path) -> None:
    _seed_cluster_with_docs(tmp_db, [("d1", "", "Trello rocks")])
    s = Settings(db_path=tmp_db)
    n1 = sweep_competitor_mentions(s)
    n2 = sweep_competitor_mentions(s)
    assert n1 == n2
    with connect(tmp_db) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM competitor_mentions").fetchone()["c"]
    assert total == n1


def test_sweep_noop_without_cluster_runs(tmp_db: Path) -> None:
    s = Settings(db_path=tmp_db)
    assert sweep_competitor_mentions(s) == 0
