from __future__ import annotations

import time
from typing import TYPE_CHECKING

from reddit_miner.db import connect
from reddit_miner.ingest import add_subreddit, ingest_subreddit, pain_score

from .conftest import FakeComment, FakeCommentForest, FakeReddit, FakeSubmission, FakeSubreddit

if TYPE_CHECKING:
    from pathlib import Path


# ---------- pain_score ----------


def test_pain_score_zero_for_neutral_text() -> None:
    score, matched = pain_score("just sharing a thought about lunch", 10, 0)
    assert score == 0.0
    assert matched == ()


def test_pain_score_counts_distinct_pattern_hits() -> None:
    text = "I hate this and I am sick of it. I wish there was a better tool."
    score, matched = pain_score(text, 10, 0)
    # At least 3 distinct patterns: 'i hate|sick of', 'sick of' standalone, 'wish there was'
    assert score >= 2.0
    assert len(matched) == len(set(matched))


def test_pain_score_engagement_bonus_caps_at_5x_0_3() -> None:
    # comments=500, score=1 -> ratio=500, capped to 5.0, *0.3 = 1.5
    score, _ = pain_score("nothing painful", score=1, num_comments=500)
    assert score == 1.5


def test_pain_score_zero_engagement_for_comments() -> None:
    # num_comments=0 means engagement_bonus=0; pure regex score.
    score, _ = pain_score("I hate this", score=20, num_comments=0)
    assert score == 1.0


def test_pain_score_engagement_normal_range() -> None:
    # comments=10, score=5 -> ratio=2.0, *0.3 = 0.6
    score, _ = pain_score("neutral", score=5, num_comments=10)
    assert abs(score - 0.6) < 1e-9


# ---------- ingest_subreddit ----------


def _build_corpus() -> FakeReddit:
    return FakeReddit(
        subs={
            "test": FakeSubreddit(
                submissions=[
                    FakeSubmission(
                        id="post1",
                        title="I hate the current tools",
                        selftext="Nothing decent exists, willing to pay",
                        score=20,
                        num_comments=15,
                        comments=FakeCommentForest(
                            comments=[
                                FakeComment(id="c1", body="I struggle with this too", score=10),
                                FakeComment(id="c2", body="[deleted]", score=8),
                                FakeComment(id="c3", body="below threshold", score=1),
                                FakeComment(id="c4", body="just a regular comment", score=7),
                            ]
                        ),
                    ),
                    FakeSubmission(
                        id="post2",
                        title="low score post",
                        selftext="should be skipped",
                        score=2,
                        num_comments=0,
                    ),
                ]
            )
        }
    )


def test_ingest_inserts_post_and_qualifying_comments(tmp_db: Path) -> None:
    reddit = _build_corpus()
    posts, comments = ingest_subreddit(
        reddit, "test", upvote_threshold=5, posts_per_sub=10, db_path=tmp_db
    )
    assert posts == 1  # post2 below threshold
    assert comments == 2  # c1 + c4; c2 deleted, c3 below threshold

    with connect(tmp_db) as conn:
        rows = conn.execute("SELECT id, type, parent_id FROM documents ORDER BY id").fetchall()
    by_type = {r["id"]: (r["type"], r["parent_id"]) for r in rows}
    assert by_type["post1"] == ("post", None)
    assert by_type["c1"] == ("comment", "post1")
    assert by_type["c4"] == ("comment", "post1")


def test_ingest_is_idempotent(tmp_db: Path) -> None:
    reddit = _build_corpus()
    ingest_subreddit(reddit, "test", 5, 10, tmp_db)
    posts, comments = ingest_subreddit(reddit, "test", 5, 10, tmp_db)
    assert (posts, comments) == (0, 0)
    with connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
    assert n == 3  # post1 + c1 + c4


def test_ingest_records_pain_score_and_patterns(tmp_db: Path) -> None:
    reddit = _build_corpus()
    ingest_subreddit(reddit, "test", 5, 10, tmp_db)
    with connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT pain_score, matched_patterns FROM documents WHERE id = 'post1'"
        ).fetchone()
    assert row["pain_score"] > 0
    assert row["matched_patterns"].startswith("[")  # JSON-encoded list


# ---------- add_subreddit ----------


def test_add_subreddit_normalizes_and_activates(tmp_db: Path) -> None:
    add_subreddit(tmp_db, "r/Productivity")
    add_subreddit(tmp_db, "ProductivitY")  # case-different duplicate after normalize
    with connect(tmp_db) as conn:
        rows = conn.execute("SELECT name, status FROM subreddits").fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "productivity"
    assert rows[0]["status"] == "active"


def test_add_subreddit_reactivates_inactive(tmp_db: Path) -> None:
    with connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO subreddits (name, status, source, added_at) "
            "VALUES ('foo', 'inactive', 'manual', ?)",
            (time.time(),),
        )
    add_subreddit(tmp_db, "foo")
    with connect(tmp_db) as conn:
        status = conn.execute("SELECT status FROM subreddits WHERE name='foo'").fetchone()["status"]
    assert status == "active"
