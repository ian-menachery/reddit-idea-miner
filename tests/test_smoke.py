"""End-to-end smoke test: fake PRAW input -> ingest -> embed (mocked) ->
reduce -> cluster -> competitor -> alert. Exercises every stage's wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reddit_miner.config import Settings
from reddit_miner.db import connect
from reddit_miner.ingest import add_subreddit
from reddit_miner.pipeline import pipeline_run

from .conftest import FakeComment, FakeCommentForest, FakeReddit, FakeSubmission, FakeSubreddit

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _settings_for(tmp_db: Path, tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_db,
        alerts_path=tmp_path / "alerts.md",
        umap_model_path=tmp_path / "umap.joblib",
        embedding_model="dummy",
        embedding_batch_size=16,
        upvote_threshold=1,  # accept everything in fixture
        posts_per_sub=50,
        umap_n_components=3,
        umap_n_neighbors=6,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_random_state=42,
        hdbscan_min_cluster_size=3,
        new_cluster_cosine_threshold=0.30,
        signal_score_threshold=1.0,
        labeler="ctfidf",
    )


def _fake_corpus() -> FakeReddit:
    """Two themes split across two subreddits each, with tool mentions."""
    submissions = []

    # Theme A: project management pain (across r/saas and r/productivity)
    pm_posts = [
        ("Trello fails our team", "Trello is too clunky and broken for sprints. I hate it."),
        ("Asana annoyances", "Asana frustrates me daily. Why isn't there a better tool?"),
        ("Notion can't find good template", "Cannot find any decent Notion template for OKRs."),
        ("Wish there was alternative", "I wish there was a real alternative to Jira."),
        (
            "Linear.app missing feature",
            "Linear.app is great but missing recurring tasks. Frustrating.",
        ),
        ("Hate ClickUp", "I hate ClickUp performance. It's useless on big projects."),
    ]
    for i, (title, body) in enumerate(pm_posts):
        sub = "saas" if i % 2 == 0 else "productivity"
        comments = [
            FakeComment(
                id=f"pm{i}c1", body="Struggling with this too. No good way exists.", score=5
            ),
            FakeComment(id=f"pm{i}c2", body="Switched to Height.app and it's amazing!", score=8),
        ]
        submissions.append(
            (
                sub,
                FakeSubmission(
                    id=f"pm{i:02d}",
                    title=title,
                    selftext=body,
                    score=20,
                    num_comments=12,
                    comments=FakeCommentForest(comments=comments),
                ),
            )
        )

    # Theme B: AI dev tools (across r/programming and r/dev)
    ai_posts = [
        (
            "Copilot keeps suggesting garbage",
            "Copilot autocomplete is broken on TypeScript projects.",
        ),
        (
            "Cursor too expensive",
            "Cursor pricing is annoying. Willing to pay if it actually works.",
        ),
        ("ChatGPT context window", "ChatGPT keeps losing context. Useless for big codebases."),
        ("Codeium vs Copilot", "Codeium is okay but Copilot still feels better most days."),
        (
            "AI tools are frustrating",
            "Every AI coding tool I tried is frustrating. Looking for an alternative.",
        ),
        ("Tabnine struggles", "Tabnine struggling to keep up with newer models."),
    ]
    for i, (title, body) in enumerate(ai_posts):
        sub = "programming" if i % 2 == 0 else "dev"
        comments = [
            FakeComment(
                id=f"ai{i}c1", body="I'd pay for something that doesn't hallucinate.", score=6
            ),
        ]
        submissions.append(
            (
                sub,
                FakeSubmission(
                    id=f"ai{i:02d}",
                    title=title,
                    selftext=body,
                    score=25,
                    num_comments=8,
                    comments=FakeCommentForest(comments=comments),
                ),
            )
        )

    subs: dict[str, FakeSubreddit] = {}
    for name, submission in submissions:
        subs.setdefault(name, FakeSubreddit()).submissions.append(submission)
    return FakeReddit(subs=subs)


def _seed_active_subs(db_path: Path, names: list[str]) -> None:
    for name in names:
        add_subreddit(db_path, name)


def _deterministic_embed(texts: list[str], _model: str, _batch: int) -> np.ndarray:
    """Hash text into a 384-d vector. Stable across runs. Two-mode separation:
    posts/comments mentioning 'Copilot', 'Cursor', 'ChatGPT', 'AI' cluster apart from
    posts mentioning 'Trello', 'Asana', 'Notion', 'Jira', 'Linear', 'ClickUp'."""
    rng = np.random.default_rng(0)
    base_pm = rng.normal(loc=+1.0, scale=0.1, size=384).astype(np.float32)
    base_ai = rng.normal(loc=-1.0, scale=0.1, size=384).astype(np.float32)
    pm_terms = ("trello", "asana", "notion", "jira", "linear", "clickup", "height")
    ai_terms = ("copilot", "cursor", "chatgpt", "codeium", "tabnine", "ai")

    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        lowered = t.lower()
        pm_hits = sum(1 for w in pm_terms if w in lowered)
        ai_hits = sum(1 for w in ai_terms if w in lowered)
        if pm_hits >= ai_hits:
            jitter = rng.normal(loc=0.0, scale=0.02, size=384).astype(np.float32)
            out[i] = base_pm + jitter
        else:
            jitter = rng.normal(loc=0.0, scale=0.02, size=384).astype(np.float32)
            out[i] = base_ai + jitter
    return out


def test_smoke_full_pipeline(tmp_db: Path, tmp_path: Path, mocker: MockerFixture) -> None:
    # Avoid spawning osascript on dev box (test box may be either platform)
    mocker.patch("reddit_miner.alert.sys.platform", "win32")
    mocker.patch("reddit_miner.embed.encode_texts", side_effect=_deterministic_embed)

    _seed_active_subs(tmp_db, ["saas", "productivity", "programming", "dev"])
    s = _settings_for(tmp_db, tmp_path)
    reddit = _fake_corpus()

    result = pipeline_run(s, reddit=reddit)

    # Ingest happened
    assert result["ingested"] > 10
    assert result["embedded"] == result["ingested"]
    assert result["reduced"] == result["ingested"]
    # Two well-separated themes should produce at least 2 clusters
    assert result["clusters"] >= 2
    # First run for a umap_version: no priors -> no alerts
    assert result["alerts"] == 0

    with connect(tmp_db) as conn:
        # Documents exist with cluster_ids assigned
        n_with_cluster = conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE cluster_id IS NOT NULL"
        ).fetchone()["c"]
        assert n_with_cluster > 0

        # Competitor mentions populated
        mentions = conn.execute("SELECT tool_name, sentiment FROM competitor_mentions").fetchall()
        tool_names = {r["tool_name"] for r in mentions}
        # At least one PM tool and one AI tool extracted
        assert any(
            t in tool_names for t in ["Trello", "Asana", "Notion", "Jira", "Linear", "ClickUp"]
        )
        assert any(t in tool_names for t in ["Copilot", "Cursor", "ChatGPT", "Codeium", "Tabnine"])


def test_smoke_pipeline_idempotent_on_rerun(
    tmp_db: Path, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "win32")
    mocker.patch("reddit_miner.embed.encode_texts", side_effect=_deterministic_embed)

    _seed_active_subs(tmp_db, ["saas", "productivity", "programming", "dev"])
    s = _settings_for(tmp_db, tmp_path)
    reddit = _fake_corpus()

    first = pipeline_run(s, reddit=reddit)
    second = pipeline_run(s, reddit=reddit)

    # Re-ingest sees no new docs (INSERT OR IGNORE)
    assert second["ingested"] == 0
    # No new embeddings
    assert second["embedded"] == 0
    # No new reductions (model already fit, all docs already reduced)
    assert second["reduced"] == 0
    # Cluster runs are NOT idempotent (each call creates a new run_id) — that's
    # correct: re-clustering with the same data writes a new cluster_run row.
    assert second["run_id"] > first["run_id"]
    # But alerts shouldn't double-fire (same cluster_ids would dedup, but cluster
    # IDs are unique per run, so this checks the new-cluster-detection logic
    # against priors). Second run has the first run as a prior; if centroids are
    # stable (same UMAP model, same docs), distances should be near-zero -> no alerts.
    assert second["alerts"] == 0
