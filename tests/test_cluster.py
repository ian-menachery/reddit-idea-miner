from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from reddit_miner.cluster import cluster_documents, signal_score
from reddit_miner.config import Settings
from reddit_miner.db import connect, transaction
from reddit_miner.label import CtfidfLabeler, ctfidf_labels, get_labeler
from reddit_miner.reduce import reduce_pending

if TYPE_CHECKING:
    from pathlib import Path


# ---------- signal_score ----------


def test_signal_score_zero_with_zero_pain() -> None:
    assert signal_score(size=10, avg_pain=0.0, subreddit_spread=3) == 0.0


def test_signal_score_zero_with_zero_size() -> None:
    assert signal_score(size=0, avg_pain=5.0, subreddit_spread=3) == 0.0


def test_signal_score_rewards_spread_logarithmically() -> None:
    s1 = signal_score(10, 2.0, subreddit_spread=1)
    s2 = signal_score(10, 2.0, subreddit_spread=4)
    s3 = signal_score(10, 2.0, subreddit_spread=16)
    assert s1 < s2 < s3
    # log1p damping: 16x spread isn't 16x signal
    assert s3 / s1 < 5.0


# ---------- ctfidf_labels ----------


def test_ctfidf_labels_picks_class_specific_words() -> None:
    clusters = {
        0: ["python pandas dataframe", "python jupyter notebook", "python numpy array"],
        1: ["car engine repair", "engine oil change", "car transmission"],
    }
    labels = ctfidf_labels(clusters, top_n=3)
    assert "python" in labels[0]
    # 'engine' or 'car' should be in cluster 1's label
    assert "engine" in labels[1] or "car" in labels[1]


def test_ctfidf_labels_handles_empty_dict() -> None:
    assert ctfidf_labels({}) == {}


def test_ctfidf_labels_handles_stopword_only_docs() -> None:
    clusters = {0: ["the and a"], 1: ["of to in"]}
    labels = ctfidf_labels(clusters)
    assert set(labels) == {0, 1}


def test_get_labeler_returns_ctfidf() -> None:
    assert isinstance(get_labeler("ctfidf"), CtfidfLabeler)


def test_get_labeler_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown labeler"):
        get_labeler("claude")


# ---------- cluster_documents (real HDBSCAN on synthetic data) ----------


def _seed_two_cluster_corpus(db_path: Path, n_per_cluster: int = 12) -> None:
    """Two well-separated clusters in raw embedding space, two subreddits."""
    rng = np.random.default_rng(0)

    with connect(db_path) as conn, transaction(conn):
        # cluster A in r/python
        for i in range(n_per_cluster):
            doc_id = f"A{i:02d}"
            text = f"python pandas dataframe issue {i}"
            vec = rng.normal(loc=+1.0, scale=0.05, size=384).astype(np.float32)
            conn.execute(
                "INSERT INTO documents (id, type, parent_id, subreddit, title, body, "
                "score, num_comments, created_utc, permalink, pain_score, ingested_at) "
                "VALUES (?, 'post', NULL, 'python', ?, ?, 10, 5, ?, '/r/python/x', 1.5, ?)",
                (doc_id, text, text, time.time(), time.time()),
            )
            conn.execute(
                "INSERT INTO embeddings (document_id, vector) VALUES (?, ?)",
                (doc_id, vec.tobytes()),
            )
        # cluster B split across r/cars and r/mechanics
        for i in range(n_per_cluster):
            doc_id = f"B{i:02d}"
            sub = "cars" if i % 2 == 0 else "mechanics"
            text = f"engine oil leak problem {i}"
            vec = rng.normal(loc=-1.0, scale=0.05, size=384).astype(np.float32)
            conn.execute(
                "INSERT INTO documents (id, type, parent_id, subreddit, title, body, "
                "score, num_comments, created_utc, permalink, pain_score, ingested_at) "
                "VALUES (?, 'post', NULL, ?, ?, ?, 10, 5, ?, '/r/x/y', 2.0, ?)",
                (doc_id, sub, text, text, time.time(), time.time()),
            )
            conn.execute(
                "INSERT INTO embeddings (document_id, vector) VALUES (?, ?)",
                (doc_id, vec.tobytes()),
            )


def _settings_for(tmp_path: Path, db_path: Path) -> Settings:
    return Settings(
        db_path=db_path,
        umap_model_path=tmp_path / "umap.joblib",
        umap_n_components=3,
        umap_n_neighbors=8,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_random_state=42,
        hdbscan_min_cluster_size=4,
        labeler="ctfidf",
    )


def test_cluster_documents_end_to_end(tmp_db: Path, tmp_path: Path) -> None:
    _seed_two_cluster_corpus(tmp_db, n_per_cluster=12)
    s = _settings_for(tmp_path, tmp_db)
    reduce_pending(s)

    n_clusters, run_id = cluster_documents(s)
    assert n_clusters >= 2
    assert run_id > 0

    with connect(tmp_db) as conn:
        # cluster_run row exists
        runs = conn.execute("SELECT * FROM cluster_runs WHERE id = ?", (run_id,)).fetchall()
        assert len(runs) == 1
        assert runs[0]["n_clusters"] == n_clusters
        assert runs[0]["umap_version"] == 1

        clusters = conn.execute(
            "SELECT id, label_tfidf, size, subreddit_spread, signal_score, centroid "
            "FROM clusters WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert len(clusters) == n_clusters

        # Centroids in reduced space (5-d or umap_n_components)
        for c in clusters:
            arr = np.frombuffer(c["centroid"], dtype=np.float32)
            assert arr.shape == (s.umap_n_components,)
            assert c["size"] > 0
            assert c["signal_score"] >= 0

        # All non-outlier docs have a cluster_id
        unassigned = conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE cluster_id IS NULL AND is_outlier = 0"
        ).fetchone()["c"]
        # 0 unassigned non-outliers (every assigned doc has cluster_id set)
        # In practice some docs may end up outliers and reassigned; both have cluster_id
        assert unassigned == 0


def test_cluster_documents_reassignment_clears_old(tmp_db: Path, tmp_path: Path) -> None:
    """Two consecutive cluster runs should not leave stale cluster_id values."""
    _seed_two_cluster_corpus(tmp_db, n_per_cluster=8)
    s = _settings_for(tmp_path, tmp_db)
    reduce_pending(s)

    _, run1 = cluster_documents(s)
    _, run2 = cluster_documents(s)
    assert run2 > run1

    with connect(tmp_db) as conn:
        # Documents reference run2's clusters, not run1's
        rows = conn.execute(
            "SELECT DISTINCT cluster_id FROM documents WHERE cluster_id IS NOT NULL"
        ).fetchall()
        cluster_ids = {r["cluster_id"] for r in rows}
        run2_cluster_ids = {
            r["id"] for r in conn.execute("SELECT id FROM clusters WHERE run_id = ?", (run2,))
        }
        assert cluster_ids <= run2_cluster_ids


def test_cluster_documents_noop_without_reduced_embeddings(tmp_db: Path, tmp_path: Path) -> None:
    s = _settings_for(tmp_path, tmp_db)
    n, run_id = cluster_documents(s)
    assert (n, run_id) == (0, 0)


def test_signal_score_persisted_matches_formula(tmp_db: Path, tmp_path: Path) -> None:
    _seed_two_cluster_corpus(tmp_db, n_per_cluster=10)
    s = _settings_for(tmp_path, tmp_db)
    reduce_pending(s)
    cluster_documents(s)

    with connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT size, avg_pain_score, subreddit_spread, signal_score FROM clusters"
        ).fetchall()
    for r in rows:
        expected = signal_score(r["size"], r["avg_pain_score"], r["subreddit_spread"])
        assert abs(r["signal_score"] - expected) < 1e-6
