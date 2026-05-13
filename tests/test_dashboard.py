from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from reddit_miner import dashboard
from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    from pathlib import Path


def _seed(db_path: Path) -> None:
    """One cluster_run, one cluster, three documents, two mentions, one alert."""
    now = time.time()
    with connect(db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO umap_models (version, fitted_at, n_documents, params_json) "
            "VALUES (1, ?, 1, '{}')",
            (now,),
        )
        cur = conn.execute(
            "INSERT INTO cluster_runs (run_at, n_documents, n_clusters, umap_version) "
            "VALUES (?, 3, 1, 1)",
            (now,),
        )
        run_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO clusters (run_id, label_tfidf, label_llm, size, avg_pain_score, "
            "subreddit_spread, signal_score, centroid, created_at) "
            "VALUES (?, 'pain, tool', NULL, 3, 1.5, 2, 5.0, X'00', ?)",
            (run_id, now),
        )
        cluster_id = cur.lastrowid
        # 3 docs, one of which is an outlier
        for i, (doc_id, pain, outlier) in enumerate(
            [("d1", 2.0, 0), ("d2", 1.5, 0), ("d3", 0.5, 1)]
        ):
            conn.execute(
                "INSERT INTO documents (id, type, parent_id, subreddit, title, body, "
                "score, num_comments, created_utc, permalink, pain_score, "
                "cluster_id, is_outlier, ingested_at) "
                "VALUES (?, 'post', NULL, 'r/t', ?, ?, 10, 5, ?, '/r/t/x', ?, ?, ?, ?)",
                (doc_id, f"title{i}", f"body{i}", now, pain, cluster_id, outlier, now),
            )
        conn.execute(
            "INSERT INTO competitor_mentions "
            "(cluster_id, tool_name, sentiment, context, source_document_id) "
            "VALUES (?, 'Trello', 'negative', 'hate Trello', 'd1')",
            (cluster_id,),
        )
        conn.execute(
            "INSERT INTO competitor_mentions "
            "(cluster_id, tool_name, sentiment, context, source_document_id) "
            "VALUES (?, 'Asana', 'positive', 'love Asana', 'd2')",
            (cluster_id,),
        )
        conn.execute(
            "INSERT INTO alerts (cluster_id, signal_score, triggered_at, notified) "
            "VALUES (?, 5.0, ?, 1)",
            (cluster_id, now),
        )


def test_cluster_runs_returns_rows(tmp_db: Path) -> None:
    _seed(tmp_db)
    df = dashboard.cluster_runs(str(tmp_db))
    assert len(df) == 1
    assert df.iloc[0]["n_clusters"] == 1


def test_clusters_for_run_ordered_by_signal_score(tmp_db: Path) -> None:
    _seed(tmp_db)
    runs = dashboard.cluster_runs(str(tmp_db))
    run_id = int(runs.iloc[0]["id"])
    df = dashboard.clusters_for_run(str(tmp_db), run_id)
    assert len(df) == 1
    assert df.iloc[0]["label_tfidf"] == "pain, tool"


def test_docs_for_cluster_excludes_outliers_by_default(tmp_db: Path) -> None:
    _seed(tmp_db)
    runs = dashboard.cluster_runs(str(tmp_db))
    clusters = dashboard.clusters_for_run(str(tmp_db), int(runs.iloc[0]["id"]))
    cid = int(clusters.iloc[0]["id"])
    df_no_outliers = dashboard.docs_for_cluster(str(tmp_db), cid)
    assert len(df_no_outliers) == 2
    assert "d3" not in df_no_outliers["id"].tolist()


def test_docs_for_cluster_includes_outliers_when_asked(tmp_db: Path) -> None:
    _seed(tmp_db)
    runs = dashboard.cluster_runs(str(tmp_db))
    clusters = dashboard.clusters_for_run(str(tmp_db), int(runs.iloc[0]["id"]))
    cid = int(clusters.iloc[0]["id"])
    df = dashboard.docs_for_cluster(str(tmp_db), cid, include_outliers=True)
    assert len(df) == 3


def test_mentions_for_run_joins_correctly(tmp_db: Path) -> None:
    _seed(tmp_db)
    runs = dashboard.cluster_runs(str(tmp_db))
    df = dashboard.mentions_for_run(str(tmp_db), int(runs.iloc[0]["id"]))
    assert len(df) == 2
    assert set(df["tool_name"]) == {"Trello", "Asana"}
    assert "label_tfidf" in df.columns


def test_recent_alerts_returns_rows(tmp_db: Path) -> None:
    _seed(tmp_db)
    df = dashboard.recent_alerts(str(tmp_db))
    assert len(df) == 1
    assert df.iloc[0]["notified"] == 1


def test_fmt_ts_handles_none_and_float() -> None:
    assert dashboard.fmt_ts(None) == ""
    s = dashboard.fmt_ts(1700000000.0)
    assert s.startswith("2023-11-14")  # 1700000000 == 2023-11-14T22:13:20+00:00


def test_dashboard_helpers_handle_empty_db(tmp_db: Path) -> None:
    """Fresh DB should yield empty DataFrames, not errors."""
    assert dashboard.cluster_runs(str(tmp_db)).empty
    assert dashboard.recent_alerts(str(tmp_db)).empty
    assert dashboard.clusters_for_run(str(tmp_db), 999).empty
    assert dashboard.mentions_for_run(str(tmp_db), 999).empty


def test_dashboard_data_dtype_signal_score(tmp_db: Path) -> None:
    _seed(tmp_db)
    runs = dashboard.cluster_runs(str(tmp_db))
    df = dashboard.clusters_for_run(str(tmp_db), int(runs.iloc[0]["id"]))
    assert np.issubdtype(df["signal_score"].dtype, np.floating)
