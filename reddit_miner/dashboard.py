"""Pure data-loading helpers for the Streamlit dashboard. No streamlit imports here
so these can be unit-tested without spinning up an app context."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd


def cluster_runs(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT id, run_at, n_documents, n_clusters, umap_version "
            "FROM cluster_runs ORDER BY id DESC",
            conn,
        )


def clusters_for_run(db_path: str, run_id: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT id, label_tfidf, label_llm, size, avg_pain_score, "
            "subreddit_spread, signal_score "
            "FROM clusters WHERE run_id = ? ORDER BY signal_score DESC",
            conn,
            params=(run_id,),
        )


def docs_for_cluster(db_path: str, cluster_id: int, include_outliers: bool = False) -> pd.DataFrame:
    outlier_filter = "" if include_outliers else "AND is_outlier = 0"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT id, type, subreddit, title, body, score, num_comments, "
            "pain_score, permalink, is_outlier "
            f"FROM documents WHERE cluster_id = ? {outlier_filter} "
            "ORDER BY pain_score DESC LIMIT 50",
            conn,
            params=(cluster_id,),
        )


def mentions_for_run(db_path: str, run_id: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT m.cluster_id, c.label_tfidf, m.tool_name, m.sentiment, m.context, "
            "m.source_document_id "
            "FROM competitor_mentions m JOIN clusters c ON c.id = m.cluster_id "
            "WHERE c.run_id = ? ORDER BY m.tool_name",
            conn,
            params=(run_id,),
        )


def recent_alerts(db_path: str, limit: int = 100) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT a.id, a.cluster_id, c.label_tfidf, a.signal_score, "
            "a.triggered_at, a.notified "
            "FROM alerts a LEFT JOIN clusters c ON c.id = a.cluster_id "
            "ORDER BY a.triggered_at DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec="seconds")
