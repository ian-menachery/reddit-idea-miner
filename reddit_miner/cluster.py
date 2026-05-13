from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from reddit_miner.db import connect, transaction
from reddit_miner.label import get_labeler
from reddit_miner.reduce import current_umap_version, load_reduced_embeddings

if TYPE_CHECKING:
    from pathlib import Path

    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)


def _run_hdbscan(
    matrix: npt.NDArray[np.float32], min_cluster_size: int
) -> tuple[Any, npt.NDArray[np.int64], npt.NDArray[np.int8]]:
    """Fit HDBSCAN, reassign noise to nearest cluster, return (model, labels, is_outlier).
    is_outlier[i] == 1 means HDBSCAN flagged it noise but we glued it to nearest cluster."""
    import hdbscan  # lazy: numba JIT

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    raw_labels = clusterer.fit_predict(matrix).astype(np.int64)
    is_outlier = (raw_labels == -1).astype(np.int8)

    if is_outlier.any() and (raw_labels != -1).any():
        noise_idx = np.flatnonzero(is_outlier)
        soft_labels, _ = hdbscan.approximate_predict(clusterer, matrix[noise_idx])
        # approximate_predict can also emit -1 for very distant points; force-assign
        # those to the closest cluster centroid by euclidean distance.
        unresolved = soft_labels == -1
        if unresolved.any():
            valid_mask = raw_labels != -1
            valid_centroids: dict[int, npt.NDArray[np.float32]] = {}
            for cid in sorted({int(c) for c in raw_labels[valid_mask]}):
                valid_centroids[cid] = matrix[raw_labels == cid].mean(axis=0)
            cid_arr = np.array(sorted(valid_centroids.keys()), dtype=np.int64)
            centroid_stack = np.stack([valid_centroids[int(c)] for c in cid_arr])
            for i, is_unresolved in enumerate(unresolved):
                if not is_unresolved:
                    continue
                point = matrix[noise_idx[i]]
                d = np.linalg.norm(centroid_stack - point, axis=1)
                soft_labels[i] = cid_arr[int(np.argmin(d))]
        raw_labels[noise_idx] = soft_labels

    return clusterer, raw_labels, is_outlier


def _load_doc_meta(db_path: Path, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, type, title, body, subreddit, pain_score "
            f"FROM documents WHERE id IN ({placeholders})",
            doc_ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _text_for_labeling(row: dict[str, Any]) -> str:
    if row["type"] == "post":
        return f"{row.get('title') or ''} {row.get('body') or ''}".strip()
    return str(row.get("body") or "")


def _compute_cluster_stats(
    cluster_id: int,
    member_indices: list[int],
    matrix: npt.NDArray[np.float32],
    is_outlier: npt.NDArray[np.int8],
    doc_ids: list[str],
    doc_meta: dict[str, dict[str, Any]],
) -> tuple[npt.NDArray[np.float32], int, float, int, float]:
    """Returns (centroid, size, avg_pain, subreddit_spread, signal_score).
    Centroid + stats use NON-outlier members only (falls back to all if cluster
    happens to be all-outlier)."""
    core = [i for i in member_indices if not is_outlier[i]]
    if not core:
        core = member_indices

    core_matrix = matrix[core]
    centroid = core_matrix.mean(axis=0).astype(np.float32)
    pain = np.array([float(doc_meta[doc_ids[i]]["pain_score"]) for i in core], dtype=np.float64)
    subs = {doc_meta[doc_ids[i]]["subreddit"] for i in core}
    size = len(core)
    avg_pain = float(pain.mean()) if pain.size else 0.0
    spread = len(subs)
    signal = float(size * avg_pain * np.log1p(spread))
    return centroid, size, avg_pain, spread, signal


def cluster_documents(settings: Settings) -> tuple[int, int]:
    """Cluster all reduced embeddings, label, persist. Returns (n_clusters, run_id)."""
    doc_ids, matrix = load_reduced_embeddings(settings.db_path)
    if matrix.shape[0] == 0:
        logger.warning("No reduced embeddings — run `reduce` first.")
        return 0, 0

    version = current_umap_version(settings.db_path)
    if version is None:
        raise RuntimeError("No UMAP version recorded — run `reduce` first.")

    _, labels, is_outlier = _run_hdbscan(matrix, settings.hdbscan_min_cluster_size)

    unique_clusters = sorted({int(c) for c in labels if c != -1})
    if not unique_clusters:
        logger.warning(
            "HDBSCAN found no clusters at min_cluster_size=%d. Skipping persist.",
            settings.hdbscan_min_cluster_size,
        )
        return 0, 0

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        if c == -1:
            continue
        cluster_to_indices[int(c)].append(i)

    doc_meta = _load_doc_meta(settings.db_path, doc_ids)

    docs_by_cluster_text: dict[int, list[str]] = {
        cid: [_text_for_labeling(doc_meta[doc_ids[i]]) for i in idxs]
        for cid, idxs in cluster_to_indices.items()
    }
    labeler = get_labeler(settings.labeler)
    cluster_labels = labeler.label(docs_by_cluster_text)

    stats = {
        cid: _compute_cluster_stats(
            cid, cluster_to_indices[cid], matrix, is_outlier, doc_ids, doc_meta
        )
        for cid in unique_clusters
    }

    now = time.time()
    with connect(settings.db_path) as conn, transaction(conn):
        cur = conn.execute(
            "INSERT INTO cluster_runs (run_at, n_documents, n_clusters, umap_version) "
            "VALUES (?, ?, ?, ?)",
            (now, len(doc_ids), len(unique_clusters), version),
        )
        run_id = cur.lastrowid
        if run_id is None:
            raise RuntimeError("Failed to insert cluster_runs row")

        cid_to_db_id: dict[int, int] = {}
        for cid in unique_clusters:
            centroid, size, avg_pain, spread, signal = stats[cid]
            cur = conn.execute(
                "INSERT INTO clusters (run_id, label_tfidf, size, avg_pain_score, "
                "subreddit_spread, signal_score, centroid, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    cluster_labels.get(cid, ""),
                    size,
                    avg_pain,
                    spread,
                    signal,
                    centroid.tobytes(),
                    now,
                ),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Failed to insert cluster row")
            cid_to_db_id[cid] = int(cur.lastrowid)

        conn.execute("UPDATE documents SET cluster_id = NULL, is_outlier = 0")
        for i, c in enumerate(labels):
            if c == -1:
                continue
            conn.execute(
                "UPDATE documents SET cluster_id = ?, is_outlier = ? WHERE id = ?",
                (cid_to_db_id[int(c)], int(is_outlier[i]), doc_ids[i]),
            )

    logger.info(
        "Clustered %d documents into %d clusters (run_id=%d, umap_version=%d)",
        len(doc_ids),
        len(unique_clusters),
        run_id,
        version,
    )
    return len(unique_clusters), int(run_id)


def signal_score(size: int, avg_pain: float, subreddit_spread: int) -> float:
    """Public helper: size * avg_pain * log1p(spread). Spread is the prize."""
    return float(size * avg_pain * np.log1p(subreddit_spread))
