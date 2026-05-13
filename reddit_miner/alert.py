from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt

from reddit_miner.db import connect, transaction
from reddit_miner.embed import EMBED_DTYPE

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)


class AlertCandidate(TypedDict):
    cluster_id: int
    label: str
    signal_score: float
    min_distance: float


def cosine_distances(
    point: npt.NDArray[np.float32], others: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """1 - cosine_similarity between `point` (1d) and each row of `others` (2d)."""
    p = point.astype(np.float64)
    o = others.astype(np.float64)
    p_norm = p / max(float(np.linalg.norm(p)), 1e-12)
    o_norm = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
    result: npt.NDArray[np.float64] = 1.0 - (o_norm @ p_norm)
    return result


def _as_applescript_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(title: str, message: str) -> bool:
    """Fire an osascript notification on macOS. Returns True iff dispatched.
    On non-Darwin platforms, returns False without erroring (alerts.md still records the event)."""
    if sys.platform != "darwin":
        return False
    script = f"display notification {_as_applescript_str(message)} with title {_as_applescript_str(title)}"
    try:
        subprocess.run(
            ["osascript", "-"],
            input=script,
            text=True,
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.exception("osascript notification failed")
        return False


def _latest_run(conn: sqlite3.Connection) -> tuple[int, int] | None:
    row = conn.execute(
        "SELECT id, umap_version FROM cluster_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return int(row["id"]), int(row["umap_version"])


def detect_new_clusters(settings: Settings) -> list[AlertCandidate]:
    """For the latest cluster_run, find clusters whose centroid is >threshold cosine
    distance from every centroid in earlier runs (in the same umap_version) AND whose
    signal_score >= signal_score_threshold.

    First run for a given umap_version returns no candidates — there are no priors to compare to.
    """
    with connect(settings.db_path) as conn:
        latest = _latest_run(conn)
        if latest is None:
            return []
        latest_run_id, umap_version = latest

        current_rows = conn.execute(
            "SELECT id, label_tfidf, signal_score, centroid FROM clusters WHERE run_id = ?",
            (latest_run_id,),
        ).fetchall()
        prior_rows = conn.execute(
            "SELECT c.centroid FROM clusters c "
            "JOIN cluster_runs r ON c.run_id = r.id "
            "WHERE r.umap_version = ? AND c.run_id < ?",
            (umap_version, latest_run_id),
        ).fetchall()

    if not current_rows or not prior_rows:
        return []

    prior_centroids = np.stack(
        [np.frombuffer(r["centroid"], dtype=EMBED_DTYPE) for r in prior_rows]
    )

    candidates: list[AlertCandidate] = []
    for c in current_rows:
        centroid = np.frombuffer(c["centroid"], dtype=EMBED_DTYPE)
        d = cosine_distances(centroid, prior_centroids)
        min_dist = float(d.min())
        if (
            min_dist > settings.new_cluster_cosine_threshold
            and float(c["signal_score"]) >= settings.signal_score_threshold
        ):
            candidates.append(
                AlertCandidate(
                    cluster_id=int(c["id"]),
                    label=str(c["label_tfidf"]),
                    signal_score=float(c["signal_score"]),
                    min_distance=min_dist,
                )
            )
    return candidates


def _append_markdown(path: Path, candidates: list[AlertCandidate], when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromtimestamp(when, tz=UTC).isoformat()
    with path.open("a", encoding="utf-8") as f:
        for c in candidates:
            f.write(
                f"## {ts} — cluster {c['cluster_id']}\n"
                f"- **Label:** {c['label']}\n"
                f"- **Signal score:** {c['signal_score']:.2f}\n"
                f"- **Min cosine distance to prior centroid:** {c['min_distance']:.3f}\n\n"
            )


def fire_alerts(settings: Settings) -> int:
    """Detect new high-signal clusters and dispatch. Returns count of alerts created.

    Idempotency: skips clusters that already have an `alerts` row (a re-run on the same
    cluster_run won't double-alert).
    """
    candidates = detect_new_clusters(settings)
    if not candidates:
        logger.info("No new-cluster candidates.")
        return 0

    cluster_ids = [c["cluster_id"] for c in candidates]
    placeholders = ",".join("?" * len(cluster_ids))
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            f"SELECT cluster_id FROM alerts WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        ).fetchall()
    already = {int(r["cluster_id"]) for r in rows}
    candidates = [c for c in candidates if c["cluster_id"] not in already]
    if not candidates:
        logger.info("All candidate clusters already alerted on. Skipping.")
        return 0

    now = time.time()
    alert_ids: list[int] = []
    with connect(settings.db_path) as conn, transaction(conn):
        for cand in candidates:
            cur = conn.execute(
                "INSERT INTO alerts (cluster_id, signal_score, triggered_at, notified) "
                "VALUES (?, ?, ?, 0)",
                (cand["cluster_id"], cand["signal_score"], now),
            )
            if cur.lastrowid is None:
                raise RuntimeError("Failed to insert alert row")
            alert_ids.append(int(cur.lastrowid))

    _append_markdown(settings.alerts_path, candidates, now)

    notified_flags: list[int] = []
    for cand in candidates:
        ok = notify(
            title="Reddit Miner: new high-signal cluster",
            message=f"{cand['label']} (signal={cand['signal_score']:.1f})",
        )
        notified_flags.append(1 if ok else 0)

    with connect(settings.db_path) as conn, transaction(conn):
        for alert_id, flag in zip(alert_ids, notified_flags, strict=True):
            conn.execute("UPDATE alerts SET notified = ? WHERE id = ?", (flag, alert_id))

    logger.info(
        "Fired %d alerts (%d notified via osascript)",
        len(candidates),
        sum(notified_flags),
    )
    return len(candidates)
