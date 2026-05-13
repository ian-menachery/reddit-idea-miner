from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from reddit_miner.alert import (
    cosine_distances,
    detect_new_clusters,
    fire_alerts,
    notify,
)
from reddit_miner.config import Settings
from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


# ---------- cosine_distances ----------


def test_cosine_distance_zero_when_parallel() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    d = cosine_distances(a, b)
    assert d[0] < 1e-6


def test_cosine_distance_two_when_antiparallel() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    d = cosine_distances(a, b)
    assert abs(d[0] - 2.0) < 1e-6


def test_cosine_distance_one_when_orthogonal() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    d = cosine_distances(a, b)
    assert abs(d[0] - 1.0) < 1e-6


# ---------- notify (platform guard) ----------


def test_notify_returns_false_on_non_darwin(mocker: MockerFixture) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "win32")
    assert notify("title", "message") is False


def test_notify_invokes_osascript_on_darwin(mocker: MockerFixture) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "darwin")
    run_mock = mocker.patch("reddit_miner.alert.subprocess.run")
    assert notify("Reddit Miner", "new cluster") is True
    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    assert args[0] == ["osascript", "-"]
    assert "display notification" in kwargs["input"]


def test_notify_returns_false_when_osascript_missing(mocker: MockerFixture) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "darwin")
    mocker.patch("reddit_miner.alert.subprocess.run", side_effect=FileNotFoundError)
    assert notify("t", "m") is False


# ---------- new-cluster detection ----------


def _seed_run(
    db_path: Path,
    run_id_target: int,
    umap_version: int,
    clusters: list[tuple[np.ndarray, float, str]],
) -> int:
    """Insert a cluster_run + its clusters. Returns the actual run_id assigned."""
    now = time.time()
    with connect(db_path) as conn, transaction(conn):
        # Ensure umap_version row exists
        conn.execute(
            "INSERT OR IGNORE INTO umap_models (version, fitted_at, n_documents, params_json) "
            "VALUES (?, ?, 1, '{}')",
            (umap_version, now),
        )
        cur = conn.execute(
            "INSERT INTO cluster_runs (run_at, n_documents, n_clusters, umap_version) "
            "VALUES (?, ?, ?, ?)",
            (now, 100, len(clusters), umap_version),
        )
        run_id = int(cur.lastrowid)
        for centroid, signal, label in clusters:
            conn.execute(
                "INSERT INTO clusters (run_id, label_tfidf, size, avg_pain_score, "
                "subreddit_spread, signal_score, centroid, created_at) "
                "VALUES (?, ?, 10, 1.0, 2, ?, ?, ?)",
                (run_id, label, signal, centroid.astype(np.float32).tobytes(), now),
            )
    return run_id


def _settings_for(tmp_db: Path, tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "db_path": tmp_db,
        "alerts_path": tmp_path / "alerts.md",
        "new_cluster_cosine_threshold": 0.30,
        "signal_score_threshold": 5.0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_detect_new_clusters_first_run_returns_empty(tmp_db: Path, tmp_path: Path) -> None:
    # Only one cluster_run exists — no priors to compare against.
    _seed_run(
        tmp_db,
        1,
        umap_version=1,
        clusters=[(np.array([1.0, 0, 0]), 100.0, "a")],
    )
    s = _settings_for(tmp_db, tmp_path)
    assert detect_new_clusters(s) == []


def test_detect_new_clusters_flags_far_high_signal(tmp_db: Path, tmp_path: Path) -> None:
    # Prior run: centroid at (1,0,0)
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    # Current run: centroid at (-1,0,0) — cosine distance ~2.0, well above 0.30
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([-1.0, 0, 0]), 100.0, "current_new")])

    s = _settings_for(tmp_db, tmp_path)
    cands = detect_new_clusters(s)
    assert len(cands) == 1
    assert cands[0]["label"] == "current_new"
    assert cands[0]["min_distance"] > 0.3


def test_detect_new_clusters_skips_close_centroids(tmp_db: Path, tmp_path: Path) -> None:
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([1.0, 0.01, 0]), 100.0, "current_near")])
    s = _settings_for(tmp_db, tmp_path)
    assert detect_new_clusters(s) == []


def test_detect_new_clusters_skips_low_signal(tmp_db: Path, tmp_path: Path) -> None:
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    # Far but below signal threshold
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([-1.0, 0, 0]), 1.0, "low_signal")])
    s = _settings_for(tmp_db, tmp_path)
    assert detect_new_clusters(s) == []


def test_detect_new_clusters_ignores_different_umap_version(tmp_db: Path, tmp_path: Path) -> None:
    # Prior run in umap_version=1
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    # Current run in umap_version=2 — different geometry, prior centroids irrelevant
    _seed_run(tmp_db, 2, 2, clusters=[(np.array([-1.0, 0, 0]), 100.0, "current")])
    s = _settings_for(tmp_db, tmp_path)
    assert detect_new_clusters(s) == []  # no prior in v2, treated as first run


# ---------- fire_alerts ----------


def test_fire_alerts_writes_markdown_and_db_rows(
    tmp_db: Path, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "win32")  # skip osascript
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([-1.0, 0, 0]), 100.0, "novel theme")])
    s = _settings_for(tmp_db, tmp_path)

    n = fire_alerts(s)
    assert n == 1

    with connect(tmp_db) as conn:
        alerts = conn.execute("SELECT cluster_id, signal_score, notified FROM alerts").fetchall()
    assert len(alerts) == 1
    assert alerts[0]["notified"] == 0  # not on darwin

    md = s.alerts_path.read_text(encoding="utf-8")
    assert "novel theme" in md
    assert "Signal score:" in md


def test_fire_alerts_is_idempotent_on_same_cluster(
    tmp_db: Path, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "win32")
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([-1.0, 0, 0]), 100.0, "new")])
    s = _settings_for(tmp_db, tmp_path)

    n1 = fire_alerts(s)
    n2 = fire_alerts(s)
    assert n1 == 1
    assert n2 == 0
    with connect(tmp_db) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()["c"]
    assert total == 1


def test_fire_alerts_sets_notified_on_darwin(
    tmp_db: Path, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch("reddit_miner.alert.sys.platform", "darwin")
    mocker.patch("reddit_miner.alert.subprocess.run")  # successful no-op
    _seed_run(tmp_db, 1, 1, clusters=[(np.array([1.0, 0, 0]), 50.0, "prior")])
    _seed_run(tmp_db, 2, 1, clusters=[(np.array([-1.0, 0, 0]), 100.0, "new")])
    s = _settings_for(tmp_db, tmp_path)

    fire_alerts(s)
    with connect(tmp_db) as conn:
        row = conn.execute("SELECT notified FROM alerts").fetchone()
    assert row["notified"] == 1
