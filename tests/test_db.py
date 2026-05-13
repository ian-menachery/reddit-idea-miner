from __future__ import annotations

from typing import TYPE_CHECKING

from reddit_miner.db import SCHEMA_VERSION, _current_version, connect, init_db, transaction

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_TABLES = {
    "schema_version",
    "subreddits",
    "documents",
    "embeddings",
    "reduced_embeddings",
    "umap_models",
    "cluster_runs",
    "clusters",
    "competitor_mentions",
    "alerts",
}


def _tables(db_path: Path) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {r["name"] for r in rows}


def test_init_db_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "miner.db"
    init_db(db)
    assert _tables(db) >= EXPECTED_TABLES


def test_init_db_sets_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "miner.db"
    version = init_db(db)
    assert version == SCHEMA_VERSION
    with connect(db) as conn:
        assert _current_version(conn) == SCHEMA_VERSION


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "miner.db"
    v1 = init_db(db)
    v2 = init_db(db)
    assert v1 == v2 == SCHEMA_VERSION
    with connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) AS c FROM schema_version").fetchone()
        assert rows["c"] == 1


def test_connect_enables_foreign_keys_and_wal(tmp_db: Path) -> None:
    with connect(tmp_db) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_foreign_key_cascade_on_document_delete(tmp_db: Path) -> None:
    with connect(tmp_db) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO documents (id, type, subreddit, score, created_utc, "
            "permalink, pain_score, ingested_at) "
            "VALUES ('d1', 'post', 'r/test', 10, 1700000000, '/r/test/d1', 1.0, 1700000000)"
        )
        conn.execute("INSERT INTO embeddings (document_id, vector) VALUES ('d1', X'00')")
    with connect(tmp_db) as conn, transaction(conn):
        conn.execute("DELETE FROM documents WHERE id = 'd1'")
    with connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
        assert n == 0


def test_transaction_rolls_back_on_error(tmp_db: Path) -> None:
    try:
        with connect(tmp_db) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO documents (id, type, subreddit, score, created_utc, "
                "permalink, pain_score, ingested_at) "
                "VALUES ('d2', 'post', 'r/x', 1, 1700000000, '/r/x/d2', 0.0, 1700000000)"
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    with connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        assert n == 0


def test_subreddit_status_check_constraint(tmp_db: Path) -> None:
    import sqlite3

    with connect(tmp_db) as conn, transaction(conn):
        try:
            conn.execute(
                "INSERT INTO subreddits (name, status, source, added_at) "
                "VALUES ('r/x', 'bogus', 'seed', 1700000000)"
            )
        except sqlite3.IntegrityError:
            return
    raise AssertionError("expected CHECK constraint to reject 'bogus' status")
