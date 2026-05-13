from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS subreddits (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'inactive')),
    source TEXT NOT NULL,
    overlap_score REAL,
    evidence_json TEXT,
    added_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('post', 'comment')),
    parent_id TEXT,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    score INTEGER NOT NULL,
    num_comments INTEGER NOT NULL DEFAULT 0,
    created_utc REAL NOT NULL,
    permalink TEXT NOT NULL,
    pain_score REAL NOT NULL,
    matched_patterns TEXT NOT NULL DEFAULT '[]',
    cluster_id INTEGER,
    is_outlier INTEGER NOT NULL DEFAULT 0 CHECK (is_outlier IN (0, 1)),
    ingested_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_subreddit ON documents (subreddit);
CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents (cluster_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (type);

CREATE TABLE IF NOT EXISTS embeddings (
    document_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reduced_embeddings (
    document_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS umap_models (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    fitted_at REAL NOT NULL,
    n_documents INTEGER NOT NULL,
    params_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at REAL NOT NULL,
    n_documents INTEGER NOT NULL,
    n_clusters INTEGER NOT NULL,
    umap_version INTEGER NOT NULL,
    FOREIGN KEY (umap_version) REFERENCES umap_models(version)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    label_tfidf TEXT NOT NULL,
    label_llm TEXT,
    label_llm_model TEXT,
    size INTEGER NOT NULL,
    avg_pain_score REAL NOT NULL,
    subreddit_spread INTEGER NOT NULL,
    signal_score REAL NOT NULL,
    centroid BLOB NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES cluster_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters (run_id);

CREATE TABLE IF NOT EXISTS competitor_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    sentiment TEXT NOT NULL
        CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    context TEXT NOT NULL,
    source_document_id TEXT NOT NULL,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
    FOREIGN KEY (source_document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_mentions_cluster ON competitor_mentions (cluster_id);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    signal_score REAL NOT NULL,
    triggered_at REAL NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0 CHECK (notified IN (0, 1)),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);
"""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with WAL + FK enforcement. Autocommit mode;
    callers manage transactions explicitly via the `transaction` helper."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Wrap a block of writes in an explicit BEGIN/COMMIT (or ROLLBACK on error)."""
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"]) if row and row["v"] is not None else 0


def init_db(db_path: Path) -> int:
    """Apply pending migrations. Returns the schema version after migration. Idempotent."""
    with connect(db_path) as conn:
        existing = _current_version(conn)
        if existing >= SCHEMA_VERSION:
            logger.debug("DB at version %d (current=%d)", existing, SCHEMA_VERSION)
            return existing
        logger.info("Migrating DB from version %d to %d", existing, SCHEMA_VERSION)
        # executescript() issues an implicit COMMIT, so DDL can't share a transaction
        # with subsequent DML. Run schema as one batch, then record the version.
        conn.executescript(_SCHEMA_V1)
        with transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
    return SCHEMA_VERSION
