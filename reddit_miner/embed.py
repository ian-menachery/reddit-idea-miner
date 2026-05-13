from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)

EMBED_DTYPE = np.float32


@lru_cache(maxsize=1)
def _get_model(name: str) -> Any:
    # Lazy import — sentence_transformers pulls in torch/transformers (~25s import).
    # Keeping it out of module-load means tests that mock encode_texts stay fast.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model: %s", name)
    return SentenceTransformer(name)


def encode_texts(texts: list[str], model_name: str, batch_size: int) -> npt.NDArray[np.float32]:
    """Batched encoding. NO L2-normalization (UMAP handles geometry downstream)."""
    if not texts:
        return np.zeros((0, 384), dtype=EMBED_DTYPE)
    model = _get_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=EMBED_DTYPE)


def _row_to_text(row: dict[str, object]) -> str:
    if row["type"] == "post":
        title = row.get("title") or ""
        body = row.get("body") or ""
        return f"{title}\n{body}".strip()
    return str(row.get("body") or "")


def _pending_documents(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT d.id, d.type, d.title, d.body
        FROM documents d
        LEFT JOIN embeddings e ON e.document_id = d.id
        WHERE e.document_id IS NULL
        ORDER BY d.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def embed_pending(settings: Settings) -> int:
    """Embed every document missing an embedding. Returns count embedded.
    Idempotent: documents already in `embeddings` are skipped."""
    with connect(settings.db_path) as conn:
        pending = _pending_documents(conn)
    if not pending:
        logger.info("No documents pending embedding.")
        return 0

    texts = [_row_to_text(r) for r in pending]
    ids = [str(r["id"]) for r in pending]
    logger.info("Embedding %d documents", len(ids))
    vecs = encode_texts(texts, settings.embedding_model, settings.embedding_batch_size)
    if vecs.shape[0] != len(ids):
        raise RuntimeError(
            f"Embedding count mismatch: got {vecs.shape[0]} vectors for {len(ids)} docs"
        )

    rows: list[tuple[str, bytes]] = [
        (doc_id, vec.tobytes()) for doc_id, vec in zip(ids, vecs, strict=True)
    ]
    with connect(settings.db_path) as conn, transaction(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (document_id, vector) VALUES (?, ?)",
            rows,
        )
    logger.info("Embedded %d documents", len(rows))
    return len(rows)


def load_embeddings(db_path: Path) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Load every embedding from disk. Returns (doc_ids, matrix). Caller picks order."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT document_id, vector FROM embeddings ORDER BY document_id"
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 384), dtype=EMBED_DTYPE)
    ids = [r["document_id"] for r in rows]
    matrix = np.stack([np.frombuffer(r["vector"], dtype=EMBED_DTYPE) for r in rows])
    return ids, matrix
