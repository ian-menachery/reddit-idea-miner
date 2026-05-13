from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import numpy.typing as npt

from reddit_miner.db import connect, transaction
from reddit_miner.embed import EMBED_DTYPE, load_embeddings

if TYPE_CHECKING:
    from pathlib import Path

    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)


def _umap_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "n_components": settings.umap_n_components,
        "n_neighbors": settings.umap_n_neighbors,
        "min_dist": settings.umap_min_dist,
        "metric": settings.umap_metric,
        "random_state": settings.umap_random_state,
    }


def fit_umap(matrix: npt.NDArray[np.float32], settings: Settings) -> Any:
    """Fit a UMAP model on `matrix`. Clamps n_neighbors to corpus size."""
    import umap  # lazy: numba JIT compile is slow

    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"UMAP needs at least 2 samples (got {n})")
    kwargs = _umap_kwargs(settings)
    kwargs["n_neighbors"] = min(int(kwargs["n_neighbors"]), max(2, n - 1))
    model = umap.UMAP(**kwargs)
    model.fit(matrix)
    return model


def save_umap(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_umap(path: Path) -> Any | None:
    if not path.exists():
        return None
    return joblib.load(path)


def _record_umap_version(db_path: Path, settings: Settings, n_docs: int) -> int:
    with connect(db_path) as conn, transaction(conn):
        cur = conn.execute(
            "INSERT INTO umap_models (fitted_at, n_documents, params_json) VALUES (?, ?, ?)",
            (time.time(), n_docs, json.dumps(_umap_kwargs(settings))),
        )
        version = cur.lastrowid
    if version is None:
        raise RuntimeError("UMAP model row inserted but no rowid returned")
    return int(version)


def current_umap_version(db_path: Path) -> int | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM umap_models").fetchone()
    return int(row["v"]) if row and row["v"] is not None else None


def reduce_pending(settings: Settings, refit: bool = False) -> tuple[int, int]:
    """Reduce embeddings to settings.umap_n_components.

    Returns (n_documents_reduced, umap_version).

    Behavior:
    - No saved model OR refit=True: fit new model on ALL embeddings, save it,
      record new version, transform everything. On refit, prior reduced_embeddings
      are deleted (they're in the old geometry).
    - Saved model exists and refit=False: load it, transform only docs missing
      from reduced_embeddings.
    """
    doc_ids, matrix = load_embeddings(settings.db_path)
    if matrix.shape[0] == 0:
        logger.info("No embeddings to reduce.")
        return 0, 0

    existing_model = None if refit else load_umap(settings.umap_model_path)

    if existing_model is None:
        logger.info("Fitting UMAP on %d documents (refit=%s)", matrix.shape[0], refit)
        model = fit_umap(matrix, settings)
        save_umap(model, settings.umap_model_path)
        if refit:
            with connect(settings.db_path) as conn, transaction(conn):
                conn.execute("DELETE FROM reduced_embeddings")
        version = _record_umap_version(settings.db_path, settings, matrix.shape[0])
        ids_to_reduce = doc_ids
        matrix_to_reduce = matrix
    else:
        model = existing_model
        maybe_version = current_umap_version(settings.db_path)
        if maybe_version is None:
            raise RuntimeError("UMAP model file exists but no version row in DB. Run with --refit.")
        version = maybe_version
        with connect(settings.db_path) as conn:
            done = {
                r["document_id"] for r in conn.execute("SELECT document_id FROM reduced_embeddings")
            }
        pending_idx = [i for i, doc_id in enumerate(doc_ids) if doc_id not in done]
        if not pending_idx:
            logger.info("No documents pending reduction.")
            return 0, version
        ids_to_reduce = [doc_ids[i] for i in pending_idx]
        matrix_to_reduce = matrix[pending_idx]

    logger.info("Transforming %d documents via UMAP", matrix_to_reduce.shape[0])
    reduced = np.asarray(model.transform(matrix_to_reduce), dtype=EMBED_DTYPE)
    rows = [(doc_id, vec.tobytes()) for doc_id, vec in zip(ids_to_reduce, reduced, strict=True)]
    with connect(settings.db_path) as conn, transaction(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO reduced_embeddings (document_id, vector) VALUES (?, ?)",
            rows,
        )
    logger.info("Reduced %d documents (umap_version=%d)", len(rows), version)
    return len(rows), version


def load_reduced_embeddings(
    db_path: Path,
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Return (doc_ids, reduced_matrix). Empty result has shape (0, 0)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT document_id, vector FROM reduced_embeddings ORDER BY document_id"
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=EMBED_DTYPE)
    ids = [r["document_id"] for r in rows]
    matrix = np.stack([np.frombuffer(r["vector"], dtype=EMBED_DTYPE) for r in rows])
    return ids, matrix
