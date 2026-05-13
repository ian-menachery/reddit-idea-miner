from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from reddit_miner.config import Settings
from reddit_miner.db import connect, transaction
from reddit_miner.embed import EMBED_DTYPE
from reddit_miner.reduce import (
    _umap_kwargs,
    current_umap_version,
    fit_umap,
    load_reduced_embeddings,
    load_umap,
    reduce_pending,
    save_umap,
)

if TYPE_CHECKING:
    from pathlib import Path


# A small synthetic corpus: 30 vectors x 384 dims, two well-separated clusters.
def _synthetic_corpus(n_per_cluster: int = 15, dim: int = 384) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=+1.0, scale=0.05, size=(n_per_cluster, dim)).astype(np.float32)
    b = rng.normal(loc=-1.0, scale=0.05, size=(n_per_cluster, dim)).astype(np.float32)
    matrix = np.vstack([a, b])
    ids = [f"d{i:03d}" for i in range(matrix.shape[0])]
    return ids, matrix


def _settings_for(tmp_path: Path, db_path: Path) -> Settings:
    return Settings(
        db_path=db_path,
        umap_model_path=tmp_path / "umap.joblib",
        umap_n_components=3,  # small for fast tests
        umap_n_neighbors=10,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_random_state=42,
    )


def _seed_embeddings(db_path: Path, ids: list[str], matrix: np.ndarray) -> None:
    with connect(db_path) as conn, transaction(conn):
        for doc_id, vec in zip(ids, matrix, strict=True):
            conn.execute(
                "INSERT INTO documents (id, type, subreddit, score, created_utc, "
                "permalink, pain_score, ingested_at) "
                "VALUES (?, 'post', 'r/t', 10, ?, '/r/t/x', 0.0, ?)",
                (doc_id, time.time(), time.time()),
            )
            conn.execute(
                "INSERT INTO embeddings (document_id, vector) VALUES (?, ?)",
                (doc_id, vec.astype(np.float32).tobytes()),
            )


# ---------- _umap_kwargs ----------


def test_umap_kwargs_from_settings(tmp_path: Path) -> None:
    s = Settings(
        db_path=tmp_path / "x.db",
        umap_n_components=5,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_random_state=42,
    )
    kwargs = _umap_kwargs(s)
    assert kwargs == {
        "n_components": 5,
        "n_neighbors": 15,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42,
    }


# ---------- fit_umap ----------


def test_fit_umap_clamps_neighbors_to_corpus_size(tmp_path: Path) -> None:
    s = Settings(
        db_path=tmp_path / "x.db",
        umap_n_components=2,
        umap_n_neighbors=100,  # larger than corpus
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_random_state=42,
    )
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(10, 384)).astype(np.float32)
    model = fit_umap(matrix, s)
    # UMAP exposes the effective n_neighbors via attribute
    assert model.n_neighbors <= matrix.shape[0] - 1


def test_fit_umap_raises_on_too_few_samples(tmp_path: Path) -> None:
    s = Settings(db_path=tmp_path / "x.db")
    with pytest.raises(ValueError, match="at least 2"):
        fit_umap(np.zeros((1, 384), dtype=np.float32), s)


# ---------- save_umap / load_umap ----------


def test_umap_save_load_roundtrip(tmp_path: Path) -> None:
    s = _settings_for(tmp_path, tmp_path / "x.db")
    _, matrix = _synthetic_corpus()
    model = fit_umap(matrix, s)
    save_umap(model, s.umap_model_path)

    loaded = load_umap(s.umap_model_path)
    assert loaded is not None
    # Same input -> same (or nearly same) output
    a = model.transform(matrix[:5])
    b = loaded.transform(matrix[:5])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_load_umap_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_umap(tmp_path / "nonexistent.joblib") is None


# ---------- reduce_pending ----------


def test_reduce_pending_first_call_fits_and_populates(tmp_db: Path, tmp_path: Path) -> None:
    ids, matrix = _synthetic_corpus()
    _seed_embeddings(tmp_db, ids, matrix)
    s = _settings_for(tmp_path, tmp_db)

    n, version = reduce_pending(s)
    assert n == len(ids)
    assert version == 1
    assert s.umap_model_path.exists()

    reduced_ids, reduced_matrix = load_reduced_embeddings(tmp_db)
    assert reduced_ids == ids
    assert reduced_matrix.shape == (len(ids), s.umap_n_components)
    assert reduced_matrix.dtype == EMBED_DTYPE


def test_reduce_pending_second_call_is_incremental(tmp_db: Path, tmp_path: Path) -> None:
    ids, matrix = _synthetic_corpus()
    _seed_embeddings(tmp_db, ids[:25], matrix[:25])
    s = _settings_for(tmp_path, tmp_db)

    n1, v1 = reduce_pending(s)
    assert n1 == 25

    # Add 5 more docs + embeddings, then reduce again
    _seed_embeddings(tmp_db, ids[25:], matrix[25:])
    n2, v2 = reduce_pending(s)
    assert n2 == 5
    assert v2 == v1  # no refit -> same version

    reduced_ids, _ = load_reduced_embeddings(tmp_db)
    assert len(reduced_ids) == len(ids)


def test_reduce_pending_refit_clears_and_bumps_version(tmp_db: Path, tmp_path: Path) -> None:
    ids, matrix = _synthetic_corpus()
    _seed_embeddings(tmp_db, ids, matrix)
    s = _settings_for(tmp_path, tmp_db)

    _, v1 = reduce_pending(s)
    n2, v2 = reduce_pending(s, refit=True)
    assert v2 == v1 + 1
    assert n2 == len(ids)


def test_reduce_pending_noop_when_no_embeddings(tmp_db: Path, tmp_path: Path) -> None:
    s = _settings_for(tmp_path, tmp_db)
    n, v = reduce_pending(s)
    assert (n, v) == (0, 0)
    assert current_umap_version(tmp_db) is None


def test_reduce_pending_idempotent_when_all_done(tmp_db: Path, tmp_path: Path) -> None:
    ids, matrix = _synthetic_corpus()
    _seed_embeddings(tmp_db, ids, matrix)
    s = _settings_for(tmp_path, tmp_db)

    reduce_pending(s)
    n, _ = reduce_pending(s)
    assert n == 0
