from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from reddit_miner.config import Settings
from reddit_miner.db import connect, transaction
from reddit_miner.embed import (
    EMBED_DTYPE,
    _pending_documents,
    _row_to_text,
    embed_pending,
    load_embeddings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _insert_doc(db_path: Path, doc_id: str, doc_type: str, title: str, body: str) -> None:
    with connect(db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO documents (id, type, parent_id, subreddit, title, body, "
            "score, num_comments, created_utc, permalink, pain_score, ingested_at) "
            "VALUES (?, ?, NULL, 'r/test', ?, ?, 10, 0, ?, '/r/test/x', 0.0, ?)",
            (doc_id, doc_type, title, body, time.time(), time.time()),
        )


def _settings_for(db_path: Path) -> Settings:
    return Settings(
        db_path=db_path,
        embedding_model="dummy",
        embedding_batch_size=4,
    )


# ---------- pure helpers ----------


def test_row_to_text_post_concats_title_and_body() -> None:
    assert _row_to_text({"type": "post", "title": "A", "body": "B"}) == "A\nB"


def test_row_to_text_comment_uses_body_only() -> None:
    assert _row_to_text({"type": "comment", "title": "", "body": "hello"}) == "hello"


def test_row_to_text_handles_missing_body() -> None:
    assert _row_to_text({"type": "post", "title": "T", "body": None}) == "T"


# ---------- pending_documents ----------


def test_pending_documents_excludes_already_embedded(tmp_db: Path) -> None:
    _insert_doc(tmp_db, "d1", "post", "t1", "b1")
    _insert_doc(tmp_db, "d2", "post", "t2", "b2")
    with connect(tmp_db) as conn, transaction(conn):
        conn.execute("INSERT INTO embeddings (document_id, vector) VALUES ('d1', X'00')")
    with connect(tmp_db) as conn:
        pending = _pending_documents(conn)
    assert [p["id"] for p in pending] == ["d2"]


# ---------- embed_pending (mocked encoder) ----------


def test_embed_pending_writes_correct_count(tmp_db: Path, mocker: MockerFixture) -> None:
    _insert_doc(tmp_db, "d1", "post", "t1", "b1")
    _insert_doc(tmp_db, "d2", "comment", "", "b2")

    fake_vecs = np.arange(2 * 384, dtype=np.float32).reshape(2, 384)
    mocker.patch("reddit_miner.embed.encode_texts", return_value=fake_vecs)

    settings = _settings_for(tmp_db)
    n = embed_pending(settings)
    assert n == 2

    with connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT document_id, vector FROM embeddings ORDER BY document_id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["document_id"] == "d1"
    # Roundtrip equality
    v0 = np.frombuffer(rows[0]["vector"], dtype=EMBED_DTYPE)
    np.testing.assert_array_equal(v0, fake_vecs[0])


def test_embed_pending_is_idempotent(tmp_db: Path, mocker: MockerFixture) -> None:
    _insert_doc(tmp_db, "d1", "post", "t1", "b1")

    fake_vecs = np.zeros((1, 384), dtype=np.float32)
    spy = mocker.patch("reddit_miner.embed.encode_texts", return_value=fake_vecs)

    settings = _settings_for(tmp_db)
    embed_pending(settings)
    n_second = embed_pending(settings)
    assert n_second == 0
    # Second call should see zero pending and not call encoder
    assert spy.call_count == 1


def test_embed_pending_empty_corpus_is_noop(tmp_db: Path, mocker: MockerFixture) -> None:
    spy = mocker.patch("reddit_miner.embed.encode_texts")
    settings = _settings_for(tmp_db)
    n = embed_pending(settings)
    assert n == 0
    spy.assert_not_called()


def test_embed_pending_raises_on_length_mismatch(tmp_db: Path, mocker: MockerFixture) -> None:
    _insert_doc(tmp_db, "d1", "post", "t", "b")
    _insert_doc(tmp_db, "d2", "post", "t", "b")
    mocker.patch(
        "reddit_miner.embed.encode_texts",
        return_value=np.zeros((1, 384), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        embed_pending(_settings_for(tmp_db))


# ---------- load_embeddings ----------


def test_load_embeddings_roundtrip(tmp_db: Path) -> None:
    vec = np.arange(384, dtype=np.float32)
    _insert_doc(tmp_db, "d1", "post", "t", "b")
    with connect(tmp_db) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO embeddings (document_id, vector) VALUES (?, ?)",
            ("d1", vec.tobytes()),
        )
    ids, matrix = load_embeddings(tmp_db)
    assert ids == ["d1"]
    assert matrix.shape == (1, 384)
    assert matrix.dtype == EMBED_DTYPE
    np.testing.assert_array_equal(matrix[0], vec)


def test_load_embeddings_empty(tmp_db: Path) -> None:
    ids, matrix = load_embeddings(tmp_db)
    assert ids == []
    assert matrix.shape == (0, 384)


# ---------- live integration test (gated) ----------


@pytest.mark.skipif(
    os.environ.get("EMBED_LIVE_TEST") != "1",
    reason="Set EMBED_LIVE_TEST=1 to run the real sentence-transformers model.",
)
def test_real_encoder_returns_384d_float32() -> None:
    from reddit_miner.embed import encode_texts

    out = encode_texts(
        ["hello world", "another sentence"], "sentence-transformers/all-MiniLM-L6-v2", 8
    )
    assert out.shape == (2, 384)
    assert out.dtype == EMBED_DTYPE
