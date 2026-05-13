from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from reddit_miner.db import init_db

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


@dataclass
class FakeComment:
    id: str
    body: str
    score: int
    created_utc: float = 1700000000.0
    permalink: str = "/r/test/comment"


@dataclass
class FakeCommentForest:
    comments: list[FakeComment] = field(default_factory=list)

    def replace_more(self, limit: int) -> None:
        return None

    def list(self) -> list[FakeComment]:
        return self.comments


@dataclass
class FakeSubmission:
    id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float = 1700000000.0
    permalink: str = "/r/test/submission"
    comments: FakeCommentForest = field(default_factory=FakeCommentForest)


@dataclass
class FakeSubreddit:
    submissions: list[FakeSubmission] = field(default_factory=list)

    def hot(self, limit: int) -> Iterable[FakeSubmission]:
        return iter(self.submissions[:limit])


@dataclass
class FakeReddit:
    subs: dict[str, FakeSubreddit] = field(default_factory=dict)

    def subreddit(self, name: str) -> FakeSubreddit:
        return self.subs.setdefault(name, FakeSubreddit())


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Iterator[Path]:
    """Initialize a fresh DB for a single test and yield its path."""
    db_path = tmp_path / "miner.db"
    init_db(db_path)
    yield db_path
