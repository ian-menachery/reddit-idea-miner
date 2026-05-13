from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

import praw

from reddit_miner.db import connect, transaction

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from reddit_miner.config import Settings

logger = logging.getLogger(__name__)


PAIN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi (?:hate|can'?t stand|am sick of|am tired of)\b",
        r"\b(?:can'?t|cannot) (?:find|figure out|seem to)\b",
        r"\b(?:is there|does anyone know) (?:a|any) (?:way|tool|app|service)\b",
        r"\b(?:willing to pay|would pay|happy to pay|pay (?:for|good money))\b",
        r"\b(?:wish there (?:was|were)|i wish (?:i had|there)|i'd love)\b",
        r"\b(?:frustrat|annoy|broken|useless|garbage|terrible)\w*",
        r"\b(?:waste of|hate that|sick of|tired of)\b",
        r"\bwhy (?:is|does|do)n'?t (?:there|anyone|someone)\b",
        r"\b(?:struggle|struggling|struggled) (?:with|to)\b",
        r"\bno (?:good|decent|reliable) (?:way|tool|option|alternative)\b",
        r"\b(?:nothing|none of)\s+(?:works|fits|solves)\b",
        r"\b(?:looking for|need(?:ed)?) (?:a|an|some) (?:tool|app|way|alternative)\b",
    )
)


class CommentLike(Protocol):
    id: str
    body: str
    score: int
    created_utc: float
    permalink: str


class CommentForestLike(Protocol):
    def replace_more(self, limit: int) -> Any: ...
    def list(self) -> list[CommentLike]: ...


class SubmissionLike(Protocol):
    id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str
    comments: CommentForestLike


class SubredditLike(Protocol):
    def hot(self, limit: int) -> Iterable[SubmissionLike]: ...


class RedditLike(Protocol):
    def subreddit(self, name: str) -> SubredditLike: ...


def _regex_hits(text: str) -> tuple[float, tuple[str, ...]]:
    matched = [pat.pattern for pat in PAIN_PATTERNS if pat.search(text)]
    return float(len(matched)), tuple(matched)


def pain_score(text: str, score: int, num_comments: int) -> tuple[float, tuple[str, ...]]:
    """Combined regex + engagement signal. Per spec: comment density rewards
    in-group resonance over virality. Comments have num_comments=0, so they get
    only the regex term."""
    regex_score, matched = _regex_hits(text)
    bonus = min(num_comments / max(score, 1), 5.0) * 0.3
    return regex_score + bonus, matched


def make_reddit_client(settings: Settings) -> praw.Reddit:
    settings.require_reddit_creds()
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        check_for_async=False,
    )


_INSERT_DOC_SQL = """
INSERT OR IGNORE INTO documents (
    id, type, parent_id, subreddit, title, body, score, num_comments,
    created_utc, permalink, pain_score, matched_patterns, ingested_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _is_deleted(body: str) -> bool:
    return body in {"[deleted]", "[removed]"}


def ingest_subreddit(
    reddit: RedditLike,
    sub_name: str,
    upvote_threshold: int,
    posts_per_sub: int,
    db_path: Path,
) -> tuple[int, int]:
    """Pull a subreddit's hot posts (+ their comments) and insert above-threshold rows.
    Returns (posts_added, comments_added). Each submission is one transaction."""
    n_posts = 0
    n_comments = 0
    now = time.time()
    sub = reddit.subreddit(sub_name)

    with connect(db_path) as conn:
        for submission in sub.hot(limit=posts_per_sub):
            if submission.score < upvote_threshold:
                continue
            body = submission.selftext or ""
            ps, matched = pain_score(
                f"{submission.title}\n{body}", submission.score, submission.num_comments
            )

            try:
                submission.comments.replace_more(limit=0)
                comments = submission.comments.list()
            except Exception:
                logger.exception("Failed to fetch comments for %s", submission.id)
                comments = []

            with transaction(conn):
                cur = conn.execute(
                    _INSERT_DOC_SQL,
                    (
                        submission.id,
                        "post",
                        None,
                        sub_name,
                        submission.title,
                        body,
                        int(submission.score),
                        int(submission.num_comments),
                        float(submission.created_utc),
                        submission.permalink,
                        ps,
                        json.dumps(list(matched)),
                        now,
                    ),
                )
                if cur.rowcount:
                    n_posts += 1

                for c in comments:
                    if c.score < upvote_threshold or _is_deleted(c.body):
                        continue
                    cps, cmatched = pain_score(c.body, c.score, 0)
                    cur = conn.execute(
                        _INSERT_DOC_SQL,
                        (
                            c.id,
                            "comment",
                            submission.id,
                            sub_name,
                            "",
                            c.body,
                            int(c.score),
                            0,
                            float(c.created_utc),
                            c.permalink,
                            cps,
                            json.dumps(list(cmatched)),
                            now,
                        ),
                    )
                    if cur.rowcount:
                        n_comments += 1

    logger.info(
        "Ingested r/%s: %d posts, %d comments (above score>=%d)",
        sub_name,
        n_posts,
        n_comments,
        upvote_threshold,
    )
    return n_posts, n_comments


def ingest_active(reddit: RedditLike, settings: Settings) -> dict[str, tuple[int, int]]:
    """Ingest every subreddit with status='active'. Returns per-sub counts."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM subreddits WHERE status = 'active' ORDER BY name"
        ).fetchall()
    active = [r["name"] for r in rows]
    if not active:
        logger.warning(
            "No active subreddits in DB. Add some via "
            "`python -m reddit_miner add-sub <name>` first."
        )
        return {}

    results: dict[str, tuple[int, int]] = {}
    for name in active:
        try:
            results[name] = ingest_subreddit(
                reddit, name, settings.upvote_threshold, settings.posts_per_sub, settings.db_path
            )
        except Exception:
            logger.exception("Failed to ingest r/%s", name)
            results[name] = (0, 0)
    return results


def add_subreddit(db_path: Path, name: str, source: str = "manual") -> None:
    """Seed a subreddit row marked active. Idempotent."""
    name = name.removeprefix("r/").removeprefix("/r/").strip().lower()
    with connect(db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO subreddits (name, status, source, added_at) VALUES (?, 'active', ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET status='active'",
            (name, source, time.time()),
        )
    logger.info("Added subreddit r/%s (source=%s)", name, source)
