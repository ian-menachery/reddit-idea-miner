from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    POST = "post"
    COMMENT = "comment"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    type: DocumentType
    parent_id: str | None
    subreddit: str
    title: str
    body: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str
    pain_score: float
    matched_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClusterMeta:
    id: int
    run_id: int
    label_tfidf: str
    label_llm: str | None
    label_llm_model: str | None
    size: int
    avg_pain_score: float
    subreddit_spread: int
    signal_score: float


@dataclass(frozen=True, slots=True)
class Mention:
    cluster_id: int
    tool_name: str
    sentiment: Sentiment
    context: str
    source_document_id: str
