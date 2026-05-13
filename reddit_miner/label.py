from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


class Labeler(Protocol):
    """Produce a short human-readable label for each cluster.

    Implementations:
      - CtfidfLabeler (v1, default)
      - ClaudeLabeler (v2)
      - OllamaLabeler (v3)
    """

    def label(self, docs_by_cluster: dict[int, list[str]]) -> dict[int, str]: ...


def ctfidf_labels(docs_by_cluster: dict[int, list[str]], top_n: int = 5) -> dict[int, str]:
    """Class-based TF-IDF labeling (BERTopic-style).

    For each cluster, concatenate all its documents into one pseudo-document.
    Fit a CountVectorizer across pseudo-documents. Compute class-based TF-IDF:
        tf[c,w] = count[c,w] / total_count[c]
        idf[w]  = log(1 + A / f[w])  where A = avg words/class, f[w] = total occurrences of w
    The top-N terms per cluster by ctfidf score become the label.
    """
    if not docs_by_cluster:
        return {}

    cluster_ids = sorted(docs_by_cluster.keys())
    pseudo_docs = [" ".join(docs_by_cluster[cid]) for cid in cluster_ids]

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=1,
        max_features=5000,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
    )
    try:
        counts = vectorizer.fit_transform(pseudo_docs)
    except ValueError:
        logger.warning("c-TF-IDF: empty vocabulary; falling back to empty labels.")
        return dict.fromkeys(cluster_ids, "")
    feature_names = vectorizer.get_feature_names_out()

    counts_array = counts.toarray().astype(np.float64)
    row_sums = counts_array.sum(axis=1, keepdims=True)
    tf = counts_array / np.maximum(row_sums, 1.0)

    n_classes = counts_array.shape[0]
    avg_words = counts_array.sum() / max(n_classes, 1)
    freq_per_term = counts_array.sum(axis=0)
    idf = np.log(1.0 + avg_words / np.maximum(freq_per_term, 1.0))

    ctfidf = tf * idf

    labels: dict[int, str] = {}
    for i, cid in enumerate(cluster_ids):
        scores = ctfidf[i]
        top_idx = np.argsort(-scores)[:top_n]
        words = [str(feature_names[j]) for j in top_idx if scores[j] > 0]
        labels[cid] = ", ".join(words[:top_n])
    return labels


class CtfidfLabeler:
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def label(self, docs_by_cluster: dict[int, list[str]]) -> dict[int, str]:
        return ctfidf_labels(docs_by_cluster, self.top_n)


def get_labeler(name: str) -> Labeler:
    if name == "ctfidf":
        return CtfidfLabeler()
    raise ValueError(f"Unknown labeler: {name!r} (v1 supports: 'ctfidf')")
