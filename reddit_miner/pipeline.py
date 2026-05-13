from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from reddit_miner.alert import fire_alerts
from reddit_miner.cluster import cluster_documents
from reddit_miner.competitor import sweep_competitor_mentions
from reddit_miner.embed import embed_pending
from reddit_miner.ingest import ingest_active, make_reddit_client
from reddit_miner.reduce import reduce_pending

if TYPE_CHECKING:
    from reddit_miner.config import Settings
    from reddit_miner.ingest import RedditLike

logger = logging.getLogger(__name__)


class PipelineResult(TypedDict):
    ingested: int
    embedded: int
    reduced: int
    clusters: int
    competitor_mentions: int
    alerts: int
    run_id: int


def pipeline_run(settings: Settings, reddit: RedditLike | None = None) -> PipelineResult:
    """Chain ingest -> embed -> reduce -> cluster -> competitor -> alert.

    If `reddit` is None, builds a real PRAW client from settings (requires creds).
    Tests inject a FakeReddit.
    """
    if reddit is None:
        reddit = make_reddit_client(settings)

    logger.info("Pipeline: starting")

    ingest_results = ingest_active(reddit, settings)
    n_ingested = sum(p + c for p, c in ingest_results.values())
    logger.info("Pipeline: ingest +%d docs across %d subs", n_ingested, len(ingest_results))

    n_embedded = embed_pending(settings)
    logger.info("Pipeline: embed +%d docs", n_embedded)

    n_reduced, _umap_version = reduce_pending(settings)
    logger.info("Pipeline: reduce +%d docs", n_reduced)

    n_clusters, run_id = cluster_documents(settings)
    logger.info("Pipeline: cluster %d clusters (run_id=%d)", n_clusters, run_id)

    n_mentions = sweep_competitor_mentions(settings)
    logger.info("Pipeline: competitor %d mentions", n_mentions)

    n_alerts = fire_alerts(settings)
    logger.info("Pipeline: alerts %d fired", n_alerts)

    return PipelineResult(
        ingested=n_ingested,
        embedded=n_embedded,
        reduced=n_reduced,
        clusters=n_clusters,
        competitor_mentions=n_mentions,
        alerts=n_alerts,
        run_id=run_id,
    )
