from __future__ import annotations

import logging

import typer

from reddit_miner.alert import fire_alerts
from reddit_miner.cluster import cluster_documents
from reddit_miner.competitor import sweep_competitor_mentions
from reddit_miner.config import get_settings
from reddit_miner.db import init_db
from reddit_miner.embed import embed_pending
from reddit_miner.ingest import add_subreddit, ingest_active, make_reddit_client
from reddit_miner.logging_setup import configure_logging
from reddit_miner.pipeline import pipeline_run
from reddit_miner.reduce import reduce_pending

logger = logging.getLogger("reddit_miner")

app = typer.Typer(
    name="reddit-miner",
    help="Mine vertical subreddits for unmet-demand signals.",
    no_args_is_help=True,
    add_completion=False,
)


def _setup() -> None:
    settings = get_settings()
    configure_logging(settings.log_dir)


@app.command("init-db")
def init_db_cmd() -> None:
    """Initialize the database schema."""
    _setup()
    settings = get_settings()
    version = init_db(settings.db_path)
    logger.info("DB initialized at %s (schema version %d)", settings.db_path, version)
    typer.echo(f"DB initialized at {settings.db_path} (schema version {version})")


@app.command()
def ingest() -> None:
    """Pull posts and comments from active subreddits."""
    _setup()
    settings = get_settings()
    reddit = make_reddit_client(settings)
    results = ingest_active(reddit, settings)
    for name, (posts, comments) in results.items():
        typer.echo(f"r/{name}: +{posts} posts, +{comments} comments")
    if not results:
        typer.echo("No active subreddits.")


@app.command("add-sub")
def add_sub_cmd(
    name: str = typer.Argument(..., help="Subreddit name (with or without r/ prefix)"),
    source: str = typer.Option("manual", help="Where this seed came from."),
) -> None:
    """Seed a subreddit as active in the DB."""
    _setup()
    settings = get_settings()
    add_subreddit(settings.db_path, name, source=source)
    typer.echo(f"Added r/{name.removeprefix('r/').lower()} (source={source})")


@app.command()
def embed() -> None:
    """Compute embeddings for new documents."""
    _setup()
    settings = get_settings()
    n = embed_pending(settings)
    typer.echo(f"Embedded {n} documents")


@app.command()
def reduce(
    refit: bool = typer.Option(False, "--refit", help="Refit UMAP from scratch."),
) -> None:
    """Reduce embeddings via UMAP."""
    _setup()
    settings = get_settings()
    n, version = reduce_pending(settings, refit=refit)
    typer.echo(f"Reduced {n} documents (umap_version={version})")


@app.command()
def cluster() -> None:
    """Cluster reduced embeddings via HDBSCAN and label clusters."""
    _setup()
    settings = get_settings()
    n_clusters, run_id = cluster_documents(settings)
    typer.echo(f"Clustered into {n_clusters} clusters (run_id={run_id})")


@app.command()
def competitor() -> None:
    """Extract competitor tool mentions."""
    _setup()
    settings = get_settings()
    n = sweep_competitor_mentions(settings)
    typer.echo(f"Extracted {n} competitor mentions")


@app.command()
def alert() -> None:
    """Detect new clusters and fire alerts."""
    _setup()
    settings = get_settings()
    n = fire_alerts(settings)
    typer.echo(f"Fired {n} alerts")


@app.command()
def pipeline() -> None:
    """Run ingest -> embed -> reduce -> cluster -> competitor -> alert."""
    _setup()
    settings = get_settings()
    result = pipeline_run(settings)
    typer.echo(
        f"Pipeline: ingested={result['ingested']} embedded={result['embedded']} "
        f"reduced={result['reduced']} clusters={result['clusters']} "
        f"mentions={result['competitor_mentions']} alerts={result['alerts']} "
        f"run_id={result['run_id']}"
    )


if __name__ == "__main__":
    app()
