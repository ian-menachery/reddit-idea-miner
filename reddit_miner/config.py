from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "reddit-idea-miner/0.1"

    data_dir: Path = Path("data")
    db_path: Path = Path("data/miner.db")
    log_dir: Path = Path("data/logs")
    alerts_path: Path = Path("data/alerts.md")
    umap_model_path: Path = Path("data/umap_model.joblib")

    upvote_threshold: int = 5
    posts_per_sub: int = 100

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    umap_random_state: int = 42

    hdbscan_min_cluster_size: int = 5

    new_cluster_cosine_threshold: float = 0.30
    signal_score_threshold: float = 50.0

    labeler: str = "ctfidf"

    def require_reddit_creds(self) -> None:
        if not self.reddit_client_id or not self.reddit_client_secret:
            raise RuntimeError(
                "Reddit credentials missing. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in .env (copy from .env.example)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
