# Reddit Idea Miner

Personal tool. Mine vertical subreddits for unmet-demand signals, cluster into themes, identify competitor tools per cluster, surface via Streamlit dashboard. Daily cron on Mac Mini, dashboard via Tailscale.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# fill in REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT
uv run python -m reddit_miner --help
```

Python 3.11 or 3.12 required (hdbscan + numba wheel availability).

## Commands

```bash
uv run python -m reddit_miner init-db    # create schema
uv run python -m reddit_miner ingest     # pull posts + comments
uv run python -m reddit_miner embed      # sentence-transformer embeddings
uv run python -m reddit_miner reduce     # UMAP -> 5-d
uv run python -m reddit_miner cluster    # HDBSCAN + c-TF-IDF label
uv run python -m reddit_miner competitor # extract tool mentions
uv run python -m reddit_miner alert      # new-cluster detection
uv run python -m reddit_miner pipeline   # all of the above, in order
```

Dashboard:

```bash
uv run streamlit run app.py
```

## Tests / lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy reddit_miner
```

## Platform support

- **macOS (Mac Mini prod)**: full pipeline + launchd cron + osascript notifications.
- **Windows (dev)**: full pipeline; notifications are skipped silently, launchd unavailable.
