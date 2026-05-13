from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def configure_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()

    if any(getattr(h, "_reddit_miner", False) for h in root.handlers):
        return

    root.setLevel(level)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler._reddit_miner = True  # type: ignore[attr-defined]
    root.addHandler(stderr_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "miner.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler._reddit_miner = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
