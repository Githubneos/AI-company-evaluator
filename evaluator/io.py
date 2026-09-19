"""Crash- and concurrency-safe file writes.

Every cache and artifact in this project is written by one process and may be
read by another at the same moment: the dashboard fetches a payload and a price
chart in parallel, the scheduler refreshes caches while the API serves, and a
retrain rewrites metadata that serving reads. Writing a file in place lets a
reader see half of it, and two interleaved writers can leave a file that is
permanently corrupt -- which happened to a price cache in production.

So nothing writes in place. Each write goes to a temp file in the *same*
directory (``os.replace`` is only atomic within one filesystem) and is renamed
over the target, so a reader sees either the old complete file or the new one.
Reads of caches treat an unreadable file as a miss: the caller refetches rather
than serving an exception.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _atomic_write(path: Path | str, write: Callable[[str], None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real extension last: some writers (np.savez_compressed) append
    # their own extension to a name that lacks it, which would silently write
    # to a different file than the one renamed into place.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=f".tmp{path.suffix}")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_parquet(frame: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
    _atomic_write(path, lambda tmp: frame.to_parquet(tmp, **kwargs))


def atomic_write_text(text: str, path: Path | str) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)

    _atomic_write(path, write)


def atomic_write_json(obj: Any, path: Path | str, **kwargs: Any) -> None:
    atomic_write_text(json.dumps(obj, **{"indent": 2, **kwargs}), path)


def atomic_write_bytes(write: Callable[[str], None], path: Path | str) -> None:
    """For writers that take a filename (e.g. ``np.savez_compressed``, LightGBM ``save_model``)."""
    _atomic_write(path, write)


def read_parquet_or_none(path: Path | str, **kwargs: Any) -> pd.DataFrame | None:
    """A cache that is missing or cannot be read is a miss, never an error."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any unreadable file means "refetch"
        log.warning("discarding unreadable cache %s: %s", path, exc)
        return None


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
    "read_parquet_or_none",
]
