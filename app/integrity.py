"""Cached SHA-256 hashing for immutable model and stimulus artifacts.

The public helper normalizes a file path and delegates hashing to a cache
keyed by file metadata, avoiding repeated reads of unchanged large artifacts.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=256)
def _file_sha256_cached(resolved_path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _file_sha256_cached(str(resolved), stat.st_size, stat.st_mtime_ns)
