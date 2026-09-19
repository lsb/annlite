"""On-disk format for embedding matrices.

Raw little-endian float32, row-major, with a JSON sidecar — not `.npy` — because
these files are read by Rust during index construction and by the browser at query
time. A headerless matrix plus a separate manifest can be memory-mapped from any
language without a parser, and a byte range of it maps to rows by arithmetic alone,
which matters when the reader is fetching ranges over HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def manifest_path(path: str | Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".json")


def write_manifest(path: str | Path, rows: int, dim: int, **extra) -> None:
    meta = {
        "rows": rows,
        "dim": dim,
        "dtype": "float32",
        "byte_order": "little",
        "layout": "row-major",
        "bytes_per_row": dim * 4,
        **extra,
    }
    manifest_path(path).write_text(json.dumps(meta, indent=2) + "\n")


def open_memmap(path: str | Path, rows: int, dim: int, mode: str = "r+") -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(rows, dim))


def read(path: str | Path) -> np.ndarray:
    meta = json.loads(manifest_path(path).read_text())
    return np.fromfile(path, dtype=np.float32).reshape(meta["rows"], meta["dim"])
