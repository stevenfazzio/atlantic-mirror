"""Shared paths, HTTP helpers, and config for the sibling-cities pipeline.

Scripts in this directory import from here directly (``from _common import ...``);
that works because Python puts the running script's directory on ``sys.path``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

for _d in (RAW, INTERIM, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

# --- Config --------------------------------------------------------------
# Wikimedia REQUIRES a descriptive User-Agent with real contact info -- requests with a
# placeholder/generic UA get rate-limited (429) after a handful of calls. Keep a real
# contact here.
USER_AGENT = (
    "international-sibling-cities/0.1 (research project; fazzios@gmail.com) python-requests"
)

COUNTRIES = {
    "US": {"name": "United States", "wikidata_qid": "Q30"},
    "UK": {"name": "United Kingdom", "wikidata_qid": "Q145"},
}

N_SUPERSET = 150  # keep a generous superset; curate down to working sets later

# --- HTTP ----------------------------------------------------------------
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def http_get(
    url: str,
    params: dict[str, Any],
    *,
    accept: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> requests.Response:
    """GET with explicit timeout and exponential backoff (handles 429/5xx/network)."""
    headers = {"Accept": accept} if accept else {}
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"retryable status {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            wait = min(2**attempt * 3, 60)
            print(f"  request failed ({exc}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {max_retries} attempts: {url}") from last_exc


def cached_json(path: Path, producer: Callable[[], Any], *, force: bool = False) -> Any:
    """Return JSON cached at ``path``; compute via ``producer`` and cache if missing."""
    path = Path(path)
    if path.exists() and not force:
        return json.loads(path.read_text())
    data = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)  # atomic on same filesystem
    return data


def write_df(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to .parquet/.csv atomically: temp file -> verify rows -> rename."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix + ".tmp")
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(tmp, index=False)
            n = len(pd.read_parquet(tmp))
        elif path.suffix == ".csv":
            df.to_csv(tmp, index=False)
            n = len(pd.read_csv(tmp))
        else:
            raise ValueError(f"unsupported suffix: {path.suffix}")
        assert n == len(df), f"row-count mismatch writing {path}: {n} vs {len(df)}"
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
