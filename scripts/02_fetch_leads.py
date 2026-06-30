"""Stage 02: fetch each city's Wikipedia lead (intro) section.

Uses the MediaWiki extracts API (exintro + explaintext) to get the lead as clean text.
One request per page, each cached under data/raw/leads/<country>/ so a re-run only hits
the network for pages we haven't seen (and an interrupted run resumes for free).

Reads:  data/interim/city_lists.parquet
Writes: data/processed/cities.parquet  (city_lists + lead_text + lead_chars)
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from _common import INTERIM, PROCESSED, RAW, cached_json, http_get, write_df

WIKI_API = "https://en.wikipedia.org/w/api.php"


def safe_filename(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:150]


def fetch_lead(title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|coordinates",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "maxlag": 5,
        "titles": title,
    }
    return http_get(WIKI_API, params=params, timeout=30.0).json()


def extract_text(payload: dict) -> str:
    for page in payload.get("query", {}).get("pages", {}).values():
        if "extract" in page:
            return page["extract"].strip()
    return ""


def extract_coords(payload: dict) -> tuple[float | None, float | None]:
    for page in payload.get("query", {}).get("pages", {}).values():
        coords = page.get("coordinates")
        if coords:
            return coords[0].get("lat"), coords[0].get("lon")
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch even if cached")
    ap.add_argument("--sleep", type=float, default=0.15, help="delay between live requests (s)")
    args = ap.parse_args()

    cities = pd.read_parquet(INTERIM / "city_lists.parquet")
    assert len(cities) > 0, "stage 01 output is empty -- run 01_fetch_city_lists.py first"

    leads, lats, lons, n_live, n_cached = [], [], [], 0, 0
    for row in cities.itertuples(index=False):
        cache_path = RAW / "leads" / row.country / f"{safe_filename(row.wikipedia_title)}.json"
        was_cached = cache_path.exists() and not args.force
        payload = cached_json(
            cache_path,
            lambda t=row.wikipedia_title: fetch_lead(t),
            force=args.force,
        )
        if was_cached:
            n_cached += 1
        else:
            n_live += 1
            time.sleep(args.sleep)
            if n_live % 25 == 0:
                print(f"  fetched {n_live} live...")
        leads.append(extract_text(payload))
        lat, lon = extract_coords(payload)
        lats.append(lat)
        lons.append(lon)

    cities = cities.assign(lead_text=leads, lat=lats, lon=lons)
    cities["lead_chars"] = cities["lead_text"].str.len()

    print(f"\nfetched: {n_live} live, {n_cached} cached")
    for code, grp in cities.groupby("country"):
        print(f"  {code}: {len(grp)} cities, median lead {int(grp['lead_chars'].median())} chars")

    empty = cities[cities["lead_chars"] == 0]
    if len(empty):
        print(f"  WARNING: {len(empty)} cities with empty leads (check titles):")
        for r in empty.itertuples(index=False):
            print(f"    [{r.country}] {r.city} -> {r.wikipedia_title}")

    write_df(cities, PROCESSED / "cities.parquet")
    print(f"\nWrote {len(cities)} rows -> {PROCESSED / 'cities.parquet'}")

    sample = cities.iloc[0]
    print(f"\nSample lead [{sample.city}]:\n{sample.lead_text[:300]}...")


if __name__ == "__main__":
    main()
