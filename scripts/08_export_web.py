"""Stage 08: emit the slim static JSON the D3 web map consumes.

Distills the captioned matches product down to the minimum the map needs -- every city keyed by
Wikidata QID with its group, real country, prominence rank, coordinates, and top-3 analogs; each is
just the target's QID plus the shared-character caption. The map resolves each target's name,
country, and coordinates from the same city table, so the redundant per-match city name and the
computed `similarity` (never displayed -- no arcs, no numbers, no colour scale) are dropped, and
coordinates are rounded. The result is a single minified, key-free-of-API static file that a
GitHub Pages site can load with no runtime keys.

Any city whose coordinates are missing upstream (stage 02 lifts them from the MediaWiki GeoData
`coordinates` prop, which has occasional gaps -- e.g. Krakow, Odesa, Des Moines) is backfilled
from the authoritative Wikidata P625 claim, cached under data/raw/ like every other network fetch.

Reads:  data/processed/matches_<model>[_profile_<key>]_captioned.json
        data/raw/wikidata_backfill_coords.json            (incremental P625 coord cache)
Writes: docs/data/atlantic-mirror.json                     (slim product; committed for Pages)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from _common import INTERIM, PROCESSED, RAW, ROOT, http_get

WD_API = "https://www.wikidata.org/w/api.php"
COORD_DECIMALS = 4  # ~11 m; far finer than a continental map needs
WEB_DATA = ROOT / "docs" / "data"


def fetch_p625(qids: list[str]) -> dict[str, tuple[float, float]]:
    """QID -> (lat, lon) from the Wikidata P625 coordinate claim, in <=50-id batches."""
    out: dict[str, tuple[float, float]] = {}
    for i in range(0, len(qids), 50):
        batch = qids[i : i + 50]
        payload = http_get(
            WD_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "claims",
                "format": "json",
            },
            timeout=60.0,
        ).json()
        for q, ent in payload.get("entities", {}).items():
            claims = ent.get("claims", {}).get("P625")
            if not claims:
                continue
            val = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")
            if val:
                out[q] = (val["latitude"], val["longitude"])
    return out


def backfill_coords(missing: list[str], *, force: bool) -> dict[str, tuple[float, float]]:
    """{qid: (lat, lon)} for the missing cities; fetch only uncached ones (incremental cache)."""
    cache_path = RAW / "wikidata_backfill_coords.json"
    cache = {} if force or not cache_path.exists() else json.loads(cache_path.read_text())
    need = [q for q in missing if q not in cache]
    if need:
        print(f"  backfilling coordinates from Wikidata P625 for {len(need)}: {need}")
        cache.update({q: list(v) for q, v in fetch_p625(need).items()})
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        tmp.replace(cache_path)  # atomic on same filesystem
    return {q: tuple(cache[q]) for q in missing if q in cache}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic")
    ap.add_argument("--source", choices=["lead", "profile"], default="profile")
    ap.add_argument("--profile-key", default="haiku")
    ap.add_argument(
        "--force", action="store_true", help="re-fetch coordinate backfill, ignoring cache"
    )
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    src = PROCESSED / f"matches_{args.model}{suffix}_captioned.json"
    matches = json.loads(src.read_text())

    # rank (1 = most prominent per group) drives dot size; wiki title builds the article link
    meta = pd.read_parquet(INTERIM / "city_lists.parquet").set_index("qid")
    ranks, wiki = meta["rank"].to_dict(), meta["wikipedia_title"].to_dict()

    missing = [q for q, v in matches.items() if v["lat"] is None or v["lon"] is None]
    coords = backfill_coords(missing, force=args.force) if missing else {}

    cities: dict[str, dict] = {}
    still_missing: list[tuple[str, str]] = []
    for q, v in matches.items():
        lat, lon = v["lat"], v["lon"]
        if lat is None or lon is None:
            if q in coords:
                lat, lon = coords[q]
            else:
                still_missing.append((q, v["city"]))
                continue
        cities[q] = {
            "city": v["city"],
            "group": v["group"],
            "country": v["country"],
            "rank": int(ranks[q]),
            "wiki": wiki[q],
            "lat": round(lat, COORD_DECIMALS),
            "lon": round(lon, COORD_DECIMALS),
            "matches": [{"qid": m["qid"], "caption": m["caption"]} for m in v["matches"]],
        }

    if still_missing:
        raise SystemExit(
            f"ERROR: {len(still_missing)} cities still missing coordinates: {still_missing}"
        )

    # Every analog target must exist as a city, or the map can't resolve its name/coords.
    dangling = sorted(
        {m["qid"] for c in cities.values() for m in c["matches"] if m["qid"] not in cities}
    )
    if dangling:
        raise SystemExit(
            f"ERROR: {len(dangling)} analog targets absent from the city table: {dangling}"
        )

    payload = {
        "meta": {
            "source": src.name,
            "n_cities": len(cities),
            "groups": sorted({c["group"] for c in cities.values()}),
        },
        "cities": cities,
    }
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    out_path = WEB_DATA / "atlantic-mirror.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))  # minified
    tmp.replace(out_path)

    gc = Counter(c["group"] for c in cities.values())
    kb = out_path.stat().st_size / 1024
    print(f"Wrote {len(cities)} cities {dict(gc)} -> {out_path.relative_to(ROOT)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
