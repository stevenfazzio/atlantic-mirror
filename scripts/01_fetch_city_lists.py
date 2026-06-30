"""Stage 01: fetch a superset of populous cities per country.

Ranking comes from Wikidata via SPARQL, kept deliberately *bare* -- just (item, population).
WDQS reliably 504s on this workload the moment you add the label/sitelink joins or a
redundant country filter, but the bare property-path closure runs in ~1s. We then resolve
QIDs to English Wikipedia titles via the Wikidata `wbgetentities` REST API (fast, batched),
which sidesteps the slow SPARQL joins entirely.

Per-country type anchors (tuned to stay under WDQS's 60s server limit):
  US -- subclasses of "city in the United States" (Q1093829); country-specific, so no P17.
  UK -- subclasses of city (Q515) or town (Q3957), filtered to P17 = United Kingdom (Q145).

Caveat: population is MAX over a city's statements, so a declining city whose Wikidata item
lacks a preferred-rank current value can rank by its historical peak. Fine for a
curate-later superset; revisit if exact ranking starts to matter.

Outputs:
  data/raw/wikidata_<code>.json       cached SPARQL responses (verbatim)
  data/raw/wd_entities_<code>.json    cached QID -> {title,label} resolutions
  data/interim/city_lists.parquet     clean ranked table (both countries)
  data/interim/city_lists.csv         human-readable copy
"""

from __future__ import annotations

import argparse

import pandas as pd

from _common import INTERIM, N_SUPERSET, RAW, cached_json, http_get, write_df

WDQS = "https://query.wikidata.org/sparql"
WD_API = "https://www.wikidata.org/w/api.php"

SOURCES = {
    "US": {
        "where": "?item wdt:P31/wdt:P279* wd:Q1093829 ; wdt:P1082 ?pop .",
        # NYC boroughs are typed as subclasses of "city in the US"; they aren't cities here.
        "exclude_titles": {"Manhattan", "Brooklyn", "Queens", "The Bronx", "Staten Island"},
    },
    "UK": {
        "where": (
            "{ ?item wdt:P31/wdt:P279* wd:Q515 } UNION { ?item wdt:P31/wdt:P279* wd:Q3957 } "
            "?item wdt:P17 wd:Q145 ; wdt:P1082 ?pop ."
        ),
        "exclude_titles": set(),
    },
}


def fetch_ranked(where: str, limit: int) -> dict:
    """Bare WDQS query: (item, max population), ranked. No labels/sitelinks (they 504)."""
    query = (
        f"SELECT ?item (MAX(?pop) AS ?population) WHERE {{ {where} }} "
        f"GROUP BY ?item ORDER BY DESC(?population) LIMIT {limit}"
    )
    return http_get(
        WDQS,
        {"query": query, "format": "json"},
        accept="application/sparql-results+json",
        timeout=90.0,
    ).json()


def resolve_entities(qids: list[str]) -> dict:
    """Map each QID -> {'title','label'} via wbgetentities (batched; enwiki sitelink only)."""
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        batch = qids[i : i + 50]
        payload = http_get(
            WD_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|sitelinks",
                "languages": "en",
                "sitefilter": "enwiki",
                "format": "json",
            },
            timeout=30.0,
        ).json()
        for qid, ent in payload.get("entities", {}).items():
            title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")
            if title:
                out[qid] = {
                    "title": title,
                    "label": ent.get("labels", {}).get("en", {}).get("value", title),
                }
    return out


def build_country(code: str, cfg: dict, limit: int, *, force: bool) -> pd.DataFrame:
    ranked = cached_json(
        RAW / f"wikidata_{code}.json",
        lambda: fetch_ranked(cfg["where"], limit),
        force=force,
    )
    pairs = [
        (b["item"]["value"].rsplit("/", 1)[-1], int(float(b["population"]["value"])))
        for b in ranked["results"]["bindings"]
    ]
    ents = cached_json(
        RAW / f"wd_entities_{code}.json",
        lambda: resolve_entities([q for q, _ in pairs]),
        force=force,
    )

    rows = []
    for qid, pop in pairs:  # already population-sorted by the SPARQL ORDER BY
        ent = ents.get(qid)
        if ent is None or ent["title"] in cfg["exclude_titles"]:
            continue
        rows.append(
            {
                "country": code,
                "city": ent["label"],
                "wikipedia_title": ent["title"],
                "population": pop,
                "qid": qid,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="wikipedia_title").drop_duplicates(subset="qid")
    df = df.head(N_SUPERSET).reset_index(drop=True)
    df.insert(1, "rank", range(1, len(df) + 1))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-query Wikidata, ignoring cache")
    ap.add_argument(
        "--limit",
        type=int,
        default=N_SUPERSET + 60,
        help="rows to request from Wikidata before resolve/dedupe/trim",
    )
    args = ap.parse_args()

    frames = []
    for code, cfg in SOURCES.items():
        df = build_country(code, cfg, args.limit, force=args.force)
        print(f"{code}: {len(df)} cities (top: {', '.join(df['city'].head(8))})")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out[["country", "rank", "city", "population", "wikipedia_title", "qid"]]
    write_df(out, INTERIM / "city_lists.parquet")
    write_df(out, INTERIM / "city_lists.csv")
    print(f"\nWrote {len(out)} rows -> {INTERIM / 'city_lists.parquet'}")


if __name__ == "__main__":
    main()
