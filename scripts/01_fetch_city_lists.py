"""Stage 01: the two working city pools -- top-N US and top-N European cities by PROMINENCE.

Two groups (the map's two panels):
  US      -- subclasses of "city in the United States" (Q1093829).
  Europe  -- city / town / municipality (Q515 / Q3957 / Q15284) across the 44 European
             sovereign states (Wikidata continent P30 = Europe Q46, minus the transcontinental
             five), pooled as ONE list. Municipality typing matters: many continental cities
             (esp. Spanish -- Seville, Valencia, Bilbao...) are typed only "municipality" and a
             city/town-only net misses them, leaving sitelink-less "X city" stubs + suburbs.

Selection = PROMINENCE, not raw population: among places above POP_FLOOR, rank by how many
Wikipedia language editions cover the city (a boundary-independent notability signal, unlike
administrative "city-proper" population which is wildly inconsistent across countries -- French
communes tiny, German/Ukrainian units large). Reported as `n_wikis` = presence among the 40
biggest human-curated Wikipedias (bot-farms Cebuano/Waray/Minangkabau excluded), tiebroken by
total languages then population.

Requiring an English-Wikipedia article drops the empty duplicate/stub entities for free.

Off-continent territories are excluded on the same geographic principle as the transcontinental
states: Hawaii (mid-Pacific) and the Canary Islands (off NW Africa) are politically US/Spanish but
not on the North American / European landmass. Alaska and Iceland are kept -- far, but on-continent.

Outputs (data/interim/city_lists.{parquet,csv}):
  country       group: "US" | "Europe"  (drives stage 04 neutralization & stage 05 direction)
  country_name  real country            (display / diagnostics)
  rank          1..N within the group, by prominence
  city, population, wikipedia_title, qid, n_wikis, n_langs
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd

from _common import INTERIM, RAW, cached_json, http_get, write_df

WDQS = "https://query.wikidata.org/sparql"
WD_API = "https://www.wikidata.org/w/api.php"

N_PER_GROUP = 250
POP_FLOOR = 100_000  # selection floor; prominence ranks within it
US_QUERY_FLOOR = 40_000  # US pool cache is fetched at 40k (superset); filtered to POP_FLOOR in code

# Broadened type net -- city OR town OR municipality (the last recovers municipality-typed cities).
CITY_TYPES = (
    "{ ?item wdt:P31/wdt:P279* wd:Q515 } UNION { ?item wdt:P31/wdt:P279* wd:Q3957 } "
    "UNION { ?item wdt:P31/wdt:P279* wd:Q15284 }"
)
US_CITY_CLOSURE = "?item wdt:P31/wdt:P279* wd:Q1093829 ; wdt:P1082 ?pop ."
# North America = US (clean "city in the US" anchor) + Canada & Mexico (city/town/municipality net).
NA_EXTRA = {"Q16": "Canada", "Q96": "Mexico"}

# European set = sovereign states (Q3624078) with continent (P30) = Europe (Q46) ...
EUROPE, SOVEREIGN_STATE = "Q46", "Q3624078"
REALM_FIX = {
    "Q756617": "Q35",
    "Q29999": "Q55",
}  # Kingdom of {Denmark, Netherlands} -> country-proper
TRANSCONTINENTAL = {
    "Q159",
    "Q43",
    "Q232",
    "Q230",
    "Q229",
}  # Russia, Turkey, Kazakhstan, Georgia, Cyprus

# Off-continent territories: politically US/Spanish but geographically part of ANOTHER continent, so
# dropped on the same geographic principle as the transcontinental states above (cf. Cyprus, an EU
# member excluded for sitting off Asia). Hawaii is mid-Pacific (Oceania); the Canary Islands sit on
# the African plate ~100 km off Morocco. Alaska (North American landmass) and Iceland (a North
# Atlantic European state) are KEPT -- far from the mainland, but genuinely on-continent. Listed by
# QID: only these three reach the prominence cut, and stage 08 re-checks coordinates as a backstop.
OFF_CONTINENT = {
    "Q18094",  # Honolulu -- Hawaii (Pacific / Oceania)
    "Q14328",  # Santa Cruz de Tenerife -- Canary Islands (off NW Africa)
    "Q11974",  # Las Palmas de Gran Canaria -- Canary Islands (off NW Africa)
}

# ~40 largest human-curated Wikipedias (European + global mix); NO bot-farms.
MAJOR40 = [
    "en",
    "de",
    "fr",
    "es",
    "it",
    "ru",
    "pt",
    "nl",
    "pl",
    "uk",
    "sv",
    "cs",
    "ro",
    "hu",
    "fi",
    "da",
    "no",
    "ca",
    "el",
    "ja",
    "zh",
    "ar",
    "fa",
    "he",
    "tr",
    "ko",
    "id",
    "vi",
    "hi",
    "th",
    "sr",
    "hr",
    "sk",
    "sl",
    "bg",
    "lt",
    "et",
    "lv",
    "eu",
    "gl",
]
MAJOR_KEYS = {f"{c}wiki" for c in MAJOR40}
BOT_WIKIS = {"cebwiki", "warwiki", "minwiki"}  # Lsjbot / bot mass-creations -- pure noise
NON_LANG = {
    "commonswiki",
    "specieswiki",
    "metawiki",
    "mediawikiwiki",
    "wikidatawiki",
    "sourceswiki",
    "foundationwiki",
    "incubatorwiki",
}

NA_EXCLUDE_TITLES = {
    "Manhattan",
    "Brooklyn",
    "Queens",
    "The Bronx",
    "Staten Island",  # NYC boroughs, not cities
    "Tenochtitlan",  # the ancient Aztec capital -- famous, but not a modern city
}
EU_EXCLUDE_TITLES = {
    # Q239, the 195k-person core commune. Q240 (enwiki "Brussels", the 1.26M capital region readers
    # mean) stays; both are distinct Wikidata entities with distinct enwiki titles, so the
    # wikipedia_title dedup below can't catch the pair.
    "City of Brussels",
}
# Display-name overrides where the Wikidata en label isn't what a map should say.
LABEL_FIX = {
    "Q240": "Brussels",  # label is "Brussels-Capital Region"
}
BAD_TITLE_RE = re.compile(
    r"(urban area|metropolitan area|metropolitan region|conurbation|agglomeration"
    r"|\((?:state|region|province|oblast|voivodeship|county|district)\))$",
    re.I,
)


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def sparql(query: str, timeout: float = 120.0) -> dict:
    return http_get(
        WDQS,
        {"query": query, "format": "json"},
        accept="application/sparql-results+json",
        timeout=timeout,
    ).json()


def resolve_and_score(qids: list[str]) -> dict:
    """QID -> {title, label, n_wikis, n_langs} via wbgetentities; drops entities w/o an enwiki article."""
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        batch = qids[i : i + 50]
        payload = http_get(
            WD_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "sitelinks|labels",
                "languages": "en",
                "format": "json",
            },
            timeout=60.0,
        ).json()
        for q, ent in payload.get("entities", {}).items():
            keys = set(ent.get("sitelinks", {}).keys())
            title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")
            if not title:  # no English article -> stub/duplicate; skip
                continue
            langs = {
                k for k in keys if k.endswith("wiki") and k not in NON_LANG and k not in BOT_WIKIS
            }
            out[q] = {
                "title": title,
                "label": ent.get("labels", {}).get("en", {}).get("value", title),
                "n_wikis": len(langs & MAJOR_KEYS),
                "n_langs": len(langs),
            }
    return out


def european_country_qids(*, force: bool) -> list[str]:
    q = f"SELECT ?c WHERE {{ ?c wdt:P31 wd:{SOVEREIGN_STATE} ; wdt:P30 wd:{EUROPE} . }}"
    raw = cached_json(RAW / "wikidata_eu_countries.json", lambda: sparql(q), force=force)
    qids = {qid(b["c"]["value"]) for b in raw["results"]["bindings"]}
    return sorted({REALM_FIX.get(q, q) for q in qids} - TRANSCONTINENTAL)


def eu_country_query(country_qid: str) -> str:
    return (
        f"SELECT ?item (MAX(?pop) AS ?population) WHERE {{ {CITY_TYPES} "
        f"?item wdt:P17 wd:{country_qid} ; wdt:P1082 ?pop . FILTER(?pop >= {POP_FLOOR}) }} "
        f"GROUP BY ?item ORDER BY DESC(?population) LIMIT 800"
    )


def fetch_europe(country_qids: list[str], force: bool) -> list[dict]:
    """Per-country (city/town/municipality) queries, individually cached, pooled."""
    bindings, failed = [], []
    floor_k = POP_FLOOR // 1000
    for i, cq in enumerate(country_qids, 1):
        cache = RAW / f"wikidata_euc{floor_k}k_{cq}.json"
        live = not cache.exists() or force
        try:
            raw = cached_json(cache, lambda c=cq: sparql(eu_country_query(c)), force=force)
        except Exception as exc:  # noqa: BLE001 - collect & re-raise so one 504 doesn't lose the rest
            failed.append(cq)
            print(f"  [{i}/{len(country_qids)}] {cq}: FAILED ({str(exc)[:50]})")
            continue
        for b in raw["results"]["bindings"]:
            b = dict(b)
            b["country"] = {"value": f"http://www.wikidata.org/entity/{cq}"}
            bindings.append(b)
        if live:
            time.sleep(0.4)
    if failed:
        raise RuntimeError(
            f"{len(failed)} European countries failed (successes cached; re-run to backfill): {failed}"
        )
    return bindings


def fetch_us(force: bool) -> list[dict]:
    q = (
        f"SELECT ?item (MAX(?pop) AS ?population) WHERE {{ {US_CITY_CLOSURE} "
        f"FILTER(?pop >= {US_QUERY_FLOOR}) }} GROUP BY ?item ORDER BY DESC(?population) LIMIT 500"
    )
    raw = cached_json(RAW / "wikidata_us_pool.json", lambda: sparql(q, timeout=180), force=force)
    return raw["results"]["bindings"]


def fetch_north_america(force: bool) -> list[dict]:
    """US via the 'city in the United States' anchor + Canada/Mexico via the city/town/municipality
    net, pooled into one candidate set (each binding tagged with its country QID)."""
    bindings = []
    for b in fetch_us(force):
        b = dict(b)
        b["country"] = {"value": "http://www.wikidata.org/entity/Q30"}
        bindings.append(b)
    bindings += fetch_europe(list(NA_EXTRA), force)  # Q16 Canada, Q96 Mexico
    return bindings


def build_group(
    bindings: list[dict], *, group: str, name_of: dict, exclude_titles: set[str], force: bool
) -> pd.DataFrame:
    pairs = [
        (
            qid(b["item"]["value"]),
            int(float(b["population"]["value"])),
            qid(b["country"]["value"]) if "country" in b else None,
        )
        for b in bindings
        if int(float(b["population"]["value"])) >= POP_FLOOR
    ]
    scored = cached_json(
        RAW / f"wd_scores_{group.lower()}.json",
        lambda: resolve_and_score([q for q, _, _ in pairs]),
        force=force,
    )
    rows = []
    for q, pop, cqid in pairs:
        s = scored.get(q)
        if (
            s is None
            or q in OFF_CONTINENT
            or s["title"] in exclude_titles
            or BAD_TITLE_RE.search(s["title"])
        ):
            continue
        rows.append(
            {
                "country": group,
                "country_name": name_of.get(cqid, name_of.get(group, group)),
                "city": LABEL_FIX.get(q, s["label"]),
                "population": pop,
                "wikipedia_title": s["title"],
                "qid": q,
                "n_wikis": s["n_wikis"],
                "n_langs": s["n_langs"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["n_wikis", "n_langs", "population"], ascending=False)
    df = df.drop_duplicates("wikipedia_title").drop_duplicates(
        "qid"
    )  # keeps the most-prominent twin
    df = df.head(N_PER_GROUP).reset_index(drop=True)
    df["city"] = df["city"].str.replace(
        r"\s+Municipality$", "", regex=True
    )  # Mexican "X Municipality"
    # US namesakes (Columbus OH vs GA, three Springfields) share one Wikidata label; the enwiki
    # title carries the disambiguating state, so use it wherever a label collides within the group.
    collide = df["city"].duplicated(keep=False)
    df.loc[collide, "city"] = df.loc[collide, "wikipedia_title"]
    df.insert(1, "rank", range(1, len(df) + 1))
    if len(df) < N_PER_GROUP:
        print(f"  WARNING: {group} only yielded {len(df)} < {N_PER_GROUP}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-query Wikidata, ignoring cache")
    args = ap.parse_args()

    eu_qids = european_country_qids(force=args.force)
    country_names = cached_json(
        RAW / "wd_entities_eu_countries.json",
        lambda: {q: {"label": e["label"]} for q, e in resolve_and_score(eu_qids).items()},
        force=args.force,
    )
    name_of = {q: e["label"] for q, e in country_names.items()}
    name_of.update({"Q30": "United States", **NA_EXTRA})
    print(f"Europe: {len(eu_qids)} sovereign states (transcontinental excluded)")

    na_df = build_group(
        fetch_north_america(args.force),
        group="North America",
        name_of=name_of,
        exclude_titles=NA_EXCLUDE_TITLES,
        force=args.force,
    )
    eu_df = build_group(
        fetch_europe(eu_qids, args.force),
        group="Europe",
        name_of=name_of,
        exclude_titles=EU_EXCLUDE_TITLES,
        force=args.force,
    )

    out = pd.concat([na_df, eu_df], ignore_index=True)
    out = out[
        [
            "country",
            "country_name",
            "rank",
            "city",
            "population",
            "wikipedia_title",
            "qid",
            "n_wikis",
            "n_langs",
        ]
    ]
    write_df(out, INTERIM / "city_lists.parquet")
    write_df(out, INTERIM / "city_lists.csv")

    for grp, g in out.groupby("country", sort=False):
        print(f"\n{grp}: {len(g)} cities  (pop floor {int(g.population.min()):,})")
        print("  top: " + ", ".join(g["city"].head(12)))
        print("  bottom: " + ", ".join(g["city"].tail(6)))
    print(f"\nEurope spans {eu_df['country_name'].nunique()} countries in the top {N_PER_GROUP}.")
    print(f"Wrote {len(out)} rows -> {INTERIM / 'city_lists.parquet'}")


if __name__ == "__main__":
    main()
