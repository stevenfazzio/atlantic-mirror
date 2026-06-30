"""Stage 06: interactive UK map of US analogs (Plotly).

Each UK city is plotted at its coordinates; hovering shows its top US analogs (from stage 05)
with similarity. Dots are sized by population and colored by the top match's similarity.

Reads:  data/processed/matches_<model>.json, data/processed/cities.parquet
Writes: output/uk_map_<model>.html  (self-contained)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from _common import PROCESSED, ROOT

MODEL_KEY = "nomic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    args = ap.parse_args()

    matches = json.loads((PROCESSED / f"matches_{args.model}.json").read_text())
    pop = pd.read_parquet(PROCESSED / "cities.parquet").set_index("qid")["population"]

    rows, skipped = [], []
    for rec in matches.values():
        if rec["lat"] is None or rec["lon"] is None:
            skipped.append(rec["city"])
            continue
        ms = rec["matches"]
        hover = f"<b>{rec['city']}</b><br>most like:<br>" + "<br>".join(
            f"{k + 1}. {m['city']} ({m['similarity']:.2f})" for k, m in enumerate(ms)
        )
        rows.append(
            {
                "city": rec["city"],
                "lat": rec["lat"],
                "lon": rec["lon"],
                "pop": int(pop.get(rec["qid"], 1000)),
                "top_sim": ms[0]["similarity"] if ms else 0.0,
                "hover": hover,
            }
        )
    df = pd.DataFrame(rows)
    if skipped:
        print(f"skipped {len(skipped)} cities without coords: {', '.join(skipped)}")

    lp = np.log10(df["pop"].clip(lower=1000))
    sizes = 7 + 25 * (lp - lp.min()) / (lp.max() - lp.min() + 1e-9)

    fig = go.Figure(
        go.Scattergeo(
            lon=df["lon"],
            lat=df["lat"],
            text=df["hover"],
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=df["top_sim"],
                colorscale="Viridis",
                colorbar=dict(title="top match<br>similarity"),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
        )
    )
    fig.update_geos(
        fitbounds="locations",
        resolution=50,
        projection_type="mercator",
        showland=True,
        landcolor="rgb(246,246,243)",
        showocean=True,
        oceancolor="rgb(224,235,245)",
        showcountries=True,
        countrycolor="rgb(185,185,185)",
        showcoastlines=True,
        coastlinecolor="rgb(170,170,170)",
        showlakes=False,
    )
    fig.update_layout(
        title=f"UK cities & their most similar US cities (hover) — {args.model}",
        height=820,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
    )

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"uk_map_{args.model}.html"
    fig.write_html(out_path, include_plotlyjs=True)
    print(f"Wrote {len(df)} cities -> {out_path}")


if __name__ == "__main__":
    main()
