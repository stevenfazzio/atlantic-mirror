"""Head-to-head: lead vs profile embeddings, one row per distillation key.

Runs the same reduction (PCA-50 + centroid) on each source and reports the name-collision
metric, the country residual, and CSLS matches for the focus cities -- so we can compare
distillation models (e.g. opus vs haiku) and confirm the wins hold.

Run after embedding each source (stage 03 --source lead / --source profile --profile-key K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analyze_name_collisions import find_namesakes
from prototype_name_fixes import build_reps, show
from prototype_name_subspace import country_diag

from _common import PROCESSED

VARIANTS = {
    "lead": "embeddings_nomic.parquet",
    "profile-opus": "embeddings_nomic_profile_opus.parquet",
    "profile-haiku": "embeddings_nomic_profile_haiku.parquet",
}
FOCUS = [
    "Oxford",
    "Lincoln",
    "Birmingham",
    "Worcester",
    "Newport",
    "Stockton-on-Tees",
    "Brighton and Hove",
    "Blackpool",
    "Edinburgh",
    "Liverpool",
    "Cambridge",
    "Manchester",
    "York",
    "Bath",
]


def main() -> None:
    for tag, fname in VARIANTS.items():
        path = PROCESSED / fname
        if not path.exists():
            print(f"skip {tag}: {fname} not found")
            continue
        df = pd.read_parquet(path).reset_index(drop=True)
        emb = np.vstack(df["embedding"].to_numpy()).astype("float64")
        country, names = df["country"].to_numpy(), df["city"].to_numpy()
        reps = build_reps(emb, country)
        pairs = find_namesakes(list(names[country == "UK"]), list(names[country == "US"]))
        same, acc = country_diag(reps, country)
        print(f"\n########## {tag.upper()}  ({fname}) ##########")
        print(f"  country: NN same-country={same:.1%}  separability={acc:.1%}")
        show(tag, reps, country, names, pairs, FOCUS)


if __name__ == "__main__":
    main()
