"""Stage 03: embed city text with a text-embedding model.

Model is selected by ``--model`` from a small registry (key -> HF path + how to prompt it), so
several embedders coexist for a bake-off; each model+source writes its own file and never clobbers.

  nomic  nomic-embed-text-v1.5   open, 8192-ctx, needs a "search_document: " task prefix (control)
  bge    bge-large-en-v1.5       open, strong general embedder, standard BERT (512-ctx, no prefix);
                                  our texts are well under 512 tokens. (Replaces gte-large-en-v1.5,
                                  whose custom RoPE code IndexErrors against our transformers version.)
  qwen3  Qwen3-Embedding-0.6B     instruction-tuned: we steer it toward transferable CHARACTER (not
                                  name/country) via an Instruct: ... prompt -- the encoder-side analog
                                  of the 02b distillation prompt (see CHARACTER_INSTRUCTION)

Each model's task string is prepended to every text, then SentenceTransformer.encode L2-normalizes.

  --source lead     embed Wikipedia leads (cities.parquet)      -> embeddings_<model>.parquet
  --source profile  embed character profiles (profiles.parquet) -> embeddings_<model>_profile_<key>.parquet

Writes: identity cols + `model` + `embedding` (L2-normalized, one list[float] per row)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import PROCESSED, write_df

# Instruction for instruction-tuned embedders (qwen3). Same intent as the 02b distillation prompt,
# applied at read-time: weight the representation toward transferable character, not identity.
CHARACTER_INSTRUCTION = (
    "Represent this city's transferable character for matching it to a similar city in another "
    "country: its economic base and main industries, geographic and physical setting, size and "
    "regional role, the historical era that shaped it, and cultural identity. Ignore its specific "
    "name, country, and identifying proper nouns."
)

# key -> HF model, task string prepended to every text, trust_remote_code, and an optional device
# pin (None = SentenceTransformer auto). All three run on auto device; `device` is kept as an escape
# hatch for models with backend quirks.
MODELS: dict[str, dict] = {
    "nomic": {
        "hf": "nomic-ai/nomic-embed-text-v1.5",
        "pre": "search_document: ",
        "trc": True,
        "device": None,
    },
    "bge": {"hf": "BAAI/bge-large-en-v1.5", "pre": "", "trc": False, "device": None},
    "qwen3": {
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "pre": f"Instruct: {CHARACTER_INSTRUCTION}\nQuery:",
        "trc": False,
        "device": None,
    },
}

ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", choices=sorted(MODELS), default="nomic", help="embedder from the registry"
    )
    ap.add_argument(
        "--source",
        choices=["lead", "profile"],
        default="lead",
        help="embed Wikipedia leads or LLM character profiles",
    )
    ap.add_argument(
        "--profile-key",
        default="opus",
        help="distillation key for --source profile (e.g. opus, haiku)",
    )
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--device",
        default=None,
        help="torch device override (cpu|mps|cuda); default: registry/auto",
    )
    args = ap.parse_args()

    spec = MODELS[args.model]

    if args.source == "lead":
        in_path, text_col, suffix = PROCESSED / "cities.parquet", "lead_text", ""
    else:
        in_path = PROCESSED / f"profiles_{args.profile_key}.parquet"
        text_col, suffix = "profile_text", f"_profile_{args.profile_key}"
    out_path = PROCESSED / f"embeddings_{args.model}{suffix}.parquet"

    df = pd.read_parquet(in_path)
    assert len(df) > 0, f"{in_path.name} is empty -- run upstream stages first"

    if out_path.exists() and not args.force:
        prev = pd.read_parquet(out_path)
        if set(prev["qid"]) == set(df["qid"]):
            print(f"{out_path.name} already covers all {len(df)} rows; --force to recompute.")
            return
        print("row set changed since last run; recomputing.")

    from sentence_transformers import SentenceTransformer  # heavy import, defer past cache check

    device = args.device or spec.get("device")
    print(f"loading {spec['hf']} (model key={args.model!r}, device={device or 'auto'}) ...")
    model = SentenceTransformer(spec["hf"], trust_remote_code=spec["trc"], device=device)

    pre = spec["pre"]
    texts = [pre + t for t in df[text_col].tolist()]
    print(
        f"embedding {len(texts)} {args.source}s (prepend={pre[:40]!r}{'...' if len(pre) > 40 else ''}) ..."
    )
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    assert emb.shape[0] == len(df), f"row mismatch: {emb.shape[0]} vs {len(df)}"
    assert not np.isnan(emb).any(), "NaNs in embeddings"
    print(f"embeddings: shape={emb.shape}, mean L2 norm={np.linalg.norm(emb, axis=1).mean():.4f}")

    out = df[ID_COLS].copy()
    out["model"] = args.model
    out["embedding"] = [row.tolist() for row in emb]
    write_df(out, out_path)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
