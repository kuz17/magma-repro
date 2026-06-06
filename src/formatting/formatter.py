# src/formatting/formatter.py
"""
Top-level formatter. Reads filtered_annotations.json, builds Magma-style
conversation dicts for each sample, writes to processed JSONL.

Usage:
    python -m src.formatting.formatter
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

from src.formatting.conversation import build_conversation

# ── paths ─────────────────────────────────────────────────────────────
ANNOTATIONS_PATH = Path("data/raw/seeclick_web/filtered_annotations.json")
SOM_RENDER_DIR   = Path("data/interim/renders/batch")
OUTPUT_DIR       = Path("data/processed/seeclick_web")
OUTPUT_FILE      = OUTPUT_DIR / "conversations.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────
print("Loading annotations...")
with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data):,} samples")

# ── build image→som_path index ────────────────────────────────────────
# Maps img_filename stem to rendered SoM path if it exists
som_index = {
    p.stem: p
    for p in SOM_RENDER_DIR.glob("*.png")
}

# ── format ────────────────────────────────────────────────────────────
rng = random.Random(42)

skipped = 0
written = 0

print(f"Writing to {OUTPUT_FILE} ...")

with open(OUTPUT_FILE, "w") as out_f:
    for sample in tqdm(data):

        # match sample to its SoM render if available
        stem = Path(sample["img_filename"]).stem
        som_path = som_index.get(stem)

        result = build_conversation(
            sample,
            som_image_path=som_path,
            rng=rng,
        )

        if result is None:
            skipped += 1
            continue

        out_f.write(json.dumps(result) + "\n")
        written += 1

print(f"Done. Written: {written:,}  Skipped: {skipped:,}")
print(f"Output: {OUTPUT_FILE}")