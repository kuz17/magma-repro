# src/formatting/formatter.py
"""
Top-level formatter. Reads filtered_annotations.json + per-image
_marks.json sidecars, builds Magma-style conversation dicts,
writes to processed JSONL.

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
SOM_RENDER_DIR   = Path("data/interim/renders/seeclick_web/batch")
OUTPUT_DIR       = Path("data/processed/seeclick_web")
OUTPUT_FILE      = OUTPUT_DIR / "conversations.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load annotations ──────────────────────────────────────────────────
print("Loading annotations...")
with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data):,} samples")

# ── build indexes ─────────────────────────────────────────────────────
# stem → path for rendered SoM images  (e.g. abc123_som.png)
som_index = {
    p.stem.replace("_som", ""): p
    for p in SOM_RENDER_DIR.glob("*_som.png")
}

# stem → path for placement sidecars   (e.g. abc123_marks.json)
marks_index = {
    p.stem.replace("_marks", ""): p
    for p in SOM_RENDER_DIR.glob("*_marks.json")
}

print(f"Found {len(som_index):,} SoM images")
print(f"Found {len(marks_index):,} sidecar files")

# ── format ────────────────────────────────────────────────────────────
rng = random.Random(42)

skipped_missing  = 0   # no sidecar found for this sample
skipped_empty    = 0   # sidecar exists but build_conversation returned None
written          = 0

print(f"Writing to {OUTPUT_FILE} ...")

with open(OUTPUT_FILE, "w") as out_f:
    for sample in tqdm(data):

        stem = Path(sample["img_filename"]).stem

        # sidecar is required — skip if render didn't produce one
        marks_path = marks_index.get(stem)
        if marks_path is None:
            skipped_missing += 1
            continue

        with open(marks_path) as f:
            placed = json.load(f)

        som_path = som_index.get(stem)   # None is fine; recorded in output

        result = build_conversation(
            sample,
            placed=placed,
            som_image_path=som_path,
            rng=rng,
        )

        if result is None:
            skipped_empty += 1
            continue

        out_f.write(json.dumps(result) + "\n")
        written += 1

# ── summary ───────────────────────────────────────────────────────────
total = written + skipped_missing + skipped_empty

print(
    f"\nDone."
    f"\n  Written          : {written:,}"
    f"\n  Skipped (no sidecar) : {skipped_missing:,}"
    f"\n  Skipped (empty)  : {skipped_empty:,}"
    f"\n  Total processed  : {total:,}"
    f"\n  Output           : {OUTPUT_FILE}"
)