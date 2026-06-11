# src/clean/split.py
"""
Split conversations.jsonl into train.jsonl (90%) and val.jsonl (10%).
Seed fixed at 42 for reproducibility.

Usage:
    python -m src.data.split
"""

import json
import random
from pathlib import Path

INPUT  = Path("data/processed/seeclick_web/conversations.jsonl")
OUTDIR = Path("data/processed/seeclick_web")
SEED   = 42
VAL_RATIO = 0.10

# load
lines = INPUT.read_text().strip().splitlines()
print(f"Total samples: {len(lines):,}")

# shuffle in place
rng = random.Random(SEED)
rng.shuffle(lines)

# split
n_val   = int(len(lines) * VAL_RATIO)
n_train = len(lines) - n_val

train_lines = lines[n_val:]   # larger chunk
val_lines   = lines[:n_val]

# write
(OUTDIR / "train.jsonl").write_text("\n".join(train_lines) + "\n")
(OUTDIR / "val.jsonl").write_text("\n".join(val_lines)   + "\n")

print(f"Train : {n_train:,}  →  data/processed/seeclick_web/train.jsonl")
print(f"Val   : {n_val:,}   →  data/processed/seeclick_web/val.jsonl")