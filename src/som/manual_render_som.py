# src/som/manual_render_som.py
#
# Manual single-sample SoM render for eyeballing apply_som's output.
# Runs directly (IDE "run" button, `python src/som/manual_render_som.py`,
# or `python -m src.som.manual_render_som`) regardless of working directory.
#
# Usage:
#   python src/som/manual_render_som.py [sample_index]

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from src.som.render_som import apply_som

FILTERED_PATH = REPO_ROOT / "data/raw/seeclick_web/filtered_annotations.json"
IMAGE_ROOT    = REPO_ROOT / "data/raw/seeclick_web"
OUTPUT_DIR    = REPO_ROOT / "data/interim/renders"

sample_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

with open(FILTERED_PATH, "r") as f:
    data = json.load(f)

sample = data[sample_index]

matches = list(IMAGE_ROOT.rglob(sample["img_filename"]))
image_path = matches[0]

print("Using image:")
print(image_path)

image = Image.open(image_path).convert("RGB")

rendered, placed = apply_som(image, sample["elements"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"som_render_{sample_index:03d}.png"

rendered.save(output_path)

print(f"Placed {len(placed)}/{len(sample['elements'])} elements")
print("Saved render to:")
print(output_path)
