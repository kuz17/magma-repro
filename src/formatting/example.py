# example.py
"""
Concrete walkthrough: raw annotation → SoM render → formatted conversation.
Takes one real sample from your dataset and shows the full transformation.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from src.som.render_som import apply_som
from src.formatting.conversation import build_conversation

#load one real sample

FILTERED_PATH = REPO_ROOT / "data/raw/seeclick_web/filtered_annotations.json"
IMAGE_ROOT    = REPO_ROOT / "data/raw/seeclick_web"

with open(FILTERED_PATH) as f:
    data = json.load(f)

sample = data[0]

#step 1: what the raw annotation says


print("STEP 1: RAW ANNOTATION")
print(f"Image : {sample['img_filename']}")
print(f"URL   : {sample.get('url', 'N/A')}")
print(f"Elements ({len(sample['elements'])} total):")
print()

for i, el in enumerate(sample["elements"]):
    x1, y1, x2, y2 = el["bbox"]
    print(f"  [{i+1:02d}] text      : {el['instruction']}")
    print(f"       bbox      : ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    print(f"       data_type : {el['data_type']}")
    print()

#step 2: apply SoM to get the placed marks build_conversation expects


print("STEP 2: SOM PLACEMENT")

image_path = list(IMAGE_ROOT.rglob(sample["img_filename"]))[0]
image = Image.open(image_path).convert("RGB")
_, placed = apply_som(image, sample["elements"])

print(f"Placed {len(placed)}/{len(sample['elements'])} elements")
print()

#step 3: what formatting produces


print("STEP 3: FORMATTED CONVERSATION")


rng = random.Random(42)
result = build_conversation(sample, placed=placed, som_image_path=None, rng=rng)

print(f"image     : {result['image']}")
print(f"som_image : {result['som_image']}")
print()

for i, turn in enumerate(result["conversations"]):
    role = turn["from"].upper()
    print(f"--- Turn {i+1} [{role}] ---")
    print(turn["value"])
    print()
