# example.py
"""
Concrete walkthrough: raw annotation → formatted conversation.
Takes one real sample from your dataset and shows the transformation.
"""

import json
import pprint
import random
from pathlib import Path

from src.formatting.conversation import build_conversation

#load one real sample

with open("data/raw/seeclick_web/filtered_annotations.json") as f:
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

#step 2: what formatting produces 


print("STEP 2: FORMATTED CONVERSATION")


rng = random.Random(42)
result = build_conversation(sample, som_image_path=None, rng=rng)

print(f"image     : {result['image']}")
print(f"som_image : {result['som_image']}")
print()

for i, turn in enumerate(result["conversations"]):
    role = turn["from"].upper()
    print(f"--- Turn {i+1} [{role}] ---")
    print(turn["value"])
    print()