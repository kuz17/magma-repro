import json
import random
from pathlib import Path

from PIL import Image

from src.som.render_som import apply_som


from collections import Counter

random.seed(42)

FILTERED_PATH = (
    "data/raw/seeclick_web/"
    "filtered_annotations.json"
)


### TEST

with open(FILTERED_PATH, "r") as f:
    data = json.load(f)

# inspect schema

### TEST



IMAGE_ROOT = Path(
    "data/raw/seeclick_web"
)

OUTPUT_DIR = Path(
    "data/interim/renders/batch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(FILTERED_PATH, "r") as f:
    data = json.load(f)


samples = random.sample(data, 10)


for idx, sample in enumerate(samples):

    matches = list(
        IMAGE_ROOT.rglob(sample["img_filename"])
    )

    if not matches:
        continue

    image_path = matches[0]

    image = Image.open(image_path)

    rendered = apply_som(
        image,
        sample["elements"],
    )

    output_path = (
        OUTPUT_DIR /
        f"render_{idx:03d}.png"
    )

    rendered.save(output_path)

    print(f"Saved {output_path}")