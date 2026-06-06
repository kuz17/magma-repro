#test_render_som.py
import json
from pathlib import Path

from PIL import Image

from src.som.render_som import apply_som


FILTERED_PATH = (
    "data/raw/seeclick_web/"
    "filtered_annotations.json"
)

IMAGE_ROOT = Path(
    "data/raw/seeclick_web"
)

OUTPUT_PATH = (
    "data/interim/renders/"
    "som_render_001.png"
)


with open(FILTERED_PATH, "r") as f:
    data = json.load(f)


sample = data[0]

matches = list(
    IMAGE_ROOT.rglob(sample["img_filename"])
)

image_path = matches[0]

print("Using image:")
print(image_path)

image = Image.open(image_path)

rendered = apply_som(
    image,
    sample["elements"],
)

Path(
    "data/interim/renders"
).mkdir(
    parents=True,
    exist_ok=True,
)

rendered.save(OUTPUT_PATH)

print(f"Saved render to:")
print(OUTPUT_PATH)
