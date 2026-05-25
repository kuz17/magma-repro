import json
from pathlib import Path

from PIL import Image

from src.som.render_ui import apply_som


FILTERED_PATH = (
    "data/raw/seeclick_web/"
    "filtered_annotations.json"
)

IMAGE_ROOT = Path(
    "data/raw/seeclick_web"
)

OUTPUT_PATH = (
    "data/interim/renders/first_render.png"
)


with open(FILTERED_PATH, "r") as f:
    data = json.load(f)


sample = data[0]

print("Using sample:")
print(sample["img_filename"])


matches = list(
    IMAGE_ROOT.rglob(sample["img_filename"])
)

image_path = matches[0]

print("Resolved image path:")
print(image_path)


image = Image.open(image_path)

rendered = apply_som(
    image,
    sample["elements"],
)

Path("outputs").mkdir(exist_ok=True)

rendered.save(OUTPUT_PATH)

print(f"Saved render to {OUTPUT_PATH}")
