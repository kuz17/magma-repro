import json
from pathlib import Path
from tqdm import tqdm


ANNOTATION_PATH = "data/raw/seeclick_web/seeclick_web.json"

IMAGE_ROOT = Path(
    "data/raw/seeclick_web"
)

OUTPUT_PATH = (
    "data/raw/seeclick_web/"
    "filtered_annotations.json"
)


with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)


# Build set of locally available images
existing_images = {
    path.name
    for path in IMAGE_ROOT.rglob("*.png")
}

print(f"Found {len(existing_images)} local images")


filtered = []

for sample in tqdm(data):
    if sample["img_filename"] in existing_images:
        filtered.append(sample)


print(f"Kept {len(filtered)} samples")


with open(OUTPUT_PATH, "w") as f:
    json.dump(filtered, f)


print(f"Saved filtered annotations to:")
print(OUTPUT_PATH)
