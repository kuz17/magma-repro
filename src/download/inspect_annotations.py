import json

ANNOTATION_PATH = "data/raw/seeclick_web/seeclick_web.json"

with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

print("\nFirst sample:")
print(data[0])

print("\nKeys:")
print(data[0].keys())
