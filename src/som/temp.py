import json
from collections import Counter

with open("data/processed/seeclick_web/conversations.jsonl") as f:
    records = [json.loads(l) for l in f]

print(f"Total records: {len(records)}")

# Task type distribution
task_types = Counter(r["conversations"][0]["value"].split("\n")[0][:40] for r in records)
# or however your task type is stored — check it matches ~40/40/10/10 split

# Check no empty conversations
empty = [i for i, r in enumerate(records) if not r["conversations"]]
print(f"Empty records: {len(empty)}")

# Spot-check a few
for r in records[:3]:
    print(json.dumps(r, indent=2))