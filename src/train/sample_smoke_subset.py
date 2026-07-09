"""
Stratified subset sampler for the smoke-test fine-tune.

Each line in train.jsonl is one whole page: `conversations` holds every
turn-pair (one per element) for that page, per the Phase 3 "one turn-pair
per element" rework. Goal: pull ~900 pages that preserve the distribution
of page density (mark count), so the smoke test spans sparse and dense
pages rather than accidentally sampling into a degenerate subset.

NOTE: input-turn stratification was dropped. SeeClick-Web's observed
element types are text/hover only (per schemas.md) -- no "input" type
turns were found in a scan of this data, so that axis wasn't recoverable
and likely isn't exercised by this dataset regardless of how it's sampled.
If your formatter's input-turn branch keys off something else, flag it
and this can be added back properly.
"""
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

SRC = Path("data/processed/seeclick_web/train.jsonl")
OUT = Path("data/processed/seeclick_web/train_smoke.jsonl")
TARGET_N = 900
SEED = 42

def mark_count(row: dict) -> int:
    # Each row is one whole page: conversations holds all turn-pairs for it.
    # Number of assistant turns == number of marks/elements on this page.
    convs = row.get("conversations", row.get("turns", []))
    return sum(1 for t in convs if t.get("from", t.get("role")) == "assistant")

def bucket(n: int) -> str:
    if n <= 5:
        return "few"
    if n <= 15:
        return "medium"
    return "many"

def main():
    random.seed(SEED)
    rows = [json.loads(l) for l in SRC.open()]

    strata = defaultdict(list)
    for r in rows:
        key = bucket(mark_count(r))
        strata[key].append(r)

    print("Strata sizes:")
    for k, v in strata.items():
        print(f"  {k}: {len(v)}")

    total = len(rows)
    sampled = []
    for key, items in strata.items():
        random.shuffle(items)
        n_take = max(1, round(len(items) / total * TARGET_N))
        sampled.extend(items[:n_take])

    random.shuffle(sampled)
    sampled = sampled[:TARGET_N]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(sampled)} pages ({sum(mark_count(r) for r in sampled)} total turn-pairs) to {OUT}")

    final_buckets = Counter(bucket(mark_count(r)) for r in sampled)
    print(f"  final bucket distribution: {dict(final_buckets)}")

if __name__ == "__main__":
    main()
