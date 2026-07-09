"""
Stratified val subset for the post-smoke-fine-tune collapse check.

Same stratification logic as sample_smoke_subset.py, applied to val.jsonl.
Also supports forcing in specific pages you already know triggered
"Mark: 0" collapse during the earlier web-integration debugging -- add
their image filenames to FORCE_INCLUDE so the collapse check directly
covers the failure cases you're worried about, not just average-case data.
"""
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

SRC = Path("data/processed/seeclick_web/val.jsonl")
OUT = Path("data/processed/seeclick_web/val_smoke.jsonl")
TARGET_N = 100
SEED = 42

FORCE_INCLUDE: set[str] = set()


def mark_count(row: dict) -> int:
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
    by_image = {r["image"]: r for r in rows}

    forced = [by_image[img] for img in FORCE_INCLUDE if img in by_image]
    missing = FORCE_INCLUDE - set(by_image.keys())
    if missing:
        print(f"WARNING: {len(missing)} forced pages not found in val.jsonl: {missing}")

    remaining = [r for r in rows if r["image"] not in FORCE_INCLUDE]

    strata = defaultdict(list)
    for r in remaining:
        strata[bucket(mark_count(r))].append(r)

    print("Strata sizes (excluding forced pages):")
    for k, v in strata.items():
        print(f"  {k}: {len(v)}")

    n_remaining_target = max(0, TARGET_N - len(forced))
    total = len(remaining)
    sampled = list(forced)
    for key, items in strata.items():
        random.shuffle(items)
        n_take = max(1, round(len(items) / total * n_remaining_target)) if total else 0
        sampled.extend(items[:n_take])

    seen = set()
    deduped = []
    for r in sampled:
        if r["image"] not in seen:
            seen.add(r["image"])
            deduped.append(r)
    sampled = deduped[:TARGET_N]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(sampled)} pages ({len(forced)} forced) to {OUT}")
    final_buckets = Counter(bucket(mark_count(r)) for r in sampled)
    print(f"  final bucket distribution: {dict(final_buckets)}")


if __name__ == "__main__":
    main()
