"""
sweep_checkpoints.py

Re-evaluates every checkpoint (base Qwen + every LoRA checkpoint dir you point
it at) through the CURRENT src.eval.eval harness against ONE fixed val set,
so every row in the resulting table is apples-to-apples.

Solves two problems at once:
  1. Some of your old rows (smoke_finetuned, full_finetuned, etc.) were likely
     evaluated with an older extract_first_element-based harness or a
     different val split. This script forces every row through the harness
     that exists on disk RIGHT NOW.
  2. You don't know which checkpoint in attempt 2's history (steps ~450-900,
     saved every 50) is actually the best one before whatever degradation
     kicks in. This sweeps all of them and lets you compare accuracy vs step.

Usage:
    python sweep_checkpoints.py \
        --checkpoint-root models/lora_adapter_v2 \
        --include-base \
        --extra-adapter models/lora_adapter=smoke \
        --max-samples 300 \
        --out-csv outputs/sweep_results.csv

REVISION NOTES:
    - PATCH: eval.py actually writes clean JSON to results/eval_{name}.json
      per run (confirmed from source) -- this version reads THAT file
      directly instead of guessing at stdout regex. Much more reliable.
    - PATCH: subprocess now streams eval.py's stdout live (no
      capture_output=True) so you see eval.py's own tqdm progress bar and
      aren't staring at a seemingly-frozen terminal during a multi-minute
      inference run.
    - PATCH: --max-samples is now a required-in-practice knob (defaults to
      300 below) so a sweep of N checkpoints doesn't mean N full passes over
      the entire val.jsonl (~7,000+ pages) on local hardware. 300 samples is
      plenty to see a real mark!=0 gap; bump it later once you've narrowed
      down which checkpoints matter.
    - Sorts final table by click_acc_gt_mark_non0 (the real collapse
      diagnostic), not blended click_accuracy.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("results")

RESULT_FIELDS = [
    "click_accuracy", "iou_hit_50", "mean_iou", "mean_dist_to_gt", "no_pred",
    "click_acc_gt_mark_0", "click_acc_gt_mark_non0",
    "n_gt_mark_0", "n_gt_mark_non0",
]


def run_eval(mode: str, adapter, max_samples: int, name: str) -> dict:
    """
    Runs eval.py as a subprocess (NOTE: --jsons is only meaningful for
    --mode compare in eval.py; baseline/finetuned modes ignore it, so it's
    correctly omitted here). Streams output live so you can watch eval.py's
    own tqdm progress bar instead of staring at a blank terminal.
    Then reads the JSON eval.py writes to results/eval_{name}.json directly
    -- far more reliable than parsing stdout.
    """
    cmd = [sys.executable, "-m", "src.eval.eval", "--mode", mode,
           "--name", name, "--max-samples", str(max_samples)]
    if adapter:
        cmd += ["--adapter", adapter]

    print(f"\n{'='*70}\nRunning: {name}\n  cmd: {' '.join(cmd)}\n{'='*70}")
    # No capture_output -- let eval.py's stdout/stderr (including its tqdm
    # bar) print straight to the terminal in real time.
    proc = subprocess.run(cmd)

    if proc.returncode != 0:
        print(f"!!! FAILED (exit {proc.returncode}) -- see output above")
        return {"name": name, "status": "FAILED", **{k: None for k in RESULT_FIELDS}}

    out_path = RESULTS_DIR / f"eval_{name}.json"
    if not out_path.exists():
        print(f"!!! eval.py exited 0 but {out_path} was not created")
        return {"name": name, "status": "MISSING_OUTPUT", **{k: None for k in RESULT_FIELDS}}

    with open(out_path) as f:
        data = json.load(f)

    result = {k: data.get(k) for k in RESULT_FIELDS}
    result["name"] = name
    result["status"] = "OK"
    return result


def discover_checkpoints(checkpoint_root: str) -> list[tuple[str, str]]:
    """Find checkpoint-N subdirs under checkpoint_root, sorted by step number."""
    root = Path(checkpoint_root)
    if not root.exists():
        print(f"WARNING: checkpoint root {root} does not exist, skipping")
        return []
    ckpts = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[-1])
                ckpts.append((step, str(d)))
            except ValueError:
                continue
    ckpts.sort(key=lambda x: x[0])
    return [(f"ckpt_{step}", path) for step, path in ckpts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", default=None,
                     help="Directory containing checkpoint-N subdirs to sweep (e.g. models/lora_adapter_v2)")
    ap.add_argument("--include-base", action="store_true",
                     help="Also run --mode baseline (no adapter) as a reference row")
    ap.add_argument("--extra-adapter", action="append", default=[],
                     help="Extra single adapter path to include, format 'path=label' "
                          "(e.g. models/lora_adapter=smoke). Repeatable.")
    ap.add_argument("--max-samples", type=int, default=300,
                     help="Samples per eval run (default 300 -- plenty to see a real "
                          "mark!=0 gap without a multi-hour pass over the full val set "
                          "on local hardware; raise later for a final confirmatory run)")
    ap.add_argument("--out-csv", default="outputs/sweep_results.csv")
    args = ap.parse_args()

    rows = []

    if args.include_base:
        rows.append(run_eval("baseline", None, args.max_samples, "base_qwen"))

    for a in args.extra_adapter:
        path, _, label = a.partition("=")
        label = label or Path(path).name
        rows.append(run_eval("finetuned", path, args.max_samples, label))

    if args.checkpoint_root:
        for name, ckpt_path in discover_checkpoints(args.checkpoint_root):
            rows.append(run_eval("finetuned", ckpt_path, args.max_samples, name))

    if not rows:
        print("Nothing to sweep -- pass --include-base, --extra-adapter, and/or --checkpoint-root")
        return

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "status"] + RESULT_FIELDS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"\n\nWrote {len(rows)} rows to {out_path}")
    print("\nSorted by click_acc_gt_mark_non0 (the real collapse diagnostic):")
    for r in sorted(rows, key=lambda r: (r.get("click_acc_gt_mark_non0")
                                          if r.get("click_acc_gt_mark_non0") is not None else -1),
                     reverse=True):
        nonzero = r.get("click_acc_gt_mark_non0")
        overall = r.get("click_accuracy")
        iou50   = r.get("iou_hit_50")
        print(f"  {r['name']:30s} "
              f"mark!=0={nonzero if nonzero is not None else '-'}  "
              f"overall={overall if overall is not None else '-'}  "
              f"iou50={iou50 if iou50 is not None else '-'}  "
              f"status={r['status']}")


if __name__ == "__main__":
    main()