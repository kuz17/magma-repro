# src/eval/eval.py
"""
Eval harness for UIAgent on val.jsonl.

Metric: click accuracy — predicted point falls inside ground-truth bbox.
Also reports mean IoU for bbox predictions and per-task-type breakdown.

Each val sample produces ONE agent call (first grounding element only),
keeping total eval time predictable (~1 hr on GTX 1650 4GB).

Usage:
    python -m src.eval.eval                        # baseline (no adapter)
    python -m src.eval.eval --adapter models/lora  # fine-tuned

Output:
    results/eval_baseline.json   or   results/eval_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────────
VAL_JSONL    = Path("data/processed/seeclick_web/val.jsonl")
RENDERS_DIR  = Path("data/interim/renders/seeclick_web/batch")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Task parsing — detect type and extract (instruction, gt_bbox) from a sample
# ══════════════════════════════════════════════════════════════════════════════

def detect_task_type(user_turn: str) -> str:
    """
    Infer grounding task type from user turn text.
    Returns one of: text_to_point, text_to_bbox, bbox_to_text, point_to_text, unknown
    """
    t = user_turn
    if "where do I direct my attention" in t:
        return "text_to_point"
    if "Predict each element's bounding box" in t:
        return "text_to_bbox"
    if "I'll provide bounding boxes" in t:
        return "bbox_to_text"
    if "contribute to the overall user experience" in t:
        return "point_to_text"
    return "unknown"


def _parse_gt_coords(assistant_turn: str) -> Optional[tuple]:
    """
    Extract the FIRST ground-truth coordinate from an assistant turn.
    Handles both:
        "Coordinate: (0.27, 0.29). Mark: 2."          ← point (2 floats)
        "Coordinate: (0.27, 0.29, 0.37, 0.30). Mark: 2."  ← bbox (4 floats)
    Returns (x1, y1, x2, y2) bbox always.
    For 2-float point output, bbox is None (can't reconstruct without mark).
    Returns None if no coordinate found.
    """
    m = re.search(r"[Cc]oordinate[:\s]+\(([0-9.,\s]+)\)", assistant_turn)
    if not m:
        return None
    nums = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
    if len(nums) == 4:
        return tuple(nums)
    if len(nums) == 2:
        # point only — return as a degenerate bbox centered on that point
        # (will be matched against GT bbox via point-in-bbox test)
        return (nums[0], nums[1], nums[0], nums[1])
    return None


def _parse_gt_mark(assistant_turn: str) -> Optional[int]:
    m = re.search(r"[Mm]ark[:\s]+(\d+)", assistant_turn)
    return int(m.group(1)) if m else None


def _load_sidecar_bbox(marks_path: Path, mark_id: int) -> Optional[tuple]:
    """
    Look up the actual normalised bbox for a given mark_id in a _marks.json sidecar.
    Returns (x1, y1, x2, y2) or None.
    """
    try:
        with open(marks_path) as f:
            placed = json.load(f)
        for mid, el in placed:
            if mid == mark_id:
                return tuple(el["bbox"])
    except Exception:
        pass
    return None


def _resolve_gt_bbox(
    gt_bbox: tuple,
    gt_mark: Optional[int],
    marks_path: Path,
) -> tuple:
    """
    Fix degenerate GT bboxes (x1==x2 or y1==y2) that arise from
    text_to_point tasks where only a center point is stored.
    Strategy: look up the actual element bbox from the sidecar.
    Falls back to a small epsilon expansion around the point.
    """
    x1, y1, x2, y2 = gt_bbox
    is_degenerate = (abs(x2 - x1) < 1e-6) or (abs(y2 - y1) < 1e-6)

    if not is_degenerate:
        return gt_bbox

    # Try sidecar lookup first
    if gt_mark is not None and marks_path.exists():
        actual = _load_sidecar_bbox(marks_path, gt_mark)
        if actual is not None:
            return actual

    # Fallback: expand point by epsilon (2% of screen in each direction)
    eps = 0.02
    return (
        max(0.0, x1 - eps),
        max(0.0, y1 - eps),
        min(1.0, x2 + eps),
        min(1.0, y2 + eps),
    )


def _normalize_prediction(
    bbox: Optional[tuple],
    point: Optional[tuple],
    image_path: Path,
) -> tuple[Optional[tuple], Optional[tuple]]:
    """
    Detect pixel-scale predictions (any value > 2.0) and normalize
    them to [0, 1] by dividing by the image dimensions.
    Returns (normalized_bbox, normalized_point).
    """
    def _is_pixel_scale(coords: tuple) -> bool:
        return any(abs(v) > 2.0 for v in coords)

    def _img_size(p: Path) -> tuple[int, int]:
        try:
            from PIL import Image as _I
            with _I.open(p) as im:
                return im.size  # (width, height)
        except Exception:
            return (1, 1)

    w, h = None, None

    if bbox is not None and _is_pixel_scale(bbox):
        if w is None:
            w, h = _img_size(image_path)
        bbox = (bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h)

    if point is not None and _is_pixel_scale(point):
        if w is None:
            w, h = _img_size(image_path)
        point = (point[0]/w, point[1]/h)

    return bbox, point


def extract_first_element(sample: dict, marks_path: Optional[Path] = None) -> Optional[dict]:
    """
    From a JSONL record, extract the first grounding element as:
        {
            "task_type": str,
            "instruction": str,      ← text shown to the agent
            "gt_bbox": (x1,y1,x2,y2) ← normalised ground-truth bbox
            "gt_mark": int | None
        }
    Returns None for non-grounding tasks (point_to_text, bbox_to_text).
    If marks_path is provided, degenerate GT points are expanded to actual
    element bboxes via sidecar lookup.
    """
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None

    user_raw  = convs[0]["value"]          # "<image>\n..."
    asst_raw  = convs[1]["value"]

    # strip the "<image>\n" prefix
    user_text = re.sub(r"^<image>\n?", "", user_raw).strip()
    task_type = detect_task_type(user_text)

    if task_type not in ("text_to_point", "text_to_bbox"):
        return None   # skip non-grounding tasks for coordinate eval

    gt_bbox = _parse_gt_coords(asst_raw)
    if gt_bbox is None:
        return None

    gt_mark = _parse_gt_mark(asst_raw)

    # Fix degenerate bboxes using sidecar
    if marks_path is not None and gt_mark is not None:
        gt_bbox = _resolve_gt_bbox(gt_bbox, gt_mark, marks_path)

    # Extract the first instruction line
    if task_type == "text_to_point":
        # "To execute the step "X", where do I direct my attention?..."
        # Could be multiple such lines — take the first one
        first_line = user_text.splitlines()[0]
        m = re.search(r'To execute the step "(.+?)"', first_line)
        instruction = m.group(1) if m else first_line
    else:
        # text_to_bbox: "From this webpage capture, ... \nElement text\n..."
        lines = [l.strip() for l in user_text.splitlines() if l.strip()]
        # skip the prefix line(s), take first element description
        instruction = lines[1] if len(lines) > 1 else lines[0]

    return {
        "task_type":   task_type,
        "instruction": instruction,
        "gt_bbox":     gt_bbox,
        "gt_mark":     gt_mark,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def iou(box_a: tuple, box_b: tuple) -> float:
    """IoU of two (x1,y1,x2,y2) normalised boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(point: tuple, bbox: tuple) -> bool:
    """True if (cx, cy) falls inside (x1, y1, x2, y2)."""
    cx, cy = point
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    sample_id:    int
    task_type:    str
    instruction:  str
    gt_bbox:      tuple
    gt_mark:      Optional[int]
    pred_mark:    Optional[int]
    pred_point:   Optional[tuple]
    pred_bbox:    Optional[tuple]
    raw_response: str
    click_hit:    bool    # predicted point falls inside gt_bbox
    iou_score:    float   # IoU of pred_bbox vs gt_bbox (0 if no pred_bbox)
    iou_hit:      bool    # IoU ≥ 0.5
    elapsed_s:    float


# ══════════════════════════════════════════════════════════════════════════════
# Main eval loop
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(
    adapter_path: Optional[str] = None,
    max_samples:  Optional[int] = None,
    output_name:  str = "baseline",
) -> None:
    from src.agent.ui_agent import UIAgent

    print("=" * 60)
    print(f"Eval run  : {output_name}")
    print(f"Val set   : {VAL_JSONL}")
    print(f"Adapter   : {adapter_path or 'none (baseline)'}")
    print("=" * 60)

    # ── load agent ───────────────────────────────────────────────────
    agent = UIAgent.with_qwen(load_in_4bit=True)

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path} ...")
        agent.backend.model = PeftModel.from_pretrained(
            agent.backend.model, adapter_path
        )
        agent.backend.model.eval()
        print("Adapter loaded.")

    # ── load val set ──────────────────────────────────────────────────
    samples = []
    with open(VAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    print(f"\nEvaluating {len(samples):,} samples ...\n")

    # ── eval loop ─────────────────────────────────────────────────────
    results: list[SampleResult] = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc="Eval")):
        # resolve SoM image + marks paths from sample
        stem = Path(sample["image"]).stem
        som_path   = RENDERS_DIR / f"{stem}_som.png"
        marks_path = RENDERS_DIR / f"{stem}_marks.json"

        if not som_path.exists() or not marks_path.exists():
            skipped += 1
            continue

        elem = extract_first_element(sample, marks_path=marks_path)
        if elem is None:
            skipped += 1
            continue

        t0 = time.time()
        try:
            action = agent.act(
                elem["instruction"],
                som_image_path=som_path,
                marks_path=marks_path,
            )
        except Exception as e:
            tqdm.write(f"[{idx}] inference error: {e}")
            skipped += 1
            continue
        elapsed = time.time() - t0

        # ── normalize pixel-scale predictions → [0,1] ────────────────
        action.bbox, action.point = _normalize_prediction(
            action.bbox, action.point, som_path
        )

        # ── score ────────────────────────────────────────────────────
        gt_bbox = elem["gt_bbox"]

        # click_hit: predicted point inside GT bbox
        click_hit = False
        if action.point is not None:
            click_hit = point_in_bbox(action.point, gt_bbox)
        elif action.bbox is not None:
            # use predicted bbox center
            px = (action.bbox[0] + action.bbox[2]) / 2
            py = (action.bbox[1] + action.bbox[3]) / 2
            click_hit = point_in_bbox((px, py), gt_bbox)

        # iou score
        iou_score = 0.0
        if action.bbox is not None:
            # skip degenerate gt_bbox (point-only GT)
            if gt_bbox[0] != gt_bbox[2] and gt_bbox[1] != gt_bbox[3]:
                iou_score = iou(action.bbox, gt_bbox)

        results.append(SampleResult(
            sample_id    = idx,
            task_type    = elem["task_type"],
            instruction  = elem["instruction"],
            gt_bbox      = gt_bbox,
            gt_mark      = elem["gt_mark"],
            pred_mark    = action.mark_id,
            pred_point   = action.point,
            pred_bbox    = action.bbox,
            raw_response = action.raw_response,
            click_hit    = click_hit,
            iou_score    = iou_score,
            iou_hit      = iou_score >= 0.5,
            elapsed_s    = elapsed,
        ))

    # ── aggregate metrics ─────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results — check val.jsonl and renders paths.")
        return

    click_acc  = sum(r.click_hit  for r in results) / n
    iou_hit_50 = sum(r.iou_hit    for r in results) / n
    mean_iou   = sum(r.iou_score  for r in results) / n
    mean_time  = sum(r.elapsed_s  for r in results) / n

    # per-task breakdown
    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r)

    print("\n" + "=" * 60)
    print(f"Results ({output_name})  —  {n} samples evaluated, {skipped} skipped")
    print("=" * 60)
    print(f"  Click accuracy (point in GT bbox) : {click_acc:.3f}  ({click_acc*100:.1f}%)")
    print(f"  IoU hit @ 0.5                     : {iou_hit_50:.3f}  ({iou_hit_50*100:.1f}%)")
    print(f"  Mean IoU                           : {mean_iou:.3f}")
    print(f"  Mean inference time                : {mean_time:.2f}s / sample")
    print()
    for ttype, rs in sorted(by_task.items()):
        acc = sum(r.click_hit for r in rs) / len(rs)
        print(f"  [{ttype}]  n={len(rs)}  click_acc={acc:.3f}")
    print("=" * 60)

    # ── save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"eval_{output_name}.json"
    summary = {
        "run":            output_name,
        "n_evaluated":    n,
        "n_skipped":      skipped,
        "click_accuracy": round(click_acc,  4),
        "iou_hit_50":     round(iou_hit_50, 4),
        "mean_iou":       round(mean_iou,   4),
        "mean_time_s":    round(mean_time,  2),
        "by_task": {
            t: {
                "n":            len(rs),
                "click_acc":    round(sum(r.click_hit  for r in rs) / len(rs), 4),
                "iou_hit_50":   round(sum(r.iou_hit    for r in rs) / len(rs), 4),
                "mean_iou":     round(sum(r.iou_score  for r in rs) / len(rs), 4),
            }
            for t, rs in by_task.items()
        },
        "samples": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval UIAgent on val.jsonl")
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to LoRA adapter directory (omit for baseline)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (default: all 999)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Output file name suffix (default: 'baseline' or 'finetuned')"
    )
    args = parser.parse_args()

    name = args.name or ("finetuned" if args.adapter else "baseline")
    run_eval(
        adapter_path=args.adapter,
        max_samples=args.max_samples,
        output_name=name,
    )