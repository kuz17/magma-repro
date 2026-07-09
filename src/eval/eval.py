# src/eval/eval.py
"""
Eval harness for UIAgent on val.jsonl.

TWO MODES — same metric (click accuracy), different inputs:

  baseline   Raw screenshot + native Qwen prompt (no SoM, no adapter)
             Measures: how good is the base model at UI grounding?

  finetuned  SoM image + SoM prompt + LoRA adapter
             Measures: does SoM fine-tuning improve grounding?

  Delta = finetuned - baseline = what SoM training actually adds.

Usage:
    # baseline (raw screenshot, native Qwen grounding)
    python -m src.eval.eval --mode baseline --name baseline

    # fine-tuned (SoM image, SoM prompt, LoRA adapter)
    python -m src.eval.eval --mode finetuned --adapter models/lora_adapter --name finetuned

    # compare both
    python -m src.eval.eval --mode compare

Output:
    results/eval_baseline.json
    results/eval_finetuned.json

──────────────────────────────────────────────────────────────────────────────
PATCH NOTES (this revision):

1. compare() now surfaces the gt_mark==0 vs gt_mark!=0 split in the printed
   table. Previously these were computed and saved into the JSON summary
   (click_acc_gt_mark_0 / click_acc_gt_mark_non0) but never displayed in
   compare() -- meaning the actual collapse diagnostic ("does the model just
   always predict Mark 0") was invisible in the comparison table even though
   the data existed. A model that always predicts Mark 0 can still post a
   decent blended click_accuracy if a meaningful fraction of ground truth
   happens to be Mark 0 -- the blended number alone cannot tell collapse
   apart from genuine competence.

2. Per-page element sampling now seeds on the sample's image stem (a stable
   identity) instead of its row index in val.jsonl:

       OLD: random.Random(1000 + idx).choice(elems)
       NEW: random.Random(_stable_seed(stem)).choice(elems)

   idx is just "this row's position when I iterated the file this time" --
   if val.jsonl is ever regenerated or reordered between two eval runs (this
   project's corpus WAS regenerated mid-project), "sample #47" in one run's
   file and "sample #47" in another run's file are not guaranteed to be the
   same physical page. Seeding on the image filename/stem instead ties the
   sampled element choice to the actual content, not file position, so two
   runs against differently-ordered files are still choosing the same
   instruction for the same page.

   IMPORTANT: this changes which element gets sampled per page relative to
   pre-patch runs. Old saved JSONs (results/eval_*.json produced before this
   patch) remain internally valid and comparable *to each other*, since their
   sampling was already baked in when they ran. But don't mix pre-patch and
   post-patch runs in one compare() table -- re-run everything you want in a
   single comparison through this patched version so the sampling is
   consistent across the whole table.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────────
VAL_JSONL   = Path("data/processed/seeclick_web/val.jsonl")
RENDERS_DIR = Path("data/interim/renders/seeclick_web/batch")
RAW_IMG_DIR = Path("data/raw/seeclick_web/seeclick_web/cpfs01/user/chengkanzhi/seeclick_web_imgs_part")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Baseline prompt — no SoM, no marks, tests native Qwen grounding
BASELINE_PROMPT = (
    'In this view I need to click a button to "{instruction}". '
    "Respond with ONLY this format, nothing else:\n"
    "Coordinate: (x1, y1, x2, y2)\n"
    "Where x1,y1 is top-left and x2,y2 is bottom-right, all normalized 0 to 1."
)


def _stable_seed(key: str) -> int:
    """
    Deterministic seed derived from a stable identity string (e.g. image
    stem), not from file/row position. Using a hash instead of Python's
    built-in hash() because hash() is randomized per-process (PYTHONHASHSEED)
    unless disabled -- md5 is deterministic across runs/machines, which is
    what we actually need here for cross-run comparability.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ══════════════════════════════════════════════════════════════════════════════
# Task parsing
# ══════════════════════════════════════════════════════════════════════════════

def detect_task_type(user_turn: str) -> str:
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
    m = re.search(r"[Cc]oordinate[:\s]+\(([0-9.,\s]+)\)", assistant_turn)
    if not m:
        return None
    nums = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
    if len(nums) == 4:
        return tuple(nums)
    if len(nums) == 2:
        return (nums[0], nums[1], nums[0], nums[1])
    return None


def _parse_gt_mark(assistant_turn: str) -> Optional[int]:
    m = re.search(r"[Mm]ark[:\s]+(\d+)", assistant_turn)
    return int(m.group(1)) if m else None


def _load_sidecar_bbox(marks_path: Path, mark_id: int) -> Optional[tuple]:
    try:
        with open(marks_path) as f:
            placed = json.load(f)
        for mid, el in placed:
            if mid == mark_id:
                return tuple(el["bbox"])
    except Exception:
        pass
    return None


def _resolve_gt_bbox(gt_bbox: tuple, gt_mark: Optional[int], marks_path: Path) -> tuple:
    x1, y1, x2, y2 = gt_bbox
    is_degenerate = (abs(x2 - x1) < 1e-6) or (abs(y2 - y1) < 1e-6)
    if not is_degenerate:
        return gt_bbox
    if gt_mark is not None and marks_path.exists():
        actual = _load_sidecar_bbox(marks_path, gt_mark)
        if actual is not None:
            return actual
    eps = 0.02
    return (max(0.0, x1 - eps), max(0.0, y1 - eps),
            min(1.0, x2 + eps), min(1.0, y2 + eps))


def _normalize_prediction(
    bbox: Optional[tuple],
    point: Optional[tuple],
    image_path: Path,
) -> tuple[Optional[tuple], Optional[tuple]]:
    def _is_pixel(coords): return any(abs(v) > 2.0 for v in coords)
    def _img_size(p):
        try:
            from PIL import Image as _I
            with _I.open(p) as im: return im.size
        except Exception: return (1, 1)
    w, h = None, None
    if bbox is not None and _is_pixel(bbox):
        if w is None: w, h = _img_size(image_path)
        bbox = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
    if point is not None and _is_pixel(point):
        if w is None: w, h = _img_size(image_path)
        point = (point[0] / w, point[1] / h)
    return bbox, point


def extract_first_element(
    sample: dict,
    marks_path: Optional[Path] = None,
) -> Optional[dict]:
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None
    user_raw = convs[0]["value"]
    asst_raw = convs[1]["value"]
    user_text = re.sub(r"^<image>\n?", "", user_raw).strip()
    task_type = detect_task_type(user_text)
    if task_type not in ("text_to_point", "text_to_bbox"):
        return None
    gt_bbox = _parse_gt_coords(asst_raw)
    if gt_bbox is None:
        return None
    gt_mark = _parse_gt_mark(asst_raw)
    if marks_path is not None and gt_mark is not None:
        gt_bbox = _resolve_gt_bbox(gt_bbox, gt_mark, marks_path)

    if task_type == "text_to_point":
        full_text = " ".join(user_text.splitlines())
        m = re.search(r'[Tt]o execute the step ["\u201c](.+?)["\u201d]', full_text)
        instruction = m.group(1) if m else user_text.splitlines()[0]
    else:
        lines = [l.strip() for l in user_text.splitlines() if l.strip()]
        instruction = lines[1] if len(lines) > 1 else lines[0]

    return {
        "task_type":   task_type,
        "instruction": instruction,
        "gt_bbox":     gt_bbox,
        "gt_mark":     gt_mark,
    }


def extract_all_elements(
    sample: dict,
    marks_path: Optional[Path] = None,
) -> list[dict]:
    """
    Like extract_first_element, but returns EVERY eligible (instruction, gt)
    pair on the page, not just the first.

    Needed because the original eval only ever tested Mark 0 (convs[0:2]),
    which structurally cannot detect "model always predicts Mark 0" collapse
    -- ground truth was always Mark 0 too, so a collapsed model would score
    identically to a fixed one.
    """
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return []

    # Task type is a page-level property, determined by turn 0's instruction
    # phrasing -- later turns are just the bare element label, no repeated
    # instruction, so task_type can't be re-detected per-turn.
    user_raw0 = convs[0]["value"]
    user_text0 = re.sub(r"^<image>\n?", "", user_raw0).strip()
    task_type = detect_task_type(user_text0)
    if task_type not in ("text_to_point", "text_to_bbox"):
        return []

    elements = []
    i = 0
    while i + 1 < len(convs):
        user_raw = convs[i]["value"]
        asst_raw = convs[i + 1]["value"]
        user_text = re.sub(r"^<image>\n?", "", user_raw).strip()

        gt_bbox = _parse_gt_coords(asst_raw)
        if gt_bbox is None:
            i += 2
            continue
        gt_mark = _parse_gt_mark(asst_raw)
        if marks_path is not None and gt_mark is not None:
            gt_bbox = _resolve_gt_bbox(gt_bbox, gt_mark, marks_path)

        if i == 0:
            if task_type == "text_to_point":
                full_text = " ".join(user_text.splitlines())
                m = re.search(r'[Tt]o execute the step ["\u201c](.+?)["\u201d]', full_text)
                instruction = m.group(1) if m else user_text.splitlines()[0]
            else:
                lines = [l.strip() for l in user_text.splitlines() if l.strip()]
                instruction = lines[1] if len(lines) > 1 else lines[0]
        else:
            # Later turns are already just the bare element label/text --
            # directly usable as the instruction for this subtask.
            instruction = user_text

        elements.append({
            "task_type":   task_type,
            "instruction": instruction,
            "gt_bbox":     gt_bbox,
            "gt_mark":     gt_mark,
        })
        i += 2

    return elements


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter  = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(point: tuple, bbox: tuple) -> bool:
    cx, cy = point
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def center_distance(pred: tuple, gt_bbox: tuple) -> float:
    """
    Euclidean distance between predicted point and GT bbox center.
    Both in normalised [0,1] coords. Max possible ~1.41.
    """
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
    return ((pred[0] - gt_cx) ** 2 + (pred[1] - gt_cy) ** 2) ** 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    sample_id:    int
    mode:         str
    task_type:    str
    instruction:  str
    gt_bbox:      tuple
    gt_mark:      Optional[int]
    pred_mark:    Optional[int]
    pred_point:   Optional[tuple]
    pred_bbox:    Optional[tuple]
    raw_response: str
    click_hit:    bool
    iou_score:    float
    iou_hit:      bool
    no_pred:      bool            # parser found nothing
    dist_to_gt:   Optional[float] # None if no prediction
    elapsed_s:    float


# ══════════════════════════════════════════════════════════════════════════════
# Baseline inference — raw screenshot, native Qwen grounding, no SoM
# ══════════════════════════════════════════════════════════════════════════════

def _run_baseline_inference(backend, instruction: str, raw_image_path: Path) -> tuple:
    """
    Run the base model on a raw screenshot with a plain grounding prompt.
    No SoM image, no marks. Tests native Qwen grounding capability.
    Returns (raw_response, mark_id, bbox, point).
    """
    from PIL import Image
    image  = Image.open(raw_image_path).convert("RGB")
    prompt = BASELINE_PROMPT.format(instruction=instruction)
    response = backend._infer(prompt, image)
    action   = backend.parse(response, [])
    return response, action.mark_id, action.bbox, action.point


# ══════════════════════════════════════════════════════════════════════════════
# Main eval loop
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(
    mode:         str = "baseline",
    adapter_path: Optional[str] = None,
    max_samples:  Optional[int] = None,
    output_name:  str = "baseline",
) -> None:
    from src.agent.ui_agent import UIAgent

    assert mode in ("baseline", "finetuned"), \
        "mode must be 'baseline' or 'finetuned'"
    if mode == "finetuned" and adapter_path is None:
        raise ValueError("--adapter required for finetuned mode")

    print("=" * 60)
    print(f"Eval mode : {mode}")
    print(f"Run name  : {output_name}")
    print(f"Adapter   : {adapter_path or 'none'}")
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
    print(f"\nEvaluating {len(samples):,} samples in [{mode}] mode...\n")

    # ── eval loop ─────────────────────────────────────────────────────
    results: list[SampleResult] = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"Eval [{mode}]")):
        stem       = Path(sample["image"]).stem
        som_path   = RENDERS_DIR / f"{stem}_som.png"
        marks_path = RENDERS_DIR / f"{stem}_marks.json"
        raw_path   = RAW_IMG_DIR / sample["image"]

        if mode == "baseline" and not raw_path.exists():
            skipped += 1
            continue
        if mode == "finetuned" and (not som_path.exists() or not marks_path.exists()):
            skipped += 1
            continue

        elems = extract_all_elements(
            sample,
            marks_path=marks_path if marks_path.exists() else None,
        )
        if not elems:
            skipped += 1
            continue
        # Deterministic per-*page* pick (seeded on the image stem, a stable
        # identity), spread across every mark on the page instead of always
        # testing Mark 0 (see extract_all_elements docstring for why that
        # mattered). Seeding on stem rather than idx means the same page
        # gets the same sampled element regardless of val.jsonl's row order
        # in a given run -- see PATCH NOTES at top of file.
        elem = random.Random(_stable_seed(stem)).choice(elems)

        t0 = time.time()
        try:
            if mode == "baseline":
                raw_response, pred_mark, pred_bbox, pred_point = \
                    _run_baseline_inference(
                        agent.backend, elem["instruction"], raw_path
                    )
                image_for_norm = raw_path
            else:
                action         = agent.act(
                    elem["instruction"],
                    som_image_path=som_path,
                    marks_path=marks_path,
                )
                raw_response   = action.raw_response
                pred_mark      = action.mark_id
                pred_bbox      = action.bbox
                pred_point     = action.point
                image_for_norm = som_path

        except Exception as e:
            tqdm.write(f"[{idx}] error: {e}")
            skipped += 1
            continue
        elapsed = time.time() - t0

        # normalize pixel-scale predictions → [0, 1]
        pred_bbox, pred_point = _normalize_prediction(
            pred_bbox, pred_point, image_for_norm
        )

        gt_bbox = elem["gt_bbox"]

        # ── click_hit ────────────────────────────────────────────────
        click_hit = False
        if pred_point is not None:
            click_hit = point_in_bbox(pred_point, gt_bbox)
        elif pred_bbox is not None:
            px = (pred_bbox[0] + pred_bbox[2]) / 2
            py = (pred_bbox[1] + pred_bbox[3]) / 2
            click_hit = point_in_bbox((px, py), gt_bbox)

        # ── iou ──────────────────────────────────────────────────────
        iou_score = 0.0
        if pred_bbox is not None:
            if gt_bbox[0] != gt_bbox[2] and gt_bbox[1] != gt_bbox[3]:
                iou_score = iou(pred_bbox, gt_bbox)

        # ── no prediction ─────────────────────────────────────────────
        no_pred = (pred_point is None and pred_bbox is None)

        # ── distance to GT center ─────────────────────────────────────
        dist_to_gt  = None
        pred_center = None
        if pred_point is not None:
            pred_center = pred_point
        elif pred_bbox is not None:
            pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2,
                           (pred_bbox[1] + pred_bbox[3]) / 2)
        if pred_center is not None:
            dist_to_gt = center_distance(pred_center, gt_bbox)

        results.append(SampleResult(
            sample_id    = idx,
            mode         = mode,
            task_type    = elem["task_type"],
            instruction  = elem["instruction"],
            gt_bbox      = gt_bbox,
            gt_mark      = elem["gt_mark"],
            pred_mark    = pred_mark,
            pred_point   = pred_point,
            pred_bbox    = pred_bbox,
            raw_response = raw_response,
            click_hit    = click_hit,
            iou_score    = iou_score,
            iou_hit      = iou_score >= 0.5,
            no_pred      = no_pred,
            dist_to_gt   = dist_to_gt,
            elapsed_s    = elapsed,
        ))

    # ── aggregate metrics ─────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results.")
        return

    click_acc  = sum(r.click_hit for r in results) / n
    iou_hit_50 = sum(r.iou_hit   for r in results) / n
    mean_iou   = sum(r.iou_score for r in results) / n
    no_pred_n  = sum(r.no_pred   for r in results)
    mean_time  = sum(r.elapsed_s for r in results) / n

    dists     = [r.dist_to_gt for r in results if r.dist_to_gt is not None]
    mean_dist = sum(dists) / len(dists) if dists else None

    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r)

    print("\n" + "=" * 60)
    print(f"Results [{mode}]  —  {n} evaluated, {skipped} skipped")
    print("=" * 60)
    print(f"  Click accuracy (point in GT bbox) : {click_acc:.3f}  ({click_acc*100:.1f}%)")
    print(f"  IoU hit @ 0.5                     : {iou_hit_50:.3f}  ({iou_hit_50*100:.1f}%)")
    print(f"  Mean IoU                           : {mean_iou:.3f}")
    if mean_dist is not None:
        print(f"  Mean dist to GT center            : {mean_dist:.3f}  (when pred exists, {len(dists)}/{n} samples)")
    print(f"  No prediction (parser found noth) : {no_pred_n}/{n}")
    print(f"  Mean inference time               : {mean_time:.2f}s / sample")

    # ── collapse diagnostic: does accuracy crater for gt_mark != 0? ────
    mark0_results = [r for r in results if r.gt_mark == 0]
    non0_results  = [r for r in results if r.gt_mark is not None and r.gt_mark != 0]
    acc0 = acc_non0 = None
    if mark0_results:
        acc0 = sum(r.click_hit for r in mark0_results) / len(mark0_results)
        print(f"  Click acc when gt_mark == 0        : {acc0:.3f}  (n={len(mark0_results)})")
    if non0_results:
        acc_non0 = sum(r.click_hit for r in non0_results) / len(non0_results)
        print(f"  Click acc when gt_mark != 0        : {acc_non0:.3f}  (n={len(non0_results)})")
    print()

    for ttype, rs in sorted(by_task.items()):
        acc  = sum(r.click_hit  for r in rs) / len(rs)
        ds   = [r.dist_to_gt for r in rs if r.dist_to_gt is not None]
        md   = sum(ds) / len(ds) if ds else None
        dist_str = f"  mean_dist={md:.3f}" if md is not None else ""
        print(f"  [{ttype}]  n={len(rs)}  click_acc={acc:.3f}{dist_str}")
    print("=" * 60)

    # ── save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"eval_{output_name}.json"
    summary = {
        "run":            output_name,
        "mode":           mode,
        "n_evaluated":    n,
        "n_skipped":      skipped,
        "click_accuracy": round(click_acc,  4),
        "iou_hit_50":     round(iou_hit_50, 4),
        "mean_iou":       round(mean_iou,   4),
        "mean_dist_to_gt":round(mean_dist,  4) if mean_dist is not None else None,
        "no_pred":        no_pred_n,
        "mean_time_s":    round(mean_time,  2),
        "click_acc_gt_mark_0":    round(acc0, 4)     if acc0     is not None else None,
        "click_acc_gt_mark_non0": round(acc_non0, 4) if acc_non0 is not None else None,
        "n_gt_mark_0":     len(mark0_results),
        "n_gt_mark_non0":  len(non0_results),
        "by_task": {
            t: {
                "n":            len(rs),
                "click_acc":    round(sum(r.click_hit  for r in rs) / len(rs), 4),
                "iou_hit_50":   round(sum(r.iou_hit    for r in rs) / len(rs), 4),
                "mean_iou":     round(sum(r.iou_score  for r in rs) / len(rs), 4),
                "mean_dist":    round(sum(r.dist_to_gt for r in rs
                                         if r.dist_to_gt is not None) /
                                      max(1, sum(1 for r in rs
                                                 if r.dist_to_gt is not None)), 4),
                "no_pred":      sum(r.no_pred for r in rs),
            }
            for t, rs in by_task.items()
        },
        "samples": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Compare two+ runs
# ══════════════════════════════════════════════════════════════════════════════

def compare(json_files: list[str]) -> None:
    runs = []

    for path in json_files:
        with open(path) as f:
            data = json.load(f)

        runs.append({
            "name": data.get("run", Path(path).stem),
            "click_accuracy": data.get("click_accuracy"),
            "iou_hit_50": data.get("iou_hit_50"),
            "mean_iou": data.get("mean_iou"),
            "mean_dist_to_gt": data.get("mean_dist_to_gt"),
            "no_pred": data.get("no_pred"),
            "n_evaluated": data.get("n_evaluated"),
            # PATCH: these existed in the saved JSON all along but were
            # never pulled into compare()'s table before.
            "click_acc_gt_mark_0":    data.get("click_acc_gt_mark_0"),
            "click_acc_gt_mark_non0": data.get("click_acc_gt_mark_non0"),
            "n_gt_mark_0":            data.get("n_gt_mark_0"),
            "n_gt_mark_non0":         data.get("n_gt_mark_non0"),
        })

    if not runs:
        print("No runs provided.")
        return

    # Highest click accuracy is treated as reference
    best_click = max(r["click_accuracy"] for r in runs if r["click_accuracy"] is not None)

    print("\n" + "=" * 135)
    print("Evaluation Comparison")
    print("=" * 135)
    print(
        f"{'Run':<26}"
        f"{'Click Acc':>11}"
        f"{'IoU@0.5':>10}"
        f"{'Mean IoU':>10}"
        f"{'Mean Dist':>11}"
        f"{'No Pred':>9}"
        f"{'Acc(mark=0)':>13}"
        f"{'Acc(mark!=0)':>14}"
        f"{'Δ Click':>10}"
    )
    print("-" * 135)

    for r in sorted(runs, key=lambda x: x["click_accuracy"], reverse=True):

        delta = r["click_accuracy"] - best_click

        def fmt_pct(v):
            return f"{100*v:.1f}%" if v is not None else "-"

        def fmt_num(v):
            return f"{v:.3f}" if v is not None else "-"

        click  = fmt_pct(r["click_accuracy"])
        iou50  = fmt_pct(r["iou_hit_50"])
        miou   = fmt_num(r["mean_iou"])
        mdist  = fmt_num(r["mean_dist_to_gt"])
        nopred = str(r["no_pred"]) if r["no_pred"] is not None else "-"

        mark0_str = fmt_pct(r["click_acc_gt_mark_0"])
        if r["n_gt_mark_0"] is not None:
            mark0_str += f" (n={r['n_gt_mark_0']})"

        non0_str = fmt_pct(r["click_acc_gt_mark_non0"])
        if r["n_gt_mark_non0"] is not None:
            non0_str += f" (n={r['n_gt_mark_non0']})"

        print(
            f"{r['name']:<26}"
            f"{click:>11}"
            f"{iou50:>10}"
            f"{miou:>10}"
            f"{mdist:>11}"
            f"{nopred:>9}"
            f"{mark0_str:>13}"
            f"{non0_str:>14}"
            f"{delta*100:+9.1f}%"
        )

    print("=" * 135)
    print(
        "Note: Acc(mark!=0) is the real collapse diagnostic -- a model that "
        "always predicts Mark 0 can still post a decent blended Click Acc if "
        "a meaningful share of ground truth is genuinely Mark 0. Trust the "
        "mark!=0 column over the blended one when judging whether SoM "
        "training actually generalized."
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned", "compare"],
                        default="baseline")
    parser.add_argument("--adapter",        type=str, default=None)
    parser.add_argument("--max-samples",    type=int, default=None)
    parser.add_argument("--name",           type=str, default=None)
    parser.add_argument(
    "--jsons",
    nargs="+",
    required=False,
    default=[
        "results/eval_baseline.json",
        "results/eval_full_finetuned.json",
    ],
    help="One or more evaluation JSON files."
    )
    args = parser.parse_args()

    if args.mode == "compare":
        compare(args.jsons)
    else:
        name = args.name or args.mode
        run_eval(
            mode=args.mode,
            adapter_path=args.adapter,
            max_samples=args.max_samples,
            output_name=name,
        )