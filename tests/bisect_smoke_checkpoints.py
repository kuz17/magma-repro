# tests/bisect_smoke_checkpoints.py
"""
Visual-inspection bisection across the non-fused SeeClick-Web checkpoints:
ckpt_800, lora_adapter, lora_adapter_full, lora_adapter_full_v2,
lora_adapter_smoke, lora_adapter_v2. Excludes lora_adapter_fused and
lora_adapter_fused_checkpoints -- those use a different training format
and adapter (act_fused(), not act()) -- see bisect_fused_checkpoints.py
for that track.

Loads checkpoints ONE AT A TIME (delete/load/set cycle, never stacked --
same VRAM discipline as bisect_fused_checkpoints.py). OmniParser + the
DOM-priority rebuild are run ONCE per page and cached, since neither
depends on which checkpoint is active -- only the adapter and the
_run_qwen() call vary per iteration.

Task phrasing: confirmed via train_smoke.jsonl inspection that this
training format's instructions are the LITERAL, VERBATIM on-page
text/caption of the target element -- not a description or "click on X"
framing (e.g. training data reads 'To execute the step "8 044 773 37 49",
where do I direct my attention?', never 'To execute the step "click the
phone number"'). PROMPT_TEMPLATE wraps whatever task string is given the
same way, so tasks below are bare labels matching what's actually printed
on the target element, not instructional phrasing -- this is the direct
test of that hypothesis across every non-fused checkpoint.

Mark resolution mirrors exactly what act() does internally (coordinate
parsed first, snapped to nearest mark within NEAREST_MARK_MAX_DIST, only
falling back to literal "Mark: N" text if no coordinate parsed) --
reimplemented here rather than calling act() directly so the OmniParser +
DOM-rebuild stage can be cached once per page instead of rerun per
checkpoint.

Setup:
    Screenshots + DOM sidecars expected at outputs/bisect_screenshots/
    (see tests/grab_bisect_screenshots.py).

Usage:
    python tests/bisect_smoke_checkpoints.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
from PIL import Image
from tqdm import tqdm

from src.agent.click_visualizer import (
    DemoRunner,
    PROMPT_TEMPLATE,
    _parse_point,
    _extract_mark,
    _nearest_mark,
    _draw_click,
)

# ══════════════════════════════════════════════════════════════════════════════
# Checkpoints — loaded and tested ONE AT A TIME, never stacked.
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINTS = {
    "lora_adapter":        "models/lora_adapter",         # original, STALE -- has Mark:0 collapse bug, included as a known-bad reference point
    "lora_adapter_smoke":  "models/lora_adapter_smoke",    # 25 steps -- best on gt_mark!=0 (22.6%) per handover doc
    "lora_adapter_v2":     "models/lora_adapter_v2",       # 91 steps -- best overall (17.4%) per handover doc
    "lora_adapter_full":   "models/lora_adapter_full",     # attempt 2 -- regressed (2.5%), unexplained
    "ckpt_800":            "models/ckpt_800",              # attempt 2, mid-run checkpoint -- still bad (4.3%)
    "lora_adapter_full_v2": "models/lora_adapter_full_v2", # attempt 3 -- also regressed (11.2%)
}

ADAPTER_NAME = "current"  # single reused adapter slot — swapped, never stacked
DOM_REBUILD_MAX_MARKS = 15  # matches act()'s current live value (reverted from 30 tonight)

# ══════════════════════════════════════════════════════════════════════════════
# Pages and tasks — bare, verbatim on-page text (NOT "click on X" phrasing)
# ══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "home": {
        "image": "outputs/bisect_screenshots/amazon_home.png",
        "tasks": [
            "Cart",
            "Returns\n& Orders",   # printed on two lines on the actual nav bar — try both forms if this misses
            "Amazon.in",
        ],
    },
    "results": {
        "image": "outputs/bisect_screenshots/search_results_odyssey.png",
        "tasks": [
            "Add to cart",
            "The Odyssey",
            "Amazon.in",
        ],
    },
    "product": {
        "image": "outputs/bisect_screenshots/product_detail_odyssey.png",
        "tasks": [
            "Add to cart",
            "Buy Now",
            "Hardcover",
        ],
    },
}

OUTPUT_DIR = Path("outputs/bisect_smoke")


def _task_slug(task: str) -> str:
    return task.lower().replace(" ", "_").replace("\n", "_").replace(".", "")[:40]


def main():
    ckpt_items = list(CHECKPOINTS.items())
    first_tag, first_path = ckpt_items[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base Qwen + OmniParser + first checkpoint ({first_tag})...")
    runner = DemoRunner(
        lora_path=first_path,
        training_style=True,
        tag="bisect_smoke",
    )
    runner._qwen.load_adapter(first_path, adapter_name=ADAPTER_NAME)
    runner._qwen.delete_adapter("default")
    runner._qwen.set_adapter(ADAPTER_NAME)

    # Pre-run OmniParser + DOM-priority rebuild ONCE per page — cached and
    # reused across every checkpoint, since neither depends on the adapter.
    print("\nPre-running OmniParser for all pages (once each)...")
    page_cache = {}
    for page_name, page_cfg in tqdm(PAGES.items(), desc="OmniParser pre-pass"):
        image_path = Path(page_cfg["image"])
        image = Image.open(image_path).convert("RGB")

        som_image, mark_to_center, content_list = runner._run_omniparser(image)

        dom_path = image_path.with_name(image_path.stem + "_dom.json")
        if dom_path.exists():
            with open(dom_path) as f:
                dom_elements = json.load(f)
            som_image, mark_to_center, content_list = \
                runner._rebuild_som_dom_priority(
                    image, content_list, dom_elements, max_marks=DOM_REBUILD_MAX_MARKS,
                )
            print(f"  {page_name}: DOM-priority rebuild applied "
                  f"({len(dom_elements)} DOM elements, {len(mark_to_center)} marks)")
        else:
            print(f"  {page_name}: no DOM sidecar found ({dom_path}) — "
                  f"using raw OmniParser marks only ({len(mark_to_center)} marks)")

        som_debug_path = OUTPUT_DIR / f"{page_name}_som_reference.png"
        som_image.save(som_debug_path)
        page_cache[page_name] = {
            "original_image": image,
            "som_image": som_image,
            "mark_to_center": mark_to_center,
            "content_list": content_list,
        }
    print(f"Done. SoM references saved under {OUTPUT_DIR}/\n")
    print("⚠ Check the *_som_reference.png files now — confirm each task's target "
          "element actually has a mark before trusting the sweep results below.\n")

    plan = []
    for tag, path in ckpt_items:
        for page_name, page_cfg in PAGES.items():
            for task in page_cfg["tasks"]:
                plan.append((tag, path, page_name, task))

    print(f"Running {len(ckpt_items)} checkpoints x {len(PAGES)} pages x 3 tasks "
          f"= {len(plan)} calls (checkpoints loaded one at a time).\n")

    csv_rows = []
    current_ckpt = None
    pbar = tqdm(plan, desc="Bisecting", unit="call")

    for tag, path, page_name, task in pbar:
        if tag != current_ckpt:
            if current_ckpt is not None:
                pbar.write(f"\nSwapping adapter -> {tag} ({path})")
                runner._qwen.delete_adapter(ADAPTER_NAME)
                torch.cuda.empty_cache()
                runner._qwen.load_adapter(path, adapter_name=ADAPTER_NAME)
                runner._qwen.set_adapter(ADAPTER_NAME)
            current_ckpt = tag

        pbar.set_postfix(ckpt=tag, page=page_name, refresh=False)

        cache = page_cache[page_name]
        original_image = cache["original_image"]
        som_image      = cache["som_image"]
        mark_to_center = cache["mark_to_center"]

        task_slug = _task_slug(task)
        prompt = PROMPT_TEMPLATE.format(task=task)

        t0 = time.time()
        response = runner._run_qwen(som_image, prompt)
        elapsed = time.time() - t0

        w, h = original_image.size
        raw_point = _parse_point(response, image_size=(w, h))
        mark_id = _extract_mark(response)

        point = None
        resolved_mark = None
        method = "none"
        if raw_point is not None:
            nearest = _nearest_mark(raw_point, mark_to_center)
            if nearest is not None:
                point = mark_to_center[nearest]
                resolved_mark = nearest
                method = "coord_snapped"
            else:
                point = raw_point
                method = "coord_raw"
        elif mark_id is not None:
            point = mark_to_center.get(mark_id)
            resolved_mark = mark_id
            method = "mark_text"

        status = "OK" if point is not None else "NO_POINT"

        out_path = OUTPUT_DIR / f"{page_name}_{task_slug}_{tag}.png"
        if point is not None:
            annotated = _draw_click(original_image, point, label=f"{tag}: {task[:20]}")
            annotated.save(out_path)
        else:
            original_image.save(out_path)

        pbar.write(f"  [{tag:<18}][{page_name:<8}] {task!r:<20} ({elapsed:4.1f}s) {status:<10} "
                   f"method={method} mark={resolved_mark} point={point}")

        csv_rows.append({
            "checkpoint": tag,
            "page": page_name,
            "task": task,
            "status": status,
            "method": method,
            "mark": resolved_mark,
            "point_x": point[0] if point else None,
            "point_y": point[1] if point else None,
            "elapsed_s": round(elapsed, 2),
            "raw_response": response,
            "image": str(out_path),
        })

    csv_path = OUTPUT_DIR / "bisect_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print(f"Done. {len(csv_rows)} results saved.")
    print(f"  Annotated images : {OUTPUT_DIR}/{{page}}_{{task}}_{{checkpoint}}.png")
    print(f"  SoM references   : {OUTPUT_DIR}/{{page}}_som_reference.png")
    print(f"  Summary CSV      : {csv_path}")
    print("=" * 70)
    print("\nWatch specifically for the same-mark-regardless-of-task pattern "
          "seen with lora_adapter_v2 tonight -- if two different tasks on the "
          "same page still resolve to the same mark/point even with verbatim "
          "phrasing, that rules out phrasing as the explanation.")


if __name__ == "__main__":
    main()