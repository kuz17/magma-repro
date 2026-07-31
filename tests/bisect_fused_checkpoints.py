# tests/bisect_fused_checkpoints.py
"""
Visual-inspection bisection across lora_adapter_fused checkpoints.

Loads checkpoints ONE AT A TIME, not stacked -- after each checkpoint's
full test grid finishes, its adapter is explicitly deleted and VRAM freed
before the next one loads. The base Qwen model and OmniParser are loaded
ONCE and stay loaded across the whole sweep (avoiding the ~60-100s reload
cost x 11 checkpoints) -- only the adapter itself cycles.

For every (page, task, checkpoint) combination, saves an annotated
screenshot showing exactly where that checkpoint resolved its click to,
so you can flip through and visually judge which checkpoint is most
reliably correct.

OmniParser is run ONCE per page (its output doesn't depend on which
checkpoint is active) and the resulting SoM image / mark_to_center /
content_list are reused across every checkpoint and every task on that
page.

DOM-priority rebuild: if a page's screenshot has a matching
"{stem}_dom.json" sidecar (see tests/grab_bisect_screenshots.py), DOM
elements are folded in via the same _rebuild_som_dom_priority() mechanism
act_fused() uses live. Confirmed necessary -- OmniParser alone missed
several elements these tasks actually need (Cart icon, Add to Cart/Buy
Now buttons, format tabs never got marks in the first sweep). max_marks
raised to 50 (from the live pipeline's default 15) since these test pages
have many DOM candidates (nav bar, filters, buttons) competing for slots
alongside the specific elements being tested.

Setup:
    Screenshots + DOM sidecars expected at outputs/bisect_screenshots/
    (see tests/grab_bisect_screenshots.py).

Usage:
    python tests/bisect_fused_checkpoints.py
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
    FUSED_PROMPT_TEMPLATE,
    format_fused_history,
    parse_fused_action,
    _draw_click,
)

# ══════════════════════════════════════════════════════════════════════════════
# Checkpoints — loaded and tested ONE AT A TIME, never stacked.
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINTS = {
    "ckpt_400": "models/lora_adapter_fused_checkpoints/checkpoint-400",
    "ckpt_450": "models/lora_adapter_fused_checkpoints/checkpoint-450",
    "ckpt_500": "models/lora_adapter_fused_checkpoints/checkpoint-500",
    "ckpt_550": "models/lora_adapter_fused_checkpoints/checkpoint-550",
    "ckpt_600": "models/lora_adapter_fused_checkpoints/checkpoint-600",
    "ckpt_650": "models/lora_adapter_fused_checkpoints/checkpoint-650",
    "ckpt_700": "models/lora_adapter_fused_checkpoints/checkpoint-700",
    "ckpt_750": "models/lora_adapter_fused_checkpoints/checkpoint-750",
    "ckpt_800": "models/lora_adapter_fused_checkpoints/checkpoint-800",
    "ckpt_825": "models/lora_adapter_fused_checkpoints/checkpoint-825",  # final training step
    "final":    "models/lora_adapter_fused",  # separately-saved copy — compare against ckpt_825
}

ADAPTER_NAME = "current"  # single reused adapter slot — swapped, never stacked
DOM_REBUILD_MAX_MARKS = 50  # raised from the live pipeline's default 15 — see module docstring

# ══════════════════════════════════════════════════════════════════════════════
# Pages and tasks — 3 pages x 3 tasks each
# ══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "home": {
        "image": "outputs/bisect_screenshots/amazon_home.png",
        "tasks": [
            "click on Cart",
            "click on Returns and Orders",
            "click on Amazon.in",
        ],
    },
    "results": {
        "image": "outputs/bisect_screenshots/search_results_odyssey.png",
        "tasks": [
            "click on Add to cart",
            "click on the Odyssey book cover",
            "click on Amazon.in",
        ],
    },
    "product": {
        "image": "outputs/bisect_screenshots/product_detail_odyssey.png",
        "tasks": [
            "click on Add to cart",
            "click on Buy now",
            "click on Hardcover",
        ],
    },
}

OUTPUT_DIR = Path("outputs/bisect_fused")


def _task_slug(task: str) -> str:
    return task.lower().replace(" ", "_").replace(".", "")[:40]


def main():
    ckpt_items = list(CHECKPOINTS.items())
    first_tag, first_path = ckpt_items[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base Qwen + OmniParser + first checkpoint ({first_tag})...")
    runner = DemoRunner(
        lora_path=first_path,
        training_style=True,
        tag="bisect",
    )
    # Rename the initial "default" adapter slot to our reused slot name so
    # every checkpoint (including the first) goes through the same
    # delete/load/set cycle below, with no special-casing.
    runner._qwen.load_adapter(first_path, adapter_name=ADAPTER_NAME)
    runner._qwen.delete_adapter("default")
    runner._qwen.set_adapter(ADAPTER_NAME)

    # Pre-load OmniParser output for every page ONCE, up front — reused
    # across every checkpoint below regardless of load/unload cycling.
    print("\nPre-running OmniParser for all pages (once each)...")
    page_omni_cache = {}
    for page_name, page_cfg in tqdm(PAGES.items(), desc="OmniParser pre-pass"):
        image_path = Path(page_cfg["image"])
        image = Image.open(image_path).convert("RGB")

        som_image, mark_to_center, content_list = runner._run_omniparser(image)

        # Apply DOM-priority rebuild if a sidecar JSON exists for this
        # screenshot — OmniParser alone misses several elements these
        # tasks need (Cart icon, Add to Cart/Buy Now buttons, format
        # tabs). Same mechanism act_fused() uses live; here it's driven
        # from a saved DOM snapshot instead of a live browser.
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
        page_omni_cache[page_name] = {
            "original_image": image,
            "som_image": som_image,
            "mark_to_center": mark_to_center,
            "content_list": content_list,
        }
    print(f"Done. SoM references saved under {OUTPUT_DIR}/\n")
    print("⚠ Check the *_som_reference.png files now, before the full sweep runs —")
    print("  confirm every task's target element actually has a mark this time.\n")

    # Flatten (checkpoint, page, task) into one list so a single tqdm bar
    # can track overall progress/ETA across the whole sweep, while still
    # driving the delete/load/set adapter-swap only when the checkpoint
    # actually changes between consecutive items.
    plan = []
    for tag, path in ckpt_items:
        for page_name, page_cfg in PAGES.items():
            for task in page_cfg["tasks"]:
                plan.append((tag, path, page_name, task))

    print(f"Running {len(ckpt_items)} checkpoints x {len(PAGES)} pages x 3 tasks "
          f"= {len(plan)} fused calls (checkpoints loaded one at a time).\n")

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

        cache = page_omni_cache[page_name]
        original_image = cache["original_image"]
        som_image      = cache["som_image"]
        mark_to_center = cache["mark_to_center"]
        content_list   = cache["content_list"]

        task_slug = _task_slug(task)
        prompt = FUSED_PROMPT_TEMPLATE.format(task=task, history=format_fused_history([]))

        t0 = time.time()
        response = runner._run_qwen(som_image, prompt, max_new_tokens=60)
        elapsed = time.time() - t0

        parsed = parse_fused_action(response)
        point = None
        mark = None
        label = ""
        if parsed and parsed.get("mark") is not None:
            mark = parsed["mark"]
            point = mark_to_center.get(mark)
            if mark < len(content_list):
                label = str(content_list[mark].get("content") or "")

        status = "OK" if point is not None else ("NO_MARK" if parsed else "PARSE_FAIL")

        out_path = OUTPUT_DIR / f"{page_name}_{task_slug}_{tag}.png"
        if point is not None:
            annotated = _draw_click(original_image, point, label=f"{tag}: mark {mark}")
            annotated.save(out_path)
        else:
            original_image.save(out_path)

        pbar.write(f"  [{tag:<10}][{page_name:<8}] {task:<32} ({elapsed:4.1f}s) {status:<10} "
                   f"action={parsed.get('action') if parsed else None} "
                   f"mark={mark} label={label!r}")

        csv_rows.append({
            "checkpoint": tag,
            "page": page_name,
            "task": task,
            "status": status,
            "action": parsed.get("action") if parsed else None,
            "mark": mark,
            "mark_label": label,
            "point_x": point[0] if point else None,
            "point_y": point[1] if point else None,
            "elapsed_s": round(elapsed, 2),
            "raw_response": response,
            "image": str(out_path),
        })

    # ── save summary CSV ────────────────────────────────────────────────
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
    print("\nOpen the CSV to scan status/action/mark quickly, then open the")
    print("annotated PNGs for whichever rows look promising to confirm visually.")


if __name__ == "__main__":
    main()