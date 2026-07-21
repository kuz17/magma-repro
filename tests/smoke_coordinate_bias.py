# tests/smoke_coordinate_bias.py
"""
Runs ONE fixed grounding task against every screenshot in a folder, using
the fine-tuned adapter (same DemoRunner path planner_agent.py's ground_click
uses), and prints the raw parsed coordinate for each image.

Purpose: two separate runs (Wimpy Kid, Odyssey) both produced the exact
same raw coordinate — Coordinate: (0.62, 0.71) — with the exact same
nearest-mark distance (0.1630) to a real 'Add to cart' button, on visually
different pages. This script checks whether that's a coincidence of two
similar-looking pages, or a real fixed-point bias in the adapter's
coordinate output for this instruction phrasing, by running the same task
across several genuinely different screenshots in one pass.

Does NOT touch click_visualizer.py or any pipeline logic - this only calls
the existing DemoRunner.act() the same way ground_click() in
planner_agent.py already does, once per image, with no DOM elements (so
each run is a clean read on the adapter + OmniParser, not affected by DOM
priority re-ranking).

Usage:
    python -u tests/smoke_coordinate_bias.py --dir outputs/browser
    python -u tests/smoke_coordinate_bias.py --dir outputs/browser --task "the yellow Add to cart button"
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.agent.click_visualizer import DemoRunner, _parse_point

DEFAULT_TASK = "the yellow Add to cart button"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Folder containing screenshots (e.g. outputs/browser)")
    parser.add_argument("--pattern", default="ss_*.png",
                        help="Glob pattern for screenshots within --dir")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Fixed grounding task string, applied to every image")
    parser.add_argument("--lora", default="models/lora_adapter_smoke",
                        help="LoRA adapter path")
    args = parser.parse_args()

    folder = Path(args.dir)
    images = sorted(folder.glob(args.pattern))
    if not images:
        print(f"No images matching {args.pattern!r} found in {folder}")
        return

    print(f"\n{'─'*70}")
    print(f"  Coordinate-bias check")
    print(f"  task  : {args.task!r}")
    print(f"  images: {len(images)} found in {folder}")
    print(f"{'─'*70}\n")

    print("Loading model (fine-tuned adapter, training-style)...")
    runner = DemoRunner(lora_path=args.lora, tag="coord_bias", training_style=True)

    results = []
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"  {img_path.name}")
        print(f"{'='*60}")
        response, point = runner.act(str(img_path), args.task, dom_elements=None)
        print(f"  raw response : {response!r}")
        print(f"  resolved pt  : {point}")

        # Independently re-parse the raw coordinate straight from the
        # response (bypassing nearest-mark snapping), so we see exactly
        # what the model said before any resolution logic touched it -
        # this is the number we actually care about for the bias question.
        raw = _parse_point(response, image_size=(1, 1))
        results.append((img_path.name, raw, point))

    print(f"\n{'─'*70}")
    print(f"  Summary — raw coordinate per image")
    print(f"{'─'*70}")
    print(f"  {'image':<20}  {'raw_coord':<20}  resolved_point")
    print(f"  {'-'*20}  {'-'*20}  {'-'*20}")
    for name, raw, point in results:
        raw_str = f"({raw[0]:.3f}, {raw[1]:.3f})" if raw else "None"
        pt_str = f"({point[0]:.3f}, {point[1]:.3f})" if point else "None"
        print(f"  {name:<20}  {raw_str:<20}  {pt_str}")

    # Cheap clustering check: how far apart are the raw coordinates from
    # each other? If they're all within a tight radius of each other
    # despite genuinely different page content, that's the fixed-point-
    # bias signature; if they're spread out and each roughly matches its
    # own page's actual button position, it's page-layout coincidence.
    raw_pts = [r for _, r, _ in results if r is not None]
    if len(raw_pts) >= 2:
        xs = [p[0] for p in raw_pts]
        ys = [p[1] for p in raw_pts]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        print(f"\n  raw-coordinate spread: Δx={spread_x:.4f}  Δy={spread_y:.4f}")
        if spread_x < 0.03 and spread_y < 0.03:
            print("  ⚠ Raw coordinates are nearly identical across different pages —")
            print("    consistent with a fixed-point bias in the adapter's output for")
            print("    this instruction phrasing, not a per-page coincidence.")
        else:
            print("  Raw coordinates vary meaningfully across pages — does not look")
            print("  like a fixed-point collapse; check per-image distance-to-true-button")
            print("  instead (page-layout coincidence is more likely).")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()