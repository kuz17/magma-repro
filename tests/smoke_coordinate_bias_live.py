# tests/smoke_coordinate_bias_live.py
"""
Live-browser version of smoke_coordinate_bias.py.

The static-screenshot version (smoke_coordinate_bias.py) confirmed the raw
coordinate collapse is real: across 4 different queries, the model's raw
output for "the yellow Add to cart button" was locked near x=0.62-0.63 and
y mostly 0.71. But that run passed dom_elements=None, which meant the
DOM-priority boost that normally puts "Add to cart" into the mark set
(_rebuild_som_dom_priority, _dom_order ranking buttons at tier 1) never
ran - so the resolved_point column in that run was snapping the collapsed
raw coordinate to whatever vision-only OmniParser mark happened to be
nearby, not necessarily the real Add to cart button.

This script fixes that gap: it navigates a real BrowserEnv/Playwright
session to each query's live Amazon search page, calls
browser.get_interactive_elements() for the real DOM list (same call
ground_click() makes in planner_agent.py), and passes that into
runner.act(..., dom_elements=dom_elements) - the exact same call shape the
real planner loop uses. This is the correct way to test whether the
(already-confirmed) collapsed raw coordinate lands within snapping range
of the ACTUAL Add to cart mark, now that it's reliably present.

Usage:
    python -u tests/smoke_coordinate_bias_live.py
    python -u tests/smoke_coordinate_bias_live.py --task "the yellow Add to cart button"
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from src.agent.click_visualizer import DemoRunner, _parse_point
from src.agent.browser_env import BrowserEnv

DEFAULT_TASK = "the yellow Add to cart button"

# Same 4 queries used in the earlier static-screenshot run, so results are
# directly comparable to that run's raw-coordinate table.
QUERIES = [
    "diary of a wimpy kid",
    "atomic habits",
    "the best of homer",
    "odyssey book",
    
]


def _navigate_search(browser: BrowserEnv, query: str) -> None:
    """
    Same domcontentloaded + tolerant networkidle pattern used in
    smoke_dom_elements_search_page.py - Amazon's background network
    activity never truly idles, so the default networkidle wait times out.
    """
    q = query.replace(" ", "+")
    url = f"https://www.amazon.in/s?k={q}"
    browser.navigate(url, wait="domcontentloaded")
    try:
        browser.wait_for_load("networkidle", timeout_ms=8_000)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Fixed grounding task string, applied on every search page")
    parser.add_argument("--lora", default="models/lora_adapter_smoke",
                        help="LoRA adapter path")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run browser headless (default: visible)")
    args = parser.parse_args()

    print(f"\n{'─'*70}")
    print(f"  Coordinate-bias check — LIVE browser, real DOM elements")
    print(f"  task    : {args.task!r}")
    print(f"  queries : {QUERIES}")
    print(f"{'─'*70}\n")

    print("Loading model (fine-tuned adapter, training-style)...")
    runner = DemoRunner(lora_path=args.lora, tag="coord_bias_live", training_style=True)

    results = []

    with BrowserEnv(headless=args.headless, save_screenshots=True) as browser:
        for query in QUERIES:
            print(f"\n{'='*60}")
            print(f"  query: {query!r}")
            print(f"{'='*60}")

            _navigate_search(browser, query)
            ss = browser.screenshot(wait_stable=True)
            tmp_path = f"/tmp/coord_bias_live_{query.replace(' ', '_')}.png"
            ss.image.save(tmp_path)

            # Real DOM extraction - same call ground_click() makes.
            dom_elements = browser.get_interactive_elements()
            n_buttons = sum(
                1 for e in dom_elements
                if e["tag"] == "button" or e.get("type") in ("submit", "button")
            )
            print(f"  DOM elements: {len(dom_elements)} total, {n_buttons} button-tagged")

            response, point = runner.act(tmp_path, args.task, dom_elements=dom_elements)
            print(f"  raw response : {response!r}")
            print(f"  resolved pt  : {point}")

            raw = _parse_point(response, image_size=(1, 1))
            results.append((query, raw, point))

    print(f"\n{'─'*70}")
    print(f"  Summary — raw coordinate per query (LIVE, real DOM boost)")
    print(f"{'─'*70}")
    print(f"  {'query':<24}  {'raw_coord':<20}  resolved_point")
    print(f"  {'-'*24}  {'-'*20}  {'-'*20}")
    for query, raw, point in results:
        raw_str = f"({raw[0]:.3f}, {raw[1]:.3f})" if raw else "None"
        pt_str = f"({point[0]:.3f}, {point[1]:.3f})" if point else "None"
        print(f"  {query:<24}  {raw_str:<20}  {pt_str}")

    raw_pts = [r for _, r, _ in results if r is not None]
    if len(raw_pts) >= 2:
        xs = [p[0] for p in raw_pts]
        ys = [p[1] for p in raw_pts]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        print(f"\n  raw-coordinate spread: Δx={spread_x:.4f}  Δy={spread_y:.4f}")
        if spread_x < 0.03 and spread_y < 0.03:
            print("  ⚠ Raw coordinates are nearly identical across different live pages —")
            print("    confirms a fixed-point bias in the adapter's output for this")
            print("    instruction phrasing, independent of DOM-boost/mark availability.")
        else:
            print("  Raw coordinates vary across pages under live DOM conditions.")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()