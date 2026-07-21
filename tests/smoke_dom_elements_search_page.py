# tests/smoke_dom_elements_search_page.py
"""
Dumps BrowserEnv.get_interactive_elements() on a real Amazon search-results
page, with no model involved, to answer one question directly:

    Do product-title <a> links even show up in the DOM element list, and
    where would they rank once _rebuild_som_dom_priority()'s _dom_order()
    and the max_marks=15 cap are applied?

This does NOT load Qwen or OmniParser - it's pure Playwright + the existing
BrowserEnv.get_interactive_elements() extraction, so it's fast (~seconds,
not minutes) and safe to run repeatedly while iterating on the DOM-priority
question before touching the planner prompt or _rebuild_som_dom_priority().

Usage:
    python -u tests/smoke_dom_elements_search_page.py
    python -u tests/smoke_dom_elements_search_page.py --query "diary of a wimpy kid"
    python -u tests/smoke_dom_elements_search_page.py --url "https://www.amazon.in/s?k=diary+of+a+wimpy+kid"
"""
import argparse
import sys
sys.path.insert(0, ".")

from src.agent.browser_env import BrowserEnv

# Mirrors _dom_order() in src/agent/click_visualizer.py exactly, so the
# ranking shown here matches what _rebuild_som_dom_priority() would actually
# do - if this script disagrees with that function later, one of the two
# has drifted and needs fixing to match the other.
def _dom_order(e):
    tag = e.get("tag", "")
    typ = e.get("type", "")
    if tag in ("input", "textarea") and typ not in ("submit", "button", "checkbox", "radio", "image"):
        return 0  # text inputs first
    if tag == "button" or typ in ("submit", "button"):
        return 1  # submit buttons second
    return 2      # everything else - including product <a> links


def looks_like_product_link(e) -> bool:
    """
    Heuristic only, for this smoke test's own reporting - not used by the
    real pipeline. Flags <a> elements whose visible text is long enough to
    plausibly be a product title, so they're easy to spot in the dump
    below without reading all ~80+ links by eye.
    """
    return e.get("tag") == "a" and len(e.get("label", "")) > 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None,
                        help="Full URL to a search-results page. "
                             "If omitted, --query is used against amazon.in.")
    parser.add_argument("--query", default="diary of a wimpy kid",
                        help="Search query, used to build the amazon.in "
                             "search URL if --url is not given.")
    parser.add_argument("--max-marks", type=int, default=15,
                        help="Same cap as _rebuild_som_dom_priority()'s "
                             "max_marks default - kept as a flag so you can "
                             "test what raising it would change.")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run headless. Default: visible, since seeing "
                             "the actual page while reading the dump helps.")
    args = parser.parse_args()

    if args.url:
        url = args.url
    else:
        q = args.query.replace(" ", "+")
        url = f"https://www.amazon.in/s?k={q}"

    print(f"\n{'─'*70}")
    print(f"  DOM element dump — no model loaded")
    print(f"  url: {url}")
    print(f"{'─'*70}\n")

    with BrowserEnv(headless=args.headless) as browser:
        # Amazon keeps background network activity going more or less
        # indefinitely (ads, analytics beacons, polling), so the default
        # wait_until="networkidle" in BrowserEnv.navigate() reliably times
        # out at 30s on this site. Use "domcontentloaded" for the initial
        # nav (fires as soon as the page is parseable), then do a bounded,
        # tolerant networkidle wait on top - same pattern already used
        # elsewhere in this repo (WebAgent.back(), the planner's SEARCH
        # step) to handle exactly this Amazon-specific behavior.
        browser.navigate(url, wait="domcontentloaded")
        try:
            browser.wait_for_load("networkidle", timeout_ms=8_000)
        except Exception:
            pass  # fine - page is already usable after domcontentloaded
        elems = browser.get_interactive_elements()

        print(f"Total interactive elements found: {len(elems)}\n")

        # Sort exactly as _rebuild_som_dom_priority() would (stable sort,
        # so original DOM order is preserved within each priority tier)
        ranked = sorted(elems, key=_dom_order)

        print(f"{'idx':>4}  {'rank':>4}  {'tag':<10}  {'type':<10}  {'product?':<9}  label")
        print(f"{'-'*4}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*40}")
        for i, e in enumerate(ranked):
            rank = _dom_order(e)
            is_product = "YES" if looks_like_product_link(e) else ""
            cutoff_marker = "  <-- max_marks cutoff" if i == args.max_marks else ""
            label = e.get("label", "")[:60]
            print(f"{i:>4}  {rank:>4}  {e.get('tag',''):<10}  {e.get('type',''):<10}  "
                  f"{is_product:<9}  {label}{cutoff_marker}")

        # Summary: how many plausible product links exist, and how many
        # survive under the current max_marks cap
        product_links = [e for e in ranked if looks_like_product_link(e)]
        product_links_within_cap = [
            e for i, e in enumerate(ranked) if i < args.max_marks and looks_like_product_link(e)
        ]

        print(f"\n{'─'*70}")
        print(f"Summary:")
        print(f"  total elements              : {len(elems)}")
        print(f"  plausible product-title <a> : {len(product_links)}")
        print(f"  ...surviving max_marks={args.max_marks} cap : {len(product_links_within_cap)}")
        if product_links and not product_links_within_cap:
            print(f"\n  ⚠ Product links exist in the DOM but ALL of them are ranked")
            print(f"    below the max_marks cutoff - if the planner named a product")
            print(f"    by title text, _rebuild_som_dom_priority() would currently")
            print(f"    drop it before the grounding model ever sees it as a mark.")
        elif product_links_within_cap:
            print(f"\n  ✓ At least one plausible product link would survive the cap")
            print(f"    and be available as a mark for the grounding model.")
        else:
            print(f"\n  ⚠ No plausible product-title <a> elements were found at all —")
            print(f"    product titles on this page may not be exposed as <a> tags")
            print(f"    in a way get_interactive_elements()'s selector catches, or")
            print(f"    the heuristic in this script is too strict.")
        print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()