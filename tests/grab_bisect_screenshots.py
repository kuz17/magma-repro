# tests/grab_bisect_screenshots.py
"""
Grab a fixed set of real screenshots + DOM elements for
tests/bisect_fused_checkpoints.py.

DOM elements are saved alongside each screenshot (as a JSON sidecar) so
the bisection script can apply the same DOM-priority SoM rebuild
act_fused() uses live -- OmniParser alone misses several of the elements
these test tasks actually need (confirmed visually: no mark was placed on
the Cart icon, Add to Cart/Buy Now buttons, or the Hardcover format tab
in the initial sweep's SoM references).
"""
import json
import os
import sys

sys.path.insert(0, ".")
from src.agent.browser_env import BrowserEnv

OUT_DIR = "outputs/bisect_screenshots"
PRODUCT_URL = "https://www.amazon.in/Odyssey-Homer/dp/9357943412/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.eh4tq7RHlvYTNgB93qTwKbF7j0aACvV-auiYFJMrurbjqKy3EvxN4gFUwOgZ2AFT2yV8dRhbHNrTLqTUhyfqhdG8hTNXA1qCcFAlS9XfGZepScSTjsuDTC4SVAZd3qNjg4NAeVnjtNGtsih4eVkXq4rneCHkpM5l7CmdLmZCkkJAnXc7FWSCHmOXZVuKVcQ-57qidXzRZwnFz7PQ6E10q5-ROdusOpXTVEHNL2v6N2w.f2oVCV-W_Ev7R9NT4o3CEvPvQj1HZKLQTGcdExPoCMQ&dib_tag=se&keywords=odyssey&qid=1785372301&sr=8-1-spons&aref=FHYCqcauLj&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"

os.makedirs(OUT_DIR, exist_ok=True)


def _save(env, name):
    ss = env.screenshot(wait_stable=True)
    img_path = f"{OUT_DIR}/{name}.png"
    dom_path = f"{OUT_DIR}/{name}_dom.json"
    ss.image.save(img_path)

    dom_elements = env.get_interactive_elements()
    with open(dom_path, "w") as f:
        json.dump(dom_elements, f, indent=2)

    print(f"Saved: {img_path}  ({ss.image.width}x{ss.image.height})  "
          f"+ {len(dom_elements)} DOM elements -> {dom_path}")


with BrowserEnv(headless=True, save_screenshots=False) as env:
    env.navigate("https://www.amazon.in", wait="domcontentloaded")
    _save(env, "amazon_home")

    env.navigate("https://www.amazon.in/s?k=odyssey", wait="domcontentloaded")
    _save(env, "search_results_odyssey")

    env.navigate(PRODUCT_URL, wait="domcontentloaded")
    _save(env, "product_detail_odyssey")