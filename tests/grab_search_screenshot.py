# /tmp/grab_search_screenshot.py
import sys
sys.path.insert(0, ".")
from src.agent.browser_env import BrowserEnv

with BrowserEnv(headless=True, save_screenshots=False) as env:
    # domcontentloaded is much more reliable than networkidle for
    # pages with persistent background requests (ads/analytics)
    env.navigate("https://www.amazon.in/s?k=atomic+habits", wait="domcontentloaded")
    ss = env.screenshot(wait_stable=True)
    ss.image.save("outputs/browser/search_results_atomic_habits.png")
    print(f"Saved: outputs/browser/search_results_atomic_habits.png")
    print(f"Size: {ss.image.width}x{ss.image.height}")