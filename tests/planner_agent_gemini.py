# tests/planner_agent_gemini.py
"""
Planner+executor loop, split-model variant using Google's free Gemini API
for planning instead of a paid model:

Planner  -> Gemini API (gemini-2.5-flash), free tier, no credit card needed
Executor -> POST /act on your existing inference server (Qwen + LoRA +
            OmniParser, local or Kaggle+ngrok, unchanged from before)

CLICK phrasing: confirmed via train_smoke.jsonl inspection that this
checkpoint's training data is literal text-localization, not intent-based
grounding -- every training instruction is the VERBATIM on-page
text/caption of the target element itself (e.g. "To execute the step
\"8 044 773 37 49\", where do I direct my attention?"), never a
paraphrased description or task-style phrase. This explains the whole
night's coordinate-bias pattern better than a brevity theory alone did:
long/paraphrased CLICK targets (full product titles, "the book cover")
are asking the model to do a task format it was never trained on at all,
and icon-only elements with no visible text have nothing for the model's
text-localization mechanism to anchor to. Prompt instructs Gemini to
phrase CLICK arguments as the exact visible text on the element, not a
short-but-still-paraphrased description.

DONE verification: DONE previously fired on Gemini's own narrated intent
("I clicked add to cart, so it's done") rather than checking the actual
resulting screenshot -- confirmed live: a run declared DONE while the
cart icon still read 0 and the URL had never left search results. Prompt
requires DONE to cite specific visual evidence in the CURRENT screenshot.

Sign-in wall (2026-08-09): confirmed live -- "Buy Now" (as opposed to
"Add to Cart") routes through Amazon's actual auth flow
(amazon.in/ap/signin?...), which the agent obviously can't and shouldn't
try to click through. Without an explicit rule for this, the run just
idled out at MAX_STEPS with an ambiguous CLICK("Continue") attempt on the
signin page. Added an explicit instruction to recognize the URL pattern
and respond DONE with a clear reason instead, giving a clean stopping
signal in the transcript rather than a silent MAX_STEPS timeout.

Free tier limits (subject to change, check https://ai.google.dev for current
numbers): roughly 15 requests/minute and up to 1,000 requests/day for
gemini-2.5-flash.

Requires: pip install google-genai python-dotenv
          Get a free key from https://aistudio.google.com/apikey (no card)
          Put it in .env as GEMINI_API_KEY=AIza...

Usage:
    python tests/planner_agent_gemini.py --goal "add a book to cart"
    python tests/planner_agent_gemini.py --goal "add a book to cart" \
        --server-url https://xxxx.ngrok-free.dev
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time

sys.path.insert(0, ".")

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory if present

from google import genai
from google.genai import types

from src.agent.browser_env import BrowserEnv

# ══════════════════════════════════════════════════════════════════════════════
# Grounding server location (unchanged from planner_agent.py)
# ══════════════════════════════════════════════════════════════════════════════
SERVER_URL = "http://127.0.0.1:8787"
# SERVER_URL = "https://plop-isolated-blinked.ngrok-free.dev"

# ══════════════════════════════════════════════════════════════════════════════
# Planner model — gemini-2.5-flash: stable, well-provisioned, confirmed
# working across multiple runs tonight. gemini-3.6-flash hit its 20/20
# daily RPD cap; gemini-3.5-flash-lite produced a 141.8s stall then an
# indefinite hang (transient API-side flakiness, not config); a later
# untested swap to gemini-3.1-flash-lite was reverted back here since it
# was never validated against a real run.
# ══════════════════════════════════════════════════════════════════════════════
PLANNER_MODEL = "gemini-3.5-flash-lite"

ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time on a shopping website. You are shown the CURRENT screenshot. Decide the single next action needed to make progress toward the goal.

The ONLY four valid function calls are exactly these — never invent other names:
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Your entire response must be ONE of these four calls and NOTHING else — no narration, no extra sentences.

Reasoning steps, in order:
1. Identify the page type: search results (grid of small thumbnails), product detail (one large image + Add to Cart button), or cart page (URL contains "cart").
2. "Add to Cart" only exists on a product detail page — if on search results, CLICK a specific product first.
3. Never repeat the exact action you just took if the screenshot still looks like the same page.
4. If the current URL contains "signin", "ap/signin", or the screenshot shows a sign-in / login page, respond DONE("reached a sign-in page - cannot proceed without account credentials") immediately. Do not attempt to click "Continue", enter credentials, or otherwise navigate a login flow.
5. Respond DONE ONLY if you can point to explicit visual evidence in the CURRENT screenshot that the goal is achieved — the cart icon showing a non-zero count, a green "Added to Cart" confirmation banner, or being on the actual cart page with the item listed. Do NOT respond DONE just because you attempted a click that seemed like it should have worked — a click may have missed. Verify the result is actually visible before declaring DONE. (The sign-in rule above is the one exception: DONE there regardless of whether the shopping goal itself was reached.)
6. Phrase the CLICK argument as the EXACT, VERBATIM text visible on the target element — copy it character-for-character from what's written on the button/link in the screenshot (e.g. "Add to Cart", "Sign in", "Buy Now"). Do NOT paraphrase, describe, summarize, or shorten it into your own words, and never use a full product title or sentence — only the literal text string printed on the element itself. If the element you want has no visible text at all (e.g. it's only an image or icon with no label), pick a different, text-labeled path toward the goal instead if one exists.

Goal: {goal}

Steps already taken:
{history}

Respond with ONLY one function call:"""

MAX_STEPS = 6


# ══════════════════════════════════════════════════════════════════════════════
# Planner — Gemini API
# ══════════════════════════════════════════════════════════════════════════════

class GeminiPlanner:
    def __init__(self, model: str = PLANNER_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key (no card required) from "
                "https://aistudio.google.com/apikey and put it in .env as "
                "GEMINI_API_KEY=AIza..."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def plan(self, png_bytes: bytes, goal: str, history: list) -> str:
        hist_str = "\n".join(f"- {h}" for h in history) if history else "(none - this is the first step)"
        prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    # 30s, in ms -- fail fast rather than hang indefinitely.
                    http_options=types.HttpOptions(timeout=30_000),
                ),
            )
            return (response.text or "").strip()
        except Exception as exc:
            print(f"  ✗ Gemini planner error: {exc}")
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# Executor — unchanged HTTP client for /act
# ══════════════════════════════════════════════════════════════════════════════

class GroundingClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            print(f"  [grounding server] mode={data.get('mode')}  status={data.get('status')}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach grounding server at {self.base_url}. Is it running?\n  Error: {exc}"
            )

    def act(self, png_bytes: bytes, task: str, dom_elements: list | None = None) -> dict:
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(f"{self.base_url}/act", json=payload, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "mark_id": None, "raw_response": "",
                    "elements": [], "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"click_norm": None, "mark_id": None, "raw_response": "",
                    "elements": [], "error": str(exc)}


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def ground_click(client: GroundingClient, browser: BrowserEnv, task_desc: str):
    ss = browser.screenshot(wait_stable=True)
    dom_elements = browser.get_interactive_elements()
    result = client.act(ss.png_bytes, task_desc, dom_elements=dom_elements)
    print("    grounding response: " + repr(result.get("raw_response", "")))
    if result.get("error"):
        print(f"    ✗ grounding server error: {result['error']}")
        return None
    click_norm = result.get("click_norm")
    return tuple(click_norm) if click_norm else None


def run_agent(goal: str, start_url: str, server_url: str):
    planner = GeminiPlanner()
    executor = GroundingClient(server_url)

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history = []

        for step in range(1, MAX_STEPS + 1):
            print("\n" + "=" * 60)
            print("STEP " + str(step))
            print("=" * 60)

            ss = browser.screenshot(wait_stable=True)

            print("  Planning next action (Gemini)...")
            t0 = time.time()
            response = planner.plan(ss.png_bytes, goal, history)
            action, arg = parse_action(response)
            print("  (%.1fs) planner said: %r" % (time.time() - t0, response))

            if action is None:
                print("  FAILED to parse action from: %r - stopping." % response)
                break

            print("  -> %s(%r)" % (action, arg))

            if action == "DONE":
                print("\nDONE: " + str(arg))
                break

            elif action == "SEARCH":
                elems = browser.get_interactive_elements()
                input_elem = next(
                    (e for e in elems if e["tag"] in ("input", "textarea")
                     and e["type"] not in ("submit", "button", "checkbox", "radio", "image", "hidden")),
                    None,
                )
                if input_elem is None:
                    print("  no search input found - stopping.")
                    break
                submit_elem = next(
                    (e for e in elems if e["tag"] == "button" or e.get("type") in ("submit", "button")),
                    None,
                )
                bbox = input_elem["bbox_norm"]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                browser.click(cx, cy)
                time.sleep(0.15)
                browser.press_key("Control+a")
                browser.type_text(arg)
                time.sleep(0.15)
                browser.press_key("Escape")
                time.sleep(0.1)
                if submit_elem is not None:
                    sb = submit_elem["bbox_norm"]
                    browser.click((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2)
                else:
                    browser.press_key("Enter")
                try:
                    browser.wait_for_load(timeout_ms=10000)
                except Exception:
                    pass
                history.append('Searched for "%s"; results are now showing.' % arg)

            elif action == "CLICK":
                print("  Grounding click target: %r" % arg)
                t0 = time.time()
                point = ground_click(executor, browser, arg)
                print("  (%.1fs) grounding done" % (time.time() - t0))
                if point is None:
                    print("  grounding failed to find a point for %r - stopping." % arg)
                    break
                browser.click(point[0], point[1], verify_change=False)
                try:
                    browser.wait_for_load(timeout_ms=5000)
                except Exception:
                    pass
                history.append('Clicked on "%s".' % arg)

            elif action == "SCROLL":
                direction = arg.strip().lower() if arg else "down"
                browser.scroll(delta_y=400 if direction == "down" else -400)
                history.append('Scrolled %s.' % direction)

            print("  url now: " + browser.current_url())

        else:
            print("\nHit MAX_STEPS=%d without DONE - stopping." % MAX_STEPS)

        # Keep the browser open so you can visually verify the final state --
        # especially worth it since DONE is only Gemini's own claim, not a
        # confirmed check against the actual page (a prior run fired DONE
        # while the URL had never left search results, meaning the item was
        # very likely never actually added to cart).
        print("\nBrowser left open for inspection. Press Enter to close it.")
        input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", default="https://www.amazon.in")
    parser.add_argument("--server-url", default=None,
                        help="Grounding server URL — overrides SERVER_URL constant above")
    args = parser.parse_args()

    server_url = args.server_url if args.server_url else SERVER_URL
    run_agent(args.goal, args.url, server_url)