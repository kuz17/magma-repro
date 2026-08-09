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
try to click through. Added an explicit rule to recognize the URL
pattern and respond DONE with a clear reason instead of idling out at
MAX_STEPS.

Self-evaluation + URL-as-text (2026-08-09): ported two patterns from
browser-use/browser-use after reading their system_prompt.md structure:

  1. Their per-step history format is "Evaluation of Previous Step /
     Memory / Next Goal / Action Results" -- the MODEL itself judges
     whether its last action actually worked, looking at the fresh
     screenshot, rather than the calling code just asserting success
     unconditionally. Our old history.append('...; results are now
     showing.') fired regardless of whether anything actually happened --
     confirmed this fed misleading context back into planning after the
     YouTube search-submit bug (history claimed success on a SEARCH that
     never actually navigated). Now the model evaluates its own previous
     action each turn, using the current screenshot as ground truth, and
     that evaluation becomes part of the history instead of a hardcoded
     success string.

  2. browser-use's step input explicitly includes "Current URL, open
     tabs" as text alongside the screenshot, not just the image --
     matches the current_url fix already validated in
     tests/planner_agent.py (the base-model planner kept repeat-SEARCHing
     even when the screenshot visibly showed results, because inferring
     page-type from pixels alone was too hard a sub-task -- reading a URL
     as text was much easier). Ported the same idea here; the hint text
     is written more generically than the Amazon-specific version so it
     doesn't silently assume every site's URLs look like Amazon's.

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

# Matches "ACTION: SEARCH(...)" or bare "SEARCH(...)" anywhere in the
# response -- still fine with the EVALUATION: line ahead of it, since
# re.search just finds the first action-shaped match, which is the
# ACTION: line as long as EVALUATION's own text doesn't happen to look
# like a function call (rare enough not to guard against explicitly).
ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

EVALUATION_RE = re.compile(r'EVALUATION:\s*(.+?)(?:\n|$)', re.IGNORECASE)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time. You are shown the CURRENT screenshot.

Current URL: {current_url}
(URLs often hint at page type as text, which is more reliable than trying to judge it from
the screenshot alone -- e.g. a URL containing a search query parameter usually means you're
on a results/listing page; a URL for one specific item (a product, video, article) usually
means you're on that item's detail page.)

First, evaluate whether your PREVIOUS action actually worked, based on what the CURRENT
screenshot actually shows (not what you intended or expected) -- one short sentence citing
specific visual evidence. If this is the first step, just say so.

Then decide the single next action needed to make progress toward the goal.

The ONLY four valid function calls are exactly these — never invent other names:
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Respond in EXACTLY this two-line format, nothing else:
EVALUATION: <one sentence assessment of the previous action, grounded in the current screenshot>
ACTION: <the function call>

Reasoning steps for choosing the ACTION, in order:
1. Identify the page type from the URL and screenshot together (results/listing page, item detail page, a confirmation/login/other special page).
2. Never repeat the exact action you just took if your own EVALUATION above concluded it didn't change anything.
3. If the current URL contains "signin", "ap/signin", or the screenshot shows a sign-in / login page, respond DONE("reached a sign-in page - cannot proceed without account credentials") immediately. Do not attempt to click "Continue", enter credentials, or otherwise navigate a login flow.
4. Respond DONE ONLY if you can point to explicit visual evidence in the CURRENT screenshot that the goal is achieved. Do NOT respond DONE just because you attempted an action that seemed like it should have worked — verify the result is actually visible. (The sign-in rule above is the one exception.)
5. Phrase the CLICK argument as the EXACT, VERBATIM text visible on the target element — copy it character-for-character from what's written on the button/link/title in the screenshot. Do NOT paraphrase, describe, summarize, or shorten it into your own words, and never use a full title or sentence — only the literal text string printed on the element itself, or the first few words of it if it's long. If the element you want has no visible text at all (e.g. it's only an image or icon with no label), pick a different, text-labeled path toward the goal instead if one exists.

Goal: {goal}

Steps already taken:
{history}

Respond now with the two-line EVALUATION / ACTION format:"""

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

    def plan(self, png_bytes: bytes, goal: str, history: list[dict], current_url: str) -> str:
        hist_str = _format_history(history)
        prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str, current_url=current_url)

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


def _format_history(history: list[dict]) -> str:
    """
    Each entry: {"action_desc": "SEARCH('...')", "evaluation": str | None}
    "evaluation" is filled in one turn LATER, once the model has actually
    seen the resulting screenshot and self-assessed -- so the most recent
    entry usually still shows "(not yet evaluated)" until the next call.
    """
    if not history:
        return "(none - this is the first step)"
    lines = []
    for h in history:
        ev = h["evaluation"] or "(not yet evaluated)"
        lines.append(f'- {h["action_desc"]}  [result: {ev}]')
    return "\n".join(lines)


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

    def describe(self, png_bytes: bytes, question: str | None = None, use_adapter: bool = False) -> dict:
        """POST /describe — same server. use_adapter=True skips disable_adapter()
        server-side, testing the (expected-to-be-degenerate) adapter-enabled case."""
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "use_adapter": use_adapter}
        if question:
            payload["question"] = question
        try:
            r = requests.post(f"{self.base_url}/describe", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"response": "", "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"response": "", "error": str(exc)}


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def parse_evaluation(response: str) -> str | None:
    m = EVALUATION_RE.search(response)
    return m.group(1).strip() if m else None


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
        history: list[dict] = []

        for step in range(1, MAX_STEPS + 1):
            print("\n" + "=" * 60)
            print("STEP " + str(step))
            print("=" * 60)

            ss = browser.screenshot(wait_stable=True)
            current_url = browser.current_url()
            print("  current url: " + current_url)

            print("  Planning next action (Gemini)...")
            t0 = time.time()
            response = planner.plan(ss.png_bytes, goal, history, current_url)
            action, arg = parse_action(response)
            evaluation = parse_evaluation(response)
            print("  (%.1fs) planner said: %r" % (time.time() - t0, response))

            # Fold the model's own evaluation of its PREVIOUS action into
            # the last history entry, now that we finally have the
            # resulting screenshot to judge it against.
            if history and evaluation:
                history[-1]["evaluation"] = evaluation
                print(f"  [self-eval of prior step] {evaluation}")

            if action is None:
                print("  FAILED to parse action from: %r - stopping." % response)
                break

            print("  -> %s(%r)" % (action, arg))

            if action == "DONE":
                print("\nDONE: " + str(arg))
                print("\n  Describing final page with the local VLM (/describe, ADAPTER ENABLED)...")
                print("  (expect possibly degenerate/coordinate-shaped output -- adapter was")
                print("   never trained on captioning, only click-grounding; testing it anyway)")
                # Take a FRESH screenshot rather than reusing `ss` from the top
                # of this step -- `ss` was captured before this step's action
                # ran, so it's stale by one action for CLICK/SEARCH steps.
                fresh_ss = browser.screenshot(wait_stable=True)
                t0 = time.time()
                desc_result = executor.describe(fresh_ss.png_bytes, use_adapter=True)
                print(f"  ({time.time() - t0:.1f}s) description done")
                if desc_result.get("error"):
                    print(f"  ✗ describe error: {desc_result['error']}")
                else:
                    print("\n  Page description (adapter ON):\n  " + desc_result.get("response", "").replace("\n", "\n  "))
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
                bbox = input_elem["bbox_norm"]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                browser.click(cx, cy)
                time.sleep(0.15)
                browser.press_key("Control+a")
                browser.type_text(arg)
                time.sleep(0.15)
                browser.press_key("Enter")
                try:
                    browser.wait_for_load(timeout_ms=10000)
                except Exception:
                    pass
                history.append({"action_desc": f'SEARCH("{arg}")', "evaluation": None})

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
                history.append({"action_desc": f'CLICK("{arg}")', "evaluation": None})

            elif action == "SCROLL":
                direction = arg.strip().lower() if arg else "down"
                browser.scroll(delta_y=400 if direction == "down" else -400)
                history.append({"action_desc": f'SCROLL("{direction}")', "evaluation": None})

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