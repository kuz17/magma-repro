# tests/planner_agent_gemini_fused.py
"""
Planner+executor loop: Gemini API for planning, FUSED adapter for grounding.

Planner  -> Gemini API (gemini-2.5-flash by default), free tier
Executor -> POST /act_fused on the inference server (fused adapter, single
            call decides ACTION + MARK + VALUE together)

Requires the inference server running in FUSED mode specifically:
    python -m src.agent.inference_server --mode fused \
        --fused-lora models/lora_adapter_fused --port 8787

This is NOT interchangeable with planner_agent_gemini.py, which calls
/act (smoke-adapter format). Sending /act-shaped prompts to a fused-mode
server (or vice versa) produces garbage -- confirmed live tonight: a
fused-mode server given an /act-style task returned "The instruction is
not clear enough to answer the question," because the loaded adapter was
never trained on that prompt/output format at all. Always check
`[grounding server] mode=...` on startup matches which script you're
running.

History tag vocabulary: fused_history entries use Mind2Web's semantic
element-type vocabulary ("link", "searchbox", "button", "combobox",
"textbox") via dom_element_to_mind2web_type(), not raw HTML tags --
confirmed important for correct mark resolution (see
click_visualizer.py's act_fused() docstring).

CLICK phrasing: the planning prompt instructs Gemini to phrase CLICK
targets as short, blunt commands (1-4 words) rather than full product
titles -- confirmed live that a full descriptive title ("Sherlock Holmes
Series Complete Collection 7 Books Set by Arthur Conan Doyle") caused the
grounding model to give a natural-language non-answer instead of a
JSON action, consistent with the grounding model being trained on short
human-style phrasings.

Requires: pip install google-genai python-dotenv
          Get a free key from https://aistudio.google.com/apikey (no card)
          Put it in .env as GEMINI_API_KEY=AIza...

Usage:
    python tests/planner_agent_gemini_fused.py --goal "add a book to cart"
    python tests/planner_agent_gemini_fused.py --goal "add a book to cart" \
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

load_dotenv()

from google import genai
from google.genai import types

from src.agent.browser_env import BrowserEnv
from src.agent.click_visualizer import dom_element_to_mind2web_type

# ══════════════════════════════════════════════════════════════════════════════
# Grounding server location
# ══════════════════════════════════════════════════════════════════════════════
SERVER_URL = "http://127.0.0.1:8787"
# SERVER_URL = "https://plop-isolated-blinked.ngrok-free.dev"

# ══════════════════════════════════════════════════════════════════════════════
# Planner model — gemini-2.5-flash: stable, showed available quota tonight.
# gemini-3.6-flash hit its 20/20 daily cap; gemini-3.5-flash-lite produced
# a 141.8s stall then an indefinite hang (transient API-side flakiness,
# not a config issue) -- 2.5-flash was the reliable choice tonight.
# ══════════════════════════════════════════════════════════════════════════════
PLANNER_MODEL = "gemini-3.1-flash-lite"

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
4. If the goal is already achieved (cart shows the item, or you're on the cart page with it listed), respond DONE with a short summary.
5. Phrase the CLICK argument the way a person would TYPE it as a short command — 1-4 words, blunt and direct (e.g. "sherlock holmes", "add to cart", "buy now"), NOT the full product title or a descriptive sentence. The grounding model was trained on short human phrasings and struggles with long, verbose descriptions.

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
                    http_options=types.HttpOptions(timeout=30_000),
                ),
            )
            return (response.text or "").strip()
        except Exception as exc:
            print(f"  ✗ Gemini planner error: {exc}")
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# Executor — /act_fused HTTP client
# ══════════════════════════════════════════════════════════════════════════════

class FusedGroundingClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            print(f"  [grounding server] mode={data.get('mode')}  status={data.get('status')}")
            if data.get("mode") != "fused":
                print(f"  ⚠ server mode is '{data.get('mode')}', not 'fused' — "
                      f"/act_fused will fail unless the fused adapter is loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach grounding server at {self.base_url}. Is it running?\n  Error: {exc}"
            )

    def act_fused(
        self,
        png_bytes: bytes,
        task: str,
        history: list[dict],
        dom_elements: list | None = None,
    ) -> dict:
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task, "history": history}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(f"{self.base_url}/act_fused", json=payload, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "action": None, "mark_id": None, "value": None,
                    "raw_response": "", "elements": [], "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"click_norm": None, "action": None, "mark_id": None, "value": None,
                    "raw_response": "", "elements": [], "error": str(exc)}


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def _dom_info_for_point(dom_elements: list, point: tuple) -> dict | None:
    """Find the DOM element (if any) whose bbox_norm contains the resolved point."""
    px, py = point
    for e in dom_elements:
        bbox = e.get("bbox_norm", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            return e
    return None


def _history_tag_for_action(action: str, dom_elements: list, point: tuple) -> str:
    """Resolve the Mind2Web-vocabulary tag for a fused_history entry."""
    dom_match = _dom_info_for_point(dom_elements, point)
    if dom_match is not None:
        return dom_element_to_mind2web_type(
            dom_match.get("tag", ""), dom_match.get("type", ""), dom_match.get("label", ""),
        )
    if action == "TYPE":
        return "textbox"
    if action == "SELECT":
        return "combobox"
    return "button"


def _label_for_mark(elements: list, mark_id: int | None) -> str:
    if mark_id is None:
        return "element"
    for e in elements:
        if e.get("id") == mark_id:
            return e.get("content") or e.get("type") or "element"
    return "element"


def ground_fused(
    client: FusedGroundingClient,
    browser: BrowserEnv,
    task_desc: str,
    fused_history: list[dict],
):
    """
    Single fused call for CLICK-type steps. Returns
    (parsed_action, point, elements, dom_elements).
    """
    ss = browser.screenshot(wait_stable=True)
    dom_elements = browser.get_interactive_elements()
    result = client.act_fused(ss.png_bytes, task_desc, history=fused_history, dom_elements=dom_elements)
    print("    fused response: " + repr(result.get("raw_response", "")))
    if result.get("error"):
        print(f"    ✗ fused server error: {result['error']}")
        return None, None, [], dom_elements

    parsed = None
    if result.get("action") is not None:
        parsed = {
            "action": result.get("action"),
            "mark": result.get("mark_id"),
            "value": result.get("value"),
        }
    click_norm = result.get("click_norm")
    point = tuple(click_norm) if click_norm else None
    elements = result.get("elements", [])
    return parsed, point, elements, dom_elements


def run_agent(goal: str, start_url: str, server_url: str):
    planner = GeminiPlanner()
    executor = FusedGroundingClient(server_url)

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history       = []   # plain strings, fed to Gemini's planning prompt
        fused_history = []   # {"tag","label","action","value"} dicts, fed to the fused adapter

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
                fused_history.append({
                    "tag": dom_element_to_mind2web_type(
                        input_elem.get("tag", "input"), input_elem.get("type", ""), input_elem.get("label", ""),
                    ),
                    "label": input_elem.get("label", "search box"),
                    "action": "TYPE",
                    "value": arg,
                })

            elif action == "CLICK":
                print("  Resolving via fused adapter: %r" % arg)
                t0 = time.time()
                parsed, point, elements, dom_elements = ground_fused(executor, browser, arg, fused_history)
                print("  (%.1fs) fused resolution done" % (time.time() - t0))

                if parsed is None or point is None:
                    print("  fused adapter failed to resolve %r - stopping." % arg)
                    break

                resolved_action = parsed["action"] or "CLICK"
                label = _label_for_mark(elements, parsed.get("mark"))
                history_tag = _history_tag_for_action(resolved_action, dom_elements, point)

                if resolved_action == "TYPE":
                    browser.click(point[0], point[1])
                    time.sleep(0.15)
                    browser.press_key("Control+a")
                    browser.type_text(parsed.get("value") or "")
                    time.sleep(0.1)
                else:
                    browser.click(point[0], point[1], verify_change=False)

                try:
                    browser.wait_for_load(timeout_ms=5000)
                except Exception:
                    pass

                history.append('Clicked on "%s" (%s, mark %s).' % (arg, resolved_action, parsed.get("mark")))
                fused_history.append({
                    "tag": history_tag,
                    "label": label,
                    "action": resolved_action,
                    "value": parsed.get("value"),
                })

            elif action == "SCROLL":
                direction = arg.strip().lower() if arg else "down"
                browser.scroll(delta_y=400 if direction == "down" else -400)
                history.append('Scrolled %s.' % direction)

            print("  url now: " + browser.current_url())

        else:
            print("\nHit MAX_STEPS=%d without DONE - stopping." % MAX_STEPS)

        # Keep the browser open so you can visually verify the final state --
        # DONE is only Gemini's own claim, not a confirmed check against the
        # actual page.
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