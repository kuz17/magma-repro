# tests/planner_agent_fused.py
"""
Planner+executor loop using the FUSED adapter for grounding.

Planner step  -> POST /plan        (base model, adapter disabled server-side)
Executor step -> POST /act_fused   (fused adapter, single call decides
                                     ACTION + MARK + VALUE together —
                                     replaces the separate OmniParser-render
                                     + smoke-adapter-grounding round trip
                                     that /act makes)

Server-side, only ONE adapter is loaded (the fused one) — no dual-adapter
loading, no set_adapter()/load_adapter() switching at runtime. The only
toggle that happens is disable_adapter() for the /plan call, which is the
same mechanism already validated safe in smoke_disable_adapter.py — just
applied to the fused adapter instead of the smoke adapter (re-confirmed
directly against lora_adapter_fused — with/disabled/with-again outputs
compared and matched cleanly, so the planner's occasional invented-action
outputs are a prompt-drift issue, not adapter leakage).

SEARCH and SCROLL still aren't in Mind2Web's action vocabulary, so the
base-model planner still decides those and they're still executed
DOM-direct (same as planner_agent.py) — the fused adapter only replaces
grounding for CLICK-type steps, where it may itself resolve to CLICK,
TYPE, or SELECT once it sees the actual page.

Live DOM elements are passed into /act_fused (same DOM-priority SoM
rebuild /act uses) — confirmed necessary after a real run resolved a
click to an unrelated nav element ("Mobiles" in the top bar) instead of
the intended product listing.

History tag vocabulary fix: fused_history entries now use Mind2Web's
semantic element-type vocabulary ("link", "searchbox", "button",
"combobox", "textbox") via dom_element_to_mind2web_type(), instead of the
generic literal "element" this script used previously. Confirmed via a
real training example (MagmaAI/Magma-Mind2Web-SoM row) that every
checkpoint (400-825 steps) correctly resolves the right mark when history
uses this vocabulary, but live Amazon runs showed task-blind collapse to
page-fixed marks — the mismatched history vocabulary is the most likely
cause. The SEARCH branch already has a real DOM element dict
(input_elem) to translate directly; the CLICK branch recovers the
resolved mark's real DOM tag by matching its center point against the
same-step dom_elements list (fetched inside ground_fused), falling back
to inferring type from the resolved ACTION if no DOM element matches
(the mark came from a pure-OmniParser detection, not the DOM-priority
rebuild).

Requires an inference server running in fused mode:
    python -m src.agent.inference_server --mode fused \
        --fused-lora models/lora_adapter_fused --port 8787

Usage:
    python tests/planner_agent_fused.py --goal "add a book to cart"

    # remote (Kaggle + ngrok)
    python tests/planner_agent_fused.py --goal "add a book to cart" \
        --server-url https://xxxx.ngrok-free.dev
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import time

sys.path.insert(0, ".")

import requests

from src.agent.browser_env import BrowserEnv
from src.agent.click_visualizer import dom_element_to_mind2web_type

# ══════════════════════════════════════════════════════════════════════════════
# Inference server location
# ══════════════════════════════════════════════════════════════════════════════
SERVER_URL = "http://127.0.0.1:8787"
# SERVER_URL = "https://xxxx.ngrok-free.dev"   # swap for remote — ephemeral, changes per Kaggle restart


ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

# Fallback for when the planner invents an action word outside the valid
# set (e.g. "SELECT(...)") but the shape is otherwise a recognizable
# function call — recover it as a CLICK rather than aborting the whole
# run over a vocabulary slip. Confirmed necessary: base Qwen occasionally
# emits SELECT("...") for what is clearly meant to be "open this product",
# likely bleed-through from the fused adapter's own CLICK/TYPE/SELECT
# vocabulary showing up in training-adjacent phrasing, even though
# disable_adapter() itself was independently confirmed clean.
FALLBACK_ACTION_RE = re.compile(r'\b[A-Z]+\s*\(\s*"([^"]*)"', re.IGNORECASE)

# Same planning prompt as planner_agent.py, tightened to close off two
# observed failure modes: (1) inventing action words outside the valid
# set (e.g. "SELECT(...)"), and (2) appending narrative/status text after
# the function call instead of stopping, apparently mimicking the
# `history` string format it's shown rather than emitting a bare action.
PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time on a shopping website. You are shown the CURRENT screenshot each turn. Decide the single next action needed to make progress toward the goal.

Valid function calls (pick exactly one — these are the ONLY four valid action words, nothing else is allowed):
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Reasoning steps, in order:
1. Look at the screenshot and identify what KIND of page you are on:
   - Search results page: a grid/list of many small product thumbnails and titles, no single large product image.
   - Product detail page: one large product image, a title, price, and an "Add to Cart" / "Buy Now" button.
   - Cart page: a list of items already added, URL contains "cart".
2. "Add to Cart" buttons only exist on a PRODUCT DETAIL page, never directly on search results. If you are on a search results page and the goal needs an item added to cart, you must first CLICK the specific product's title or image to open its product page. Do NOT try to click "Add to cart" while still on search results.
3. Never repeat the exact action you just completed if the screenshot still looks like the same page — if the last action doesn't seem to have changed anything useful, try a different, more specific action instead.
4. If the screenshot shows the goal is already achieved (a cart/basket icon shows a count, the page confirms an item was added, or you're on the cart page with the item listed), respond DONE with a short summary. Do not keep repeating SEARCH or CLICK once the goal is achieved.
5. To open a specific product from search results, use CLICK, never SELECT — SELECT is not a valid action here.

Respond with EXACTLY one function call and nothing else — no extra words, no explanation of what it does, no status note after the closing parenthesis.

Example sequence (one action per turn, across several turns — NOT all in one response):
  Turn 1, goal "buy Dune": SEARCH("Dune")
  Turn 2, now on search results: CLICK("the Dune book cover or title")
  Turn 3, now on product page: CLICK("the Add to Cart button")
  Turn 4, item now in cart: DONE("Added Dune to cart")

Goal: {goal}

Already completed:
{history}

Look at the CURRENT screenshot. Respond with ONLY one function call for the next step:"""
MAX_STEPS = 6


# ══════════════════════════════════════════════════════════════════════════════
# Remote inference client
# ══════════════════════════════════════════════════════════════════════════════

class PlannerFusedInferenceClient:
    """
    HTTP wrapper around /plan (base model, coarse action) and /act_fused
    (fused adapter, single-call CLICK/TYPE/SELECT resolution).
    """

    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            print(f"  [server] mode={data.get('mode')}  status={data.get('status')}")
            if data.get("mode") != "fused":
                print(f"  ⚠ server mode is '{data.get('mode')}', not 'fused' — "
                      f"/act_fused will fail unless the fused adapter is loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach inference server at {self.base_url}. "
                f"Is it running?\n  Error: {exc}"
            )

    def plan(self, png_bytes: bytes, prompt: str) -> dict:
        """POST /plan — base model, no adapter, no OmniParser."""
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        try:
            r = requests.post(
                f"{self.base_url}/plan",
                json={"image_b64": image_b64, "prompt": prompt},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"raw_response": "", "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"raw_response": "", "error": str(exc)}

    def act_fused(
        self,
        png_bytes: bytes,
        task: str,
        history: list[dict],
        dom_elements: list | None = None,
    ) -> dict:
        """POST /act_fused — single call: OmniParser + fused adapter, returns ACTION+MARK+VALUE."""
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task, "history": history}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(
                f"{self.base_url}/act_fused",
                json=payload,
                timeout=240,
            )
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
    if m:
        return m.group(1).upper(), m.group(2).strip()

    # Fallback: model produced a recognizable function-call shape but with
    # an invalid action word (e.g. SELECT("...")) — recover as CLICK rather
    # than aborting the run. See FALLBACK_ACTION_RE docstring above.
    fm = FALLBACK_ACTION_RE.search(response)
    if fm:
        print(f"  ⚠ unrecognized action word in response — treating as CLICK: {response!r}")
        return "CLICK", fm.group(1).strip()

    return None, None


def plan_next_action(client: PlannerFusedInferenceClient, png_bytes: bytes, goal: str, history: list):
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none - this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    result = client.plan(png_bytes, prompt)
    if result.get("error"):
        print(f"  ✗ planner server error: {result['error']}")
        return "", None, None
    response = result.get("raw_response", "")
    action, arg = parse_action(response)
    return response, action, arg


def _dom_info_for_point(dom_elements: list, point: tuple) -> dict | None:
    """
    Find the DOM element (if any) whose bbox_norm contains the resolved
    click point, so we can recover its real HTML tag/type for the
    history entry. Expected to match when the mark came from the
    DOM-priority rebuild; no match is expected for a pure OmniParser
    visual detection with no DOM backing.
    """
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
    """
    Resolve the Mind2Web-vocabulary tag for a fused_history entry: prefer
    the real DOM element's tag/type if the resolved mark matches one,
    otherwise infer a reasonable type from the ACTION itself.
    """
    dom_match = _dom_info_for_point(dom_elements, point)
    if dom_match is not None:
        return dom_element_to_mind2web_type(
            dom_match.get("tag", ""), dom_match.get("type", ""), dom_match.get("label", ""),
        )
    if action == "TYPE":
        return "textbox"
    if action == "SELECT":
        return "combobox"
    return "button"  # CLICK default


def ground_fused(
    client: PlannerFusedInferenceClient,
    browser: BrowserEnv,
    task_desc: str,
    fused_history: list[dict],
):
    """
    Single fused call replacing the old plan+act split for execution steps.
    Passes live DOM elements through so /act_fused can apply the same
    DOM-priority SoM rebuild /act uses — the main available mitigation
    against OmniParser's mark ordering diverging from what the fused
    adapter was trained against.

    Returns (parsed_action, point, elements, dom_elements) — parsed_action
    is {"action", "mark", "value"} or None if the model's JSON didn't
    parse. dom_elements is returned too so the caller can resolve the
    real DOM tag for the fused_history entry without re-fetching it.
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


def _label_for_mark(elements: list, mark_id: int | None) -> str:
    """Look up a human-readable label for the resolved mark, for fused_history."""
    if mark_id is None:
        return "element"
    for e in elements:
        if e.get("id") == mark_id:
            return e.get("content") or e.get("type") or "element"
    return "element"


def run_agent(goal: str, start_url: str, server_url: str):
    client = PlannerFusedInferenceClient(server_url)

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history       = []   # plain strings, fed to the coarse planner prompt
        fused_history = []   # {"tag","label","action","value"} dicts, fed to the fused adapter

        for step in range(1, MAX_STEPS + 1):
            print("\n" + "=" * 60)
            print("STEP " + str(step))
            print("=" * 60)

            ss = browser.screenshot(wait_stable=True)

            print("  Planning next action...")
            t0 = time.time()
            response, action, arg = plan_next_action(client, ss.png_bytes, goal, history)
            print("  (%.1fs) planner said: %r" % (time.time() - t0, response))

            if action is None:
                print("  FAILED to parse action from: %r - stopping." % response)
                break

            print("  -> %s(%r)" % (action, arg))

            if action == "DONE":
                print("\nDONE: " + str(arg))
                break

            elif action == "SEARCH":
                # DOM-direct, same as planner_agent.py — no VLM, fused or otherwise
                elems = browser.get_interactive_elements()
                input_elem = None
                for e in elems:
                    if e["tag"] in ("input", "textarea") and e["type"] not in (
                        "submit", "button", "checkbox", "radio", "image", "hidden"
                    ):
                        input_elem = e
                        break
                if input_elem is None:
                    print("  no search input found - stopping.")
                    break

                submit_elem = None
                for e in elems:
                    if e["tag"] == "button" or e.get("type") in ("submit", "button"):
                        submit_elem = e
                        break

                bbox = input_elem["bbox_norm"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                browser.click(cx, cy)
                time.sleep(0.15)
                browser.press_key("Control+a")
                browser.type_text(arg)
                time.sleep(0.15)
                browser.press_key("Escape")
                time.sleep(0.1)

                if submit_elem is not None:
                    sb = submit_elem["bbox_norm"]
                    sx = (sb[0] + sb[2]) / 2
                    sy = (sb[1] + sb[3]) / 2
                    browser.click(sx, sy)
                else:
                    browser.press_key("Enter")

                try:
                    browser.wait_for_load(timeout_ms=10000)
                except Exception:
                    pass
                history.append('SEARCH("%s") - completed, results now showing' % arg)
                fused_history.append({
                    "tag": dom_element_to_mind2web_type(
                        input_elem.get("tag", "input"),
                        input_elem.get("type", ""),
                        input_elem.get("label", ""),
                    ),
                    "label": input_elem.get("label", "search box"),
                    "action": "TYPE",
                    "value": arg,
                })

            elif action == "CLICK":
                print("  Resolving via fused adapter: %r" % arg)
                t0 = time.time()
                parsed, point, elements, dom_elements = ground_fused(client, browser, arg, fused_history)
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
                    # CLICK or SELECT — fire a click at the resolved point
                    browser.click(point[0], point[1], verify_change=False)

                try:
                    browser.wait_for_load(timeout_ms=5000)
                except Exception:
                    pass  # slow pages may never reach networkidle — same as planner_agent.py

                history.append(
                    '%s("%s") - completed, resolved to mark %s at (%.2f, %.2f)'
                    % (resolved_action, arg, parsed.get("mark"), point[0], point[1])
                )
                fused_history.append({
                    "tag": history_tag,
                    "label": label,
                    "action": resolved_action,
                    "value": parsed.get("value"),
                })

            elif action == "SCROLL":
                direction = arg.strip().lower() if arg else "down"
                browser.scroll(delta_y=400 if direction == "down" else -400)
                history.append('SCROLL("%s") - completed' % direction)

            print("  url now: " + browser.current_url())

        else:
            print("\nHit MAX_STEPS=%d without DONE - stopping." % MAX_STEPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", default="https://www.amazon.in")
    parser.add_argument("--server-url", default=None,
                        help="Override the SERVER_URL constant above without "
                             "editing the file — e.g. a Kaggle+ngrok URL")
    args = parser.parse_args()

    server_url = args.server_url if args.server_url else SERVER_URL
    run_agent(args.goal, args.url, server_url)