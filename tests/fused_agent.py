# tests/fused_agent.py
"""
Pure fused-adapter agent loop — no separate base-model planner at all.

Unlike planner_agent_fused.py, this passes the ACTUAL GOAL as `task` on
every step (constant across the run), matching how FUSED_PROMPT_TEMPLATE
was trained: same task string throughout a trajectory, only `history`
changes. planner_agent_fused.py was instead passing the planner's narrow
per-step CLICK description (e.g. "The Odyssey") as `task` — a materially
different, out-of-distribution prompt shape compared to training, and a
likely contributor to unreliable mark resolution independent of the
OmniParser-ordering mismatch already found.

The fused adapter decides ACTION (CLICK/TYPE/SELECT) + MARK + VALUE
itself, every step — there is no planner call, no /plan endpoint used.

History tag vocabulary fix: fused_history entries now use Mind2Web's
semantic element-type vocabulary ("link", "searchbox", "button",
"combobox", "textbox") via dom_element_to_mind2web_type(), instead of the
generic literal "element" this script used previously. Confirmed via a
real training example (MagmaAI/Magma-Mind2Web-SoM row) that every
checkpoint (400-825 steps) correctly resolves the right mark when history
uses this vocabulary, but live Amazon runs showed task-blind collapse to
page-fixed marks — the mismatched history vocabulary is the most likely
cause, since the model demonstrably CAN discriminate correctly given
training-shaped input. The resolved mark's real DOM tag is recovered by
matching its center point against the same-step dom_elements list (the
tag isn't otherwise available from /act_fused's response, since
ElementInfo only carries OmniParser's own type, not HTML semantics); if
no DOM element matches (the mark came from a pure-OmniParser detection,
not the DOM-priority rebuild), the type is inferred from the resolved
ACTION instead, as a reasonable fallback.

Two real vocabulary gaps this creates, both handled heuristically here,
NOT by the model:
  - No SCROLL action exists in Mind2Web's vocabulary. If resolution fails
    twice in a row, this script auto-scrolls and retries — a safety
    valve, not something the model asked for.
  - No DONE signal exists. This script stops on MAX_STEPS, or earlier via
    a soft heuristic: if the resolved element's label contains "cart" and
    the action was CLICK, it's treated as likely task completion.

Requires an inference server running in fused mode:
    python -m src.agent.inference_server --mode fused \
        --fused-lora models/lora_adapter_fused --port 8787

Usage:
    python tests/fused_agent.py --goal "add the odyssey book to cart"
"""
from __future__ import annotations

import argparse
import base64
import sys
import time

sys.path.insert(0, ".")

import requests

from src.agent.browser_env import BrowserEnv
from src.agent.click_visualizer import dom_element_to_mind2web_type

SERVER_URL = "http://127.0.0.1:8787"
MAX_STEPS = 10
MAX_CONSECUTIVE_FAILURES = 2   # triggers auto-scroll fallback


class FusedAgentClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"  [server] mode={data.get('mode')}  status={data.get('status')}")
        if data.get("mode") != "fused":
            print(f"  ⚠ server mode is '{data.get('mode')}', not 'fused'.")

    def act_fused(self, png_bytes: bytes, task: str, history: list[dict], dom_elements: list) -> dict:
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task, "history": history, "dom_elements": dom_elements}
        try:
            r = requests.post(f"{self.base_url}/act_fused", json=payload, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "action": None, "mark_id": None, "value": None,
                    "raw_response": "", "elements": [], "error": "Request timed out."}
        except Exception as exc:
            return {"click_norm": None, "action": None, "mark_id": None, "value": None,
                    "raw_response": "", "elements": [], "error": str(exc)}


def _label_for_mark(elements: list, mark_id: int | None) -> str:
    if mark_id is None:
        return "element"
    for e in elements:
        if e.get("id") == mark_id:
            return e.get("content") or e.get("type") or "element"
    return "element"


def _dom_info_for_point(dom_elements: list, point: tuple) -> dict | None:
    """
    Find the DOM element (if any) whose bbox_norm contains the resolved
    click point, so we can recover its real HTML tag/type for the
    history entry. The mark itself may have come from the DOM-priority
    rebuild (in which case this should reliably match) or from a pure
    OmniParser visual detection with no DOM backing (in which case no
    match is expected, and the caller should fall back to inferring type
    from the resolved ACTION instead).
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
    # No DOM backing (pure OmniParser detection) -- infer from action.
    if action == "TYPE":
        return "textbox"
    if action == "SELECT":
        return "combobox"
    return "button"  # CLICK default


def run_agent(goal: str, start_url: str, server_url: str):
    client = FusedAgentClient(server_url)

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        fused_history = []   # {"tag","label","action","value"} dicts
        consecutive_failures = 0

        for step in range(1, MAX_STEPS + 1):
            print("\n" + "=" * 60)
            print(f"STEP {step}")
            print("=" * 60)

            ss = browser.screenshot(wait_stable=True)
            dom_elements = browser.get_interactive_elements()

            print(f"  Resolving via fused adapter (goal: {goal!r})...")
            t0 = time.time()
            result = client.act_fused(ss.png_bytes, goal, fused_history, dom_elements)
            print(f"  ({time.time() - t0:.1f}s) fused response: {result.get('raw_response', '')!r}")

            if result.get("error"):
                print(f"  ✗ server error: {result['error']} - stopping.")
                break

            action = result.get("action")
            mark_id = result.get("mark_id")
            value = result.get("value")
            click_norm = result.get("click_norm")
            elements = result.get("elements", [])
            point = tuple(click_norm) if click_norm else None

            if action is None or point is None:
                consecutive_failures += 1
                print(f"  no valid resolution ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} failures)")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("  auto-scrolling down as a fallback (no SCROLL action in fused vocabulary)...")
                    browser.scroll(delta_y=400)
                    consecutive_failures = 0
                    # "page" has no real Mind2Web equivalent -- this is a
                    # pipeline-side fallback action, not something the
                    # model ever predicts, so it's left as-is rather than
                    # mapped through dom_element_to_mind2web_type().
                    fused_history.append({"tag": "page", "label": "scrolled down", "action": "SCROLL", "value": None})
                continue

            consecutive_failures = 0
            label = _label_for_mark(elements, mark_id)
            history_tag = _history_tag_for_action(action, dom_elements, point)
            print(f"  -> {action} mark {mark_id} ({label!r}) at ({point[0]:.3f}, {point[1]:.3f})  "
                  f"value={value!r}  history_tag={history_tag!r}")

            if action == "TYPE":
                browser.click(point[0], point[1])
                time.sleep(0.15)
                browser.press_key("Control+a")
                browser.type_text(value or "")
                time.sleep(0.1)
            else:
                # CLICK or SELECT
                browser.click(point[0], point[1], verify_change=False)

            try:
                browser.wait_for_load(timeout_ms=5000)
            except Exception:
                pass

            fused_history.append({
                "tag": history_tag,
                "label": label,
                "action": action,
                "value": value,
            })

            print(f"  url now: {browser.current_url()}")

            # Soft completion heuristic — no DONE signal exists in this
            # vocabulary, so this is a guess, not a confirmed success.
            if action == "CLICK" and "cart" in label.lower():
                print(f"\n  Resolved element mentions 'cart' — likely task completion.")
                print(f"  STOPPING (heuristic, not confirmed) after step {step}.")
                break

        else:
            print(f"\nHit MAX_STEPS={MAX_STEPS} without a completion signal - stopping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", default="https://www.amazon.in")
    parser.add_argument("--server-url", default=None)
    args = parser.parse_args()

    server_url = args.server_url if args.server_url else SERVER_URL
    run_agent(args.goal, args.url, server_url)