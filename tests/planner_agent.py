# tests/planner_agent.py
"""
Planner+executor loop wired to an inference server (local or remote).

Planner step  -> POST /plan  (base model, adapter disabled server-side)
Executor step -> POST /act   (fine-tuned adapter, full OmniParser+Qwen grounding)

Both endpoints are served by the SAME loaded model instance on the server
(see src/agent/inference_server.py) — no double VRAM, no load_adapter/unload
bug, matching the property the original in-process design was built around.
Just split across a network call now instead of a Python function call, so
the model can live on a remote GPU (e.g. Kaggle T4/P100 + ngrok) while the
browser stays local.

Usage:
    # local inference server (default — SERVER_URL below)
    python tests/planner_agent.py --goal "add a book to cart"

    # remote inference server
    python tests/planner_agent.py --goal "add a book to cart" \
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

# ══════════════════════════════════════════════════════════════════════════════
# Inference server location
# ══════════════════════════════════════════════════════════════════════════════
# LOCAL (default) — inference server running on this same machine.
SERVER_URL = "http://127.0.0.1:8787"

# REMOTE — Kaggle T4/P100 + ngrok tunnel. To switch: comment out the LOCAL
# line above, uncomment this one, and paste in your current ngrok URL
# (it's ephemeral — changes every time the Kaggle notebook restarts).
# SERVER_URL = "https://plop-isolated-blinked.ngrok-free.dev"

# Can also be overridden per-run with --server-url without editing this file.


ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time. You respond with ONLY a function call, nothing else - no explanation, no extra words.

Valid function calls (pick exactly one):
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Examples of correct responses:
CLICK("the blue Add to cart button")
SEARCH("wireless mouse")
SCROLL("down")

Goal: {goal}

Already completed:
{history}

Look at the screenshot. Respond with ONLY one function call for the next step:"""

MAX_STEPS = 6


# ══════════════════════════════════════════════════════════════════════════════
# Remote inference client
# ══════════════════════════════════════════════════════════════════════════════

class PlannerInferenceClient:
    """
    Thin HTTP wrapper around the inference server's /plan and /act endpoints.
    Replaces the in-process DemoRunner calls the original planner_agent.py
    made directly — same two operations (plan, ground), now over the network,
    so the model can live on a different machine than the browser.
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

    def act(self, png_bytes: bytes, task: str, dom_elements: list | None = None) -> dict:
        """POST /act — fine-tuned adapter, full OmniParser+Qwen grounding pipeline."""
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(
                f"{self.base_url}/act",
                json=payload,
                timeout=240,
            )
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


def plan_next_action(client: PlannerInferenceClient, png_bytes: bytes, goal: str, history: list):
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none - this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    result = client.plan(png_bytes, prompt)
    if result.get("error"):
        print(f"  ✗ planner server error: {result['error']}")
        return "", None, None
    response = result.get("raw_response", "")
    action, arg = parse_action(response)
    return response, action, arg


def ground_click(client: PlannerInferenceClient, browser: BrowserEnv, task_desc: str):
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
    client = PlannerInferenceClient(server_url)

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history = []

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
                browser.press_key("Escape")  # dismiss autocomplete dropdown
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

            elif action == "CLICK":
                print("  Grounding click target: %r" % arg)
                t0 = time.time()
                point = ground_click(client, browser, arg)
                print("  (%.1fs) grounding done" % (time.time() - t0))
                if point is None:
                    print("  grounding failed to find a point for %r - stopping." % arg)
                    break
                browser.click(point[0], point[1], verify_change=False)
                try:
                    browser.wait_for_load(timeout_ms=5000)
                except Exception:
                    pass  # slow pages may never reach networkidle — that's fine,
                          # same reasoning as BrowserEnv.back() elsewhere; without
                          # this try/except a slow page previously killed the
                          # entire agent process mid-run (see 2026-07-28 fix)
                history.append(
                    'CLICK("%s") - completed, clicked at (%.2f, %.2f)' % (arg, point[0], point[1])
                )

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