# tests/planner_agent.py
"""
Minimal planner+executor loop.
Planner: base Qwen (adapter disabled) decides the next action.
Executor: fine-tuned Qwen (adapter active) grounds CLICK actions to coordinates.
Single model instance, single process - no double VRAM, no load_adapter/unload bug.
"""
import re
import sys
import time
sys.path.insert(0, ".")

from src.agent.click_visualizer import DemoRunner
from src.agent.browser_env import BrowserEnv

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


def parse_action(response):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def plan_next_action(runner, image, goal, history):
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none - this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    with runner._qwen.disable_adapter():
        response = runner._run_qwen(image, prompt)
    action, arg = parse_action(response)
    return response, action, arg


def ground_click(runner, browser, task_desc):
    ss = browser.screenshot(wait_stable=True)
    tmp_path = "/tmp/planner_step.png"
    ss.image.save(tmp_path)
    dom_elements = browser.get_interactive_elements()
    response, point = runner.act(tmp_path, task_desc, dom_elements=dom_elements)
    print("    grounding response: " + repr(response))
    return point


def run_agent(goal, start_url):
    print("Loading model...")
    runner = DemoRunner(lora_path="models/lora_adapter_smoke", tag="planner_agent", training_style=True)

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
            response, action, arg = plan_next_action(runner, ss.image, goal, history)
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
                point = ground_click(runner, browser, arg)
                print("  (%.1fs) grounding done" % (time.time() - t0))
                if point is None:
                    print("  grounding failed to find a point for %r - stopping." % arg)
                    break
                browser.click(point[0], point[1], verify_change=False)
                browser.wait_for_load(timeout_ms=5000)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", default="https://www.amazon.in")
    args = parser.parse_args()
    run_agent(args.goal, args.url)