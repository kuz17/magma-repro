# tests/smoke_planning_prompt.py
"""
Fast, isolated test of the planning prompt — no browser, no loop.
Reuses your existing DemoRunner (fine-tuned adapter) but calls the
planner path via disable_adapter(). Goal: see if base Qwen reliably
outputs a parseable next-action given a goal + screen + history,
and — critically — whether it actually USES the history to avoid
repeating completed steps, rather than just pattern-matching the goal.
"""
import re
import torch
from PIL import Image
import sys
sys.path.insert(0, ".")
from src.agent.click_visualizer import DemoRunner

# ACTION: prefix is now optional — the model reliably omits it even
# when the prompt asks for it, so match with or without.
ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

PLANNING_PROMPT_TEMPLATE = """You are controlling a web browser to accomplish a goal, one step at a time.

Goal: {goal}

Actions already completed (do NOT repeat these):
{history}

Based on what's already been done, choose the SINGLE NEXT action needed.
Respond with ONLY one line in this exact format:
SEARCH("query") or CLICK("description") or SCROLL(down) or DONE("summary")

Next action:"""


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def plan_next_action(runner: DemoRunner, image: Image.Image, goal: str, history: list) -> tuple:
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none — this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    with runner._qwen.disable_adapter():
        response = runner._run_qwen(image, prompt)
    action, arg = parse_action(response)
    return response, action, arg


if __name__ == "__main__":
    print("Loading model (adapter loaded, will toggle off for planning)...")
    runner = DemoRunner(lora_path="models/lora_adapter_smoke", raw_mode=True, tag="planner_test")

    TEST_IMAGE = "outputs/browser/ss_0000.png"  # swap to a real screenshot
    image = Image.open(TEST_IMAGE).convert("RGB")

    test_cases = [
        ("buy the Atomic Habits book", []),
        ("buy the Atomic Habits book", ['SEARCH("Atomic Habits") — completed, search results now showing']),
        ("what is the weather in Delhi", []),
        ("buy a rubik's cube under 500 rupees with good reviews", []),
    ]

    for goal, history in test_cases:
        print(f"\n{'─'*60}")
        print(f"Goal: {goal}")
        print(f"History: {history}")
        response, action, arg = plan_next_action(runner, image, goal, history)
        print(f"Raw response : {response!r}")
        print(f"Parsed       : action={action}  arg={arg!r}")
        if action is None:
            print("✗ FAILED TO PARSE — prompt needs work")