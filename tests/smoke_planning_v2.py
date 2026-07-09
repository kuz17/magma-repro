import re
import torch
from PIL import Image
import sys
sys.path.insert(0, ".")
from src.agent.click_visualizer import DemoRunner

ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time. You respond with ONLY a function call, nothing else — no explanation, no extra words.

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


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def plan_next_action(runner, image, goal, history):
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none — this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    with runner._qwen.disable_adapter():
        response = runner._run_qwen(image, prompt)
    action, arg = parse_action(response)
    return response, action, arg


if __name__ == "__main__":
    print("Loading model...")
    runner = DemoRunner(lora_path="models/lora_adapter_smoke", raw_mode=True, tag="planner_test")

    image = Image.open("outputs/browser/search_results_atomic_habits.png").convert("RGB")

    goal = "buy the Atomic Habits book"
    history = ['SEARCH("Atomic Habits") — completed, search results now showing']

    print(f"\nGoal: {goal}")
    print(f"History: {history}")
    response, action, arg = plan_next_action(runner, image, goal, history)
    print(f"Raw response : {response!r}")
    print(f"Parsed       : action={action}  arg={arg!r}")
