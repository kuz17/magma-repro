# tests/probe_reasoning_prompt.py
"""
Probe: does giving the model more token budget (with or without an
explicit reasoning instruction) change grounding accuracy on two
representative search-results tasks -- a product name ("The Odyssey")
and a button with a color qualifier ("yellow add to cart")?

Confirmed from Magma's own reference inference code (microsoft/Magma
README) that their default generation_args use max_new_tokens=500 --
25x this project's current 20-token cap -- so this probe checks whether
that gap is actually costing anything, before committing to a pipeline-
wide change.

Three conditions per task:
  1. CURRENT  -- exact production prompt/token-limit (baseline, what the
                 live pipeline actually does today)
  2. REASON_200 -- reasoning-invited prompt, max_new_tokens=200
  3. REASON_400 -- reasoning-invited prompt, max_new_tokens=400

Not a permanent pipeline change -- a quick comparison to decide whether
this is worth building out properly.

Usage:
    python tests/probe_reasoning_prompt.py
"""
import sys
sys.path.insert(0, ".")
from PIL import Image
from src.agent.click_visualizer import DemoRunner

CHECKPOINT = "models/lora_adapter_v2"  # best-behaving checkpoint from tonight's sweep
IMAGE_PATH = "outputs/bisect_smoke/results_som_reference.png"  # search results page, already SoM-rendered

TASKS = [
    "The Odyssey",
    "cover of the odyssey",   # alt phrasing for the same target, in case "The Odyssey" alone is too ambiguous (matches multiple text spans on page)
    "yellow add to cart",
]

CURRENT_TEMPLATE = (
    'To execute the step "{task}", where do I direct my attention? '
    "Please provide the coordinate and the bounding box's mark index."
)

REASONING_TEMPLATE = (
    'To execute the step "{task}", where do I direct my attention? '
    'First, in one short sentence, describe which numbered mark matches this element and why. '
    'Then, on a new line, give your final answer in the exact format: '
    'Coordinate: (x, y). Mark: N.'
)

CONDITIONS = [
    ("CURRENT",     CURRENT_TEMPLATE,   20),
    ("REASON_200",  REASONING_TEMPLATE, 200),
    ("REASON_400",  REASONING_TEMPLATE, 400),
]


def main():
    print(f"Loading {CHECKPOINT}...")
    runner = DemoRunner(lora_path=CHECKPOINT, training_style=True, tag="probe")
    som_image = Image.open(IMAGE_PATH).convert("RGB")

    for task in TASKS:
        print(f"\n{'='*70}\nTask: {task!r}\n{'='*70}")

        for label, template, max_tokens in CONDITIONS:
            prompt = template.format(task=task)
            response = runner._run_qwen(som_image, prompt, max_new_tokens=max_tokens)
            print(f"\n-- {label} (max_new_tokens={max_tokens}) --")
            print(repr(response))


if __name__ == "__main__":
    main()