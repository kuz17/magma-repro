# tests/check_training_distribution.py
"""
Direct test: does the fused adapter discriminate correctly on a REAL
training-distribution example (known ground truth), or does it show the
same task-blind collapse seen on live Amazon screenshots?

This bypasses OmniParser entirely -- the image already has training-style
SoM marks baked in (it IS a MagmaAI/Magma-Mind2Web-SoM row), so we feed
the exact image + exact prompt from that row straight to each checkpoint
and compare against the known correct answer.

Ground truth for this example: MARK 13 ("IL" in the state dropdown),
following from history: [link] Store Locator -> CLICK, [searchbox]
SEARCH BY KEYWORD -> CLICK, [searchbox] SEARCH BY KEYWORD -> TYPE: Chicago,
[input] -> CLICK (opened the state dropdown). Task: "Find all Uniqlo's
located in Chicago, IL."

If checkpoints get 13 (or land visually on "IL") -- the model discriminates
fine on in-distribution data, and the live-Amazon failures are a
generalization gap (different problem, different fix).

If checkpoints STILL collapse to some fixed answer regardless of this
being real training-shaped data -- that's a real methodology bug, most
likely a MARK-distribution skew in the training set itself, not a
generalization gap.

Usage:
    python tests/check_training_distribution.py --image outputs/bisect_fused/training_example_uniqlo.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
from PIL import Image

from src.agent.click_visualizer import DemoRunner, parse_fused_action

CHECKPOINTS = {
    "ckpt_400": "models/lora_adapter_fused_checkpoints/checkpoint-400",
    "ckpt_450": "models/lora_adapter_fused_checkpoints/checkpoint-450",
    "ckpt_500": "models/lora_adapter_fused_checkpoints/checkpoint-500",
    "ckpt_550": "models/lora_adapter_fused_checkpoints/checkpoint-550",
    "ckpt_600": "models/lora_adapter_fused_checkpoints/checkpoint-600",
    "ckpt_650": "models/lora_adapter_fused_checkpoints/checkpoint-650",
    "ckpt_700": "models/lora_adapter_fused_checkpoints/checkpoint-700",
    "ckpt_750": "models/lora_adapter_fused_checkpoints/checkpoint-750",
    "ckpt_800": "models/lora_adapter_fused_checkpoints/checkpoint-800",
    "ckpt_825": "models/lora_adapter_fused_checkpoints/checkpoint-825",
    "final":    "models/lora_adapter_fused",
}

ADAPTER_NAME = "current"

# Exact verbatim prompt from the training example — NOT reconstructed via
# FUSED_PROMPT_TEMPLATE, to guarantee byte-for-byte match with what the
# model actually saw during training for this row.
PROMPT = (
    'Imagine that you are imitating humans doing web navigation for a task step by step. '
    'At each stage, you can see the webpage like humans by a screenshot and know the previous '
    'actions before the current step decided by yourself through recorded history. You need to '
    'decide on the following action to take. You can click an element with the mouse, select an '
    'option, or type text with the keyboard. The output format should be a dictionary like: \n'
    '"{"ACTION": "CLICK" or "TYPE" or "SELECT", "MARK": a numeric id, e.g., 5, "VALUE": a string '
    'value for the action if applicable, otherwise None}".\n'
    'You are asked to complete the following task: Find all Uniqlo\'s located in Chicago, IL.. '
    'The previous actions you have taken: \n\n'
    '[link] Store Locator -> CLICK\n'
    '[searchbox] SEARCH BY KEYWORD -> CLICK\n'
    '[searchbox] SEARCH BY KEYWORD -> TYPE: Chicago\n'
    '[input] -> CLICK\n'
    'For your convinience, I have labeled the candidates with numeric marks and bounding boxes on '
    'the screenshot. What is the next action you would take?'
)

GROUND_TRUTH_MARK = 13


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    ckpt_items = list(CHECKPOINTS.items())
    first_tag, first_path = ckpt_items[0]

    print(f"Loading base Qwen + first checkpoint ({first_tag})...")
    runner = DemoRunner(lora_path=first_path, training_style=True, tag="train_dist_check", raw_mode=True)
    runner._qwen.load_adapter(first_path, adapter_name=ADAPTER_NAME)
    runner._qwen.delete_adapter("default")
    runner._qwen.set_adapter(ADAPTER_NAME)

    print(f"\nGround truth: MARK {GROUND_TRUTH_MARK} (\"IL\" in the state dropdown)\n")
    print("=" * 70)

    results = []
    for i, (tag, path) in enumerate(ckpt_items):
        if i > 0:
            runner._qwen.delete_adapter(ADAPTER_NAME)
            torch.cuda.empty_cache()
            runner._qwen.load_adapter(path, adapter_name=ADAPTER_NAME)
            runner._qwen.set_adapter(ADAPTER_NAME)

        response = runner._run_qwen(image, PROMPT, max_new_tokens=60)
        parsed = parse_fused_action(response)
        mark = parsed.get("mark") if parsed else None
        correct = "✓ CORRECT" if mark == GROUND_TRUTH_MARK else "✗ wrong"

        print(f"[{tag:<10}] raw={response!r}")
        print(f"             parsed mark={mark}  {correct}\n")

        results.append((tag, mark, mark == GROUND_TRUTH_MARK))

    print("=" * 70)
    n_correct = sum(1 for _, _, c in results if c)
    print(f"Correct: {n_correct}/{len(results)}")
    marks_seen = {m for _, m, _ in results}
    print(f"Distinct marks predicted across all checkpoints: {sorted(x for x in marks_seen if x is not None)}")
    if len(marks_seen) == 1:
        print("⚠ EVERY checkpoint gave the identical mark on real training-shaped "
              "data too — this points to a training-data/methodology issue, "
              "not a live-page generalization gap.")
    else:
        print("Checkpoints vary on this example — behavior differs from the "
              "live-Amazon collapse, suggesting a generalization gap rather "
              "than a training-data issue.")


if __name__ == "__main__":
    main()