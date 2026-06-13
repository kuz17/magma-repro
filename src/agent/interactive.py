# src/agent/interactive.py
"""
Interactive text-only inference with OmniParser → Qwen pipeline.
For click rendering use click_visualizer.py instead.

Usage:
    python -m src.agent.interactive --image outputs/demo/screenshot.png
    python -m src.agent.interactive --image outputs/demo/screenshot.png --lora models/lora_adapter
    python -m src.agent.interactive --image outputs/demo/screenshot.png --raw
"""

from __future__ import annotations

import argparse
import sys

# re-use the DemoRunner and _parse_point from click_visualizer
from src.agent.click_visualizer import DemoRunner, _parse_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to screenshot")
    parser.add_argument("--lora",  default=None,  help="Path to LoRA adapter dir")
    parser.add_argument("--raw",   action="store_true",
                        help="Skip OmniParser, run Qwen directly on raw image")
    args = parser.parse_args()

    runner = DemoRunner(lora_path=args.lora, raw_mode=args.raw)

    print(f"\nImage  : {args.image}")
    mode = "raw" if args.raw else "OmniParser"
    model = f"fine-tuned [{args.lora}]" if args.lora else "base"
    print(f"Mode   : {mode}  |  Model: {model}")
    print("Type your prompt and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            task = input(">>> ").strip()
            if not task:
                continue

            response, point = runner.act(args.image, task)
            print(f"Response : {response}")
            if point:
                print(f"Point    : ({point[0]:.3f}, {point[1]:.3f})")
            print()

        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()