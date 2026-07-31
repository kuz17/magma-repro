#tests/smoke_fused_mark_check.py
"""
Empirical check: does the fused adapter's MARK output actually point at
something sensible on a real OmniParser-rendered screenshot?

This is the one open risk from the handover doc — Magma's own SoM
rendering pipeline is not this project's OmniParser + render_som.py, and
visual similarity doesn't guarantee matching detection/numbering order.
Run this BEFORE building the /act_fused endpoint or wiring history.

Usage:
    python -m tests.smoke_fused_mark_check --image outputs/demo/screenshot.png --task "click the search button"
"""
import argparse
import sys
sys.path.insert(0, ".")

from src.agent.click_visualizer import DemoRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lora", default="models/lora_adapter_smoke")
    parser.add_argument("--fused-lora", default="models/lora_adapter_fused")
    args = parser.parse_args()

    print("Loading model with both adapters...")
    runner = DemoRunner(
        lora_path=args.fused_lora,
        #fused_lora_path=args.fused_lora,
        training_style=True,
        tag="fused_check",
    )

    print(f"\nTask: {args.task!r}\n")
    response, parsed, point = runner.act_fused(args.image, args.task)

    print(f"Raw response : {response!r}")
    print(f"Parsed       : {parsed}")
    print(f"Resolved pt  : {point}")

    if parsed is None:
        print("\n>>> FAILED TO PARSE JSON — check max_new_tokens / prompt format first.")
    elif point is None:
        print(f"\n>>> Mark {parsed.get('mark')} did not resolve to any element — "
              f"either MARK is out of range or OmniParser detected fewer elements "
              f"than the fused adapter expects.")
    else:
        print(f"\n>>> Mark resolved to a real point. Manually check the screenshot: "
              f"does ({point[0]:.3f}, {point[1]:.3f}) actually land on \"{args.task}\"?")


if __name__ == "__main__":
    main()