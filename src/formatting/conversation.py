# src/formatting/conversation.py
"""
Assembles final Magma-style conversation JSON from a raw annotation sample.

Output format per sample:
{
    "image": "seeclick_web/images/foo.png",
    "som_image": "interim/renders/batch/foo.png",
    "conversations": [
        {"from": "user",      "value": "<image>\n..."},
        {"from": "assistant", "value": "..."},
        {"from": "user",      "value": "..."},
        {"from": "assistant", "value": "..."},
    ]
}

Structure:
    - First user turn carries <image>\n prefix
    - Grounding task turns come first
    - Input field task turns appended after
"""

import random
from pathlib import Path

from src.formatting.task_samplers import Element, sample_grounding_task
from src.formatting.input_field import sample_input_task


def build_elements(sample: dict) -> list[Element]:
    """
    Convert raw annotation dict to Element list.
    mark_id is 1-based to match SoM renderer numbering.
    """
    return [
        Element(
            text=el["instruction"],
            bbox=tuple(el["bbox"]),
            mark_id=idx + 1,
            data_type=el.get("data_type", "text"),
        )
        for idx, el in enumerate(sample["elements"])
    ]


def build_conversation(
    sample: dict,
    som_image_path: str | Path | None = None,
    rng: random.Random | None = None,
) -> dict:
    """
    Takes a single raw annotation sample and returns a Magma-style
    conversation dict.

    Args:
        sample:          one entry from filtered_annotations.json
        som_image_path:  path to the pre-rendered SoM image for this sample
        rng:             optional seeded Random for reproducibility

    Returns:
        {
            "image": str,
            "som_image": str,
            "conversations": list of {"from": ..., "value": ...} dicts
        }
    """
    elements = build_elements(sample)

    if not elements:
        return None

    input_elements = [el for el in elements if el.data_type == "input"]

    # ── grounding task ────────────────────────────────────────────────
    grounding_user, grounding_assistant = sample_grounding_task(
        elements, rng=rng
    )

    # ── input field task ──────────────────────────────────────────────
    input_user, input_assistant = sample_input_task(
        input_elements, rng=rng
    )

    # ── assemble turns ────────────────────────────────────────────────
    # <image> token goes on the very first user turn only
    conversations = [
        {"from": "user",      "value": f"<image>\n{grounding_user}"},
        {"from": "assistant", "value": grounding_assistant},
        {"from": "user",      "value": input_user},
        {"from": "assistant", "value": input_assistant},
    ]

    return {
        "image": sample["img_filename"],
        "som_image": str(som_image_path) if som_image_path else None,
        "conversations": conversations,
    }


# ── quick smoke test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import pprint

    sample = {
        "img_filename": "web_000042.png",
        "url": "https://example.com",
        "elements": [
            {"instruction": "Search", "bbox": [0.1, 0.05, 0.6, 0.08], "data_type": "text"},
            {"instruction": "Submit", "bbox": [0.65, 0.05, 0.8, 0.08], "data_type": "text"},
            {"instruction": "Home",   "bbox": [0.02, 0.01, 0.1, 0.04], "data_type": "hover"},
        ],
    }

    rng = random.Random(42)
    result = build_conversation(sample, som_image_path="interim/renders/batch/render_000.png", rng=rng)
    pprint.pprint(result)