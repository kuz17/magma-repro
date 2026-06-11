# src/formatting/conversation.py

import random
from pathlib import Path

from src.formatting.task_samplers import Element, sample_grounding_task
from src.formatting.input_field import sample_input_task


def build_elements_from_placed(placed: list) -> list[Element]:
    return [
        Element(
            text=el["instruction"],
            bbox=tuple(el["bbox"]),
            mark_id=mark_id,
            data_type=el.get("data_type", "text"),
        )
        for mark_id, el in placed
    ]


def build_conversation(
    sample: dict,
    placed: list,
    som_image_path: str | Path | None = None,
    rng: random.Random | None = None,
) -> dict | None:
    elements = build_elements_from_placed(placed)

    if not elements:
        return None

    input_elements = [el for el in elements if el.data_type == "input"]

    grounding_user, grounding_assistant = sample_grounding_task(
        elements, rng=rng
    )
    input_user, input_assistant = sample_input_task(
        input_elements, rng=rng
    )

    conversations = [
        {"from": "user",      "value": f"<image>\n{grounding_user}"},
        {"from": "assistant", "value": grounding_assistant},
        {"from": "user",      "value": input_user},
        {"from": "assistant", "value": input_assistant},
    ]

    return {
        "image":         sample["img_filename"],
        "som_image":     str(som_image_path) if som_image_path else None,
        "conversations": conversations,
    }


if __name__ == "__main__":
    import pprint

    placed = [
        (0, {"instruction": "Search", "bbox": [0.1, 0.05, 0.6, 0.08], "data_type": "text"}),
        (1, {"instruction": "Submit", "bbox": [0.65, 0.05, 0.8, 0.08], "data_type": "text"}),
        (2, {"instruction": "Home",   "bbox": [0.02, 0.01, 0.1, 0.04], "data_type": "hover"}),
    ]
    sample = {
        "img_filename": "web_000042.png",
        "url": "https://example.com",
        "elements": [],
    }
    rng = random.Random(42)
    result = build_conversation(
        sample,
        placed=placed,
        som_image_path="interim/renders/batch/render_000.png",
        rng=rng,
    )
    pprint.pprint(result)