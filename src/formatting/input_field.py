# src/formatting/input_field.py


import random
from src.formatting.task_samplers import Element, _fmt_bbox, _fmt_point, _center


def input_to_bbox(elements: list[Element]) -> tuple[str, str]:
    
    #User asks for input field bounding boxes.
    #Assistant returns bbox + mark index for each, or signals none found.

    
    user_turn = (
        "Analyze the screenshot for input fields and provide "
        "their bounding box coordinates and mark index if they exist."
    )

    if not elements:
        assistant_turn = "No input areas found."
    else:
        lines = [
            f"Coordinate: {_fmt_bbox(el.bbox)}. Mark: {el.mark_id}."
            for el in elements
        ]
        assistant_turn = "\n".join(lines)

    return user_turn, assistant_turn


def input_to_point(elements: list[Element]) -> tuple[str, str]:
    
    #User asks for input field center points.
    #Assistant returns center point + mark index for each, or signals none found.

    user_turn = (
        "Identify the presence of input fields in the provided screenshot "
        "and return their click point coordinates along with the "
        "mark index if applicable."
    )

    if not elements:
        assistant_turn = "No input areas found."
    else:
        lines = []
        for el in elements:
            cx, cy = _center(el.bbox)
            lines.append(
                f"Coordinate: {_fmt_point(cx, cy)}. Mark: {el.mark_id}."
            )
        assistant_turn = "\n".join(lines)

    return user_turn, assistant_turn


INPUT_TASKS = [input_to_point, input_to_bbox]


def sample_input_task(
    input_elements: list[Element],
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """
    Sample one input field task with equal probability.
    input_elements should be pre-filtered to only input-type elements.
    """
    r = rng or random
    task_fn = r.choice(INPUT_TASKS)
    return task_fn(input_elements)