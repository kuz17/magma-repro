# src/formatting/task_samplers.py
"""
Four grounding task builders for SeeClick-Web / Vision2UI formatting.

Each function takes a list of (text, bbox, mark_id) tuples for a single
screenshot and returns (user_turn: str, assistant_turn: str).

Coordinate format throughout: normalized (x1, y1, x2, y2) in [0, 1].
Center point format: (cx, cy).

Sampling weights from paper (Section B.1.1):
    text→point : 0.4
    text→bbox  : 0.4
    point→text : 0.1
    bbox→text  : 0.1
"""

import random
from typing import NamedTuple


# ─────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────

class Element(NamedTuple):
    text: str               # element label / content
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    mark_id: int            # integer mark drawn on the SoM image
    data_type: str = "text"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _center(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _fmt_bbox(bbox: tuple) -> str:
    x1, y1, x2, y2 = bbox
    return f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"


def _fmt_point(cx: float, cy: float) -> str:
    return f"({cx:.2f}, {cy:.2f})"


# ─────────────────────────────────────────────
# Task builders
# Each returns (user_turn_text, assistant_turn_text)
# The caller is responsible for prepending <image>\n to the first user turn.
# ─────────────────────────────────────────────

def text_to_bbox(elements: list[Element]) -> tuple[str, str]:
    """
    User gives element text descriptions one per turn.
    Assistant returns bbox + mark index.

    Paper example (Figure 12b):
        User:      "November 7, 2013"
        Assistant: "Coordinate: (0.27, 0.29, 0.37, 0.30). Mark: 2."
    """
    user_lines = []
    assistant_lines = []

    for el in elements:
        user_lines.append(el.text)
        assistant_lines.append(
            f"Coordinate: {_fmt_bbox(el.bbox)}. Mark: {el.mark_id}."
        )

    # First user turn carries the task instruction prefix
    prefix = (
        "From this webpage capture, I'll give descriptions. "
        "Predict each element's bounding box and mark index, if present."
    )
    user_turn = prefix + "\n" + "\n".join(user_lines)
    assistant_turn = "\n".join(assistant_lines)

    return user_turn, assistant_turn


def text_to_point(elements: list[Element]) -> tuple[str, str]:
    """
    User gives element text / task instruction.
    Assistant returns center point + mark index.

    Paper example (Figure 12c):
        User:      "execute the step 'its a stopwatch that measures time',
                    where do I direct my attention?"
        Assistant: "Coordinate: (0.04, 0.12). Mark: 12"
    """
    user_lines = []
    assistant_lines = []

    for el in elements:
        cx, cy = _center(el.bbox)
        user_lines.append(
            f'To execute the step "{el.text}", '
            f"where do I direct my attention? "
            f"Please provide the coordinate and the bounding box's mark index."
        )
        assistant_lines.append(
            f"Coordinate: {_fmt_point(cx, cy)}. Mark: {el.mark_id}."
        )

    user_turn = "\n".join(user_lines)
    assistant_turn = "\n".join(assistant_lines)

    return user_turn, assistant_turn


def bbox_to_text(elements: list[Element]) -> tuple[str, str]:
    """
    User gives bbox coordinates.
    Assistant returns text content + mark index.

    Paper example (Figure 12a):
        User:      "(0.10, 0.29, 0.13, 0.30)"
        Assistant: "Text: News. Mark: 1."
    """
    prefix = (
        "I'll provide bounding boxes on this webpage screenshot. "
        "Please describe the text content within each box, "
        "adding the mark index when relevant."
    )

    user_lines = [prefix]
    assistant_lines = []

    for el in elements:
        user_lines.append(_fmt_bbox(el.bbox))
        assistant_lines.append(
            f"Text: {el.text}. Mark: {el.mark_id}."
        )

    user_turn = "\n".join(user_lines)
    assistant_turn = "\n".join(assistant_lines)

    return user_turn, assistant_turn


def point_to_text(elements: list[Element]) -> tuple[str, str]:
    """
    User gives center point + mark index.
    Assistant returns text description of the element.

    Paper example (Figure 12d):
        User:      "How does the element at (0.93, 0.41) (Mark 0)
                    contribute to the overall user experience?"
        Assistant: "make selection"
    """
    user_lines = []
    assistant_lines = []

    for el in elements:
        cx, cy = _center(el.bbox)
        user_lines.append(
            f"How does the element at {_fmt_point(cx, cy)} "
            f"(Mark {el.mark_id}) contribute to the overall user experience?"
        )
        assistant_lines.append(el.text)

    user_turn = "\n".join(user_lines)
    assistant_turn = "\n".join(assistant_lines)

    return user_turn, assistant_turn


# ─────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────

TASK_WEIGHTS = [0.4, 0.4, 0.1, 0.1]
TASKS = [text_to_point, text_to_bbox, point_to_text, bbox_to_text]


def sample_grounding_task(
    elements: list[Element],
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """
    Sample one grounding task according to paper weights [0.4, 0.4, 0.1, 0.1]
    and apply it to the given elements.

    Returns (user_turn, assistant_turn).
    """
    r = rng or random
    task_fn = r.choices(TASKS, weights=TASK_WEIGHTS, k=1)[0]
    return task_fn(elements)