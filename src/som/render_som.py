# src/som/render_som.py

from PIL import ImageDraw, ImageFont
from src.utils.bbox import bbox_center, bbox_area, denormalize_bbox

MIN_AREA_FLOOR = 1e-6
MAX_MARKS      = 100
# Re-tuned for the box-outline renderer (was 1.3, tuned for the old center-dot
# style). This gate is a fixed-radius circular distance check around element
# centers — it doesn't know about actual box size/shape, so 1.3 was dropping
# plainly non-overlapping elements once boxes replaced dots. Swept against the
# full 10k corpus: 0.7 cuts overall drop from 5.9% to 1.5%, with every density
# bucket under 1.5% except the 81+-element tail (21 pages), which stays
# 26-35% at every spacing tested and needs a different fix (label offset
# instead of hard-drop), not further tuning here.
MIN_SPACING    = 0.7

FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _nearest_corner_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    corners_a = [(ax1, ay1), (ax1, ay2), (ax2, ay1), (ax2, ay2)]
    corners_b = [(bx1, by1), (bx1, by2), (bx2, by1), (bx2, by2)]
    return min(
        ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5
        for cax, cay in corners_a for cbx, cby in corners_b
    )


def _find_least_overlapping_corner(box, bboxes, drawn_boxes, text_size, image_size):
    """
    Port of Magma's plot_boxes_with_marks._find_least_overlapping_corner
    (agents/ui_agent/util/som.py): try each of the 8 positions around `box`
    where a `text_size`-shaped label could sit, and pick whichever minimizes
    the worst-case overlap area against every other box outline (`bboxes`)
    and label already drawn (`drawn_boxes`).
    """
    x1, y1, x2, y2 = box
    tw, th = text_size
    width, height = image_size

    candidate_corners = [
        (x1,      y1 - th),   # top-left
        (x2 - tw, y1 - th),   # top-right
        (x2,      y1),        # right-top
        (x2,      y2 - th),   # right-bottom
        (x2 - tw, y2),        # bottom-right
        (x1,      y2),        # bottom-left
        (x1 - tw, y2 - th),   # left-bottom
        (x1 - tw, y1),        # left-top
    ]

    other_boxes = bboxes + drawn_boxes

    best_corner = candidate_corners[0]
    best_score = float("inf")

    for cx, cy in candidate_corners:
        if cx < 0 or cy < 0 or cx + tw > width or cy + th > height:
            continue
        label_box = (cx, cy, cx + tw, cy + th)
        worst = -(width + height)
        for other in other_boxes:
            score = (
                _intersection_area(label_box, other)
                - 0.0001 * _nearest_corner_distance(label_box, other)
            )
            worst = max(worst, score)
        if worst < best_score:
            best_score = worst
            best_corner = (cx, cy)

    return best_corner


def apply_som(image, elements, radius=9, preserve_order=False):
    """
    Draw Set-of-Mark overlays onto image.

    Args:
        image: PIL image to annotate (mutated in place, and also returned).
        elements: list of dicts, each with a "bbox" key — normalized
            [x1, y1, x2, y2] coordinates.
        radius: used only to derive the non-overlap gate distance
            (min_dist_sq below) — not for drawing, since marks are boxes
            with corner labels, not circles.
        preserve_order: if True, skip the area-sort and place elements in
            the order given. Used when the caller has already prioritized
            order (e.g. DOM elements before OmniParser elements) and that
            order must map directly to mark IDs 0, 1, 2, ... .

    Returns:
        image   — the annotated PIL image
        placed  — list of (mark_id, element_dict) for every mark drawn,
                  in placement order (0-indexed, matching drawn labels)
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 1. drop degenerate bboxes
    candidates = [e for e in elements if bbox_area(e["bbox"]) > MIN_AREA_FLOOR]

    # 2. larger elements win placement ties — unless caller already ordered
    # them (e.g. DOM-priority ordering, which must be preserved so DOM
    # elements land on the lowest mark IDs regardless of on-screen size)
    if not preserve_order:
        candidates.sort(key=lambda e: -bbox_area(e["bbox"]))

    # 3. greedy non-overlap placement (gate only — drawing happens below,
    # once the final placed set and every box's pixel coords are known)
    placed_px    = []
    placed       = []
    min_dist_sq  = (radius * 2 * MIN_SPACING) ** 2
    mark_id      = 0

    for el in candidates:
        if mark_id >= MAX_MARKS:
            break

        # Live OmniParser detections can exceed [0,1] on edge-touching boxes;
        # clamp here (normalized space, by construction) so every downstream
        # consumer of `placed` (mark_to_center, formatter, eval) sees valid coords.
        x1, y1, x2, y2 = el["bbox"]
        el = {**el, "bbox": [
            min(max(x1, 0.0), 1.0),
            min(max(y1, 0.0), 1.0),
            min(max(x2, 0.0), 1.0),
            min(max(y2, 0.0), 1.0),
        ]}

        cx, cy = bbox_center(el["bbox"])
        px, py = round(cx * width), round(cy * height)

        if any((px - x) ** 2 + (py - y) ** 2 < min_dist_sq
               for x, y in placed_px):
            continue

        placed_px.append((px, py))
        placed.append((mark_id, el))
        mark_id += 1

    # 4. draw box outlines for every placed element
    boxes_px = [
        tuple(denormalize_bbox(el["bbox"], width, height))
        for _, el in placed
    ]
    for box_px in boxes_px:
        draw.rectangle(list(box_px), outline="red", width=2)

    # 5. label each box at whichever corner overlaps least with other boxes
    # and labels already drawn (paper's SoM style: box outline + corner mark,
    # not a center dot)
    drawn_label_boxes_px = []
    for idx, (mark_id, el) in enumerate(placed):
        box_px = boxes_px[idx]
        other_boxes_px = boxes_px[:idx] + boxes_px[idx + 1:]

        label = str(mark_id)
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        cx, cy = _find_least_overlapping_corner(
            box_px, other_boxes_px, drawn_label_boxes_px,
            (tw, th), (width, height),
        )

        draw.rectangle([cx, cy, cx + tw, cy + th], fill="red")
        draw.text((cx, cy), label, fill="white", font=FONT)

        drawn_label_boxes_px.append((cx, cy, cx + tw, cy + th))

    return image, placed