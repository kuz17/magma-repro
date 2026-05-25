from PIL import ImageDraw, ImageFont
from src.utils.bbox import (
    bbox_center,
    bbox_area,
)

MIN_AREA_FLOOR = 1e-6   # reject only degenerate/zero-area bboxes
MAX_MARKS = 20
MIN_SPACING = 1.3       # multiplier on marker diameter


def apply_som(image, elements, radius=9):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    # 1. drop only degenerate bboxes; MAX_MARKS + non-overlap handle clutter
    candidates = [
        e for e in elements
        if bbox_area(e["bbox"]) > MIN_AREA_FLOOR
        # and e.get("tag", "").lower() in {"a", "button", "input", ...}
    ]

    # 2. prioritize larger elements so they win ties
    candidates.sort(key=lambda e: -bbox_area(e["bbox"]))

    # 3. greedy non-overlap placement
    placed_px = []
    min_dist_sq = (radius * 2 * MIN_SPACING) ** 2
    mark_id = 0

    for el in candidates:
        if mark_id >= MAX_MARKS:
            break

        cx, cy = bbox_center(el["bbox"])
        px, py = round(cx * width), round(cy * height)

        if any((px - x) ** 2 + (py - y) ** 2 < min_dist_sq
               for x, y in placed_px):
            continue
        placed_px.append((px, py))

        # marker circle
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill="red",
            outline="white",
            width=1,
        )

        # label
        label = str(mark_id)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text(
            (px - tw / 2, py - th / 2 - 1),
            label,
            fill="white",
            font=font,
        )

        mark_id += 1

    return image