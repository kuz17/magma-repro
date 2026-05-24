def denormalize_bbox(bbox, width, height):
    """
    Convert [0,1] normalized bbox to pixel coordinates.
    Uses round() to minimize sub-pixel drift at edges.
    """
    left, top, right, bottom = bbox

    return [
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    ]


def bbox_center(bbox):
    """
    Returns center point of bbox.

    Unit-agnostic:
    normalized input -> normalized center
    pixel input -> pixel center
    """
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def bbox_area(bbox):
    """
    Returns bbox area.

    Safe against malformed boxes.
    """
    x1, y1, x2, y2 = bbox

    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(box1, box2):
    """
    Computes Intersection over Union (IoU)
    between two bounding boxes.

    Returns value in [0,1].
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0
