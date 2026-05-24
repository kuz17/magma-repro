from src.utils.bbox import (
    denormalize_bbox,
    bbox_center,
    bbox_area,
    bbox_iou
)


def test_denormalize_bbox():
    bbox = [0.1, 0.2, 0.5, 0.6]

    result = denormalize_bbox(bbox, 1000, 500)

    assert result == [100, 100, 500, 300]


def test_bbox_center():
    bbox = [0, 0, 10, 20]

    center = bbox_center(bbox)

    assert center == (5, 10)


def test_bbox_area():
    bbox = [0, 0, 10, 20]

    assert bbox_area(bbox) == 200


def test_bbox_area_invalid():
    bbox = [10, 10, 5, 5]

    assert bbox_area(bbox) == 0


def test_bbox_iou_same():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]

    assert bbox_iou(box1, box2) == 1.0


def test_bbox_iou_no_overlap():
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]

    assert bbox_iou(box1, box2) == 0.0
