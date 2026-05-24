# SeeClick-Web

## Source
https://github.com/njucckevin/SeeClick

## Coordinate format

bbox = [left, top, right, bottom]

Normalized coordinates in [0,1].

Coordinates represent ratios relative to image width/height.

Example:

bbox = [0.1, 0.2, 0.5, 0.6]

For image:
- width = 1000
- height = 500

Pixel bbox becomes:
- left = 100
- top = 100
- right = 500
- bottom = 300

## Data types

- text
- hover

## Sample schema

{
  "img_filename": "...",
  "url": "...",
  "elements": [
    {
      "instruction": "...",
      "bbox": [...],
      "data_type": "text" | "hover"
    }
  ]
}

## Notes

- Multiple elements per webpage screenshot
- Bounding boxes are already normalized
- Need denormalization before rendering SoM overlays
- UI screenshots are static images (SoM only, no ToM)

## Internal utilities

### bbox.py

Implemented:
- denormalize_bbox
- bbox_center
- bbox_area

Planned:
- bbox_iou

## Planned preprocessing pipeline

raw annotations
    ↓
validate bbox format
    ↓
load screenshot
    ↓
denormalize bbox
    ↓
apply SoM rendering
    ↓
generate training sample
