# SeeClick-Web

## Source

https://github.com/njucckevin/SeeClick

## Dataset Type

GUI grounding / webpage interaction dataset.

Used for:
- UI grounding
- instruction following
- SoM prompting
- multimodal agent training

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
- UI screenshots are static images
- SoM is applied to UI screenshots
- ToM is NOT applied to UI screenshots

## Internal utilities

### bbox.py

Implemented:
- denormalize_bbox
- bbox_center
- bbox_area
- bbox_iou

### denormalize_bbox

Purpose:
Convert normalized coordinates into pixel coordinates for rendering.

Output:
[x1, y1, x2, y2] in pixel space.

### bbox_center

Purpose:
Compute bbox center point.

Used for:
- click-point generation
- SoM label placement
- nearest-element matching

### bbox_area

Purpose:
Compute bbox area safely.

Handles malformed boxes using:
max(0, dimension)

### bbox_iou

Intersection over Union:

IoU = intersection_area / union_area

Used for:
- duplicate filtering
- overlap analysis
- evaluation metrics
- annotation matching

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

## Planned SoM rendering pipeline

image
    +
elements
    ↓
draw bbox
    ↓
draw numeric marks
    ↓
save rendered image

## Current Project Status

Completed:
- environment setup
- dataset schema inspection
- bbox utilities
- unit tests

Current focus:
- first visual SoM rendering
