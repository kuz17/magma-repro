# schemas.md
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

## Dataset Status

### Local Assets
Downloaded:
- 10k webpage screenshot subset
- full annotation metadata

Observed:
- annotation metadata contains 271121 webpage samples
- local image subset contains 10000 images

Therefore:
- annotation filtering by local image existence is required

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
Observed across first 1000 samples:
- text  (18437, ~95%)
- hover (1066,  ~5%)

Both are valid grounding targets and are kept during rendering.

## Real observed sample schema
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

## Observed dataset properties
- Multiple elements per webpage screenshot
- Some webpages contain dense annotations
- Elements include:
  - navigation items
  - buttons
  - search bars
  - links
  - text blocks
  - hover targets

## Notes
- Bounding boxes are already normalized
- Need denormalization before rendering SoM overlays
- UI screenshots are static images
- SoM is applied to UI screenshots
- ToM is NOT applied to UI screenshots
- **Dataset is pre-filtered to actionable grounding targets.** Every
  element has an associated `instruction`, and there are no wrapper-div
  or layout-container entries polluting the candidate list. This means
  area-based filtering in the SoM renderer is a stylistic clutter
  control, not a correctness requirement.
- **No HTML tag information.** Schema is `instruction` / `bbox` /
  `data_type` only. Tag-based actionability scoring (button / a /
  input) is not applicable to this dataset — those fields do not
  exist. Saliency scoring would be relevant for downstream evaluation
  on raw proposals (e.g. OmniParser on ScreenSpot), not for SeeClick
  pretraining renders.

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
- SoM marker placement
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

## Current preprocessing pipeline
raw annotations
    ↓
filter images that exist locally
    ↓
load screenshot
    ↓
denormalize bbox
    ↓
render debug overlays
    ↓
visual verification

## Renderer architecture

### Debug renderer
Purpose:
Geometry validation and annotation debugging.

Style:
- visible rectangles
- visible labels
- dense overlays

Status:
Implemented and visually validated.

### Paper-style SoM renderer
Purpose:
Training-style Set-of-Mark representation.

Target style:
- compact numbered markers
- minimal visual obstruction
- marker placement at bbox centers

Parameters (current):
- radius = 9 px
- font = DejaVuSans-Bold, 12pt (fallback: PIL default)
- marker = red fill, white 1px outline, white numeric label
- MIN_AREA_FLOOR = 1e-6 (rejects only degenerate / zero-area bboxes)
- MAX_MARKS = 20
- MIN_SPACING = 1.3 (multiplier on marker diameter)

Status:
Implemented and visually validated on 10-sample batch.

## Paper-style SoM pipeline
image + elements
    ↓
filter degenerate bboxes (area > MIN_AREA_FLOOR)
    ↓
sort candidates by area descending
    ↓
greedy non-overlap placement, capped at MAX_MARKS
    ↓
draw numbered markers at bbox centers
    ↓
save SoM image

## Decisions log
- Tried fixed `MIN_AREA = 0.0005`: too aggressive on sparse pages
  (e.g. pages with only 1–2 small text links got 0 marks). Switched to
  `MIN_AREA_FLOOR = 1e-6` and rely on `MAX_MARKS` + non-overlap for
  clutter control.
- Tried `MIN_SPACING = 2.0`: dropped legitimate stacked nav items in
  vertical menus. Lowered to 1.3.
- Considered tag-based actionability filtering (keep only
  `a` / `button` / `input`). Not applicable: SeeClick schema has no
  tag field, and the dataset is already filtered to interactive
  grounding targets upstream.
- Considered semantic/saliency scoring to favor buttons/inputs over
  large containers. Rejected for this dataset: it has no tag metadata
  to score on, and area-descending sort + MAX_MARKS already produces
  paper-quality renders. Saliency scoring belongs in the
  ScreenSpot/OmniParser evaluation pipeline where raw proposals lack
  curation.
- Considered dynamic / percentile-based `MIN_AREA`. Rejected in favor
  of static floor + MAX_MARKS, which is simpler and behaves correctly
  on both sparse and dense pages.

## Current Project Status

Completed:
- environment setup
- dataset schema inspection
- annotation loading
- local image filtering
- bbox utilities
- unit tests
- debug renderer
- first visual alignment validation
- paper-style SoM renderer
- 10-sample SoM batch visual validation

Validated:
- normalized → pixel conversion
- annotation parsing
- image matching
- geometric correctness
- SoM marker placement and density

Current focus:
- scale SoM rendering to larger batches for sanity checking
- decide on radius/font scaling policy for screenshots that deviate
  significantly from typical resolution