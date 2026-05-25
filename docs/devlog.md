# devlog.md
### Till 2026-05-25

## Goal
Bootstrap the Magma reproduction environment and complete the
SeeClick-Web preprocessing rendering pipeline.

---

## Completed

### Environment
- Created project structure
- Created Python virtual environment
- Installed core dependencies
- Added .gitignore
- Initialized git repository
- Connected local repo to GitHub
- Configured pytest infrastructure
- Created raw/interim/processed data layout

### HuggingFace
- Logged into HuggingFace CLI
- Investigated dataset access failure
- Determined SeeClick is not hosted as a HuggingFace dataset package

### Dataset Research
Located official SeeClick repository:
https://github.com/njucckevin/SeeClick

Identified:
- 10k webpage screenshot subset
- full annotation metadata source
- bbox schema format

### Dataset Inspection
Loaded annotation metadata locally.

Observed:
- 271121 annotation samples
- 10000 local webpage screenshots

Discovered:
- annotations and local images are mismatched subsets
- filtering by image existence is required

Confirmed:
- bbox format = [left, top, right, bottom]
- coordinates normalized to [0,1]
- multiple UI elements per screenshot
- nested element structure
- element types:
  - text  (~95%)
  - hover (~5%)

Confirmed via schema inspection:
- no HTML tag field present (schema is instruction / bbox / data_type)
- dataset is pre-filtered upstream to actionable grounding targets

### Utilities
Implemented:
- denormalize_bbox
- bbox_center
- bbox_area
- bbox_iou

Added:
- unit tests
- pytest configuration
- package structure (__init__.py)

### Testing
- All bbox utility tests passing

### Cleaning Pipeline
Implemented:
- local image existence filtering

Generated:
- filtered annotation subset matching local images

### Rendering — Debug Renderer
Implemented first debug renderer:
- bbox overlays
- numeric labels
- screenshot rendering

Validated visually:
- normalized → pixel conversion
- annotation parsing
- image matching
- geometric alignment correctness

Observed:
- dense annotation clutter on some webpages
- need for paper-style compact SoM markers

### Rendering — Paper-style SoM Renderer
Implemented compact SoM renderer:
- numbered markers at bbox centers
- red fill + white outline, white numeric label
- radius = 9 px, DejaVuSans-Bold 12pt (PIL default fallback)

Implemented clutter control:
- MIN_AREA_FLOOR = 1e-6 (rejects only degenerate bboxes)
- area-descending sort so larger elements win placement ties
- greedy non-overlap rejection (MIN_SPACING = 1.3 × diameter)
- MAX_MARKS = 20 hard cap

Validated on 10-sample batch:
- marker density now matches paper-style reference
- sparse pages render correctly (no over-imposed marks)
- dense pages render legibly (no overlapping markers)

Decisions recorded:
- rejected fixed MIN_AREA (over-aggressive on sparse pages)
- rejected MIN_SPACING ≥ 2.0 (dropped legitimate stacked nav items)
- rejected tag-based actionability filter (schema has no tag field;
  dataset already filtered upstream)
- rejected semantic/saliency scoring (not applicable to pre-curated
  data; relevant later for ScreenSpot / OmniParser evaluation)
- rejected dynamic/percentile MIN_AREA (static floor + MAX_MARKS
  behaves correctly on both sparse and dense pages)

---

## Current Status

Current pipeline stage:

raw annotations
    ↓
local image filtering
    ↓
bbox utilities
    ↓
debug rendering
    ↓
paper-style SoM rendering  ← validated on 10-sample batch
    ↓
formatted training samples (next)

---

## Next

### Immediate
- Scale SoM rendering to larger batches (100, then 1000) for sanity
  checking and spot-check sampling
- Decide on radius / font scaling policy for screenshots that deviate
  significantly from typical resolution

### After validation
- Generate instruction ↔ mark index mappings
- Create Magma-style formatted training samples (text2point /
  text2bbox / point2text / bbox2text per the Vision2UI subtasks)
- Run preprocessing over full local 10k subset
- Save finalized artifacts into data/processed/

### Later
- Move into the code-study phase of the Magma reproduction roadmap

---

## Notes
- UI screenshots use SoM only
- ToM is not applicable for discrete UI screenshots
- Current focus is preprocessing and dataset engineering
- Model training has NOT started yet
- Rendering pipeline is now validated; next bottleneck is converting
  rendered samples into training-format instruction data