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

### 2026-06-07

## Goal
Complete SeeClick-Web data pipeline end-to-end and pivot to model phase.

---

## Completed

### SoM Rendering — Full Scale
- Scaled paper-style SoM renderer to full 10k local image subset
- All 10k images rendered successfully
- SoM parameters unchanged from validated 10-sample config:
  radius=9, MIN_AREA_FLOOR=1e-6, MAX_MARKS=20, MIN_SPACING=1.3

### Formatting Pipeline
Implemented full Magma-style conversation formatter:

Modules:
- task_samplers.py  — samples one grounding task per screenshot
- input_field.py    — input→point and input→bbox subtasks
- conversation.py   — assembles turn-by-turn conversation structure
- formatter.py      — top-level pipeline: annotations → JSONL

Four grounding tasks implemented with paper-specified sampling weights:
- text→bbox   (0.4)
- text→point  (0.4)
- bbox→text   (0.1)
- point→text  (0.1)

Input field subtasks merged into the same conversation at equal weight.
Multiple subtasks from the same webpage merged into one example
(matches paper Section B.1.1).

Output:
- data/processed/seeclick_web/conversations.jsonl
- Format validated against paper Figure 12
- All 10k screenshots processed

### Project Pivot
SeeClick-Web data pipeline is complete. Project transitions from
data engineering to model work.

Key open question surfaced: reproduce the methodology (train own model
on generated data) vs reproduce the released results (validate against
Magma-8B weights). Decision needs mentor input before proceeding.

---

## Current Status

Pipeline stage: COMPLETE for SeeClick-Web

raw annotations
    ↓
local image filtering        ✓
    ↓
bbox utilities               ✓
    ↓
debug rendering              ✓
    ↓
paper-style SoM rendering    ✓  (10k images)
    ↓
Magma-style formatting       ✓  (conversations.jsonl)
    ↓
model / evaluation           ← next

---

## Next

### Immediate
- Statistical validation pass on conversations.jsonl:
  record count, task distribution (~40/40/10/10), bbox range checks,
  empty conversation check
- Clarify reproduction target with mentor:
  methodology reproduction vs result reproduction
- Clone official Magma repo and run SeeClick UI grounding demo
  to establish a concrete benchmark baseline number

### After mentor alignment
Option A — Full methodology reproduction:
- Move to ShareGPT4V + LLaVA-Instruct formatting (next pipeline phase)
- Then SeeClick-Mobile
- Then ToM generation (CoTracker-based)
- Then training run on full data mix

Option B — Small demo first:
- Fine-tune Phi-3.5-Vision or LLaVA-1.5-7B on conversations.jsonl
- Evaluate on SeeClick grounding subset
- Compare against Magma-8B reference numbers
- Use as proof-of-concept before requesting heavy compute

### Later
- Cloud infrastructure scale-up
- Docker containerization
- ToM generation pipeline (CoTracker)
- SeeClick-Mobile + Vision2UI formatting

---

## Notes
- UI screenshots use SoM only; ToM is not applicable
- Reproduction target (methodology vs results) is the key
  open question going into the model phase
- GPU resource request may be needed depending on mentor decision
- conversations.jsonl is the primary artifact from Phase 1