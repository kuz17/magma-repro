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
  exist.

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
render SoM overlay
    ↓
format Magma-style conversations
    ↓
write to conversations.jsonl  ← COMPLETE

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

Parameters (settled):
- radius = 9 px
- font = DejaVuSans-Bold, 12pt (fallback: PIL default)
- marker = red fill, white 1px outline, white numeric label
- MIN_AREA_FLOOR = 1e-6 (rejects only degenerate / zero-area bboxes)
- MAX_MARKS = 20
- MIN_SPACING = 1.3 (multiplier on marker diameter)

Status:
Implemented, validated on 10-sample batch, scaled to full 10k.

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
save SoM image + _marks.json sidecar

## Formatting pipeline

### Modules
- task_samplers.py — samples one grounding task per screenshot
- input_field.py  — handles input→point and input→bbox subtasks
- conversation.py — assembles turn-by-turn conversation structure
- formatter.py    — top-level: reads cleaned annotations, calls
                    renderer + sampler, writes output JSONL

### Grounding tasks and sampling weights
| Task         | Weight |
|--------------|--------|
| text→bbox    | 0.4    |
| text→point   | 0.4    |
| bbox→text    | 0.1    |
| point→text   | 0.1    |

Input field subtasks (input→point, input→bbox) sampled with equal
probability (0.5/0.5) and merged into the same conversation.

### Output schema (conversations.jsonl)
One JSON object per line, one object per screenshot:

{
  "image": "<img_filename>",
  "conversations": [
    {"from": "user",      "value": "<image>\n<task prompt>"},
    {"from": "assistant", "value": "<answer>"},
    ...                   // optional merged input-field turns
  ]
}

Output location: data/processed/seeclick_web/conversations.jsonl

Format validated against paper Figure 12.

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
  paper-quality renders.
- Considered dynamic / percentile-based `MIN_AREA`. Rejected in favor
  of static floor + MAX_MARKS, which is simpler and behaves correctly
  on both sparse and dense pages.
- SoM rendering is decoupled from formatter.py by design. Rendering
  happens as a separate preprocessing step; formatter reads pre-rendered
  images. This keeps the two concerns independent.

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
- full 10k SoM rendering
- Magma-style conversation formatting (all four grounding tasks +
  input field subtasks)
- conversations.jsonl written and validated against Figure 12
- train/val split (90/10, seed=42)
- UIAgent implementation (ui_agent.py)
- Eval harness (eval.py) — two modes: baseline and finetuned
- Baseline eval: 402/500 samples, click accuracy 1.7%
  (results/eval_baseline.json)
- finetune.py: QLoRA training script for Kaggle T4
- LoRA adapter trained (Kaggle T4, 3 epochs, ~8.9k samples)
  → models/lora_adapter/adapter_model.safetensors (148.7MB)
- OmniParser-v2.0 installed (models/omniparser/)
- click_visualizer.py demo pipeline: OmniParser + Qwen, baseline and
  finetuned modes, click-point rendering, confirmed accurate on real
  screenshots

Current focus:
- Run finetuned eval (eval.py --mode finetuned) to get delta
- Obtain Magma-8B reference numbers

---

## Experiment Pipeline (Phase 2)

### Design
Focused ablation: does SoM-formatted conversation training improve
spatial grounding in a small VLM?

```
conversations.jsonl
      ↓
train/val split (90/10, seed=42)
      ↓
┌─────────────────────┐    ┌──────────────────────┐
│ baseline eval       │    │ QLoRA fine-tune       │
│ (raw screenshot,    │    │ (trl.SFTTrainer,      │
│  no SoM, no adapter)│    │  Kaggle T4, 3 epochs) │
└──────────┬──────────┘    └──────────┬────────────┘
           │                          │ LoRA adapter (148.7MB)
           └──────────── delta ───────┘
                  click accuracy @ IoU 0.5
```

### Model
- Qwen/Qwen2.5-VL-3B-Instruct
- 4-bit NF4 quantization (bitsandbytes)
- Local path: models/qwen2_5_vl_3b/
- Coordinate output: normalized 0–1

### Eval modes
TWO modes — same metric (click accuracy), different inputs:

  baseline   Raw screenshot + native Qwen prompt (no SoM, no adapter)
             Tests: how good is the base model at UI grounding?

  finetuned  SoM image + SoM prompt + LoRA adapter
             Tests: does SoM fine-tuning improve grounding?

  Delta = finetuned − baseline = what SoM training actually adds.

### Eval metric
click accuracy = predicted point falls inside GT bbox

GT bbox sourced from:
1. text_to_bbox tasks: 4-coord bbox from assistant turn
2. text_to_point tasks: center point → resolved to actual element
   bbox via _marks.json sidecar lookup (gt_mark index)

Pixel-scale predictions (any value > 2.0) auto-normalized by
dividing by SoM image dimensions.

### Results

| Mode       | Samples evaluated | Click accuracy | IoU@0.5 | Mean IoU |
|------------|-------------------|----------------|---------|----------|
| baseline   | 402 / 500         | 1.7%           | 0.0%    | 0.005    |
| finetuned  | —                 | TBD            | TBD     | TBD      |
| Magma-8B   | —                 | TBD            | TBD     | TBD      |

### Results location
results/eval_baseline.json   ← complete
results/eval_finetuned.json  ← pending
results/eval_magma.json      ← pending

## Demo pipeline architecture

### click_visualizer.py
Full OmniParser → Qwen pipeline for live interaction.

Components:
- OmniParser: YOLO (icon_detect) + Florence-2 (icon_caption) + EasyOCR
  Detection thresholds: YOLO=0.10, OCR=0.75, IOU=0.4, MAX_ELEMENTS=35
  (YOLO lowered from 0.25 to catch more UI elements; OCR lowered from
  0.92 to surface placeholder text without producing noise)
- Qwen2.5-VL-3B-Instruct: 4-bit NF4, loaded from models/qwen2_5_vl_3b/
- Optional LoRA adapter: PeftModel wraps base model

Two modes:
  baseline (--tag baseline):
    OmniParser renders bbox-style SoM → Qwen receives screenshot
    + BASELINE_PROMPT_TEMPLATE → parser extracts "Coordinate: (x,y)"

  finetuned (--lora models/lora_adapter --training-style --tag finetuned):
    OmniParser detections re-rendered as training-style red circles
    → Qwen prompted with exact text_to_point training format
    → model responds "Coordinate: (cx, cy). Mark: N."
    → pipeline looks up Mark N's center from mark_to_center dict

Prompt templates:
  PROMPT_TEMPLATE (finetuned):
    'To execute the step "{task}", where do I direct my attention?
     Please provide the coordinate and the bounding box\'s mark index.'
  
  BASELINE_PROMPT_TEMPLATE:
    'On this software\'s interface, to execute the step "{task}",
     where do I direct my attention? Please provide the coordinate
     and the bounding box\'s mark index if applicable.'

Key design principle: mark-lookup, not coordinate prediction.
The model selects a mark by ID; the pipeline resolves the exact
pixel center from mark_to_center. Falls back to coordinate parsing
if mark lookup fails.

VLM input debug image: always saved to /tmp/vlm_input_{tag}.png
after each inference. Inspect this to see what marks OmniParser
placed before Qwen runs.

Usage:
  # fine-tuned
  python -m src.agent.click_visualizer \
    --image outputs/demo/screenshot.png \
    --interactive --lora models/lora_adapter \
    --tag finetuned --training-style

  # baseline
  python -m src.agent.click_visualizer \
    --image outputs/demo/screenshot.png \
    --interactive --tag baseline

---

### inference_server.py
FastAPI server wrapping DemoRunner. Runs as a separate process.

Start:
  python -m src.agent.inference_server \
    --mode finetuned --lora models/lora_adapter --port 8787

  python -m src.agent.inference_server --mode baseline --port 8787

Endpoints:
  GET  /health
  POST /act

#### POST /act request schema
```json
{
  "image_b64": "<base64-encoded PNG>",
  "task": "click the search bar",
  "dom_elements": [
    {
      "tag": "input",
      "type": "text",
      "label": "Search Amazon.in",
      "bbox_norm": [0.29, 0.02, 0.63, 0.07]
    }
  ]
}
```

dom_elements is optional (defaults to []). Populated by the web agent
from Playwright's live DOM before each inference call.

#### POST /act response schema
```json
{
  "click_norm": [0.46, 0.04],
  "mark_id": 0,
  "raw_response": "Coordinate: (0.46, 0.04). Mark: 0.",
  "elements": [
    {"id": 0, "content": "Search Amazon.in", "type": "input", "bbox": [...]},
    ...
  ],
  "error": null
}
```

click_norm: normalised [x, y] in [0,1], or null if no click resolved.
elements: list in content_list order (OmniParser order, NOT mark ID order).

---

### web_agent.py
Interactive Playwright agent wired to inference_server via HTTP.

Start:
  # inference server must already be running
  python -m src.agent.web_agent --url https://www.amazon.in

Per-step loop:
  1. Playwright screenshot (wait_stable)
  2. BrowserEnv.get_interactive_elements() → DOM element list
  3. POST /act with PNG + task + dom_elements
  4. Receive click_norm
  5. Playwright click at (x_norm * vp_w, y_norm * vp_h)

REPL commands:
  <task>             grounding task → screenshot → OmniParser → Qwen → click
  search <query>     DOM-direct: find text input, clear, type, submit (no VLM)
  go <url>           navigate (also accepts "go to <url>")
  back / go back     browser back (networkidle timeout silenced)
  type <text>        type at current focus
  enter              press Enter + wait for load
  scroll [up]        scroll 400px down (default) or up
  screenshot         save to outputs/browser/
  url                print current URL
  quit / exit        close browser and exit

search <query> command detail:
  1. get_interactive_elements() → first visible text input
  2. click(cx, cy)              → focus input
  3. press_key("Control+a")    → select-all to clear existing text
  4. type_text(query)
  5. press_key("Escape")       → dismiss autocomplete (prevents Enter interception)
  6. click submit button if found in DOM; else press_key("Enter")
  7. wait_for_load(10s)        → silenced on timeout (slow pages acceptable)

  Why DOM-direct: OmniParser + Qwen takes 90s for a task whose answer is
  already known from the DOM. VLM reserved for ambiguous element selection.

Known bugs fixed (2026-06-29):
  - "go back" now calls browser.back() instead of navigating to "https://back/"
  - back() no longer crashes on slow pages (networkidle timeout caught)

---

### DOM element injection
Problem: OmniParser cannot detect blank input fields (search bars with
no text, no icon). YOLO and OCR both miss them.

Solution: query the live DOM via Playwright JS before each inference.

BrowserEnv.get_interactive_elements() returns:
```python
[
  {
    "tag":       "input",
    "type":      "text",
    "label":     "Search Amazon.in",   # placeholder / aria-label / name / id
    "bbox_norm": [x1, y1, x2, y2]     # normalised to [0,1] over viewport
  },
  ...
]
```

Queried selectors:
  input:not([type="hidden"]), textarea, select, button,
  [role="searchbox"], [role="combobox"], [role="textbox"]

Invisible elements filtered: display:none, visibility:hidden, opacity:0,
elements outside viewport.

In training_style mode, DemoRunner re-renders the entire SoM image with
DOM elements first (_rebuild_som_dom_priority):

  Mark 0, 1, ... → DOM inputs / textareas    (text inputs first)
  Mark N, N+1, ... → DOM buttons
  Mark M, M+1, ... → top OmniParser elements
  Total capped at 15 marks

This converts the model's Mark:0 bias from a bug into a feature: the
search bar is always Mark 0, so "Mark: 0" responses click correctly.

DOM elements drawn with blue bounding-box outline + red circle marker
(blue distinguishes them from OmniParser detections visually).