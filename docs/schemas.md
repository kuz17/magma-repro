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

Target style (reworked 2026-07-02 to match the paper's actual box-outline
style, replacing the earlier center-dot marker):
- red box outline around each placed element (`width=2`)
- numeric label placed at whichever of the box's 8 candidate corners
  overlaps least with other boxes/labels already drawn — ported from
  Magma's own `_find_least_overlapping_corner`
  (`agents/ui_agent/util/som.py` in `microsoft/Magma`)

Parameters (settled):
- font = DejaVuSans-Bold, 12pt
- label = red-filled rectangle behind white numeric text
- MIN_AREA_FLOOR = 1e-6 (rejects only degenerate / zero-area bboxes)
- MAX_MARKS = 100
- MIN_SPACING = 0.7 (multiplier on marker diameter; retuned from 1.3 for
  the box-outline style — see Decisions log)
- coordinates clamped to [0,1] before placement (handles OmniParser
  detections that exceed image bounds on edge-touching boxes)

Status:
Implemented, validated on 10-sample batch, scaled to full 10k (10,000/10,000
succeeded, 0 failures, 2026-07-02 — see `render_log.txt`).

## Paper-style SoM pipeline
image + elements
    ↓
filter degenerate bboxes (area > MIN_AREA_FLOOR)
    ↓
sort candidates by area descending
    ↓
clamp bbox coords to [0,1]
    ↓
greedy non-overlap placement, capped at MAX_MARKS
    ↓
draw box outline per placed element
    ↓
label each box at its least-overlapping corner
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
One task type is sampled **per screenshot** (not per element) and applied
to every element on that page, producing one turn pair per element:

| Task         | Weight |
|--------------|--------|
| text→bbox    | 0.4    |
| text→point   | 0.4    |
| bbox→text    | 0.1    |
| point→text   | 0.1    |

Input field subtasks (input→point, input→bbox) sampled with equal
probability (0.5/0.5) and appended as one additional turn pair — but only
when the page actually has `data_type == "input"` elements. Verified
empirically: SeeClick-Web has **zero** such elements across the full
113,142-element / 10k-sample corpus, so in practice this turn never
appears in the current corpus (reworked 2026-07-02; previously every
conversation ended on a "No input areas found." filler turn regardless).

**Downstream consequence for training/eval (confirmed 2026-07-02):**
because task type is sampled per-page rather than per-element, a page's
entire conversation is *either* grounding (`text→bbox`/`text→point`,
answers contain `Coordinate:`) *or* one of the other two types (no
`Coordinate:` anywhere on that page). Any pipeline that filters on
`Coordinate:` presence — the training dataset loader and the eval
harness both do — is therefore filtering at the *page* level, not
dropping individual elements within an otherwise-usable page. Of the
8,997 train pages, **728 are grounding-type** and eligible for the
current grounding-accuracy experiment; the remaining 8,269 are valid,
correctly-formatted `bbox→text`/`point→text` data that's simply out of
scope for this particular metric (see Fine-tuning section below).

### Output schema (conversations.jsonl)
One JSON object per line, one object per screenshot (reworked 2026-07-02:
one turn pair per element, matching paper Figure 12, rather than one
merged multi-line assistant turn covering every element):

{
  "image": "<img_filename>",
  "som_image": "<path to rendered SoM png, or null>",
  "conversations": [
    {"from": "user",      "value": "<image>\n<task prompt for element 0>"},
    {"from": "assistant", "value": "<answer for element 0>"},
    {"from": "user",      "value": "<task prompt for element 1>"},
    {"from": "assistant", "value": "<answer for element 1>"},
    ...                   // one pair per element, then optional
                           // input-field pair if input elements exist
  ]
}

Output location: data/processed/seeclick_web/conversations.jsonl
(regenerated 2026-07-02 from the reworked renders: 9,996 written, 4 skipped
empty, 0 skipped missing-sidecar; split 8,997 train / 999 val, seed=42).

Format validated against paper Figure 12.

**Historical note — the bug this format fixes:** the *previous* format
(pre-2026-07-02) merged every element on a page into a single assistant
turn (e.g. one long string: `"Coordinate: (...). Mark: 0.\nCoordinate:
(...). Mark: 1.\n..."`). Training on that format taught the model that
its answer always *starts* with `Mark: 0`, which is the confirmed root
cause of the "Mark: 0" collapse observed during web-agent testing (see
devlog.md, 2026-06-25 and 2026-07-02 continued). The one-turn-pair-per-
element rework, combined with a training script updated to use every
turn-pair (not just the first) and mask loss across all assistant spans,
is the fix — confirmed working via a controlled smoke test (see Results
below).

## Decisions log
- Tried fixed `MIN_AREA = 0.0005`: too aggressive on sparse pages
  (e.g. pages with only 1–2 small text links got 0 marks). Switched to
  `MIN_AREA_FLOOR = 1e-6` and rely on `MAX_MARKS` + non-overlap for
  clutter control.
- Tried `MIN_SPACING = 2.0`: dropped legitimate stacked nav items in
  vertical menus. Lowered to 1.3, then to 0.7 after switching from
  center-dot to box-outline markers (1.3 was tuned for the dot style and
  over-dropped once boxes replaced dots — see renderer section above).
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
- (2026-07-02) Considered training on the full 8,997-page corpus by
  including `bbox→text`/`point→text` pages. Rejected for the current
  ablation: those subtasks require a free-text loss target instead of
  `Coordinate:`/`Mark:`, and wouldn't move the grounding-accuracy metric
  this experiment reports. Logged as candidate future work (see
  TODO.md) rather than folded in under time pressure.

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
- full 10k SoM rendering (reworked box-outline style, re-rendered
  2026-07-02: 10,000/10,000 succeeded, 0 failures)
- Magma-style conversation formatting (all four grounding tasks +
  input field subtasks; reworked 2026-07-02 to one turn-pair per
  element, matching Figure 12)
- conversations.jsonl written and validated against Figure 12
  (regenerated 2026-07-02: 9,996 samples)
- train/val split (90/10, seed=42) (regenerated 2026-07-02: 8,997/999)
- UIAgent implementation (ui_agent.py)
- Eval harness (eval.py) — two modes: baseline and finetuned
- **Eval harness Mark-0 blind spot fixed (2026-07-02):**
  `extract_first_element` only ever read the first turn-pair per page,
  meaning ground truth was always Mark 0 — a collapsed model would score
  identically to a genuinely fixed one on this eval. Added
  `extract_all_elements()` (samples any turn-pair per page) and a
  `gt_mark==0` vs `gt_mark!=0` accuracy breakdown, which is the actual
  diagnostic for collapse (see Results below).
- Baseline eval (old, Mark-0-only methodology): 402/500 samples, click
  accuracy 1.7% (results/eval_baseline.json) — **SUPERSEDED, see Results**
- Baseline eval (fixed methodology): 80/100 samples, click accuracy 0.0%
  (results/eval_baseline_v2.json)
- finetune.py: original QLoRA training script for Kaggle T4 (local repo
  script) — superseded for this round by an updated Kaggle-notebook
  training pipeline (multi-turn-per-page dataset/collator); finetune.py
  itself not yet updated to match — see TODO.md
- LoRA adapter trained (Kaggle T4, 3 epochs, ~8.9k pages, old merged-
  mega-turn data) → models/lora_adapter/ (148.7MB) — **STALE**, root
  cause of Mark:0 collapse confirmed to be this adapter's training data
- Smoke-test LoRA adapter (200 pages, 25 steps, new multi-turn pipeline)
  → results/eval_smoke_finetuned.json: 27.5% overall, 22.6% on
  gt_mark!=0 — collapse fix CONFIRMED (non-zero-mark accuracy is not
  near-zero, which is what a still-collapsed model would show)
- Full retrain LoRA adapter (728 pages, 91 steps, same pipeline)
  → models/lora_adapter_v2/ — training COMPLETE, eval PENDING
- OmniParser-v2.0 installed (models/omniparser/)
- click_visualizer.py demo pipeline: OmniParser + Qwen, baseline and
  finetuned modes, click-point rendering, confirmed accurate on real
  screenshots (using the now-stale models/lora_adapter/ — needs
  re-pointing at models/lora_adapter_v2/ once full eval confirms it)

**Update 2026-07-09/11:** Full eval on `lora_adapter_v2` is now complete —
17.4% click accuracy, 10.3% on `gt_mark!=0` (see Results below). Two
further Kaggle retrain attempts (`lora_adapter_full`, `lora_adapter_full_v2`)
both regressed relative to it and are not currently explained.
`click_visualizer.py`'s DOM-priority rebuild now renders through the real
training renderer (`apply_som(preserve_order=True)`) instead of a diverged
dot-style drawer, DOM extraction now catches plain links/JS-clickables, and
point resolution is coordinate-first (nearest-mark snap) rather than
trusting literal "Mark: N" text. A planner+executor prototype
(`tests/planner_agent.py`) exists but is not yet validated end-to-end. A
possibly-distinct raw-coordinate bias (separate from the fixed Mark:0 text
collapse) is under investigation, uncommitted. See devlog.md 2026-07-02
(continued further) and 2026-07-09 entries for full detail.

Current focus:
- Decide on the `lora_adapter_full`/`lora_adapter_full_v2` regression:
  root-cause it (`sweep_checkpoints.py` exists, not yet run to completion)
  or report `lora_adapter_v2` as the result and move on
- Land the coordinate-bias investigation: run
  `smoke_coordinate_bias_live.py`, write up the conclusion, commit the
  three untracked smoke scripts
- Statistical validation pass on conversations.jsonl (still open, not
  blocking)
- Fix the `max_marks=15` cap risk for product/content links flagged by
  `smoke_dom_elements_search_page.py`
- Obtain Magma-8B reference numbers (live Kaggle run if time allows
  before the demo; otherwise cite paper-reported figures, clearly
  labeled as such rather than reproduced)

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
│  no SoM, no adapter)│    │  Kaggle T4)           │
└──────────┬──────────┘    └──────────┬────────────┘
           │                          │ LoRA adapter
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

**Element sampling (reworked 2026-07-02):** previously always evaluated
the first turn-pair on each page (`extract_first_element`), which is
almost always Mark 0 — meaning every eval sample's ground truth was
Mark 0, structurally unable to detect a model that always predicts
Mark 0. Now samples one turn-pair per page via
`extract_all_elements()` + `random.Random(1000 + idx)` (deterministic,
spread across every mark on the page). Results additionally broken down
by `gt_mark == 0` vs `gt_mark != 0` — a large gap where non-zero-mark
accuracy is near zero is the collapse signature; comparable accuracy
across both means the model isn't just defaulting to the first answer.

### Results

| Mode                                | Samples evaluated | Click accuracy | gt_mark==0 | gt_mark!=0 |
|--------------------------------------|--------------------|-----------------|------------|------------|
| baseline (old, Mark-0-only)         | 402 / 500          | 1.7% *(superseded)* | —      | —          |
| baseline (v2, fixed eval)           | 80 / 100           | 0.0%            | 0.0%       | 0.0%       |
| smoke finetuned (200 pages)         | 80 / 100           | 27.5%           | 37.0%      | 22.6%      |
| **full finetuned (728 pages, lora_adapter_v2)** | 161 / 200 | **17.4%**  | **35.6%**  | **10.3%**  |
| retrain attempt 2 (lora_adapter_full) | 161 / 200        | 2.5% *(regressed)* | 6.7%    | 0.9%       |
| retrain attempt 2, ckpt_800          | 47 / 60            | 4.3% *(regressed)* | —      | —          |
| retrain attempt 3 (lora_adapter_full_v2) | 161 / 200      | 11.2% *(regressed)* | 17.8% | 8.6%       |
| Magma-8B                            | —                  | TBD             | TBD        | TBD        |

**`lora_adapter_v2` (728-page retrain) is the reportable result.** Two
further retrain attempts on Kaggle (2026-07-02/03) both scored worse on
every metric — attempt 2 also spiked no-prediction rate to 25% (vs 3% for
`lora_adapter_v2`). Root cause of the regression is not diagnosed; training
happened on Kaggle notebooks with no log checked into this repo. See
devlog.md 2026-07-02/03 (continued further) for detail. Unless this gets
root-caused, cite `lora_adapter_v2` / `eval_full_finetuned.json`, not the
later attempts, in the writeup.

Interpretation: the smoke-test result (200-page) was the first evidence
that the Mark:0 collapse is fixed; the 728-page `lora_adapter_v2` result
confirms it holds at full retrain scale — a still-collapsed model would
show `gt_mark!=0` accuracy near 0% (predicting Mark 0's location essentially
never falls inside a different element's bbox by chance), and 10.3% is
clearly not that. The mark0-vs-non0 gap that persists across every
finetuned row (e.g. 35.6% vs 10.3% for `lora_adapter_v2`) is a plausible
ordinary pattern (header/logo/nav elements at Mark 0 tend to be more
visually distinctive), not the collapse signature.

### Results location
results/eval_baseline.json          ← superseded (Mark-0-only methodology)
results/eval_baseline_v2.json       ← complete (fixed methodology)
results/eval_smoke_finetuned.json   ← complete (200-page smoke test)
results/eval_full_finetuned.json    ← complete (lora_adapter_v2, 728-page retrain) — BEST RESULT
results/eval_true_full_finetuned.json ← complete (lora_adapter_full, attempt 2) — regressed
results/eval_ckpt_800.json          ← complete (attempt 2, step 800 checkpoint) — regressed
results/eval_full_finetuned_v2.json ← complete (lora_adapter_full_v2, attempt 3) — regressed
results/eval_magma.json             ← pending

**Gap:** none of the above JSONs record which adapter path produced them
(`eval.py` saves `run`/`mode` only) — the mapping above was reconstructed
from file mtimes, not read directly off disk. TODO: add an `"adapter"`
field to `eval.py`'s saved output.

### eval.py — compare() and per-page sampling (updated 2026-07-03)
`compare()` no longer takes a fixed baseline/finetuned pair; it accepts
`--jsons <path> [<path>...]` (any number of result files) and prints one
row per run, sorted by click accuracy, now including the `gt_mark==0`/
`gt_mark!=0` columns in the printed table (previously only in the saved
JSON, not surfaced in `compare()`'s output).

Per-page element sampling reseeded on the image filename stem
(`_stable_seed()`, md5-based) instead of the row index in val.jsonl —
index-based seeding wasn't a stable identity across runs if val.jsonl was
ever regenerated/reordered (it was, earlier in this project). **Runs from
before this patch and after it should not be mixed in one `compare()`
table** — the element sampled per page can differ.

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

  NOTE (2026-07-02): `--lora` currently points at the stale
  `models/lora_adapter/`. Repoint at `models/lora_adapter_v2/` once the
  full eval confirms it — see Current Focus above.

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

  NOTE (2026-07-02): same stale-adapter caveat as click_visualizer.py
  above — repoint --lora at models/lora_adapter_v2 once confirmed.

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

BrowserEnv.get_interactive_elements() returns (updated 2026-07-09 — see
below):
```python
[
  {
    "tag":       "input",
    "type":      "text",
    "label":     "Search Amazon.in",   # placeholder / aria-label / name / id / innerText
    "bbox_norm": [x1, y1, x2, y2]     # normalised to [0,1] over viewport
  },
  ...
]
```

Queried selectors (extended 2026-07-09):
  input:not([type="hidden"]), textarea, select, button,
  a[href], [role="searchbox"], [role="combobox"], [role="textbox"],
  [role="link"], [role="button"], [onclick]

**2026-07-09 change:** previously had no `<a>` tag or `[onclick]` in the
selector, meaning plain nav links (e.g. "Hello, sign in") and JS-driven
fake-buttons were invisible to DOM extraction entirely — root-caused as
the reason the agent couldn't target them. Also added: `disabled` /
`aria-disabled="true"` filtering, and an `innerText`/`textContent`
fallback (trimmed, capped 60 chars) for the `label` field, since links and
JS-clickables usually carry their label as visible text rather than
`placeholder`/`name`.

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

**2026-07-09 change:** the rebuild now renders through
`apply_som(preserve_order=True)` (`src.som.render_som`) — the same
box-outline + corner-label renderer used to produce training data —
instead of a separate dot-style drawer (`_draw_som_mark`/`_inject_dom_mark`)
that had silently diverged from the training format since the 2026-07-02
box-outline rework (that rework landed in `render_som.py` but was never
ported into this function until now). `apply_som` gained a
`preserve_order` param for this: when `True`, it skips the area-descending
sort and places elements in the order given, so DOM-priority ordering maps
directly onto the lowest mark IDs.

DOM elements drawn with blue bounding-box outline + box-outline mark
(previously a dot; see above).

**NOTE (2026-07-02):** this was a workaround built around the Mark:0
bias, not a fix for it — see devlog.md 2026-06-25. Now that the actual
root cause (training data format) is confirmed and fixed, it's worth
re-evaluating whether this rebuild is still necessary or can be
simplified once lora_adapter_v2 is validated (see Current Focus above).

**Known cap risk (2026-07-09/10):** product/content `<a>` links rank in
DOM-priority tier 2 (lowest, after inputs and buttons). On pages with many
buttons/inputs, `tests/smoke_dom_elements_search_page.py` confirmed
plausible product-title links can be pushed past `max_marks=15` before the
grounding model ever sees them as a mark. Not yet fixed.

### Point resolution — coordinate-first, mark-text fallback (2026-07-09)
`click_visualizer.py`'s `act()` previously trusted the model's literal
"Mark: N" text first, falling back to coordinate parsing only if no mark
was found. Given the documented residual Mark:0 bias in that literal text
channel, this let a biased mark ID override an otherwise-correct
coordinate guess. Flipped: parse the raw `(x, y)` first, snap it to the
nearest placed mark within `NEAREST_MARK_MAX_DIST = 0.15` (normalized
euclidean distance) via new `_nearest_mark()`, and only fall back to the
literal mark text if no coordinate parsed at all. Explicitly flagged in
code as a demo-time workaround, not a model fix — changes what "grounding
accuracy" measures for this checkpoint if these numbers ever feed
`eval.py` or the results table above.

### Planner + executor prototype (new, 2026-07-09)
`tests/planner_agent.py`: a minimal multi-step loop built on top of the
existing single-shot grounding pipeline. One model instance, one process —
role-switches via PEFT's `disable_adapter()` context manager instead of
loading two model copies or using the confirmed-buggy `load_adapter`/
`unload` path:

  Planner  = base Qwen (adapter disabled) → picks next action
  Executor = fine-tuned Qwen (adapter enabled) → grounds CLICK targets
             via the existing DemoRunner.act() path

Action grammar (regex-parsed from the planner's raw text response):
```
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")
```
`SEARCH` is handled DOM-direct (find first visible text input, clear,
type, submit) — same pattern as `web_agent.py`'s `search` command, not
routed through the VLM. `CLICK` calls `runner.act()` with live DOM
elements from `browser.get_interactive_elements()`. Loop runs up to
`MAX_STEPS = 6`, maintaining a `history` list appended to the planning
prompt each step so the model can (in principle) avoid repeating a
completed action.

**Status:** the `disable_adapter()` toggle mechanism is validated
(`tests/smoke_disable_adapter.py` — adapter-on / adapter-off /
adapter-on-again outputs compared, confirmed stable). The planning prompt
itself is still being iterated (`smoke_planning_prompt.py`,
`smoke_planning_prompt_case2.py`, `smoke_planning_v2.py`) — specifically
testing whether the model actually uses the `history` list rather than
just pattern-matching the goal text — but no end-to-end run of the full
loop against a live multi-step task has been recorded yet.

### Coordinate-bias investigation (in progress, uncommitted, 2026-07-09/11)
Manual live-agent testing surfaced a pattern distinct from the (already
-fixed) Mark:0 *text* collapse: the model's raw `(x, y)` coordinate output
appears to cluster around a near-fixed point for certain instruction
phrasings, largely independent of actual page content — e.g. "the yellow
Add to cart button" reportedly produced the same coordinate `(0.62, 0.71)`
on two visually different product pages. Three untracked scripts exist to
characterize this (`tests/smoke_coordinate_bias.py`, `smoke_coordinate_bias_live.py`,
`smoke_dom_elements_search_page.py`) — see devlog.md 2026-07-09 to 07-11
for detail. **Not yet confirmed as a root cause** — treat as a working
hypothesis pending review of the live-browser script's output.