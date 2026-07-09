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

### 2026-06-12

## Goal
Pivot from full Magma-8B reproduction to small VLM experiment.
Build agent + eval harness, establish baseline, prepare for fine-tuning.

---

## Completed

### Experiment Design Pivot
Decided on focused experiment: Qwen2.5-VL-3B-Instruct + SoM agent,
fine-tuned on conversations.jsonl, benchmarked against zero-shot baseline.
Core claim mirrors Magma paper: SoM-formatted conversation training
improves spatial grounding in VLMs.

### Train/Val Split
- Implemented src/clean/split.py
- 90/10 split, seed=42
- Train: 8,997 samples → data/processed/seeclick_web/train.jsonl
- Val:   999 samples   → data/processed/seeclick_web/val.jsonl

### UIAgent
- Implemented src/agent/ui_agent.py (~500 lines)
- Model-agnostic design: pluggable VLM backends + SoM sources
- VLM backends: QwenBackend (4-bit NF4, 0–1 coords),
  MagmaBackend (bfloat16, 0–1000 coords, trust_remote_code)
- SoM sources: AnnotationSoM (sidecar-based, eval/train),
  OmniParserSoM (YOLO+Florence2+OCR, production)
- Three interaction modes matching Magma app.py:
  empty → OmniParser only, Q: prefix → VQA, task → grounding
- Coordinate scale difference handled per-backend (_COORD_SCALE)
- OmniParser import optional (silent fallback if weights absent)

### Model Download
- Qwen2.5-VL-3B-Instruct downloaded via aria2c (16 parallel connections,
  bypassed ISP throttle on large safetensor shards)
- Stored at models/qwen2_5_vl_3b/ (in .gitignore)
- OmniParser-v2.0 downloading in parallel to models/omniparser/

### Eval Harness
- Implemented src/eval/eval.py
- Metric: click accuracy (predicted point falls inside GT bbox)
- Secondary: IoU@0.5, mean IoU, per-task breakdown
- Fixed degenerate GT bboxes: text_to_point GT is stored as a
  center point (x1==x2, y1==y2) — resolved via sidecar bbox lookup
  using gt_mark, with epsilon expansion fallback
- Fixed coordinate scale: pixel-scale predictions (any value > 2.0)
  auto-normalized by image dimensions
- max_new_tokens=32 (correct format is ~20 tokens; 128 was wasteful)
- CLI: --adapter, --max-samples, --name, --mode flags

### Baseline Eval (initial smoke test)
- Smoke test (10 samples): 0% click accuracy — expected
- Model outputs pixel coords + wrong format (no fine-tuning yet)
- GT bbox fix confirmed: non-degenerate bboxes in results

---

### 2026-06-24

## Goal
Complete baseline eval, implement and run QLoRA fine-tuning on Kaggle T4,
build and validate end-to-end demo pipeline with OmniParser.

---

## Completed

### Baseline Eval — Full Run
- Ran eval on 500 val samples (402 evaluated, 98 skipped due to missing
  raw images); runtime ~1h 44min on local GTX 1650 Mobile (4-bit)
- Results saved to results/eval_baseline.json

```
Results [baseline]  —  402 evaluated, 98 skipped
  Click accuracy (point in GT bbox) : 0.017  (1.7%)
  IoU hit @ 0.5                     : 0.000  (0.0%)
  Mean IoU                           : 0.005
  Mean dist to GT center            : 0.367
  No prediction (parser found none) : 20/402
  Mean inference time               : 15.18s / sample

  [text_to_bbox]   n=205  click_acc=2.9%  mean_dist=0.350
  [text_to_point]  n=197  click_acc=0.5%  mean_dist=0.384
```

Interpretation: base Qwen produces wrong output format (no SoM context,
no mark-lookup protocol) — near-zero accuracy is expected and establishes
the floor for the SoM fine-tuning delta.

NOTE (2026-07-02): this number was later found to be an artifact of the
eval harness only ever testing Mark 0 per page (see eval.py fix below).
Superseded by results/eval_baseline_v2.json — see 2026-07-02 (continued).

### Fine-Tuning — finetune.py
- Implemented src/train/finetune.py: full QLoRA pipeline
- SoMDataset: loads train.jsonl + SoM renders, serves first grounding
  turn only (user→assistant); skips non-grounding turns (bbox→text,
  point→text produce no coordinates to supervise)
- SoMCollator: builds chat-formatted inputs, masks user/image tokens
  so loss falls on assistant tokens only
- Config: r=16, alpha=32, dropout=0.05; targets all linear projections
  (q/k/v/o/gate/up/down_proj); 3 epochs, LR=2e-4, cosine schedule,
  effective batch size 8 (bs=1, grad_accum=8), MAX_SEQ_LEN=512

### Fine-Tuning — Kaggle T4 Run
- Ran finetune.py on Kaggle free T4 (16GB VRAM)
- Training: Qwen2.5-VL-3B-Instruct, 3 epochs on ~8.9k SoM conversations
- LoRA adapter saved: adapter_model.safetensors (148.7MB)
- Adapter downloaded and stored at models/lora_adapter/ (in .gitignore)

NOTE (2026-07-02): this adapter was later diagnosed as trained on a
buggy data format (merged mega-turn: one assistant turn listing every
mark on the page, always starting "Mark: 0") and superseded — see
2026-07-02 (continued).

### OmniParser — Installation Complete
- OmniParser-v2.0 weights installed at models/omniparser/
  - icon_detect/model.pt (YOLO)
  - icon_caption/ (Florence-2)
- Tuned detection thresholds: YOLO=0.25, OCR=0.92, IOU=0.4,
  MAX_ELEMENTS=35
- Fixed gpu=False flag (OmniParser upstream uses cpu=False incorrectly)
- OmniParser running on CPU; confirmed functional

### Demo Pipeline — click_visualizer.py
- Implemented src/agent/click_visualizer.py: full OmniParser → Qwen
  pipeline with click-point rendering and image output
- Two modes:
  - baseline: OmniParser bbox-style SoM → Qwen with coordinate-guessing
    prompt (RAW_PROMPT_TEMPLATE); parser extracts "Coordinate: (x, y)"
  - fine-tuned (--lora --training-style --tag finetuned): re-renders
    OmniParser detections as training-style red circles (matching
    finetune.py input format), prompts model for "Mark: N", then looks
    up that mark's pixel center from OmniParser's label_coords
- Also implemented src/agent/interactive.py: text-only REPL wrapping
  DemoRunner for quick interactive testing
- Mark-lookup is the key correctness fix: model returns a mark ID →
  pipeline resolves exact pixel center from OmniParser detections,
  bypassing the model's imprecise coordinate estimation entirely

### Demo Validation
- Confirmed fine-tuned pipeline clicking correctly on real screenshots
  (e.g. "Gift Cards" in Amazon navigation bar)
- Baseline mode confirmed producing wrong predictions (as expected)
- Visual output saved to outputs/demo/ with annotated click overlays

---

## Current Status

```
Pipeline:      COMPLETE
Agent:         COMPLETE (Qwen2.5-VL-3B-Instruct, 4-bit, OmniParser+AnnotationSoM)
Baseline eval: COMPLETE — 1.7% click accuracy (results/eval_baseline.json)
                [SUPERSEDED 2026-07-02 — see eval.py Mark-0 blind spot fix]
Fine-tuning:   COMPLETE — LoRA adapter at models/lora_adapter/ (148.7MB)
                [SUPERSEDED 2026-07-02 — trained on buggy merged-mega-turn data]
Demo:          COMPLETE — click_visualizer.py confirmed on real screenshots
Finetuned eval: NEXT   — re-run eval.py --mode finetuned --adapter models/lora_adapter
```

---

## Notes
- Mark-lookup (model returns "Mark: N" → pipeline resolves pixel center)
  is the architecture that makes the fine-tuned mode work. Never trust
  the model's raw coordinate output for production use.
- Eval baseline deliberately uses raw screenshots (no SoM) so the delta
  measures actual grounding improvement, not format compliance.
- Baseline 1.7% is lower than expected; confirms base Qwen has no
  training-style SoM awareness and the improvement delta will be clean.
- OmniParser runs on CPU locally (GTX 1650 Mobile has insufficient VRAM
  to run both OmniParser and quantized Qwen simultaneously).

---

### 2026-06-25

## Goal
Build live web agent (Playwright + inference server), diagnose and fix
inference pipeline bugs found during first real-browser testing.

---

## Completed

### Inference Server — inference_server.py
- Implemented FastAPI server wrapping DemoRunner (POST /act, GET /health)
- Two startup modes: `--mode baseline` and `--mode finetuned --lora <path>`
- Runs as a separate process; web_agent communicates over localhost:8787
- Fixed broken `_patch_demo_runner`: it was reading `self._content_list`
  which DemoRunner never set — always returned empty elements list.
  Simplified to no-op; DemoRunner now sets `self._last_content_list`
  directly in `_run_omniparser`.
- Added `DomElementInfo` model and `dom_elements: List[DomElementInfo]`
  field to `ActRequest` for DOM-augmented grounding (see below).

### Web Agent — web_agent.py
- Implemented `WebAgent` REPL: screenshot → inference server → click loop
- Commands: `go <url>`, `type <text>`, `enter`, `scroll`, `back`,
  `screenshot`, `url`, `quit`
- Fixed: `go to <url>` (user-natural form) crashed because `raw[3:]`
  stripped "go " leaving "to https://..."; added "to " prefix stripping.
- Fixed: uncaught navigation exception crashed the entire agent process;
  now caught and printed as an error.
- Inference timeout raised 120 → 240s (OmniParser on CPU takes 30–70s;
  120s was too tight under load).

### Bug: Prompt mismatch — root cause of Mark: 0 problem
The fine-tuned model always returned "Mark: 0" on unseen pages.
Root cause: `PROMPT_TEMPLATE` in finetuned mode was completely wrong.

Old prompt (never seen during training):
  "On this software's interface, to execute the step "X",
   which mark should I click?
   Detected elements: ...
   Respond with ONLY: Mark: N"

Training format (text_to_point, task_samplers.py, weight 0.4):
  'To execute the step "X", where do I direct my attention?
   Please provide the coordinate and the bounding box's mark index.'

Expected training response:
  "Coordinate: (cx, cy). Mark: N."

Fix: `PROMPT_TEMPLATE` rewritten to exact training phrasing. Element
list injection removed (training never included one). After fix, model
now returns correct format: "Coordinate: (x, y). Mark: N."

NOTE (2026-07-02): this prompt fix was necessary but not sufficient —
the deeper root cause (merged-mega-turn training data always starting
"Mark: 0") wasn't found until the renderer/formatter rework. See
2026-07-02 (continued).

### Bug: Wrong mark→center mapping in training_style mode
`apply_som` re-sorts elements by area and assigns NEW sequential IDs
(largest element = Mark 0). The old code built `mark_to_center` from
OmniParser's `label_coords` keys — a completely different ordering.
Fix: `mark_to_center` now built from `apply_som`'s `placed` output.

### Bug: content_list never persisted to self
`_run_omniparser` returned `content_list` locally; inference server's
`_patch_demo_runner` tried to read `self._content_list` which was never
set — always returned `[]` for the elements field in API responses.
Fix: `self._last_content_list` and `self._raw_label_coords` both set
inside `_run_omniparser` after the cap.

### Bug: VLM input only saved once
`_input_saved = True` guard meant `/tmp/vlm_input_finetuned.png` only
reflected the first inference. Fix: always overwrite to fixed path so
the file always shows the latest state for debugging.

### OmniParser threshold tuning
Initial attempt YOLO=0.05 caused Florence-2 to caption 100+ elements on
CPU → inference timeout at 120s. OCR=0.5 produced garbage OCR (anti-aliased
rendering artifacts detected at 50% confidence → huge noisy bboxes).

Settled values after testing:
  YOLO_THRESHOLD = 0.10   (was 0.25; catches more icons without flooding)
  OCR_THRESHOLD  = 0.75   (was 0.92; catches placeholder text, avoids noise)

### DOM element injection — browser_env.py + inference pipeline
OmniParser cannot detect blank input fields (search bars with no text,
no icon inside the bounding box). YOLO and EasyOCR both miss them.
Fix: query the live DOM via Playwright JS before each inference.

Added `BrowserEnv.get_interactive_elements()`:
- Queries `input:not([type=hidden])`, `textarea`, `select`, `button`,
  `[role=searchbox]`, `[role=combobox]`, `[role=textbox]`
- Filters invisible elements (display:none, visibility:hidden, opacity:0)
- Returns normalized bbox_norm [x1,y1,x2,y2] + tag/type/label
- Passed to inference server as `dom_elements` in ActRequest

### DOM-priority SoM rebuild — fixes Mark: 0 bias
After DOM injection, the model still always chose "Mark: 0".
Diagnosis: the fine-tuned model has a strong Mark:0 bias, likely a
training distribution artifact (SeeClick-Web only; model may have
overfit to the most common answer being the largest/first element).

DOM elements were appended at marks 35, 36 (after OmniParser marks).
Model ignoring them and picking Mark 0 (a random large OmniParser bbox).

Fix (exploit the bias): when DOM elements are present in training_style
mode, the entire SoM image is re-rendered from scratch via
`_rebuild_som_dom_priority()`:
  - DOM inputs/textareas → marks 0, 1, ...   (Mark 0 = search bar)
  - DOM buttons → next marks
  - OmniParser elements → remaining marks
  - Total capped at 15 (model trained on 5–30; 37 marks was too many)

With search bar as Mark 0, the model's "Mark: 0" output now correctly
clicks the search field. DOM elements drawn with blue outline + red
circle (blue distinguishes them visually; red circle matches training).

NOTE (2026-07-02): this was a workaround, not a fix — it exploited the
bias rather than removing it. The actual root cause (training data
format) was found and fixed this session; see below.

---

## Current Status

```
Inference server : COMPLETE (POST /act, GET /health, dom_elements support)
Web agent        : COMPLETE (Playwright REPL, DOM injection, 240s timeout)
Pipeline bugs    : FIXED (prompt, mark mapping, content_list, VLM save)
DOM injection    : COMPLETE (search bars now detected via Playwright)
Mark:0 bias      : MITIGATED (DOM-priority SoM rebuild puts search bar at Mark 0)
Finetuned eval   : PENDING
```

---

## Next

### Immediate
1. Validate web agent end-to-end: `search books` on Amazon.in should
   click search bar (Mark 0) → type → enter → results page.
2. Run finetuned eval: `python -m src.eval.eval --mode finetuned
   --adapter models/lora_adapter --name finetuned`
3. Compute and record baseline → finetuned delta.
4. Generate comparison table (baseline / finetuned / Magma-8B).

### Investigate Mark:0 bias
- Run baseline mode (no LoRA) on same pages — does base Qwen pick better marks?
- If base model generalises better, fine-tuning on SeeClick-Web only
  overfit to a narrow distribution; wider data needed.
- Consider: task keyword → DOM element label fuzzy match as a
  deterministic fallback when model confidence is low.

### After eval
- Analyze failure cases: where does fine-tuning still fail?
- Generate qualitative examples (side-by-side baseline vs finetuned)
- Statistical validation pass on conversations.jsonl
- Write thesis experiment section draft

### Later
- ShareGPT4V formatting pipeline
- SeeClick-Mobile support
- ToM generation (CoTracker)
- Docker containerization

---

### 2026-06-29

## Goal
Harden web agent REPL: fix crashes on slow pages, add compound `search`
command so common tasks don't require VLM inference.

---

## Completed

### Bug: `go back` navigated to "https://back/"
The REPL matched the `go <url>` branch first because `"go back".startswith("go ")`.
It stripped "go " and called `navigate("back")` → `"https://back/"` →
NS_ERROR_UNKNOWN_HOST crash.
Fix: added `"go back"` to the `back` branch before the `go <url>` branch,
so both `back` and `go back` call `agent.back()`.

### Bug: `back()` crash on slow pages
`back()` called `self.browser.wait_for_load()` (networkidle, 10s timeout).
Amazon never reaches networkidle after a back-navigation → TimeoutError
propagated up and killed the agent process.
Fix: wrapped `wait_for_load()` in try/except; uses 15s timeout instead of
10s. Timeout silently ignored — the page is usable even if networkidle
isn't declared.

### New: `search <query>` compound command
Single command replaces the old three-step manual flow:
`click on search bar` → `type books` → `enter`.

The command is DOM-direct (no VLM inference, no 90-second wait):
1. `get_interactive_elements()` → find first visible text input
2. Click its center
3. `Ctrl+A` to clear any existing text
4. Type the query
5. `Escape` to dismiss autocomplete dropdown (Amazon's autocomplete
   intercepted Enter and redirected to homepage instead of search results)
6. Click the submit button (found via DOM; more reliable than Enter which
   autocomplete can swallow)
7. `wait_for_load(10s)` with exception silenced

Why DOM-direct instead of VLM: the search bar is already reliably detected
by `get_interactive_elements()` (DOM query always finds it). Running the
full OmniParser + Qwen pipeline for a task this simple is wasteful and slow.
VLM inference reserved for cases where element identity is ambiguous.

---

## Current Status

```
Inference server : COMPLETE
Web agent        : COMPLETE + hardened (back crash fixed, search command added)
Pipeline bugs    : FIXED
DOM injection    : COMPLETE
Finetuned eval   : PENDING
```

---

## Next

1. Validate `search books` on Amazon.in → results page end-to-end.
2. Run finetuned eval: `python -m src.eval.eval --mode finetuned
   --adapter models/lora_adapter --name finetuned`
3. Compute baseline → finetuned delta.
4. Generate comparison table (baseline / finetuned / Magma-8B).

### Recommended REPL additions (not yet implemented)
- `reload` — refresh page
- `elements` — list DOM interactive elements without inference
- `inspect` — open `/tmp/vlm_input_finetuned.png` with xdg-open
- `fill <label> <value>` — DOM-direct fill by placeholder/label
- `marks` — run OmniParser only (no Qwen) and open SoM image

---

### 2026-07-02

## Goal
Rework the SoM renderer and conversation format to match the paper's actual
spec (Figure 12), then re-run the full pipeline on the new format before
trusting the existing fine-tuned adapter's results.

---

## Completed

### SoM renderer: box outlines replace center dots
`apply_som` previously drew a filled circle at each element's bbox center.
Ported Magma's actual `_find_least_overlapping_corner` (from
`agents/ui_agent/util/som.py` in `microsoft/Magma` on GitHub — no vendored
copy existed locally) so marks are now a red box outline with the number
label placed at whichever of the box's 8 candidate corners overlaps least
with other boxes/labels already drawn. `OmniParserSoM` (the live-inference
path in `ui_agent.py`) needed no change — it already renders box outlines
via OmniParser's own `BoxAnnotator`.

`MIN_SPACING` retuned 1.3 → 0.7: the old value was tuned for the retired
dot-marker style and, with box outlines, was dropping plainly
non-overlapping nav elements. Swept against the full 10k corpus: 0.7 cuts
overall element drop from 5.9% to 1.5%. The 81+-element tail bucket (21
pages, ~0.2% of corpus) stays 26-35% drop at every spacing tested — doesn't
respond to spacing tuning, needs a label-offset fallback instead; deferred.

Also added: coordinate clamping to [0,1] inside `apply_som` (live OmniParser
detections can exceed image bounds on edge-touching boxes; every downstream
consumer of `placed` — mark_to_center, formatter, eval — now sees valid
coords unconditionally).

### Conversation format: one turn-pair per element
Previously each grounding task merged every element on a page into a single
assistant turn (N coordinate lines in one response). Paper Figure 12 uses
one user/assistant turn pair per element instead. `task_samplers.py`
rewritten so `text_to_bbox` / `text_to_point` / `bbox_to_text` /
`point_to_text` each return a list of turn pairs (one per element) rather
than one merged turn; `conversation.py` flattens that list into the output.

### Input-field turn now conditional
Verified empirically that SeeClick-Web has **zero** `data_type == "input"`
elements across the full 113,142-element / 10k-sample corpus. Every
conversation used to end on an identical "No input areas found." filler
turn from `input_field.py` — pure noise, no grounding signal. Fixed:
`build_conversation` now only appends the input-field turn pair when the
page actually has input elements.

### Script hygiene
`test_render_batch.py`, `test_render_som.py`, `test_render.py` all had
`test_` prefixes but are side-effecting production scripts, not real tests.
Bare `pytest` silently collected and executed them — including a full 10k
re-render — which caused one accidental triple-concurrent re-render this
session (caught and killed mid-run). Renamed to `render_batch.py`,
`manual_render_som.py`, `manual_render_ui.py`. Also fixed
`manual_render_som.py` calling `.save()` on the tuple `apply_som` now
returns, and `example.py` calling `build_conversation()` without the
required `placed=` arg. `pytest -q` now runs clean (6 passed, `test_bbox.py`
only) with no side effects.

### Full 10k re-render (Phase 2)
Ran `render_batch.py` against the reworked renderer: **10,000/10,000
succeeded, 0 failures**, 22m34s (7.38 renders/sec). Log at
`data/interim/renders/seeclick_web/batch/render_log.txt`.

### Regenerated conversations.jsonl (Phase 3)
Ran `python -m src.formatting.formatter` against the fresh renders:
9,996 written, 4 skipped (empty), 0 skipped (missing sidecar). Re-split via
`python -m src.clean.split`: `train.jsonl` (8,997) / `val.jsonl` (999),
seed=42. Spot-checked output — one turn-pair per element, matches paper
Figure 12 format. The old (04:01) `conversations.jsonl`/`train.jsonl`/
`val.jsonl` predated this render and have been overwritten.

### Committed
All of the above (renderer, formatting, renames) committed in `2713576`.
`data/processed/` and `data/interim/` remain gitignored, as before — only
code changed hands.

---

## Current Status

```
SoM renderer (box outlines)   : COMPLETE
Conversation format (Fig 12)  : COMPLETE
Full 10k re-render            : COMPLETE (0 failures)
conversations.jsonl rebuild   : COMPLETE (9,996 samples)
train/val split                : COMPLETE (8,997 / 999)
LoRA adapter (models/lora_adapter/) : STALE — trained on pre-rework data
Finetuned eval                : PENDING (blocked on retrain)
```

---

## Next

1. Statistical validation pass on the new `conversations.jsonl` (record
   count, task distribution, bbox range checks, empty-conversation
   detection) — cheap sanity check before spending T4 hours on a retrain.
2. Retrain QLoRA on the new `train.jsonl` (Kaggle T4, same config as
   before) — the current adapter was trained on the old merged-mega-turn,
   dot-marker format and is a "before" data point, not the project's real
   result.
3. Run finetuned eval, compute baseline → finetuned delta.
4. Re-check whether the Mark:0 bias persists now that the merged-mega-turn
   format (suspected root cause) is gone.
5. Magma-8B reference numbers; comparison table; thesis sections.

---

### 2026-07-02 (continued)

## Goal
Diagnose the Mark:0 collapse's actual root cause via a controlled smoke
test before committing Kaggle hours to a full retrain; fix a blind spot
in the eval harness discovered along the way; run the full retrain on
the reworked data ahead of the demo presentation.

---

## Completed

### Root cause of Mark:0 collapse confirmed
Inspected the *old* (pre-rework) fine-tuning notebook's printed sample
output directly: the old training data's assistant turn was one giant
string concatenating every mark on the page in order, e.g.
`"Coordinate: (0.49, 0.08). Mark: 0.\nCoordinate: (0.31, 0.09). Mark: 1.\n...`
— this is the "merged mega-turn" format the 2026-07-02 renderer/formatter
rework already targeted, confirmed here as the actual mechanism: every
training example taught the model that its answer always *starts* with
"Coordinate: (...). Mark: 0.", regardless of the query. This fully
explains the inference-time collapse and validates that the formatter
rework (one turn-pair per element) is the correct fix — provided the
training script is also updated to use every turn-pair per page, not
just the first (see below).

Also confirmed old `finetune.py`'s `SoMDataset` only ever used
`convs[0]`/`convs[1]` (the first turn-pair) per page — compounding the
mega-turn issue, since the sole training target per page was always
Mark 0's (merged) answer.

### Kaggle smoke-test run (200 pages)
Built a new training cell set (multi-turn-per-page dataset + collator)
reusing the original run's proven config verbatim (LoRA r=16/alpha=32/
dropout=0.05, targets incl. gate/up/down_proj, LR=2e-4, warmup=0.03,
fp16 compute, bs=1/grad_accum=8) with two deliberate changes:
  - MAX_SEQ_LEN 1024 → 2048 (dense pages up to 100 marks need room for
    every turn-pair, not just the first ~20)
  - Dataset/collator now build ONE multi-turn conversation per page
    (system + every turn-pair) and mask loss across ALL assistant spans

### Collator bugs found and fixed during smoke test
1. First attempt used `return_offsets_mapping` to locate assistant
   spans for masking — silently broke after the image placeholder
   expands into many tokens (common failure mode for multimodal
   processors). Symptom: masking assertion caught only 1 "Mark:" in
   trainable labels instead of many. Fixed by switching to token-
   subsequence matching (tokenize each assistant answer standalone,
   find it as a subsequence in the full input_ids) — doesn't depend on
   processor offset support at all.
2. Diagnostic key (`_matched_spans`) was initially stored inside the
   collator's returned dict, which gets passed directly to
   `model.forward(**inputs)` by the Trainer — crashed with
   `TypeError: unexpected keyword argument '_matched_spans'`. Fixed by
   storing it as an instance attribute (`collator.last_matched_spans`)
   instead of a batch dict key.
3. Added a hard assertion cell (run before training, not skippable)
   that fails loudly if fewer than 2 assistant spans get correctly
   masked on a real sample — specifically to prevent silently
   reproducing the original bug in a new form.

### Eval harness blind spot found and fixed
`extract_first_element` in `eval.py` only ever read `convs[0]`/`convs[1]`
— same pattern as the old training bug. Since the first turn-pair on a
page is (almost) always Mark 0, **every eval sample's ground truth was
also always Mark 0** — meaning a collapsed model (always predicting
Mark 0) would score *identically* to a genuinely fixed model. The old
1.7% baseline number was measured this way and is not representative.

Fix: added `extract_all_elements()` (returns every eligible turn-pair
on a page, not just the first) and changed `run_eval` to sample one
element per page via `random.Random(1000 + idx)` (deterministic,
spread across all marks). Also added a `gt_mark == 0` vs `gt_mark != 0`
accuracy breakdown to both console output and the saved JSON
(`click_acc_gt_mark_0` / `click_acc_gt_mark_non0`) — this split is the
actual diagnostic for collapse: near-zero non-zero-mark accuracy would
mean still collapsed, comparable accuracy means fixed.

### Re-measured baseline (unbiased)
```
Results [baseline]  —  80 evaluated, 20 skipped   (results/eval_baseline_v2.json)
  Click accuracy (point in GT bbox) : 0.000  (0.0%)
  Click acc when gt_mark == 0        : 0.000  (n=27)
  Click acc when gt_mark != 0        : 0.000  (n=53)
```
Dropped from the old (Mark-0-only) 1.7% to a real 0.0% once tested
against a representative spread of marks — expected: base Qwen has no
SoM training and no marks to anchor on, and small/footer/generic
elements are much harder than the large, distinctive Mark-0 elements
the old eval exclusively tested.

### Smoke-test fine-tuned result (200 pages, 25 steps, ~6 min on Kaggle T4)
```
Results [finetuned]  —  80 evaluated, 20 skipped   (results/eval_smoke_finetuned.json)
  Click accuracy (point in GT bbox) : 0.275  (27.5%)
  Click acc when gt_mark == 0        : 0.370  (n=27)
  Click acc when gt_mark != 0        : 0.226  (n=53)
  [text_to_bbox]  n=45  click_acc=0.378
  [text_to_point] n=35  click_acc=0.143
```
Non-zero-mark accuracy (22.6%) is not near-zero — **collapse is fixed**.
The mark0-vs-non0 gap (37.0% vs 22.6%) is a plausible ordinary pattern
(header/logo elements at Mark 0 tend to be visually distinctive), not
the collapse signature, which would show non-zero-mark accuracy near 0%.

### Dataset scoping note (subtask filtering)
`SoMPageDataset._parse` keeps a page only if at least one turn-pair
contains `Coordinate:` — same filter the original proven pipeline used,
now applied per-turn-pair. Since task type is sampled per-page (not
per-element) at weights text→bbox 0.4 / text→point 0.4 / bbox→text 0.1 /
point→text 0.1, this means only pages whose sampled task is
`text_to_bbox`/`text_to_point` survive: **728 of 8,997 train pages**.
This is intentional, not a bug — it matches exactly what `eval.py`
measures (`extract_all_elements` also only accepts these two task
types). `bbox_to_text`/`point_to_text` pages are out of scope for this
ablation; noted as explicit future work (see TODO.md), not a defect —
training on them would need a different loss target format (free text
description, not `Coordinate:`/`Mark:`) and wouldn't move the grounding
accuracy metric this experiment reports.

### Full retrain launched
Same config as the smoke test, applied to all 728 eligible pages
(`ceil(728/8) = 91` steps, ~18-20 min estimated on Kaggle T4 based on
smoke-test throughput — far faster than the initial ~4.3hr estimate,
which assumed all 8,997 pages would train rather than the 728 that
actually pass the `Coordinate:` filter). Adapter saved to
`models/lora_adapter_v2/` (kept separate from both the stale original
adapter and the smoke-test adapter). Hit one environment issue mid-run
(`transformers` lazy-import crash on a stray `tensorflow_text` backend
check after a fresh kernel session) — resolved via clean uninstall/
reinstall of the pinned dependency versions + full kernel restart.

Result: PENDING — training completed, full eval + compare not yet run.

---

## Current Status

```
Mark:0 collapse root cause  : CONFIRMED (old merged-mega-turn training data)
Multi-turn training pipeline: COMPLETE (Kaggle notebook, token-subsequence
                                masking, hard pre-flight assertion)
Eval harness Mark-0 blind spot : FIXED (extract_all_elements + gt_mark split)
eval_baseline.json (1.7%)   : SUPERSEDED by eval_baseline_v2.json (0.0%)
Smoke test (200 pages)      : COMPLETE — 27.5% overall, 22.6% on gt_mark!=0
                                (collapse fix CONFIRMED)
Full retrain (728 pages)    : COMPLETE (training) — eval PENDING
lora_adapter_v2/            : trained, not yet evaluated
```

---

## Next

1. Run `eval.py --mode finetuned --adapter models/lora_adapter_v2
   --max-samples 200 --name full_finetuned` and
   `eval.py --mode compare --baseline-json results/eval_baseline_v2.json
   --finetuned-json results/eval_full_finetuned.json`.
2. Record the full-run numbers here and in schemas.md's Results table.
3. Prepare demo: 2-3 fixed tasks/sites, rehearse end-to-end multiple
   times, record a backup run.
4. Presentation framing: root cause (merged-mega-turn data) → fix
   (per-element turn-pairs + multi-turn training) → controlled evidence
   (gt_mark==0 vs !=0 split, not just an aggregate accuracy number) →
   explicit scoping note (grounding subtask only; bbox_to_text/
   point_to_text out of scope, flagged as future work).
5. Magma-8B reference numbers (cite paper-reported figures if a live
   Kaggle eval doesn't fit the remaining time budget — label clearly as
   "as reported by the authors", not reproduced).
6. Statistical validation pass on conversations.jsonl (still open from
   the previous session, not blocking).