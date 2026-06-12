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
- CLI: --adapter, --max-samples, --name flags

### Baseline Eval
- Smoke test (10 samples): 0% click accuracy — expected
- Model outputs pixel coords + wrong format (no fine-tuning yet)
- GT bbox fix confirmed: non-degenerate bboxes in results
- Full baseline eval running: 200 samples, ~50 min, nohup

---

## Current Status

Pipeline:    COMPLETE
Agent:       COMPLETE (Qwen2.5-VL-3B-Instruct, 4-bit, AnnotationSoM)
Baseline:    RUNNING (200 samples, results/eval_baseline.json)
Fine-tuning: NEXT (Colab T4, QLoRA via trl.SFTTrainer)

---

## Next
- Write src/train/finetune.py (QLoRA, SFTTrainer, train.jsonl)
- Run fine-tuning on Colab free T4 (~4–6 hrs, 3 epochs)
- Download LoRA adapter (~100MB) to models/lora_adapter/
- Re-run eval with --adapter → record finetuned score
- Report delta: baseline% → finetuned% = result