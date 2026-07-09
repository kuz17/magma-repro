# TODO.md

## Dataset
- [x] Create dataset directory structure
- [x] Identify correct SeeClick data source
- [x] Download 10k subset
- [x] Download annotation metadata
- [x] Inspect repository schema
- [x] Load annotation JSON locally
- [x] Inspect real annotation samples
- [x] Verify nested element structure
- [x] Verify normalized bbox format
- [x] Inspect image naming consistency
- [x] Detect annotation/image subset mismatch
- [x] Build image existence filtering pipeline
- [x] Validate multiple samples visually
- [x] Confirm element data_type distribution (text ~95%, hover ~5%)
- [x] Confirm absence of HTML tag metadata in schema

## Utilities
- [x] Implement denormalize_bbox
- [x] Implement bbox_center
- [x] Implement bbox_area
- [x] Implement bbox_iou
- [x] Add bbox unit tests
- [x] Configure pytest infrastructure

## Cleaning
- [x] Filter annotations to locally available images
- [x] Validate bbox ranges
- [x] Detect malformed boxes
- [x] Remove invalid elements
- [x] Remove duplicate elements
- [x] Compute dataset statistics
- [x] Save cleaned schema

## SoM

### Debug Renderer
- [x] Create render_ui.py skeleton
- [x] Draw bbox overlays
- [x] Add numeric mark labels
- [x] Generate first rendered visualization
- [x] Verify geometric alignment manually

### Paper-style SoM Renderer
- [x] Create render_som.py
- [x] Render compact numbered markers
- [x] Place markers at bbox centers (superseded — see box-outline rework below)
- [x] Add circular marker backgrounds (superseded — see box-outline rework below)
- [x] Improve label readability
- [x] Prevent overlap between nearby marks
- [x] Filter degenerate / zero-area UI elements
- [x] Limit maximum marks per image
- [x] Save paper-style SoM renders
- [x] Validate renderer on 10-sample batch
- [x] Tune MIN_AREA_FLOOR / MIN_SPACING / MAX_MARKS to paper-quality
- [x] Record decisions log
- [x] Scale SoM rendering to full 10k subset
- [x] Rework marker style to box outline + corner label (matches paper,
      ported from Magma's `_find_least_overlapping_corner`) — 2026-07-02
- [x] Retune MIN_SPACING 1.3 → 0.7 for box-outline style (5.9% → 1.5% drop
      on full 10k corpus) — 2026-07-02
- [x] Clamp element coords to [0,1] in apply_som — 2026-07-02
- [x] Re-render full 10k subset with reworked renderer — 2026-07-02
      (10,000/10,000 succeeded, 0 failures)
- [ ] Spot-check statistical distribution of marks per image across full 10k
- [ ] Decide adaptive scaling policy for very high/low resolutions (deferred)
- [ ] Label-offset fallback for 81+-element tail pages (21 pages stay at
      26-35% element drop regardless of MIN_SPACING; deferred, not blocking)

## Formatting
- [x] Implement task_samplers.py
- [x] Implement input_field.py
- [x] Implement conversation.py
- [x] Implement formatter.py
- [x] text → bbox task
- [x] text → point task
- [x] bbox → text task
- [x] point → text task
- [x] Map instructions to mark IDs
- [x] Merge multiple tasks per screenshot into one conversation
- [x] Apply Vision2UI subtask sampling weights (0.4 / 0.4 / 0.1 / 0.1)
- [x] Implement input-field subtasks
- [x] Generate Magma-style conversations
- [x] Write conversations.jsonl
- [x] Validate output format against paper Figure 12
- [x] Rework to one turn-pair per element instead of merged mega-turns
      (matches Figure 12 exactly) — 2026-07-02
- [x] Make input-field turn conditional on page having input elements
      (SeeClick-Web has zero — confirmed empirically across full corpus) — 2026-07-02
- [x] Regenerate conversations.jsonl from reworked renders — 2026-07-02
      (9,996 written, 4 skipped empty, 0 skipped missing-sidecar)
- [x] Re-split train/val from regenerated conversations.jsonl — 2026-07-02
      (8,997 train / 999 val, seed=42)
- [ ] Statistical validation pass on conversations.jsonl
      - record count
      - task distribution
      - bbox range checks
      - empty conversation detection

## Model

### Experiment Design
- [x] Pivot from full Magma-8B reproduction to small-VLM experiment
- [x] Define baseline vs fine-tuned comparison
- [x] Define click accuracy metric (point ∈ GT bbox)
- [x] Train/validation split (90/10, seed=42)

### Agent
- [x] Implement src/agent/ui_agent.py
- [x] Implement backend abstraction
- [x] Implement QwenBackend
- [x] Implement MagmaBackend
- [x] Implement AnnotationSoM
- [x] Install OmniParser-v2.0 weights (models/omniparser/)
- [x] Implement OmniParserSoM integration
- [x] Implement three prompting modes
- [x] Implement coordinate parser
- [x] Handle backend-specific coordinate scaling
- [x] Download Qwen2.5-VL-3B-Instruct

### Demo Pipeline
- [x] Implement src/agent/click_visualizer.py
- [x] Baseline mode (raw screenshot, coordinate-guessing prompt)
- [x] Fine-tuned mode (training-style red circles, mark-lookup)
- [x] Click-point rendering with annotated output images
- [x] Implement src/agent/interactive.py (text-only REPL)
- [x] Validate fine-tuned pipeline on real screenshots (Amazon nav bar)
- [x] Fix prompt mismatch (PROMPT_TEMPLATE now matches text_to_point training format exactly)
- [x] Fix mark→center mapping (built from apply_som placed output, not label_coords)
- [x] Fix content_list persistence (self._last_content_list set in _run_omniparser)
- [x] Fix VLM input always saved to /tmp/vlm_input_{tag}.png
- [x] Tune YOLO_THRESHOLD 0.25→0.10, OCR_THRESHOLD 0.92→0.75
- [ ] Wire up lora_adapter_v2 (post-rework adapter) once eval confirms it — was
      DOM-priority workaround for Mark:0 bias still needed with the real fix in
      place, or can it be simplified/removed now that root cause is fixed?

### Inference Server
- [x] Implement src/agent/inference_server.py (FastAPI, POST /act, GET /health)
- [x] Baseline and finetuned startup modes
- [x] Fix _patch_demo_runner (was reading non-existent attributes; now no-op)
- [x] Add DomElementInfo schema and dom_elements field to ActRequest

### Web Agent
- [x] Implement src/agent/web_agent.py (Playwright REPL)
- [x] go / type / enter / scroll / back / screenshot / url commands
- [x] Fix: go to <url> navigation crash (strip "to " prefix)
- [x] Fix: navigation exception crashes agent (now caught and printed)
- [x] Raise inference timeout 120→240s
- [x] Fix: go back navigated to "https://back/" (now calls browser.back())
- [x] Fix: back() crash on slow pages (networkidle timeout caught, 10→15s)
- [x] Add: search <query> compound command (DOM-direct, no VLM inference)
      click input → Ctrl+A → type → Escape → click submit button
- [ ] Add: reload command (refresh current page)
- [ ] Add: elements command (list DOM interactive elements, no inference)
- [ ] Add: inspect command (open /tmp/vlm_input_finetuned.png with xdg-open)
- [ ] Add: fill <label> <value> command (DOM-direct fill by label/placeholder)
- [ ] Add: marks command (run OmniParser only, open SoM image, no Qwen)

### DOM Injection
- [x] Add BrowserEnv.get_interactive_elements() (JS DOM query)
- [x] Pass dom_elements through InferenceClient → inference server → DemoRunner
- [x] Implement _rebuild_som_dom_priority(): DOM elements at marks 0,1,… then OmniParser
- [x] Cap rebuilt SoM at 15 marks total
- [ ] Re-evaluate whether DOM-priority rebuild is still needed now that the
      root cause of Mark:0 bias (training data format) is fixed, or whether
      it can revert to simpler mark ordering — 2026-07-02

### Evaluation
- [x] Implement src/eval/eval.py (two modes: baseline / finetuned)
- [x] Click accuracy metric
- [x] IoU@0.5 metric
- [x] Mean IoU metric
- [x] Per-task breakdown
- [x] Degenerate GT bbox fix
- [x] Prediction normalization
- [x] Smoke test (10 samples)
- [x] Run baseline evaluation (500 samples → 402 evaluated)
- [x] Save results/eval_baseline.json  ← 1.7% click accuracy
      **SUPERSEDED 2026-07-02** — only ever tested Mark 0 per page (see fix below)
- [x] Fix Mark-0 blind spot: extract_first_element only ever read convs[0:2],
      meaning every eval ground truth was Mark 0 — a collapsed model would
      score identically to a fixed one. Added extract_all_elements() (samples
      any turn-pair per page) + gt_mark==0 vs gt_mark!=0 accuracy breakdown
      (click_acc_gt_mark_0 / click_acc_gt_mark_non0 in saved JSON) — 2026-07-02
- [x] Re-run baseline with fixed eval → results/eval_baseline_v2.json
      (0.0% click accuracy, both gt_mark==0 and !=0) — 2026-07-02
- [x] Run smoke-test finetuned evaluation (200 pages / 25 steps)
      → results/eval_smoke_finetuned.json
      27.5% overall, 37.0% gt_mark==0, 22.6% gt_mark!=0 — 2026-07-02
      Collapse fix CONFIRMED: non-zero-mark accuracy is not near-zero.
- [ ] Run full finetuned evaluation on lora_adapter_v2 (728-page retrain)
      python -m src.eval.eval --mode finetuned \
        --adapter models/lora_adapter_v2 --max-samples 200 --name full_finetuned
- [ ] Save results/eval_full_finetuned.json
- [ ] Run compare: baseline_v2 vs full_finetuned
- [ ] Compute baseline → finetuned delta (using v2 numbers, not the
      superseded 1.7%/old-finetuned pair)
- [ ] Generate comparison table (baseline / finetuned / Magma-8B)
- [x] Investigate Mark:0 bias — ROOT CAUSE FOUND: old training data was one
      merged assistant turn per page listing every mark in order, always
      starting "Mark: 0". Confirmed via old notebook's printed sample output.
      Fixed by Phase 3 data rework (one turn-pair per element) + new
      multi-turn training pipeline (all turn-pairs per page, not just first).
- [ ] Analyze failure cases (once full eval numbers are in)
- [ ] Generate qualitative examples (side-by-side baseline vs finetuned)

### Fine-tuning
- [x] Implement src/train/finetune.py
- [x] SoMDataset: load train.jsonl + renders, first grounding turn only
      **Known limitation, confirmed as collapse root cause 2026-07-02** —
      only ever training on convs[0:2] meant every page's sole training
      target was Mark 0's (merged) answer
- [x] SoMCollator: chat format, loss masked on user/image tokens
- [x] Configure QLoRA (r=16, alpha=32, all linear projections)
- [x] Configure TRL SFTTrainer
- [x] Run Kaggle T4 training (3 epochs, ~8.9k samples)
- [x] Export LoRA adapter (adapter_model.safetensors, 148.7MB)
- [x] Save adapter to models/lora_adapter/
      **STALE as of 2026-07-02** — trained on pre-rework data (old
      dot-marker renders, merged mega-turn format). Treat as a "before"
      data point only; retrain on the new train.jsonl before trusting
      finetuned eval numbers.
- [x] Build new Kaggle notebook cells: multi-turn-per-page dataset +
      collator, token-subsequence label masking (offset_mapping approach
      broke on image-token expansion, switched), hard pre-flight assertion
      cell (fails if <2 assistant spans mask correctly) — 2026-07-02
- [x] Run smoke test (200 pages, 25 steps, ~6 min) — validated the
      multi-turn masking works and confirmed collapse is fixed before
      committing to the full run — 2026-07-02
- [x] Diagnose and fix Kaggle environment bug (transformers lazy-import
      crash on stray tensorflow_text backend check) — clean reinstall +
      kernel restart — 2026-07-02
- [x] Run full retrain on all eligible (Coordinate:-containing) pages:
      728 of 8,997 (task-type sampling means only text_to_bbox/text_to_point
      pages qualify — matches what eval.py measures, intentional scoping,
      not a bug) — 91 steps, ~18-20 min — 2026-07-02
- [x] Save adapter to models/lora_adapter_v2/ — 2026-07-02
- [ ] Evaluate models/lora_adapter_v2 (see Evaluation above) — PENDING
- [ ] Save results/eval_full_finetuned.json
- [ ] Compute baseline → finetuned delta
- [ ] Generate comparison table
- [ ] FUTURE: bbox_to_text / point_to_text subtask pages (8,269 of 8,997)
      are currently unused by training. Would need a different loss target
      format (free-text description, not Coordinate:/Mark:) and doesn't
      move the grounding accuracy metric this experiment reports — explicit
      scope decision, not a defect. Candidate follow-up experiment: does
      adding these as an auxiliary training signal improve grounding
      accuracy beyond grounding-only training?

### Magma Reference
- [ ] Download Magma-8B in 4-bit on Kaggle
- [ ] Run eval.py with MagmaBackend on val.jsonl
- [ ] Record reference performance
- [ ] Compare with Qwen baseline and finetuned
- [ ] FALLBACK if time doesn't allow a live run before the demo: cite
      paper-reported Magma-8B numbers directly, clearly labeled "as
      reported by the authors" rather than reproduced — 2026-07-02

### Reproduction Target
- [ ] Clarify with mentor:
      - reproduce methodology?
      - reproduce released results?
- [ ] Document decision in devlog

## Infrastructure
- [x] Setup virtual environment
- [x] Setup git repository
- [x] Setup GitHub sync
- [x] Configure .gitignore
- [x] Setup pytest.ini
- [x] Create docs structure
- [x] Create raw/interim/processed layout

## Thesis Deliverables
- [ ] Produce architecture diagram
- [ ] Produce data pipeline diagram
- [ ] Produce evaluation methodology figure
- [ ] Write experiment section draft
- [ ] Write implementation section draft
- [ ] Write limitations section
- [ ] Maintain development log

## Future Work
- [ ] ShareGPT4V formatting pipeline
- [ ] LLaVA-Instruct formatting pipeline
- [ ] SeeClick-Mobile support
- [ ] OCR enrichment
- [ ] ToM generation via CoTracker
- [ ] Visualization notebooks
- [ ] Preprocessing benchmarks
- [ ] Saliency scoring experiments
- [ ] Docker containerization
- [ ] bbox_to_text / point_to_text auxiliary training (see Fine-tuning above)
- [ ] Mind2Web integration (deferred until current pipeline is fully
      validated on SeeClick-Web alone — flagged in an earlier planning
      session, not started)

## Immediate Next Steps (demo-focused, updated 2026-07-02)
1. Run full finetuned eval on `models/lora_adapter_v2` (200 samples) and
   the compare command against `eval_baseline_v2.json`.
2. Record final numbers in devlog.md and schemas.md's Results table.
3. Pick 2-3 fixed demo tasks/sites (ideally resembling training
   distribution), screenshot and hand-check OmniParser coverage on those
   exact pages ahead of time.
4. Rehearse the full web_agent pipeline end-to-end 5-10 times on the
   chosen tasks; record one clean backup run.
5. Prepare the presentation narrative: root cause found (merged-mega-turn
   training data) → fix (per-element turn-pairs + multi-turn training) →
   controlled evidence (gt_mark==0 vs !=0 split proves it's not just
   testing the easy case) → explicit scoping note on what's out of bounds
   for this ablation (bbox_to_text/point_to_text, Mind2Web, full 3-epoch/
   full-dataset retrain).
6. Magma-8B reference: live Kaggle run if time allows, else cite paper
   numbers clearly labeled as such.
7. Statistical validation pass on conversations.jsonl (still open, not
   blocking the demo).