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
- [ ] Run finetuned evaluation
      python -m src.eval.eval --mode finetuned \
        --adapter models/lora_adapter --name finetuned
- [ ] Save results/eval_finetuned.json
- [ ] Compute baseline → finetuned delta
- [ ] Generate comparison table (baseline / finetuned / Magma-8B)
- [ ] Investigate Mark:0 bias: run baseline mode on same pages and compare mark selection
- [ ] Analyze failure cases
- [ ] Generate qualitative examples (side-by-side baseline vs finetuned)

### Fine-tuning
- [x] Implement src/train/finetune.py
- [x] SoMDataset: load train.jsonl + renders, first grounding turn only
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
- [ ] Retrain QLoRA on regenerated train.jsonl (2026-07-02 rework)
- [ ] Evaluate fine-tuned model (see Evaluation above)
- [ ] Save results/eval_finetuned.json
- [ ] Compute baseline → finetuned delta
- [ ] Generate comparison table

### Magma Reference
- [ ] Download Magma-8B in 4-bit on Kaggle
- [ ] Run eval.py with MagmaBackend on val.jsonl
- [ ] Record reference performance
- [ ] Compare with Qwen baseline and finetuned

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

## Immediate Next Steps
1. Statistical validation pass on the new conversations.jsonl (count,
   task distribution, bbox ranges, empty-conversation detection).
2. Retrain QLoRA on the regenerated train.jsonl (Kaggle T4) — current
   adapter is stale, trained on pre-rework data.
3. Run finetuned eval on local machine or Kaggle.
4. Compute and record baseline → finetuned delta.
5. Re-check whether the Mark:0 bias persists post-retrain (suspected root
   cause — the old merged-mega-turn format — is now fixed).
6. Download Magma-8B in 4-bit (Kaggle) and run reference eval.
7. Generate comparison table: baseline / finetuned / Magma-8B.
8. Analyze fine-tuned failure cases.
9. Write thesis experiment section draft.