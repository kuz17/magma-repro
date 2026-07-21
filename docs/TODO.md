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
- [x] Extend DOM selector to a[href], role=link/button, [onclick]; filter
      disabled/aria-disabled; add innerText fallback label — root-caused
      and fixed the agent's inability to target plain nav links like
      "Hello, sign in" (previously not in dom_elements at all) — 2026-07-09
- [x] apply_som gains preserve_order param (skip area-sort, place in given
      order) — needed so DOM-priority elements map directly to the lowest
      mark IDs — 2026-07-09
- [x] _rebuild_som_dom_priority() now renders through apply_som(preserve_order=True)
      instead of its own diverged dot-style drawer — live inference input
      now visually matches the training renderer (box outlines + corner
      labels), which it had silently stopped doing since the 2026-07-02
      box-outline rework landed in render_som.py but was never ported here
      — 2026-07-09
- [x] Point resolution flipped to coordinate-first: parse the raw (x,y),
      snap to nearest mark within NEAREST_MARK_MAX_DIST=0.15, fall back to
      literal "Mark: N" text only if no coordinate parsed — mitigates
      literal-mark-text Mark:0 bias overriding an otherwise-good coordinate
      guess. Flagged as a demo-time workaround, not a model fix — 2026-07-09
- [ ] max_marks=15 cap risk: on pages with many buttons/inputs, product/
      content links (DOM tier 2, lowest priority) can be pushed past the
      cap before the grounding model ever sees them — confirmed possible
      via tests/smoke_dom_elements_search_page.py on a real Amazon search
      page; not yet fixed (raise cap? separate tier for content links?) —
      2026-07-09/10

### Planner + Executor Prototype (new, 2026-07-09)
- [x] tests/planner_agent.py: single model instance, base Qwen (adapter
      disabled via peft disable_adapter()) plans next action
      (SEARCH/CLICK/SCROLL/DONE), fine-tuned Qwen (adapter enabled) grounds
      CLICK targets via existing DemoRunner.act()
- [x] Validate disable_adapter() toggles cleanly on the 4-bit adapter and
      re-enables correctly (tests/smoke_disable_adapter.py) — confirmed
      safe to build on
- [x] Iterate on planning prompt phrasing in isolation, no browser loop
      (tests/smoke_planning_prompt.py, smoke_planning_prompt_case2.py,
      smoke_planning_v2.py) — specifically testing whether the model uses
      `history` to avoid repeating a completed SEARCH once results are
      already showing
- [ ] Outcome of the prompt iteration not written back to devlog — rerun
      and record which phrasing actually passes the "doesn't repeat SEARCH"
      check (case2 script prints a pass/fail hint but the run wasn't logged)
- [ ] Run planner_agent.py end-to-end on a real multi-step task (only
      isolated/unit-level pieces validated so far, not the full loop)
- [ ] SEARCH step re-implements the same DOM-direct click→clear→type→submit
      flow already in web_agent.py's `search` command — consider sharing
      one implementation instead of two copies

### Coordinate-Bias Investigation (new, in progress, uncommitted — 2026-07-09/11)
- [x] Observed via manual live-agent testing (outputs/click_coordinate_log.csv,
      local only, not in git): task "the first product in the search
      results" repeatedly produces a raw coordinate in a narrow band
      (~0.35-0.45, ~0.38-0.42) across visibly different pages, often with
      no mark nearby (rejected_no_mark)
- [x] tests/smoke_coordinate_bias.py (untracked): static-screenshot check
      across genuinely different pages with a fixed task — docstring
      records two runs landing on the identical coordinate (0.62, 0.71)
- [x] tests/smoke_coordinate_bias_live.py (untracked): live-browser
      follow-up with real DOM elements (static version used
      dom_elements=None, which disables the DOM-priority boost entirely)
- [x] tests/smoke_dom_elements_search_page.py (untracked): pure-DOM dump,
      no model — checks whether product-title <a> links survive the
      max_marks cap (see DOM Injection cap-risk item above)
- [ ] Run smoke_coordinate_bias_live.py and write the actual conclusion
      into devlog.md — current entry is based on docstrings + a partial
      local CSV, not a reviewed result
- [ ] If confirmed: root-cause the raw-coordinate bias before trusting the
      2026-07-09 coordinate-first resolution order as a real fix rather
      than a patch over a symptom
- [ ] Rebuild click_coordinate_log.csv's logging as a real script — the
      code that produced the existing CSV isn't anywhere in src/ as of
      this check, so it was likely ad hoc/interactive and never saved
- [ ] Commit the three untracked smoke test files once the investigation's
      conclusion is written up

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
- [x] Run full finetuned evaluation on lora_adapter_v2 (728-page retrain)
      → results/eval_full_finetuned.json: 17.4% click acc, 35.6% gt_mark==0,
      10.3% gt_mark!=0 — collapse fix CONFIRMED at full scale — 2026-07-02/03
- [x] Save results/eval_full_finetuned.json
- [x] Run compare: baseline_v2 vs full_finetuned (compare() now takes N
      json paths via --jsons, not just a fixed baseline/finetuned pair) —
      also added gt_mark0/non0 columns to the printed table, which existed
      in the saved JSON but weren't surfaced in compare() before — 2026-07-03
- [x] Compute baseline → finetuned delta (using v2 numbers): 0.0% → 17.4%
      overall, 0.0% → 10.3% on gt_mark!=0
- [ ] Generate comparison table (baseline / finetuned / Magma-8B) — still
      needs the Magma-8B row
- [x] Investigate Mark:0 bias — ROOT CAUSE FOUND: old training data was one
      merged assistant turn per page listing every mark in order, always
      starting "Mark: 0". Confirmed via old notebook's printed sample output.
      Fixed by Phase 3 data rework (one turn-pair per element) + new
      multi-turn training pipeline (all turn-pairs per page, not just first).
- [ ] Analyze failure cases (once full eval numbers are in)
- [ ] Generate qualitative examples (side-by-side baseline vs finetuned)
- [ ] NEW (2026-07-03): a second retrain attempt (models/lora_adapter_full)
      and a third (models/lora_adapter_full_v2) both scored WORSE than
      lora_adapter_v2 (2.5% and 11.2% vs 17.4%; attempt 2 also spiked
      no-prediction rate to 25%). Root cause not diagnosed — training ran
      on Kaggle, no local log of what differed between runs. Either
      root-cause via sweep_checkpoints.py (built, not yet run to
      completion — no outputs/sweep_results.csv on disk) or explicitly
      report lora_adapter_v2 as the result and drop the other two.
- [ ] Add an "adapter" field to eval.py's saved JSON (run/mode are saved,
      the adapter path used is not — had to reverse-engineer which JSON
      matched which models/lora_adapter_* dir from file mtimes when
      writing this up, 2026-07-03) — 2026-07-03
- [ ] Run sweep_checkpoints.py to completion against lora_adapter_full's
      checkpoint-* dirs (steps ~450-900, saved every 50) to find whether
      any intermediate checkpoint beats lora_adapter_v2, or whether the
      whole run regressed uniformly (ckpt_800 alone was still worse:
      4.3% vs 17.4%) — 2026-07-03

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
- [x] Evaluate models/lora_adapter_v2 (see Evaluation above) — 17.4% click
      acc, best result to date — 2026-07-02/03
- [x] Second retrain attempt → models/lora_adapter_full/ — REGRESSED
      (2.5% click acc, 25% no-pred, vs 17.4%/3% for lora_adapter_v2) —
      2026-07-02/03
- [x] Pulled intermediate checkpoint (step 800) from attempt 2 to check
      for mid-run degradation → models/ckpt_800/ — still bad (4.3%) —
      2026-07-03
- [x] Third retrain attempt → models/lora_adapter_full_v2/ — also
      regressed relative to lora_adapter_v2 (11.2% vs 17.4%) — 2026-07-03
- [ ] Root-cause why attempts 2/3 regressed relative to lora_adapter_v2
      (same nominal config per devlog, but no local record of what
      actually differed — training happened on Kaggle notebooks not
      checked into this repo)
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

## Immediate Next Steps (demo-focused, updated 2026-07-09)
1. Decide on the attempt-2/3 regression: root-cause it (run
   sweep_checkpoints.py to completion) or explicitly report
   `models/lora_adapter_v2` / `eval_full_finetuned.json` (17.4% click acc,
   10.3% gt_mark!=0) as the result and stop chasing the worse follow-up runs.
2. Land the coordinate-bias investigation: run
   `smoke_coordinate_bias_live.py`, write the conclusion into devlog.md,
   commit the three untracked smoke test scripts. This affects whether the
   2026-07-09 coordinate-first mark-resolution change in click_visualizer.py
   is a real fix or a workaround over an unresolved bias.
3. Pick 2-3 fixed demo tasks/sites (ideally resembling training
   distribution), screenshot and hand-check OmniParser + DOM coverage on
   those exact pages ahead of time — including the max_marks cap risk for
   product/content links flagged by smoke_dom_elements_search_page.py.
4. Rehearse the full web_agent (or planner_agent, if finished in time)
   pipeline end-to-end 5-10 times on the chosen tasks; record one clean
   backup run.
5. Prepare the presentation narrative: root cause found (merged-mega-turn
   training data) → fix (per-element turn-pairs + multi-turn training) →
   controlled evidence (gt_mark==0 vs !=0 split proves it's not just
   testing the easy case) → full-scale confirmation (17.4% / 10.3% on
   lora_adapter_v2) → explicit scoping note on what's out of bounds for
   this ablation (bbox_to_text/point_to_text, Mind2Web, the unresolved
   attempt-2/3 regression, the coordinate-bias investigation).
6. Magma-8B reference: live Kaggle run if time allows, else cite paper
   numbers clearly labeled as such.
7. Statistical validation pass on conversations.jsonl (still open, not
   blocking the demo).
8. Add an "adapter" field to eval.py's saved JSON (see Evaluation section)
   — cheap fix, prevents future timeline reconstruction pain.