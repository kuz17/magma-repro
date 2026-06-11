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
- [x] Place markers at bbox centers
- [x] Add circular marker backgrounds
- [x] Improve label readability
- [x] Prevent overlap between nearby marks
- [x] Filter degenerate / zero-area UI elements
- [x] Limit maximum marks per image
- [x] Save paper-style SoM renders
- [x] Validate renderer on 10-sample batch
- [x] Tune MIN_AREA_FLOOR / MIN_SPACING / MAX_MARKS to paper-quality
- [x] Record decisions log (rejected fixed MIN_AREA, ≥2.0 spacing,
      tag-filter, saliency scoring, dynamic MIN_AREA)
- [x] Scale SoM rendering to full 10k subset
- [ ] Spot-check statistical distribution of marks per image across
      full 10k (mean, min, max, % hitting MAX_MARKS cap)
- [ ] Decide adaptive scaling policy for very high/low resolutions
      (deferred; not blocking training)

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
- [x] Apply Vision2UI subtask sampling weights (0.4/0.4/0.1/0.1)
- [x] Implement input field subtasks (input→point, input→bbox, equal weight)
- [x] Generate Magma-style conversation samples
- [x] Write output to data/processed/seeclick_web/conversations.jsonl
- [x] Validate output format matches paper Figure 12
- [ ] Statistical validation pass on conversations.jsonl
      (record count, task distribution, bbox range checks, empty convs)

## Model

### Reproduction Target
- [ ] Clarify with mentor: reproduce methodology vs reproduce released results
- [ ] Document decision in devlog

### Magma-8B Reference Run
- [ ] Clone official Magma repo (github.com/microsoft/Magma)
- [ ] Download Magma-8B weights
- [ ] Run official SeeClick / UI grounding demo
- [ ] Record baseline grounding accuracy (this is the benchmark target)

### Small VLM Demo (if full Magma-8B training deferred)
- [ ] Select base VLM (Phi-3.5-Vision or LLaVA-1.5-7B)
- [ ] Estimate GPU requirements for fine-tuning on 10k conversations
- [ ] Fine-tune for 1–2 epochs on conversations.jsonl
- [ ] Evaluate on SeeClick grounding subset
- [ ] Compare against Magma-8B reference numbers

### Training Infrastructure
- [ ] Estimate compute requirements for Magma-8B fine-tuning
- [ ] Request GPU resources if needed (cloud scale-up phase)
- [ ] Implement training dataloader for conversations.jsonl
- [ ] Add train/validation splits

## Infrastructure
- [x] Setup virtual environment
- [x] Setup git repository
- [x] Setup GitHub sync
- [x] Configure .gitignore
- [x] Setup pytest.ini
- [x] Setup docs structure
- [x] Create raw/interim/processed structure

## Future
- [ ] ShareGPT4V + LLaVA-Instruct data formatting (next pipeline phase)
- [ ] SeeClick-Mobile formatting (same pipeline, different schema nuances)
- [ ] ToM generation via CoTracker (heaviest lift; after static data solid)
- [ ] Add OCR enrichment pipeline
- [ ] Add visualization notebooks
- [ ] Benchmark preprocessing speed
- [ ] Saliency scoring for ScreenSpot / OmniParser eval pipeline
      (deferred; not applicable to pre-curated SeeClick data)
- [ ] Docker containerization (deferred to cloud/training phase)