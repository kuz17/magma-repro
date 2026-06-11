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
- [x] Record decisions log
- [x] Scale SoM rendering to full 10k subset
- [ ] Spot-check statistical distribution of marks per image across full 10k
- [ ] Decide adaptive scaling policy for very high/low resolutions (deferred)

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
- [ ] Finish OmniParser installation
- [x] Implement OmniParserSoM integration
- [x] Implement three prompting modes
- [x] Implement coordinate parser
- [x] Handle backend-specific coordinate scaling
- [x] Download Qwen2.5-VL-3B-Instruct
- [ ] Download OmniParser-v2.0 weights

### Evaluation
- [x] Implement src/eval/eval.py
- [x] Click accuracy metric
- [x] IoU@0.5 metric
- [x] Mean IoU metric
- [x] Per-task breakdown
- [x] Degenerate GT bbox fix
- [x] Prediction normalization
- [x] Smoke test (10 samples)
- [ ] Run 200-sample baseline evaluation
- [ ] Save results/eval_baseline.json
- [ ] Analyze failure cases
- [ ] Generate qualitative examples

### Fine-tuning
- [ ] Implement src/train/finetune.py
- [ ] Convert conversations.jsonl → Qwen SFT format
- [ ] Configure QLoRA
- [ ] Configure TRL SFTTrainer
- [ ] Run Colab T4 training
- [ ] Export LoRA adapter
- [ ] Save adapter to models/lora_adapter/
- [ ] Evaluate fine-tuned model
- [ ] Save results/eval_finetuned.json
- [ ] Compute baseline → finetuned delta
- [ ] Generate comparison table

### Magma Reference
- [ ] Clone official Magma repository
- [ ] Download Magma-8B weights
- [ ] Run official UI-grounding demo
- [ ] Record reference performance
- [ ] Compare with Qwen baseline

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
1. Finish OmniParser installation.
2. Complete 200-sample baseline evaluation.
3. Inspect baseline failures manually.
4. Implement finetune.py.
5. Run QLoRA training on Colab.
6. Evaluate LoRA adapter.
7. Compare baseline vs finetuned performance.
8. Obtain Magma reference numbers.
9. Write thesis experiment section.