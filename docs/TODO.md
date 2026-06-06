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
- [ ] Validate renderer on 100-sample batch
- [ ] Validate renderer on 1000-sample batch
- [ ] Decide adaptive scaling policy for very high/low resolutions
- [ ] Benchmark large-batch rendering throughput

## Formatting
- [ ] text -> bbox
- [ ] text -> point
- [ ] bbox -> text
- [ ] point -> text
- [ ] Map instructions to mark IDs
- [ ] Merge multiple tasks per screenshot
- [ ] Apply Vision2UI subtask sampling weights
      ([0.4, 0.4, 0.1, 0.1] for the four grounding tasks)
- [ ] Generate Magma-style conversation samples

## Infrastructure
- [x] Setup virtual environment
- [x] Setup git repository
- [x] Setup GitHub sync
- [x] Configure .gitignore
- [x] Setup pytest.ini
- [x] Setup docs structure
- [x] Create raw/interim/processed structure

## Future
- [ ] Add OCR enrichment pipeline
- [ ] Add mobile datasets (RICO, AITW)
- [ ] Add visualization notebooks
- [ ] Benchmark preprocessing speed
- [ ] Implement training dataloader
- [ ] Add train/validation splits
- [ ] Saliency scoring for ScreenSpot / OmniParser eval pipeline
      (deferred; not applicable to pre-curated SeeClick data)