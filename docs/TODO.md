# TODO

## Dataset
- [x] Create dataset directory structure
- [x] Identify correct SeeClick data source
- [x] Download 10k subset
- [x] Download annotation metadata
- [x] Inspect repository schema
- [ ] Load annotation JSON locally
- [ ] Validate bbox visually against screenshots
- [ ] Inspect image naming consistency
- [ ] Check for missing/corrupt images

## Utilities
- [x] Implement denormalize_bbox
- [x] Implement bbox_center
- [x] Implement bbox_area
- [x] Implement bbox_iou
- [x] Add bbox unit tests
- [x] Configure pytest infrastructure

## Cleaning
- [ ] Validate bbox ranges
- [ ] Detect malformed boxes
- [ ] Remove invalid elements
- [ ] Remove duplicate elements
- [ ] Compute dataset statistics
- [ ] Save cleaned schema

## SoM
- [x] Create render_ui.py skeleton
- [ ] Draw bbox overlays
- [ ] Add numeric mark labels
- [ ] Improve label placement
- [ ] Prevent overlap between labels
- [ ] Save rendered visualization samples

## Formatting
- [ ] text->bbox
- [ ] text->point
- [ ] bbox->text
- [ ] point->text
- [ ] Merge multiple tasks per screenshot
- [ ] Generate Magma-style conversation samples

## Infrastructure
- [x] Setup virtual environment
- [x] Setup git repository
- [x] Setup GitHub sync
- [x] Configure .gitignore
- [x] Setup pytest.ini
- [x] Setup docs structure

## Future
- [ ] Add OCR enrichment pipeline
- [ ] Add mobile datasets
- [ ] Add visualization notebooks
- [ ] Benchmark preprocessing speed
- [ ] Implement training dataloader
