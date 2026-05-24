# Till 2026-05-24

## Goal
Bootstrap the Magma reproduction environment and begin the SeeClick-Web preprocessing pipeline.

## Completed

### Environment
- Created project structure
- Created Python virtual environment
- Installed core dependencies
- Added .gitignore
- Initialized git repository
- Connected local repo to GitHub
- Configured pytest infrastructure

### HuggingFace
- Logged into HuggingFace CLI
- Investigated dataset access failure
- Determined SeeClick is not hosted as a HuggingFace dataset package

### Dataset Research
- Located official SeeClick repository:
  https://github.com/njucckevin/SeeClick

- Identified:
  - 10k webpage screenshot subset
  - annotation metadata source
  - bbox schema format

### Schema Understanding
Confirmed:
- bbox format = [left, top, right, bottom]
- coordinates normalized to [0,1]
- multiple UI elements per screenshot
- element types:
  - text
  - hover

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

## Current Status

Current pipeline stage:

raw annotations
    ↓
bbox utilities
    ↓
SoM rendering (next)
    ↓
formatted training samples

## Next
- Load annotation JSON locally
- Render first SoM visualization
- Validate bbox alignment on screenshots
- Generate first formatted Magma-style sample

## Notes
- UI screenshots use SoM only
- ToM is not applicable for discrete UI screenshots
- Focus is currently preprocessing and dataset engineering, NOT model training
