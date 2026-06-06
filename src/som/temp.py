#faster render

import json
import random
import time
from pathlib import Path

from PIL import Image

from src.som.render_som import apply_som

random.seed(42)

FILTERED_PATH = (
    "data/raw/seeclick_web/"
    "filtered_annotations.json"
)

IMAGE_ROOT = Path(
    "data/raw/seeclick_web"
)

OUTPUT_DIR = Path(
    "data/interim/renders/batch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = OUTPUT_DIR / "render_log.txt"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading annotations...")

with open(FILTERED_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data):,} samples")

# ============================================================
# BUILD IMAGE INDEX
# ============================================================

print("Building image index...")

image_index = {
    p.name: p
    for p in IMAGE_ROOT.rglob("*")
    if p.is_file()
}

print(
    f"Indexed {len(image_index):,} files"
)

# ============================================================
# SAMPLE DATA
# ============================================================

NUM_SAMPLES = 100

samples = random.sample(
    data,
    min(NUM_SAMPLES, len(data))
)

print(
    f"Rendering {len(samples)} samples..."
)

# ============================================================
# TIMING
# ============================================================

total_open_time = 0.0
total_som_time = 0.0
total_save_time = 0.0

# ============================================================
# RENDER
# ============================================================

start_time = time.perf_counter()

successful_renders = 0
failed_renders = 0

with open(LOG_FILE, "w") as log_f:

    log_f.write("Batch Render Log\n")
    log_f.write("=" * 80 + "\n\n")

    for idx, sample in enumerate(samples):

        image_path = image_index.get(
            sample["img_filename"]
        )

        if image_path is None:

            failed_renders += 1

            msg = (
                f"[{idx}] "
                f"MISSING IMAGE: "
                f"{sample['img_filename']}"
            )

            print(msg)
            log_f.write(msg + "\n")

            continue

        try:

            # -------------------------
            # OPEN
            # -------------------------

            t0 = time.perf_counter()

            image = Image.open(image_path)

            t1 = time.perf_counter()

            # -------------------------
            # SOM
            # -------------------------

            rendered = apply_som(
                image,
                sample["elements"],
            )

            t2 = time.perf_counter()

            # -------------------------
            # SAVE (FAST PNG)
            # -------------------------

            output_path = (
                OUTPUT_DIR /
                f"render_{idx:03d}.png"
            )

            rendered.save(
                output_path,
                optimize=False,
                compress_level=1,
            )

            t3 = time.perf_counter()

            open_time = t1 - t0
            som_time = t2 - t1
            save_time = t3 - t2

            total_open_time += open_time
            total_som_time += som_time
            total_save_time += save_time

            successful_renders += 1

            msg = (
                f"[{idx}] "
                f"image={image_path.name} "
                f"elements={len(sample['elements'])} "
                f"open={open_time:.4f}s "
                f"som={som_time:.4f}s "
                f"save={save_time:.4f}s"
            )

            print(msg)
            log_f.write(msg + "\n")

        except Exception as e:

            failed_renders += 1

            msg = (
                f"[{idx}] "
                f"ERROR: {image_path.name} "
                f"({e})"
            )

            print(msg)
            log_f.write(msg + "\n")

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = time.perf_counter() - start_time

    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    avg_open = (
        total_open_time / successful_renders
        if successful_renders else 0
    )

    avg_som = (
        total_som_time / successful_renders
        if successful_renders else 0
    )

    avg_save = (
        total_save_time / successful_renders
        if successful_renders else 0
    )

    summary = (
        "\n"
        + "=" * 80
        + "\n"
        + "SUMMARY\n"
        + "=" * 80
        + "\n"
        + f"Successful renders : {successful_renders}\n"
        + f"Failed renders     : {failed_renders}\n"
        + f"Total processed    : {successful_renders + failed_renders}\n"
        + f"Time taken         : {minutes}m {seconds:.2f}s\n"
        + f"Render rate        : "
        + f"{successful_renders / elapsed:.2f} renders/sec\n"
        + "\n"
        + "AVERAGE TIMES\n"
        + f"Open image         : {avg_open:.4f}s\n"
        + f"Apply SoM          : {avg_som:.4f}s\n"
        + f"Save image         : {avg_save:.4f}s\n"
    )

    print(summary)
    log_f.write(summary)

print(
    f"\nFinished. Log written to:\n{LOG_FILE}"
)