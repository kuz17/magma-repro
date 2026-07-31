# tests/inspect_omniparser_stages.py
"""
Diagnostic: check what each of OmniParser's three detection sources
actually produced, independently, rather than trusting the merged
content_list blindly.

Two things are directly inspectable with confidence, since they reuse
the exact same calls _run_omniparser() already makes:
  1. Raw EasyOCR output (self._check_ocr_box) -- text boxes + transcribed
     content, BEFORE merging with YOLO/Florence-2.
  2. The full merged content_list (self._get_som_labeled_img) -- split by
     "type" field into text-sourced vs icon-sourced entries, so you can
     see exactly what each contributed and whether anything got merged
     or dropped in between.

YOLO-only and Florence-2-only outputs are NOT isolated here -- doing that
safely requires knowing OmniParser's internal wrapper API precisely
(get_yolo_model's returned object's exact call signature), which isn't
confirmed in this codebase. Rather than guess and risk a silent wrong
call, this script infers YOLO/Florence-2 behavior indirectly: any
content_list entry with type != "text" (i.e. not OCR-sourced) came from
a YOLO detection, and its "content" string is Florence-2's caption for
it -- so the icon-entries section below IS effectively inspecting both,
just via the merged output rather than a raw isolated call.

Usage:
    python tests/inspect_omniparser_stages.py --image outputs/bisect_screenshots/amazon_home.png
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from PIL import Image

from src.agent.click_visualizer import DemoRunner, OCR_THRESHOLD, YOLO_THRESHOLD, IOU_THRESHOLD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--search", default="art",
                        help="Case-insensitive substring to search for across all "
                             "stages (default 'art', catches 'Cart'/'cart')")
    args = parser.parse_args()

    print("Loading OmniParser (no adapter needed for this check)...")
    runner = DemoRunner(lora_path=None, training_style=False, tag="inspect")

    image = Image.open(args.image).convert("RGB")
    print(f"\nImage: {args.image}  ({image.size[0]}x{image.size[1]})\n")

    # ── Stage 1: raw EasyOCR, before any merging ────────────────────────
    print("=" * 70)
    print("STAGE 1 — Raw EasyOCR (before merging with YOLO)")
    print("=" * 70)
    ocr_result, _ = runner._check_ocr_box(
        image,
        display_img=False,
        output_bb_format="xyxy",
        easyocr_args={"paragraph": False, "text_threshold": OCR_THRESHOLD},
        use_paddleocr=False,
    )
    text, ocr_bbox = ocr_result
    text = text or []
    ocr_bbox = ocr_bbox or []
    print(f"Total raw OCR text boxes: {len(text)}  (threshold={OCR_THRESHOLD})\n")

    matches = [(t, b) for t, b in zip(text, ocr_bbox) if args.search.lower() in t.lower()]
    if matches:
        print(f"Boxes containing {args.search!r}:")
        for t, b in matches:
            print(f"  {t!r:<40} bbox={b}")
    else:
        print(f"⚠ No raw OCR box contains {args.search!r} at all — "
              f"if this is unexpected, OCR itself is not detecting this text "
              f"(font size, contrast, or OCR_THRESHOLD={OCR_THRESHOLD} may be too strict).")

    print(f"\nFirst 15 raw OCR boxes (sample):")
    for t, b in list(zip(text, ocr_bbox))[:15]:
        print(f"  {t!r:<40} bbox={b}")

    # ── Stage 2: full merged pipeline (YOLO + OCR + Florence-2 captions) ─
    print("\n" + "=" * 70)
    print("STAGE 2 — Full merged content_list (YOLO + OCR + Florence-2)")
    print("=" * 70)

    iw, ih = image.size
    ratio = iw / 3200
    draw_cfg = {
        "text_scale":     0.8 * ratio,
        "text_thickness": max(int(2 * ratio), 1),
        "text_padding":   max(int(3 * ratio), 1),
        "thickness":      max(int(3 * ratio), 1),
    }
    encoded, label_coords, content_list = runner._get_som_labeled_img(
        image,
        runner._yolo,
        BOX_TRESHOLD=YOLO_THRESHOLD,
        output_coord_in_ratio=False,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_cfg,
        caption_model_processor=runner._caption_proc,
        ocr_text=text,
        iou_threshold=IOU_THRESHOLD,
        imgsz=640,
    )

    text_entries = [e for e in content_list if e.get("type") == "text"]
    icon_entries = [e for e in content_list if e.get("type") != "text"]

    print(f"Total merged elements : {len(content_list)}")
    print(f"  from OCR (type=='text')       : {len(text_entries)}")
    print(f"  from YOLO+Florence-2 (icons)  : {len(icon_entries)}")
    print(f"\n  (raw OCR had {len(text)} boxes -> {len(text_entries)} survived merging; "
          f"{len(text) - len(text_entries)} were merged into icon boxes or dropped)")

    print(f"\nMerged entries containing {args.search!r}:")
    merged_matches = [
        (i, e) for i, e in enumerate(content_list)
        if args.search.lower() in str(e.get("content", "")).lower()
    ]
    if merged_matches:
        for i, e in merged_matches:
            print(f"  [{i}] type={e.get('type'):<6} content={e.get('content')!r}")
    else:
        print(f"  ⚠ NOTHING in the merged content_list contains {args.search!r} — "
              f"even if raw OCR caught it (see Stage 1), it did not survive to "
              f"the final candidate set. Likely merged into a nearby YOLO icon box "
              f"and its text was discarded, or dropped by IOU_THRESHOLD={IOU_THRESHOLD}.")

    print(f"\nAll icon-sourced captions (Florence-2 output) — check these look like "
          f"real descriptions, not empty/generic junk:")
    empty_or_generic = 0
    for i, e in enumerate(icon_entries):
        content = str(e.get("content", "")).strip()
        flag = ""
        if len(content) == 0:
            flag = "  ⚠ EMPTY"
            empty_or_generic += 1
        elif content.lower() in ("icon", "image", "logo", "button"):
            flag = "  ⚠ GENERIC"
            empty_or_generic += 1
        print(f"  {content!r:<50}{flag}")

    if icon_entries:
        pct = 100 * empty_or_generic / len(icon_entries)
        print(f"\n{empty_or_generic}/{len(icon_entries)} icon captions ({pct:.0f}%) are "
              f"empty or generic — {'concerning' if pct > 30 else 'looks reasonable'}.")

    print(f"\nFirst 10 text-sourced entries (sample):")
    for e in text_entries[:10]:
        print(f"  {e.get('content')!r}")


if __name__ == "__main__":
    main()