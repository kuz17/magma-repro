# tests/smoke_gpu_headroom_check.py
"""
Checks whether there's real VRAM headroom to move OmniParser's Florence-2
captioning model from CPU to GPU, alongside the already-loaded 4-bit Qwen.

Devlog (2026-06-25) previously concluded GTX 1650 Mobile (4GB) had
insufficient VRAM to run both simultaneously. This script re-checks that
with current thresholds/caps, since MAX_ELEMENTS/YOLO_THRESHOLD tuning
may have changed the picture. Prints real allocated/reserved numbers
at each stage instead of guessing.

Usage:
    python -m tests.smoke_gpu_headroom_check --image outputs/demo/screenshot.png
"""
import argparse
import sys

sys.path.insert(0, ".")

import torch


def mem(label: str):
    if not torch.cuda.is_available():
        print(f"  [{label}] CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1e9, total / 1e9
    print(f"  [{label}] allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  "
          f"free={free_gb:.2f}GB / total={total_gb:.2f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a real screenshot to caption")
    args = ap.parse_args()

    print("=" * 60)
    print("STAGE 0: baseline (nothing loaded)")
    print("=" * 60)
    mem("baseline")

    print("\n" + "=" * 60)
    print("STAGE 1: load 4-bit Qwen2.5-VL-3B (matches click_visualizer.py)")
    print("=" * 60)
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    QWEN_PATH = "models/qwen2_5_vl_3b"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, quantization_config=bnb, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    mem("after Qwen load")

    print("\n" + "=" * 60)
    print("STAGE 2: load OmniParser YOLO (icon detector, GPU)")
    print("=" * 60)
    OMNIPARSER_CODE = "/data/Magma/OmniParser"
    sys.path.insert(0, OMNIPARSER_CODE)
    from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

    yolo = get_yolo_model("models/omniparser/icon_detect/model.pt")
    mem("after YOLO load")

    print("\n" + "=" * 60)
    print("STAGE 3: load Florence-2 caption model ON GPU (device='cuda')")
    print("  <-- this is the actual test; watch for OOM here")
    print("=" * 60)
    try:
        caption_proc = get_caption_model_processor(
            "florence2", "models/omniparser/icon_caption", device="cuda"
        )
        mem("after Florence-2 GPU load")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ OOM loading Florence-2 on GPU: {e}")
        print("  -> stick with device='cpu' for now, or try 8-bit Florence-2.")
        return

    print("\n" + "=" * 60)
    print("STAGE 4: run full OmniParser pass on a real screenshot")
    print("  (this is where captioning batch size actually matters)")
    print("=" * 60)
    from PIL import Image
    image = Image.open(args.image).convert("RGB")

    try:
        ocr_result, _ = check_ocr_box(
            image, display_img=False, output_bb_format="xyxy",
            easyocr_args={"paragraph": False, "text_threshold": 0.75},
            use_paddleocr=False,
        )
        text, ocr_bbox = ocr_result
        mem("after OCR (CPU, EasyOCR)")

        torch.cuda.reset_peak_memory_stats()
        encoded, label_coords, content_list = get_som_labeled_img(
            image, yolo, BOX_TRESHOLD=0.10, output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox, caption_model_processor=caption_proc,
            ocr_text=text, iou_threshold=0.4, imgsz=640,
        )
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  detected {len(label_coords)} elements before MAX_ELEMENTS cap")
        print(f"  peak VRAM during captioning: {peak:.2f}GB")
        mem("after full OmniParser pass")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ OOM during captioning pass: {e}")
        print("  -> apply MAX_ELEMENTS cap BEFORE captioning (fix #2) and retry,")
        print("     or fall back to device='cpu' for Florence-2.")
        return

    print("\n" + "=" * 60)
    print("RESULT: GPU captioning fit in memory for this test image.")
    print("Run this on 2-3 of your real demo pages (not just one) before")
    print("committing to device='cuda' — peak memory scales with how many")
    print("boxes YOLO_THRESHOLD=0.10 surfaces on a given page.")
    print("=" * 60)


if __name__ == "__main__":
    main()