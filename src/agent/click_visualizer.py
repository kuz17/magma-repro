# src/agent/click_visualizer.py
"""
Full OmniParser → Qwen pipeline with click-point rendering.

Usage:
    # baseline — OmniParser bbox style
    python -m src.agent.click_visualizer \
        --image outputs/demo/screenshot.png \
        --interactive \
        --tag baseline

    # fine-tuned — training circle style
    python -m src.agent.click_visualizer \
        --image outputs/demo/screenshot.png \
        --interactive \
        --lora models/lora_adapter \
        --tag finetuned \
        --training-style
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────
OMNIPARSER_CODE = "/data/Magma/OmniParser"
OMNIPARSER_YOLO = "models/omniparser/icon_detect/model.pt"
OMNIPARSER_CAP  = "models/omniparser/icon_caption"
QWEN_PATH       = "models/qwen2_5_vl_3b"

# ── visual config ──────────────────────────────────────────────────────
CIRCLE_RADIUS  = 18
CIRCLE_COLOR   = (0, 220, 80)
CIRCLE_OUTLINE = (255, 255, 255)
OUTLINE_WIDTH  = 3
CROSSHAIR_LEN  = 10
LABEL_COLOR    = (255, 255, 255)
FONT_SIZE      = 16

# ── OmniParser detection config ────────────────────────────────────────
YOLO_THRESHOLD = 0.30       # higher = fewer icon detections
OCR_THRESHOLD  = 0.95       # higher = fewer text detections
IOU_THRESHOLD  = 0.5        # higher = more overlap removal

# ── prompt templates ───────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    'On this software\'s interface, to execute the step "{task}", '
    "where do I direct my attention? "
    "Please provide the coordinate and the bounding box's mark index if applicable."
)
RAW_PROMPT_TEMPLATE = (
    'In this UI screenshot I need to perform the action: "{task}". '
    "Respond with ONLY the format: Coordinate: (x, y) "
    "where x and y are normalized 0 to 1."
)


# ──────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────

def _parse_point(response: str, image_size: tuple = (1, 1)) -> tuple | None:
    w, h = image_size

    # Format 1: Coordinate: (x, y) — fine-tuned output, normalized
    m = re.search(
        r'[Cc]oordinates?\s*[:\(]*\s*\(?([0-9.]+)\s*,\s*([0-9.]+)\)?',
        response,
    )
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    # Format 2: **X-coordinate:** 1830 — base model verbose
    mx = re.search(r'[Xx]-[Cc]oordinate[^0-9]*([0-9]+)', response, re.IGNORECASE)
    my = re.search(r'[Yy]-[Cc]oordinate[^0-9]*([0-9]+)', response, re.IGNORECASE)
    if mx and my:
        x, y = float(mx.group(1)), float(my.group(1))
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    # Format 3: bare (x, y) pair anywhere in response
    m = re.search(r'\(([0-9.]+)\s*,\s*([0-9.]+)\)', response)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return (x, y)
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    return None


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

class DemoRunner:
    """Loads OmniParser + Qwen once, runs the full pipeline per task."""

    def __init__(
        self,
        lora_path: str | None = None,
        raw_mode: bool = False,
        tag: str = "",
        training_style: bool = False,
    ):
        self.raw_mode       = raw_mode
        self.tag            = tag
        self.training_style = training_style
        self._input_saved   = False
        self._load_qwen(lora_path)
        if not raw_mode:
            self._load_omniparser()

    def _load_omniparser(self):
        sys.path.insert(0, OMNIPARSER_CODE)
        from util.utils import (
            check_ocr_box,
            get_caption_model_processor,
            get_som_labeled_img,
            get_yolo_model,
        )
        self._check_ocr_box       = check_ocr_box
        self._get_som_labeled_img = get_som_labeled_img

        print("Loading OmniParser YOLO...")
        self._yolo = get_yolo_model(OMNIPARSER_YOLO)

        print("Loading OmniParser Florence-2...")
        self._caption_proc = get_caption_model_processor(
            "florence2", OMNIPARSER_CAP, device="cpu"
        )
        print("OmniParser ready.")

    def _load_qwen(self, lora_path: str | None):
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        print("Loading Qwen2.5-VL-3B (4-bit)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_PATH,
            quantization_config=bnb,
            device_map="auto",
        )

        if lora_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter: {lora_path}")
            self._qwen = PeftModel.from_pretrained(self._qwen, lora_path)
            print("Adapter loaded.")

        self._processor = AutoProcessor.from_pretrained(QWEN_PATH)
        label = f"fine-tuned [{lora_path}]" if lora_path else "base"
        print(f"Qwen ready ({label}).")

    def _run_omniparser(self, image: Image.Image):
        iw, ih = image.size
        ratio = iw / 3200
        draw_cfg = {
            "text_scale":     0.8 * ratio,
            "text_thickness": max(int(2 * ratio), 1),
            "text_padding":   max(int(3 * ratio), 1),
            "thickness":      max(int(3 * ratio), 1),
        }

        ocr_result, _ = self._check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            easyocr_args={"paragraph": False, "text_threshold": OCR_THRESHOLD},
            use_paddleocr=False,
        )
        text, ocr_bbox = ocr_result
        print(f"  OCR boxes : {len(text)}", flush=True)

        encoded, label_coords, content_list = self._get_som_labeled_img(
            image,
            self._yolo,
            BOX_TRESHOLD=YOLO_THRESHOLD,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_cfg,
            caption_model_processor=self._caption_proc,
            ocr_text=text,
            iou_threshold=IOU_THRESHOLD,
            imgsz=640,
        )

        if self.training_style:
            # re-render in training circle style
            # label_coords: {str: [x, y, w, h]} pixel top-left xywh
            elements = [
                {"bbox": [x/iw, y/ih, (x+bw)/iw, (y+bh)/ih]}
                for (x, y, bw, bh) in label_coords.values()
            ]
            from src.som.render_som import apply_som
            som_image, _ = apply_som(image.copy(), elements)
        else:
            som_image = Image.open(io.BytesIO(base64.b64decode(encoded)))

        print(f"  Original : {image.size}  |  SoM : {som_image.size}")
        return som_image, label_coords, content_list

    def _run_qwen(self, image: Image.Image, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        ).to(self._qwen.device)

        with torch.no_grad():
            out = self._qwen.generate(**inputs, max_new_tokens=32)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(trimmed, skip_special_tokens=True).strip()

    def act(self, image_path: str, task: str) -> tuple[str, tuple | None]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if self.raw_mode:
            prompt = RAW_PROMPT_TEMPLATE.format(task=task)
            inference_image = image
        else:
            print("  Running OmniParser...", flush=True)
            som_image, label_coords, _ = self._run_omniparser(image)
            print(f"  Total elements : {len(label_coords)}", flush=True)
            prompt = PROMPT_TEMPLATE.format(task=task)
            inference_image = som_image

            # save the exact image Qwen sees — only on first task
            if not self._input_saved:
                p = Path(image_path)
                tag = f"_{self.tag}" if self.tag else ""
                save_path = p.parent / f"vlm_input{tag}_{p.stem}.png"
                inference_image.save(save_path)
                print(f"  VLM input saved: {save_path}")
                self._input_saved = True

        print("  Running Qwen...", flush=True)
        response = self._run_qwen(inference_image, prompt)
        point = _parse_point(response, image_size=(w, h))
        return response, point


# ──────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────

def _draw_click(
    image: Image.Image,
    point: tuple,
    label: str = "",
    radius: int = CIRCLE_RADIUS,
) -> Image.Image:
    img = image.copy().convert("RGBA")
    w, h = img.size
    cx = int(point[0] * w)
    cy = int(point[1] * h)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(*CIRCLE_COLOR, 120),
        outline=(*CIRCLE_OUTLINE, 255),
        width=OUTLINE_WIDTH,
    )
    draw.line([cx - CROSSHAIR_LEN, cy, cx + CROSSHAIR_LEN, cy],
              fill=(*CIRCLE_COLOR, 255), width=2)
    draw.line([cx, cy - CROSSHAIR_LEN, cx, cy + CROSSHAIR_LEN],
              fill=(*CIRCLE_COLOR, 255), width=2)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    if label:
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                FONT_SIZE)
        except Exception:
            font = ImageFont.load_default()
        bb = d.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 4
        tx = max(0, min(cx - tw // 2, w - tw - pad * 2))
        ty = max(0, cy - radius - th - pad * 2 - 4)
        d.rectangle([tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                    fill=(0, 0, 0))
        d.text((tx, ty), label, fill=LABEL_COLOR, font=font)

    return img


def _save_and_open(rendered: Image.Image, image_path: str, tag: str = "", task: str = "") -> Path:
    p = Path(image_path)
    task_slug = re.sub(r'[^a-z0-9_]', '', task.lower().replace(' ', '_'))[:40]
    parts = [x for x in ["click", tag, p.stem, task_slug] if x]
    out = p.parent / f"{'_'.join(parts)}.png"
    rendered.save(out)
    print(f"Saved  : {out}")
    try:
        import subprocess
        subprocess.Popen(["xdg-open", str(out)])
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────────

def run(image_path: str, task: str, runner: DemoRunner) -> None:
    print(f"\nTask   : {task}")
    response, point = runner.act(image_path, task)
    print(f"Model  : {response}")

    if point is None:
        print("No click point found in response.")
        return

    print(f"Point  : ({point[0]:.3f}, {point[1]:.3f})")
    base = Image.open(image_path).convert("RGB")
    rendered = _draw_click(base, point, label=task[:50])
    _save_and_open(rendered, image_path, tag=runner.tag, task=task)


def interactive(image_path: str, runner: DemoRunner) -> None:
    print(f"\nImage  : {image_path}")
    print("Type a task and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            task = input("Task >>> ").strip()
            if not task:
                continue
            run(image_path, task, runner)
            print()
        except KeyboardInterrupt:
            print("\nBye.")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",          required=True,
                        help="Path to screenshot")
    parser.add_argument("--task",           default=None,
                        help="Task description (single shot)")
    parser.add_argument("--lora",           default=None,
                        help="Path to LoRA adapter dir")
    parser.add_argument("--tag",            default="",
                        help="Output filename tag: click_TAG_screenshot_task.png")
    parser.add_argument("--raw",            action="store_true",
                        help="Skip OmniParser, run Qwen on raw image")
    parser.add_argument("--training-style", action="store_true",
                        help="Render SoM as training circles instead of OmniParser boxes")
    parser.add_argument("--interactive",    action="store_true",
                        help="Interactive loop")
    args = parser.parse_args()

    if not args.interactive and not args.task:
        parser.error("Provide --task 'your task' or --interactive")

    runner = DemoRunner(
        lora_path=args.lora,
        raw_mode=args.raw,
        tag=args.tag,
        training_style=args.training_style,
    )

    if args.interactive:
        interactive(args.image, runner)
    else:
        run(args.image, args.task, runner)


if __name__ == "__main__":
    main()