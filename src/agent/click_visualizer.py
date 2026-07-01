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
YOLO_THRESHOLD = 0.10   # lowered from 0.25; 0.05 caused Florence-2 to caption 100+ elements on CPU
OCR_THRESHOLD  = 0.75   # lowered from 0.92; 0.50 produced garbage OCR on anti-aliased text
IOU_THRESHOLD  = 0.4
MAX_ELEMENTS   = 35

# ── prompt templates ───────────────────────────────────────────────────
# Fine-tuned prompt: exact text_to_point format used during training
PROMPT_TEMPLATE = (
    'To execute the step "{task}", where do I direct my attention? '
    "Please provide the coordinate and the bounding box's mark index."
)

# Baseline prompt: no element list, model guesses coordinate
BASELINE_PROMPT_TEMPLATE = (
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
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _parse_point(response: str, image_size: tuple = (1, 1)) -> tuple | None:
    """
    Fallback coordinate parser when mark lookup fails.
    Handles normalized and absolute pixel formats.
    """
    w, h = image_size

    # Format 1: Coordinate: (x, y)
    m = re.search(
        r'[Cc]oordinates?\s*[:\(]*\s*\(?([0-9.]+)\s*,\s*([0-9.]+)\)?',
        response,
    )
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    # Format 2: X-coordinate: 1830 / Y-coordinate: 24
    mx = re.search(r'[Xx]-[Cc]oordinate[^0-9]*([0-9]+)', response, re.IGNORECASE)
    my = re.search(r'[Yy]-[Cc]oordinate[^0-9]*([0-9]+)', response, re.IGNORECASE)
    if mx and my:
        x, y = float(mx.group(1)), float(my.group(1))
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    # Format 3: bare (x, y) pair
    m = re.search(r'\(([0-9.]+)\s*,\s*([0-9.]+)\)', response)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return (x, y)
        return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))

    return None


def _draw_som_mark(image: Image.Image, bbox_norm: list, mark_id: int, radius: int = 9) -> None:
    """Draw a single mark (red circle + label) on image — same style as apply_som."""
    w, h = image.size
    cx = int((bbox_norm[0] + bbox_norm[2]) / 2 * w)
    cy = int((bbox_norm[1] + bbox_norm[3]) / 2 * h)
    draw = ImageDraw.Draw(image)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill="red", outline="white", width=1)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    label = str(mark_id)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((cx - tw / 2, cy - th / 2 - 1), label, fill="white", font=font)


def _inject_dom_mark(image: Image.Image, bbox_norm: list, mark_id: int) -> None:
    """Draw a DOM-injected element mark: blue outline around the bbox + red circle."""
    w, h = image.size
    x1, y1 = int(bbox_norm[0] * w), int(bbox_norm[1] * h)
    x2, y2 = int(bbox_norm[2] * w), int(bbox_norm[3] * h)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=2)
    _draw_som_mark(image, bbox_norm, mark_id)


def _extract_mark(response: str) -> int | None:
    """Extract mark number from model response."""
    m = re.search(r'[Mm]ark\s*:?\s*(\d+)', response)
    return int(m.group(1)) if m else None


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
        self.raw_mode           = raw_mode
        self.tag                = tag
        self.training_style     = training_style  # True = fine-tuned mode
        self._last_content_list = []
        self._load_qwen(lora_path)
        if not raw_mode:
            self._load_omniparser()

    def _load_omniparser(self):
        sys.path.insert(0, OMNIPARSER_CODE)
        from util.utils import (  # type: ignore[import-not-found]
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

        # cap to MAX_ELEMENTS — model trained on 5-30 marks per image
        if len(label_coords) > MAX_ELEMENTS:
            keys = list(label_coords.keys())[:MAX_ELEMENTS]
            label_coords   = {k: label_coords[k] for k in keys}
            content_list   = content_list[:MAX_ELEMENTS]
            print(f"  Capped to {MAX_ELEMENTS} elements", flush=True)

        self._last_content_list = content_list
        self._raw_label_coords  = label_coords   # saved for DOM-priority re-render

        if self.training_style:
            # Render in training circle style (red circles, same as training data).
            # apply_som re-sorts elements by area and assigns NEW sequential mark IDs,
            # so we must build mark_to_center from its `placed` output — not label_coords.
            elements_for_som = [
                {"bbox": [x/iw, y/ih, (x+bw)/iw, (y+bh)/ih]}
                for (x, y, bw, bh) in label_coords.values()
            ]
            from src.som.render_som import apply_som
            som_image, placed = apply_som(image.copy(), elements_for_som)
            mark_to_center = {
                mark_id: (
                    (el["bbox"][0] + el["bbox"][2]) / 2,
                    (el["bbox"][1] + el["bbox"][3]) / 2,
                )
                for mark_id, el in placed
            }
        else:
            som_image = Image.open(io.BytesIO(base64.b64decode(encoded)))
            mark_to_center = {
                int(k): ((x + bw/2) / iw, (y + bh/2) / ih)
                for k, (x, y, bw, bh) in label_coords.items()
            }

        print(f"  Original : {image.size}  |  SoM : {som_image.size}")
        return som_image, mark_to_center, content_list

    def _rebuild_som_dom_priority(
        self,
        original_image: Image.Image,
        omni_content_list: list,
        dom_elements: list,
        max_marks: int = 15,
    ) -> tuple[Image.Image, dict, list]:
        """
        Re-render the SoM from scratch with DOM elements occupying the LOWEST
        mark IDs (0, 1, ...), then OmniParser elements fill the rest.
        Capped at max_marks to avoid overwhelming the model.

        This converts the model's Mark:0 bias into an asset: the search bar
        (or most relevant input) becomes Mark 0, so "Mark: 0" responses are
        now correct for search/input tasks.

        DOM priority: text inputs → buttons → other interactive elements.
        """
        iw, ih = original_image.size

        def _dom_order(e):
            tag = e.get("tag", "")
            typ = e.get("type", "")
            if tag in ("input", "textarea") and typ not in ("submit", "button", "checkbox", "radio", "image"):
                return 0  # text inputs first
            if tag == "button" or typ in ("submit", "button"):
                return 1  # submit buttons second
            return 2

        valid_dom = [e for e in dom_elements if len(e.get("bbox_norm", [])) == 4]
        sorted_dom = sorted(valid_dom, key=_dom_order)

        img          = original_image.copy()
        mark_to_center = {}
        content_list   = []
        mark_id        = 0
        draw           = ImageDraw.Draw(img)

        # 1. DOM elements — draw blue outline + red circle mark
        for elem in sorted_dom:
            if mark_id >= max_marks:
                break
            bbox = elem["bbox_norm"]
            x1, y1 = int(bbox[0] * iw), int(bbox[1] * ih)
            x2, y2 = int(bbox[2] * iw), int(bbox[3] * ih)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=2)
            _draw_som_mark(img, bbox, mark_id)
            mark_to_center[mark_id] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            content_list.append({
                "content": elem.get("label", elem.get("tag", "input")),
                "type":    elem.get("tag", "input"),
                "bbox":    bbox,
            })
            mark_id += 1

        # 2. OmniParser elements after
        raw = getattr(self, "_raw_label_coords", {})
        for ((x, y, bw, bh), omni_item) in zip(raw.values(), omni_content_list):
            if mark_id >= max_marks:
                break
            bbox = [x / iw, y / ih, (x + bw) / iw, (y + bh) / ih]
            _draw_som_mark(img, bbox, mark_id)
            mark_to_center[mark_id] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            content_list.append(omni_item)
            mark_id += 1

        return img, mark_to_center, content_list

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
            out = self._qwen.generate(**inputs, max_new_tokens=20)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(trimmed, skip_special_tokens=True).strip()

    def act(self, image_path: str, task: str, dom_elements: list | None = None) -> tuple[str, tuple | None]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if self.raw_mode:
            prompt = RAW_PROMPT_TEMPLATE.format(task=task)
            response = self._run_qwen(image, prompt)
            point = _parse_point(response, image_size=(w, h))
            return response, point

        print("  Running OmniParser...", flush=True)
        som_image, mark_to_center, content_list = self._run_omniparser(image)
        print(f"  Total elements : {len(mark_to_center)}", flush=True)

        if self.training_style:
            # fine-tuned mode: exact text_to_point prompt used during training
            prompt = PROMPT_TEMPLATE.format(task=task)
        else:
            # baseline mode: no element list, model guesses coordinate
            prompt = BASELINE_PROMPT_TEMPLATE.format(task=task)

        if dom_elements and self.training_style:
            # Re-render the entire SoM with DOM elements at mark IDs 0, 1, ...
            # so the model's Mark:0 bias maps to the most relevant input element.
            inference_image, mark_to_center, content_list = \
                self._rebuild_som_dom_priority(image, content_list, dom_elements, max_marks=15)
            self._last_content_list = content_list
            print(f"  SoM rebuilt: {len(mark_to_center)} marks (DOM-first)", flush=True)
        else:
            inference_image = som_image
            if dom_elements:
                # baseline mode — just append DOM marks visually at the end
                next_id = max(mark_to_center.keys(), default=-1) + 1
                for elem in dom_elements:
                    bbox = elem.get("bbox_norm", [])
                    if len(bbox) != 4:
                        continue
                    mark_to_center[next_id] = (
                        (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    )
                    content_list.append({
                        "content": elem.get("label", elem.get("tag", "input")),
                        "type":    elem.get("tag", "input"),
                        "bbox":    bbox,
                    })
                    _inject_dom_mark(inference_image, bbox, next_id)
                    next_id += 1
                self._last_content_list = content_list

        # always save the exact image Qwen sees to /tmp for debugging
        tag = self.tag or "debug"
        save_path = Path(f"/tmp/vlm_input_{tag}.png")
        inference_image.save(save_path)
        print(f"  VLM input saved: {save_path}")

        print("  Running Qwen...", flush=True)
        response = self._run_qwen(inference_image, prompt)

        # resolve point: mark lookup first, coordinate parse as fallback
        point = None
        mark_id = _extract_mark(response)
        if mark_id is not None:
            point = mark_to_center.get(mark_id)
            if point:
                print(f"  Mark {mark_id} → center {point}")
            else:
                print(f"  Mark {mark_id} not in label_coords, falling back to coord parse")
                point = _parse_point(response, image_size=(w, h))
        else:
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
                        help="Fine-tuned mode: element list prompt + mark lookup")
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