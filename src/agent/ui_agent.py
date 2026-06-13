# src/agent/ui_agent.py
"""
UIAgent — model-agnostic UI grounding agent.

Mirrors the architecture of Magma's agents/ui_agent/app.py but is
decoupled from Gradio and supports two VLM backends and two SoM sources.

─── VLM backends ─────────────────────────────────────────────────────────
  MagmaBackend   microsoft/Magma-8B via AutoModelForCausalLM
                 (trust_remote_code=True, bfloat16, coords in 0–1000 range)

  QwenBackend    Qwen/Qwen2.5-VL-3B-Instruct via Qwen2_5_VLForConditionalGeneration
                 (4-bit NF4 quant, coords in 0–1 range)

─── SoM sources ──────────────────────────────────────────────────────────
  OmniParserSoM  YOLO + Florence2 + OCR — production, any screenshot.
                 Requires OmniParser weights at agents/ui_agent/weights/.
                 Optional import; agent falls back gracefully if unavailable.

  AnnotationSoM  Our render_som.py + _marks.json sidecar — training/eval.
                 No extra weights needed.

─── Interaction modes (matching Magma app.py) ────────────────────────────
  1. Empty instruction  → OmniParser-only, return parsed element list
  2. "Q: ..." prefix    → VQA, no grounding
  3. Task instruction   → SoM grounding, return Action

─── Usage ────────────────────────────────────────────────────────────────
  # Qwen backend + annotation SoM (eval / fine-tuning experiment)
  agent = UIAgent.with_qwen()
  action = agent.act("click the search button",
                     som_image_path="..._som.png",
                     marks_path="..._marks.json")

  # Qwen backend + OmniParser SoM (production-style, any screenshot)
  agent = UIAgent.with_qwen(use_omniparser=True)
  action = agent.act("click the search button", image_path="screenshot.png")

  # Magma backend + OmniParser SoM (exact Magma demo replication)
  agent = UIAgent.with_magma(use_omniparser=True)
  action = agent.act("click the search button", image_path="screenshot.png")

  # VQA mode (works with both backends)
  answer = agent.act("Q: What is the title of the page?",
                     image_path="screenshot.png")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# ── optional OmniParser import ─────────────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, "/data/Magma/OmniParser")
    from util.utils import (
        check_ocr_box,
        get_yolo_model,
        get_caption_model_processor,
        get_som_labeled_img,
    )
    _OMNIPARSER_AVAILABLE = True
except ImportError:
    _OMNIPARSER_AVAILABLE = False
# ── coordinate constants ────────────────────────────────────────────────────
_MAGMA_COORD_SCALE = 1000.0   # Magma outputs 0–1000; divide to normalise
_QWEN_COORD_SCALE  = 1.0      # Qwen outputs 0–1 directly

LOCAL_QWEN_PATH = "/data/Magma/magma-repro/models/qwen2_5_vl_3b"


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Action:
    """Result of a grounding call."""
    mark_id:      Optional[int]    = None
    bbox:         Optional[tuple]  = None   # (x1, y1, x2, y2) normalised 0–1
    point:        Optional[tuple]  = None   # (cx, cy) normalised 0–1
    raw_response: str              = ""
    parsed_elements: str           = ""     # OmniParser element list (if any)

    def __repr__(self) -> str:
        return (f"Action(mark_id={self.mark_id}, "
                f"point={self.point}, bbox={self.bbox})")


@dataclass
class SoMResult:
    """Output of a SoM source."""
    som_image: Image.Image
    # list of (mark_id, element_dict); element_dict has at minimum "bbox"
    placed: list = field(default_factory=list)
    parsed_content: str = ""      # human-readable element descriptions


# ══════════════════════════════════════════════════════════════════════════════
# SoM sources
# ══════════════════════════════════════════════════════════════════════════════

class _SoMSource(ABC):
    @abstractmethod
    def generate(self, image: Image.Image) -> SoMResult: ...


class AnnotationSoM(_SoMSource):
    """
    Uses pre-rendered _som.png + _marks.json sidecar from our pipeline.
    Requires both paths to be provided at call time via generate_from_paths().
    Intended for training / eval against SeeClick-Web ground truth.
    """

    def generate(self, image: Image.Image) -> SoMResult:
        raise RuntimeError(
            "AnnotationSoM requires explicit paths. "
            "Call generate_from_paths(som_image_path, marks_path) instead."
        )

    def generate_from_paths(
        self,
        som_image_path: str | Path,
        marks_path: str | Path,
    ) -> SoMResult:
        som_image = Image.open(som_image_path).convert("RGB")
        with open(marks_path) as f:
            placed = json.load(f)   # [[mark_id, {instruction, bbox, data_type}], ...]
        return SoMResult(som_image=som_image, placed=placed)


class OmniParserSoM(_SoMSource):
    """
    Generates SoM at runtime via OmniParser (YOLO + Florence2 + OCR).
    Replicates exactly the element-detection stage in Magma's app.py.

    Requires:
        weights/icon_detect/model.pt
        weights/icon_caption/  (Florence-2 model)
    """

    def __init__(
        self,
        yolo_path: str = "models/omniparser/icon_detect/model.pt",
        caption_model: str = "florence2",
        caption_path: str = "models/omniparser/icon_caption",
        box_threshold: float = 0.05,
        iou_threshold: float = 0.10,
        use_paddleocr: bool = True,
        imgsz: int = 640,
    ):
        if not _OMNIPARSER_AVAILABLE:
            raise ImportError(
                "OmniParser utilities not found. "
                "Clone https://github.com/microsoft/OmniParser to /data/Magma/OmniParser "
                "and add it to sys.path."
            )
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.use_paddleocr = use_paddleocr
        self.imgsz = imgsz
        #self._mark_helper = MarkHelper()

        print("Loading OmniParser YOLO model...")
        self.yolo_model = get_yolo_model(model_path=yolo_path)
        print("Loading OmniParser caption model...")
        self.caption_proc = get_caption_model_processor(
            model_name=caption_model,
            model_name_or_path=caption_path,
        )
        print("OmniParser ready.")

    def generate(self, image: Image.Image) -> SoMResult:
        import base64, io as _io

        w, h = image.size
        box_overlay_ratio = w / 3200
        draw_bbox_config = {
            "text_scale":     0.8  * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding":   max(int(3 * box_overlay_ratio), 1),
            "thickness":      max(int(3 * box_overlay_ratio), 1),
        }

        ocr_result, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=self.use_paddleocr,
        )
        text, ocr_bbox = ocr_result

        dino_img_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.yolo_model,
            BOX_TRESHOLD=self.box_threshold,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_proc,
            ocr_text=text,
            iou_threshold=self.iou_threshold,
            imgsz=self.imgsz,
        )

        som_image = Image.open(_io.BytesIO(base64.b64decode(dino_img_b64)))
        parsed_content = "\n".join(
            [f"icon {i}: " + str(v) for i, v in enumerate(parsed_content_list)]
        )

        # Convert label_coordinates (xywh, pixel) → placed format
        # placed: [(mark_id, {"bbox": [x1,y1,x2,y2] normalised}), ...]
        placed = []
        for key, val in label_coordinates.items():
            x1 = val[0] / w
            y1 = val[1] / h
            x2 = (val[0] + val[2]) / w
            y2 = (val[1] + val[3]) / h
            placed.append((int(key), {"bbox": [x1, y1, x2, y2]}))

        return SoMResult(
            som_image=som_image,
            placed=placed,
            parsed_content=parsed_content,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VLM backends
# ══════════════════════════════════════════════════════════════════════════════

class _VLMBackend(ABC):

    SYSTEM_PROMPT = "You are agent that can see, talk and act."

    # Prompt templates — subclasses override if needed
    SOM_PROMPT = (
        "<image>\nIn this view I need to click a button to \"{instruction}\"? "
        "Provide the coordinates and the mark index of the containing bounding box if applicable."
    )
    QA_PROMPT = "<image>\n{instruction} Answer the question briefly."

    # Coordinate scale: divide parsed numbers by this to get 0–1 normalised
    _COORD_SCALE: float = 1.0

    @abstractmethod
    def _infer(self, prompt_text: str, image: Image.Image) -> str: ...

    def infer_grounding(self, instruction: str, image: Image.Image) -> str:
        prompt = self.SOM_PROMPT.format(instruction=instruction)
        return self._infer(prompt, image)

    def infer_qa(self, instruction: str, image: Image.Image) -> str:
        # strip the "Q:" prefix
        question = instruction[2:].strip()
        prompt = self.QA_PROMPT.format(instruction=question)
        return self._infer(prompt, image)

    # ── shared response parsing ───────────────────────────────────────────

    def parse(self, response: str, placed: list) -> Action:
        """
        Parse raw model response into an Action.
        Handles:
          - "Mark: N" extraction
          - "Coordinate: (x, y)" or "(x, y, x2, y2)"
          - Magma-style <box>(...)</box>
          - Fallback: any two or four floats in the string
        Coordinates are divided by _COORD_SCALE to normalise to 0–1.
        """
        s = self._COORD_SCALE
        mark_id = None
        bbox    = None
        point   = None

        # mark ID
        m = re.search(r"[Mm]ark[:\s]+(\d+)", response)
        if m:
            mark_id = int(m.group(1))

        # Magma-style <box>(x1,y1),(x2,y2)</box>
        box_m = re.search(
            r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>", response
        )
        if box_m:
            nums = [int(box_m.group(i)) / s for i in range(1, 5)]
            bbox  = tuple(nums)
            point = ((nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2)

        if bbox is None:
            # "Coordinate: (a, b)" or "Coordinate: (a, b, c, d)"
            c = re.search(r"[Cc]oordinate[:\s]+\(([0-9.,\s]+)\)", response)
            if c:
                nums = [float(x.strip()) / s
                        for x in c.group(1).split(",") if x.strip()]
                if len(nums) == 4:
                    bbox  = tuple(nums)
                    point = ((nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2)
                elif len(nums) == 2:
                    point = tuple(nums)

        if bbox is None and point is None:
            # last resort: grab any 2 or 4 floats from the string
            floats = [float(x) / s for x in re.findall(r"-?\d+\.?\d*", response)]
            if len(floats) >= 4:
                bbox  = tuple(floats[:4])
                point = ((floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2)
            elif len(floats) == 2:
                point = tuple(floats[:2])

        # look up bbox from placed marks via mark_id if still missing
        if mark_id is not None:
            for mid, el in placed:
                if mid == mark_id:
                    if bbox is None:
                        bbox = tuple(el["bbox"])
                    if point is None:
                        x1, y1, x2, y2 = el["bbox"]
                        point = ((x1 + x2) / 2, (y1 + y2) / 2)
                    break

        return Action(mark_id=mark_id, bbox=bbox, point=point,
                      raw_response=response)


# ── Magma backend ─────────────────────────────────────────────────────────

class MagmaBackend(_VLMBackend):
    """
    Wraps microsoft/Magma-8B exactly as in app.py.
    Requires ~20 GB VRAM (bfloat16, no quantisation).
    """
    _COORD_SCALE = _MAGMA_COORD_SCALE

    def __init__(
        self,
        model_id: str = "microsoft/Magma-8B",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_new_tokens: int = 32, #128
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens

        print(f"Loading {model_id} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        print("Magma ready.")

    def _build_prompt(self, raw_prompt: str) -> str:
        """Apply Magma's image-token substitution + chat template."""
        if self.model.config.mm_use_image_start_end:
            raw_prompt = raw_prompt.replace(
                "<image>", "<image_start><image><image_end>"
            )
        convs = [
            {"role": "system",  "content": self.SYSTEM_PROMPT},
            {"role": "user",    "content": raw_prompt},
        ]
        return self.processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _infer(self, prompt_text: str, image: Image.Image) -> str:
        prompt = self._build_prompt(prompt_text)
        inputs = self.processor(
            images=[image], texts=prompt, return_tensors="pt"
        )
        # Magma's processor returns unbatched tensors — add batch dim
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"]  = inputs["image_sizes"].unsqueeze(0)
        inputs = {k: v.to(self.dtype).to(self.device)
                  if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items()}

        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )
        output_ids = self.model.generate(
            **inputs,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        # strip prompt from output
        prompt_decoded = self.processor.batch_decode(
            inputs["input_ids"], skip_special_tokens=True
        )[0]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return response.replace(prompt_decoded, "").strip()


# ── Qwen backend ──────────────────────────────────────────────────────────

class QwenBackend(_VLMBackend):
    """
    Wraps Qwen/Qwen2.5-VL-3B-Instruct with 4-bit NF4 quantisation.
    Fits on 4 GB VRAM (GTX 1650 / T4).
    Coordinates are 0–1 normalised.
    """
    _COORD_SCALE = _QWEN_COORD_SCALE

    # Qwen doesn't use a literal <image> token in the text prompt;
    # the image is passed separately via the processor.
    SOM_PROMPT = (
        "To execute the step \"{instruction}\", "
        "where do I direct my attention? "
        "Please provide the coordinate and the bounding box's mark index if applicable."
    )
    QA_PROMPT = "{instruction} Answer the question briefly."

    def __init__(
        self,
        model_id: str = LOCAL_QWEN_PATH,
        load_in_4bit: bool = True,
        max_new_tokens: int = 32,#128,
    ):
        from transformers import AutoProcessor, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info as _pvf
        self._process_vision_info = _pvf
        self.max_new_tokens = max_new_tokens

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _M
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as _M  # type: ignore

        quant = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if load_in_4bit else None
        )

        print(f"Loading {model_id}  (4-bit={load_in_4bit}) ...")
        self.model = _M.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        print("Qwen ready.")

    @torch.inference_mode()
    def _infer(self, prompt_text: str, image: Image.Image) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# UIAgent orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class UIAgent:
    """
    Orchestrates SoM generation + VLM inference.

    Interaction modes (matching Magma app.py):
      1. Empty instruction  → return OmniParser element list only (no VLM)
      2. "Q: ..."           → VQA on raw screenshot
      3. Task instruction   → SoM grounding → Action
    """

    def __init__(self, backend: _VLMBackend, som_source: _SoMSource):
        self.backend    = backend
        self.som_source = som_source

    # ── factory helpers ───────────────────────────────────────────────────

    @classmethod
    def with_qwen(
        cls,
        model_id: str = LOCAL_QWEN_PATH,
        load_in_4bit: bool = True,
        use_omniparser: bool = False,
        **omni_kwargs,
    ) -> "UIAgent":
        backend = QwenBackend(model_id=model_id, load_in_4bit=load_in_4bit)
        som     = OmniParserSoM(**omni_kwargs) if use_omniparser else AnnotationSoM()
        return cls(backend=backend, som_source=som)

    @classmethod
    def with_magma(
        cls,
        model_id: str = "microsoft/Magma-8B",
        use_omniparser: bool = True,
        **omni_kwargs,
    ) -> "UIAgent":
        backend = MagmaBackend(model_id=model_id)
        som     = OmniParserSoM(**omni_kwargs) if use_omniparser else AnnotationSoM()
        return cls(backend=backend, som_source=som)

    # ── main entry point ──────────────────────────────────────────────────

    def act(
        self,
        instruction: str,
        *,
        image_path: str | Path | None = None,
        som_image_path: str | Path | None = None,
        marks_path: str | Path | None = None,
    ) -> Action:
        """
        Run one agent step.

        For AnnotationSoM (eval mode):
            agent.act(task, som_image_path="..._som.png",
                            marks_path="..._marks.json")

        For OmniParserSoM (production mode):
            agent.act(task, image_path="screenshot.png")

        For VQA (either SoM source):
            agent.act("Q: What is the title?", image_path="screenshot.png")
        """
        # ── mode 2: VQA ──────────────────────────────────────────────────
        if instruction.startswith("Q:") or instruction.startswith("q:"):
            image = self._load_image(image_path, som_image_path)
            response = self.backend.infer_qa(instruction, image)
            return Action(raw_response=response)

        # ── SoM generation ───────────────────────────────────────────────
        if isinstance(self.som_source, AnnotationSoM):
            if som_image_path is None or marks_path is None:
                raise ValueError(
                    "AnnotationSoM requires som_image_path and marks_path."
                )
            som_result = self.som_source.generate_from_paths(
                som_image_path, marks_path
            )
        else:
            image = self._load_image(image_path)
            som_result = self.som_source.generate(image)

        # ── mode 1: empty instruction → element list only ─────────────────
        if not instruction.strip():
            return Action(parsed_elements=som_result.parsed_content)

        # ── mode 3: grounding ─────────────────────────────────────────────
        response = self.backend.infer_grounding(instruction, som_result.som_image)
        action   = self.backend.parse(response, som_result.placed)
        action.parsed_elements = som_result.parsed_content
        return action

    @staticmethod
    def _load_image(
        image_path: str | Path | None = None,
        fallback_path: str | Path | None = None,
    ) -> Image.Image:
        path = image_path or fallback_path
        if path is None:
            raise ValueError("Provide image_path or som_image_path.")
        return Image.open(path).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    RENDERS = Path("data/interim/renders/seeclick_web/batch")
    som_files = sorted(RENDERS.glob("*_som.png"))
    if not som_files:
        print("No SoM renders found. Run test_render_batch.py first.")
        sys.exit(1)

    som_path   = som_files[0]
    marks_path = RENDERS / (som_path.stem.replace("_som", "_marks") + ".json")

    print(f"SoM image : {som_path}")
    print(f"Marks     : {marks_path}\n")

    # Qwen + annotation SoM (eval mode — no OmniParser needed)
    agent  = UIAgent.with_qwen(load_in_4bit=True)
    action = agent.act(
        "click the search button",
        som_image_path=som_path,
        marks_path=marks_path,
    )

    print(f"Action   : {action}")
    print(f"Response : {action.raw_response}")