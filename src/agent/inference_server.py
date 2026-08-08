# src/agent/inference_server.py
"""
Inference server — wraps DemoRunner in a FastAPI endpoint.

Usage:
    python -m src.agent.inference_server \
        --mode finetuned --lora models/lora_adapter --port 8787

    python -m src.agent.inference_server --mode baseline --port 8787

Endpoints:
    GET  /health
    POST /act        { image_b64, task, dom_elements? }        -> click grounding (adapter ON)
    POST /plan       { image_b64, goal, history? }              -> next action (adapter OFF)
    POST /describe   { image_b64, question?, max_new_tokens? }  -> description/VQA (adapter OFF)

/plan and /describe both run the BASE model via disable_adapter() - the LoRA adapter
was trained only on click-to-coordinate grounding, so it's the wrong mode for picking
a next action or describing a scene. Same single DemoRunner instance serves all three;
OmniParser (needed only by /act) is already paid for once at startup, adapter state is
just toggled per-request - no reload, no extra VRAM.
"""

from __future__ import annotations

import argparse
import base64
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Schema  (module-level so Pydantic resolves refs correctly)
# ══════════════════════════════════════════════════════════════════════════════

class DomElementInfo(BaseModel):
    tag:       str
    type:      str
    label:     str
    bbox_norm: List[float]

class ActRequest(BaseModel):
    image_b64:    str
    task:         str
    dom_elements: List[DomElementInfo] = []

class ElementInfo(BaseModel):
    id:      int
    content: str
    type:    str
    bbox:    List[float]

class ActResponse(BaseModel):
    click_norm:   Optional[List[float]]
    mark_id:      Optional[int]
    raw_response: str
    elements:     List[ElementInfo]
    error:        Optional[str] = None


class PlanRequest(BaseModel):
    image_b64: str
    goal:      str
    history:   List[str] = []

class PlanResponse(BaseModel):
    action:       Optional[str]   # SEARCH / CLICK / SCROLL / DONE, or None if unparseable
    arg:          Optional[str]
    raw_response: str
    error:        Optional[str] = None


class DescribeRequest(BaseModel):
    image_b64:      str
    question:       Optional[str] = None   # omit for a generic description
    max_new_tokens: int = 200

class DescribeResponse(BaseModel):
    response: str
    error:    Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Shared prompt/parsing bits (kept identical to router_agent.py's local versions
# so remote and local behavior match)
# ══════════════════════════════════════════════════════════════════════════════

ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time. You respond with ONLY a function call, nothing else - no explanation, no extra words.

Valid function calls (pick exactly one):
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Goal: {goal}

Already completed:
{history}

Look at the screenshot. Respond with ONLY one function call for the next step:"""

QUESTION_RE = re.compile(r"^\s*(what|where|who|how many|is there|does|are there)\b", re.IGNORECASE)

# Kaggle T4/P100 have far more headroom than the 4GB local card this was tuned for,
# but still cap absurdly large uploads (e.g. a raw phone photo) so one request can't
# blow past whatever's free.
MAX_IMAGE_SIDE = 1536


def _resize_for_vram(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    log.info("Resizing %dx%d -> %dx%d to stay within VRAM budget", w, h, *new_size)
    return image.resize(new_size, Image.LANCZOS)


def _decode_image(image_b64: str) -> Image.Image:
    png_bytes = base64.b64decode(image_b64)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name
    try:
        return Image.open(tmp_path).convert("RGB")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DemoRunner patch — expose _last_content_list
# ══════════════════════════════════════════════════════════════════════════════

def _patch_demo_runner():
    # DemoRunner now sets self._last_content_list in _run_omniparser;
    # this patch is kept as a no-op for backward compatibility.
    pass


# ══════════════════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════════════════

def build_app(mode: str, lora_path: Optional[str]) -> FastAPI:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _patch_demo_runner()
    from src.agent.click_visualizer import DemoRunner

    log.info("Loading models for mode='%s'...", mode)
    runner = DemoRunner(
        lora_path=lora_path if mode == "finetuned" else None,
        raw_mode=False,
        tag=mode,
        training_style=(mode == "finetuned"),
    )
    log.info("Runner ready.")

    def _generate_prose(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Like runner._run_qwen, but with a token budget for prose instead of
        the 20-token budget _run_qwen hardcodes for coordinate/mark output."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = runner._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = runner._processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(runner._qwen.device)

        with torch.no_grad():
            out = runner._qwen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return runner._processor.decode(trimmed, skip_special_tokens=True).strip()

    app = FastAPI(title="magma-repro inference server")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": mode}

    @app.post("/act")
    def act(req: ActRequest) -> ActResponse:
        # decode image
        try:
            png_bytes = base64.b64decode(req.image_b64)
        except Exception as exc:
            return ActResponse(
                click_norm=None, mark_id=None,
                raw_response="", elements=[],
                error=f"Bad base64: {exc}",
            )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        error_msg:    Optional[str]     = None
        raw_response: str               = ""
        point:        Optional[tuple]   = None
        mark_id:      Optional[int]     = None
        elements:     List[ElementInfo] = []

        dom_elems = [
            {"tag": e.tag, "type": e.type, "label": e.label, "bbox_norm": e.bbox_norm}
            for e in req.dom_elements
        ]

        try:
            raw_response, point = runner.act(tmp_path, req.task, dom_elements=dom_elems)

            m = re.search(r'[Mm]ark\s*:?\s*(\d+)', raw_response)
            mark_id = int(m.group(1)) if m else None

            for i, elem in enumerate(getattr(runner, "_last_content_list", []) or []):
                elements.append(ElementInfo(
                    id=i,
                    content=str(elem.get("content") or ""),
                    type=str(elem.get("type", "element")),
                    bbox=list(elem.get("bbox") or [0, 0, 0, 0]),
                ))
        except Exception as exc:
            log.exception("Pipeline error")
            error_msg = str(exc)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return ActResponse(
            click_norm=list(point) if point else None,
            mark_id=mark_id,
            raw_response=raw_response,
            elements=elements,
            error=error_msg,
        )

    @app.post("/plan")
    def plan(req: PlanRequest) -> PlanResponse:
        try:
            image = _decode_image(req.image_b64)
        except Exception as exc:
            return PlanResponse(action=None, arg=None, raw_response="", error=f"Bad base64: {exc}")

        try:
            hist_str = "\n".join(f"- {h}" for h in req.history) if req.history else "(none - this is the first step)"
            prompt = PLANNING_PROMPT_TEMPLATE.format(goal=req.goal, history=hist_str)
            with runner._qwen.disable_adapter():
                raw_response = runner._run_qwen(image, prompt)
            m = ACTION_RE.search(raw_response)
            action = m.group(1).upper() if m else None
            arg = m.group(2).strip() if m else None
            return PlanResponse(action=action, arg=arg, raw_response=raw_response)
        except Exception as exc:
            log.exception("Plan error")
            return PlanResponse(action=None, arg=None, raw_response="", error=str(exc))

    @app.post("/describe")
    def describe(req: DescribeRequest) -> DescribeResponse:
        try:
            image = _decode_image(req.image_b64)
            image = _resize_for_vram(image)
        except Exception as exc:
            return DescribeResponse(response="", error=f"Bad base64: {exc}")

        try:
            if req.question and req.question.strip():
                prompt = req.question.strip()
            else:
                prompt = "Describe what you see in this image in detail."
            with runner._qwen.disable_adapter():
                response = _generate_prose(image, prompt, req.max_new_tokens)
            return DescribeResponse(response=response)
        except Exception as exc:
            log.exception("Describe error")
            return DescribeResponse(response="", error=str(exc))

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="magma-repro inference server")
    parser.add_argument("--mode", choices=["baseline", "finetuned"],
                        default="finetuned")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.mode == "finetuned" and not args.lora:
        parser.error("--lora is required when --mode finetuned")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(mode=args.mode, lora_path=args.lora)

    import uvicorn
    log.info("Starting on %s:%d  mode=%s", args.host, args.port, args.mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()