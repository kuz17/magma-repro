# src/agent/inference_server.py
"""
Inference server — wraps DemoRunner in a FastAPI endpoint.

Usage:
    python -m src.agent.inference_server \
        --mode finetuned --lora models/lora_adapter --port 8787

    python -m src.agent.inference_server --mode baseline --port 8787

POST /act    { "image_b64": "<base64 PNG>", "task": "click X" }
POST /plan   { "image_b64": "<base64 PNG>", "prompt": "<fully-formatted planning prompt>" }
GET  /health
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

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
    prompt:    str   # fully-formatted planning prompt (goal + history baked in client-side)

class PlanResponse(BaseModel):
    raw_response: str
    error:        Optional[str] = None


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
    from PIL import Image

    log.info("Loading models for mode='%s'...", mode)
    runner = DemoRunner(
        lora_path=lora_path if mode == "finetuned" else None,
        raw_mode=False,
        tag=mode,
        training_style=(mode == "finetuned"),
    )
    log.info("Runner ready.")

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
        """
        Planning-only endpoint: runs the BASE model (adapter disabled via
        PEFT's disable_adapter() context manager) on a raw image + prompt,
        with no OmniParser/SoM/mark-lookup involved.

        Mirrors exactly what tests/planner_agent.py's plan_next_action()
        did in-process before this split — the planner step here (base
        model) and the executor step at /act (fine-tuned adapter, full
        grounding pipeline) share the SAME loaded model instance
        server-side. No double VRAM, no load_adapter/unload bug — the
        property the original in-process design was built to preserve,
        just moved across a network call instead of a Python function call.

        Requires the server to have been started with --mode finetuned
        (an adapter must be loaded for disable_adapter() to have anything
        to disable). If started with --mode baseline, this endpoint still
        works, but disable_adapter() is effectively a no-op — /plan and
        /act would behave identically since there's no adapter either way.
        """
        try:
            png_bytes = base64.b64decode(req.image_b64)
        except Exception as exc:
            return PlanResponse(raw_response="", error=f"Bad base64: {exc}")

        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            if hasattr(runner._qwen, "disable_adapter"):
                with runner._qwen.disable_adapter():
                    raw_response = runner._run_qwen(image, req.prompt)
            else:
                raw_response = runner._run_qwen(image, req.prompt)
            return PlanResponse(raw_response=raw_response)
        except Exception as exc:
            log.exception("Planning error")
            return PlanResponse(raw_response="", error=str(exc))

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