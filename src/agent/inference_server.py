# src/agent/inference_server.py
"""
Inference server — wraps DemoRunner in a FastAPI endpoint.

Usage:
    python -m src.agent.inference_server \
        --mode finetuned --lora models/lora_adapter --port 8787

    python -m src.agent.inference_server --mode baseline --port 8787

    python -m src.agent.inference_server \
        --mode fused --fused-lora models/lora_adapter_fused --port 8787

POST /act        { "image_b64": "<base64 PNG>", "task": "click X" }
POST /plan       { "image_b64": "<base64 PNG>", "prompt": "<fully-formatted planning prompt>" }
POST /act_fused  { "image_b64": "<base64 PNG>", "task": "click X", "history": [...], "dom_elements": [...] }
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

class FusedHistoryEntry(BaseModel):
    tag:    str
    label:  str
    action: str
    value:  Optional[str] = None

class ActFusedRequest(BaseModel):
    image_b64:    str
    task:         str
    history:      List[FusedHistoryEntry] = []
    dom_elements: List[DomElementInfo] = []

class ActFusedResponse(BaseModel):
    click_norm:   Optional[List[float]]
    action:       Optional[str]
    mark_id:      Optional[int]
    value:        Optional[str]
    raw_response: str
    elements:     List[ElementInfo]
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

def build_app(
    mode: str,
    lora_path: Optional[str],
    fused_lora_path: Optional[str] = None,
) -> FastAPI:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _patch_demo_runner()
    from src.agent.click_visualizer import DemoRunner
    from PIL import Image

    log.info("Loading models for mode='%s'...", mode)
    if mode == "fused":
        # Fused adapter loaded SOLO as the default adapter — no dual-loading
        # with the smoke adapter. Keeps this to one adapter's VRAM footprint
        # (confirmed necessary on 4GB local hardware; harmless on Kaggle too).
        runner = DemoRunner(
            lora_path=fused_lora_path,
            raw_mode=False,
            tag=mode,
            training_style=True,
        )
    else:
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
        model) and the executor step at /act or /act_fused share the SAME
        loaded model instance server-side. No double VRAM, no
        load_adapter/unload bug — the property the original in-process
        design was built to preserve, just moved across a network call
        instead of a Python function call.

        disable_adapter() works the same regardless of WHICH adapter is
        currently loaded as default — smoke, fused, or none — so this
        endpoint needs no changes for --mode fused. Independently
        confirmed via tests/smoke_disable_adapter.py (re-run against
        lora_adapter_fused specifically): with/disabled/with-again outputs
        compared and matched cleanly, no leakage of fused-adapter behavior
        into the disabled state.
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

    @app.post("/act_fused")
    def act_fused(req: ActFusedRequest) -> ActFusedResponse:
        """
        Single-call fused-adapter endpoint: OmniParser + fused adapter
        decide ACTION (CLICK/TYPE/SELECT) + MARK + VALUE together, in one
        forward pass — replacing the separate OmniParser-render +
        smoke-adapter-grounding round trip that /act makes.

        dom_elements, when provided, triggers the same DOM-priority SoM
        rebuild /act uses — mitigates OmniParser's mark ordering diverging
        from whatever ordering Magma's own SoM pipeline used during
        training. Not a complete fix (confirmed via real-page testing:
        mark resolution can still land on an unrelated element even with
        DOM elements present) — see act_fused()'s docstring in
        click_visualizer.py for the full caveat.

        Requires the server to have been started with --mode fused (or
        any mode where a fused adapter was loaded, named either "default"
        or "fused" — see DemoRunner.act_fused()'s has_named_fused check).
        """
        try:
            png_bytes = base64.b64decode(req.image_b64)
        except Exception as exc:
            return ActFusedResponse(
                click_norm=None, action=None, mark_id=None, value=None,
                raw_response="", elements=[],
                error=f"Bad base64: {exc}",
            )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        error_msg:    Optional[str]     = None
        raw_response: str               = ""
        parsed:       Optional[dict]    = None
        point:        Optional[tuple]   = None
        elements:     List[ElementInfo] = []

        history = [
            {"tag": h.tag, "label": h.label, "action": h.action, "value": h.value}
            for h in req.history
        ]
        dom_elems = [
            {"tag": e.tag, "type": e.type, "label": e.label, "bbox_norm": e.bbox_norm}
            for e in req.dom_elements
        ]

        try:
            raw_response, parsed, point = runner.act_fused(
                tmp_path, req.task, history=history, dom_elements=dom_elems,
            )

            for i, elem in enumerate(getattr(runner, "_last_content_list", []) or []):
                elements.append(ElementInfo(
                    id=i,
                    content=str(elem.get("content") or ""),
                    type=str(elem.get("type", "element")),
                    bbox=list(elem.get("bbox") or [0, 0, 0, 0]),
                ))
        except Exception as exc:
            log.exception("Fused pipeline error")
            error_msg = str(exc)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return ActFusedResponse(
            click_norm=list(point) if point else None,
            action=parsed.get("action") if parsed else None,
            mark_id=parsed.get("mark") if parsed else None,
            value=parsed.get("value") if parsed else None,
            raw_response=raw_response,
            elements=elements,
            error=error_msg,
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="magma-repro inference server")
    parser.add_argument("--mode", choices=["baseline", "finetuned", "fused"],
                        default="finetuned")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--fused-lora", default=None,
                        help="Path to the fused adapter dir — required when --mode fused")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.mode == "finetuned" and not args.lora:
        parser.error("--lora is required when --mode finetuned")
    if args.mode == "fused" and not args.fused_lora:
        parser.error("--fused-lora is required when --mode fused")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(mode=args.mode, lora_path=args.lora, fused_lora_path=args.fused_lora)

    import uvicorn
    log.info("Starting on %s:%d  mode=%s", args.host, args.port, args.mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()