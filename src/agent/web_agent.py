# src/agent/web_agent.py
"""
Interactive web agent — wires BrowserEnv (Playwright) and the inference
server (OmniParser + Qwen) into a live browser agent loop.

The inference server must already be running before starting this script:
    python -m src.agent.inference_server --mode finetuned --lora models/lora_adapter --port 8787

Usage:
    python -m src.agent.web_agent --url https://www.google.com
    python -m src.agent.web_agent --url https://www.google.com --port 8787 --headless

Commands at the interactive prompt:
    <any text>      Send as a click task to the agent
    go <url>        Navigate to a URL
    back            Go back one page
    scroll down     Scroll down
    scroll up       Scroll up
    type <text>     Type text at the current focus
    enter           Press Enter
    screenshot      Save a screenshot to outputs/browser/
    url             Print the current URL
    quit / exit     Close the browser and exit
"""

from __future__ import annotations

import argparse
import base64
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from src.agent.browser_env import BrowserEnv, ScreenshotResult

log = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("outputs/browser")


# ══════════════════════════════════════════════════════════════════════════════
# Inference client
# ══════════════════════════════════════════════════════════════════════════════

class InferenceClient:
    """
    Thin wrapper around the inference server's /act endpoint.
    Handles base64 encoding and error reporting.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8787"):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            log.info("Inference server ready — mode: %s", data.get("mode"))
            print(f"  [server] mode={data.get('mode')}  status={data.get('status')}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach inference server at {self.base_url}. "
                f"Is it running?\n  Error: {exc}"
            )

    def act(self, png_bytes: bytes, task: str, dom_elements: list | None = None) -> dict:
        """
        Send a screenshot + task to the server.

        Returns the full response dict:
            click_norm   : [x, y] normalised 0-1, or None
            mark_id      : int or None
            raw_response : str
            elements     : list of element dicts
            error        : str or None
        """
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task}
        if dom_elements:
            payload["dom_elements"] = dom_elements

        try:
            r = requests.post(
                f"{self.base_url}/act",
                json=payload,
                timeout=240,   # OmniParser (CPU) + Qwen (4-bit GPU) takes ~60-100s
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "mark_id": None,
                    "raw_response": "", "elements": [],
                    "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"click_norm": None, "mark_id": None,
                    "raw_response": "", "elements": [],
                    "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Web agent
# ══════════════════════════════════════════════════════════════════════════════

class WebAgent:
    """
    Interactive web agent.

    Owns a BrowserEnv (Playwright) and an InferenceClient (localhost HTTP).
    On each task:
      1. Screenshot the current browser viewport
      2. Send PNG + task to inference server
      3. Receive normalised click coordinates
      4. Fire click in Playwright
      5. Print result summary
    """

    def __init__(
        self,
        server_url:  str  = "http://127.0.0.1:8787",
        headless:    bool = False,
        save_screenshots: bool = True,
        max_retries: int  = 0,
    ):
        self.client      = InferenceClient(server_url)
        self.browser     = BrowserEnv(
            headless=headless,
            save_screenshots=save_screenshots,
        )
        self.max_retries = max_retries
        self._step       = 0

        if save_screenshots:
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── navigation helpers ────────────────────────────────────────────────

    def navigate(self, url: str):
        if url.lower().startswith("to "):
            url = url[3:].strip()
        if not url.startswith("http"):
            url = "https://" + url
        print(f"  → navigating to {url}")
        try:
            self.browser.navigate(url)
            print(f"  ✓ {self.browser.current_url()}")
        except Exception as exc:
            print(f"  ✗ navigation error: {exc}")

    def back(self):
        self.browser._page.go_back()
        try:
            self.browser.wait_for_load(timeout_ms=15_000)
        except Exception:
            pass  # slow pages may never reach networkidle — that's fine
        print(f"  ✓ {self.browser.current_url()}")

    def scroll(self, direction: str = "down"):
        delta = 400 if direction == "down" else -400
        self.browser.scroll(delta_y=delta)
        print(f"  ✓ scrolled {direction}")

    def type_text(self, text: str):
        self.browser.type_text(text)
        print(f"  ✓ typed: {text!r}")

    def press_enter(self):
        self.browser.press_key("Enter")
        self.browser.wait_for_load()
        print(f"  ✓ Enter  →  {self.browser.current_url()}")

    def save_screenshot(self):
        ss = self.browser.screenshot(wait_stable=False)
        path = SCREENSHOT_DIR / f"manual_{int(time.time())}.png"
        ss.image.save(path)
        print(f"  ✓ saved → {path}")

    def search(self, query: str) -> bool:
        """
        Compound action: find the first visible text input via DOM,
        click it, clear it, type the query, then submit.
        Skips VLM inference — uses DOM coordinates directly.

        Submits by clicking the nearest submit button if one is found
        (more reliable than Enter, which autocomplete dropdowns intercept).
        """
        elems = self.browser.get_interactive_elements()
        input_elem = next(
            (e for e in elems
             if e["tag"] in ("input", "textarea")
             and e["type"] not in ("submit", "button", "checkbox", "radio", "image", "hidden")),
            None,
        )
        if input_elem is None:
            print("  ✗ no text input found on this page")
            return False

        submit_elem = next(
            (e for e in elems
             if e["tag"] == "button"
             or e.get("type") in ("submit", "button")),
            None,
        )

        # Click input, select-all to clear existing text, type query
        bbox = input_elem["bbox_norm"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        print(f"  → clicking '{input_elem['label']}' at ({cx:.3f}, {cy:.3f})")
        self.browser.click(cx, cy)
        time.sleep(0.15)
        self.browser.press_key("Control+a")   # select all existing text
        self.browser.type_text(query)
        time.sleep(0.15)
        self.browser.press_key("Escape")       # dismiss autocomplete
        time.sleep(0.1)

        # Submit: prefer clicking the submit button; fall back to Enter
        if submit_elem is not None:
            sb = submit_elem["bbox_norm"]
            sx = (sb[0] + sb[2]) / 2
            sy = (sb[1] + sb[3]) / 2
            print(f"  → submitting via '{submit_elem['label']}' at ({sx:.3f}, {sy:.3f})")
            self.browser.click(sx, sy)
        else:
            self.browser.press_key("Enter")

        try:
            self.browser.wait_for_load(timeout_ms=10_000)
        except Exception:
            pass
        print(f"  ✓ searched '{query}'  →  {self.browser.current_url()}")
        return True

    # ── main action loop ──────────────────────────────────────────────────

    def act(self, task: str) -> bool:
        """
        Execute one grounding task.  Returns True if a click was fired.
        Retries up to self.max_retries times only if no coordinate is returned.
        Visual-change checking is logged but never triggers a retry — on slow
        hardware a single inference pass takes 30-80s and retrying on visual
        change would triple the latency for no benefit.
        """
        self._step += 1
        print(f"\n  [step {self._step}] task: {task!r}")

        for attempt in range(1, self.max_retries + 2):
            # 1. Screenshot
            ss = self.browser.screenshot(wait_stable=True)
            print(f"  screenshot: {ss.image.width}×{ss.image.height}  "
                  f"viewport: {ss.viewport_w}×{ss.viewport_h}")

            # 2. Extract DOM interactive elements (inputs Playwright can see but YOLO can't)
            dom_elements = self.browser.get_interactive_elements()
            if dom_elements:
                print(f"  DOM elements: {len(dom_elements)} injected "
                      f"({', '.join(e['label'][:20] for e in dom_elements[:4])}...)"
                      if len(dom_elements) > 4
                      else f"  DOM elements: {[e['label'] for e in dom_elements]}")

            # 3. Infer
            print(f"  sending to inference server (attempt {attempt})...")
            t0 = time.time()
            result = self.client.act(ss.png_bytes, task, dom_elements=dom_elements)
            elapsed = time.time() - t0
            print(f"  inference: {elapsed:.1f}s")

            # 4. Check for server-side error
            if result.get("error"):
                print(f"  ✗ server error: {result['error']}")
                return False

            # 4. Print what the model said
            print(f"  model response: {result.get('raw_response', '').strip()!r}")
            if result.get("mark_id") is not None:
                print(f"  mark_id: {result['mark_id']}")
            _print_elements(result.get("elements", []))

            # 5. Check we got coordinates — only retry condition
            click_norm = result.get("click_norm")
            if not click_norm:
                print(f"  ✗ no click coordinate returned")
                if attempt <= self.max_retries:
                    print(f"  retrying...")
                    time.sleep(0.5)
                    continue
                return False

            x_norm, y_norm = click_norm
            print(f"  click_norm: ({x_norm:.4f}, {y_norm:.4f})")

            # 6. Fire click — no visual-change retry, just report
            cr = self.browser.click(
                x_norm, y_norm,
                ss=ss,
                verify_change=False,
            )
            print(f"  click_px:   ({cr.x_viewport:.1f}, {cr.y_viewport:.1f})")
            self.browser.wait_for_load(timeout_ms=5_000)
            print(f"  url: {self.browser.current_url()}")
            return True

        return False

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_elements(elements: list):
    if not elements:
        return
    print(f"  elements ({len(elements)}):")
    for e in elements[:8]:   # cap at 8 to keep terminal readable
        print(f"    [{e['id']}] {e['content']!r}  ({e['type']})")
    if len(elements) > 8:
        print(f"    ... +{len(elements) - 8} more")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive REPL
# ══════════════════════════════════════════════════════════════════════════════

HELP = """
Commands:
  <task>          Click task — agent screenshots and clicks
  search <query>  Find text input, type query, press Enter (no VLM)
  go <url>        Navigate to URL
  back            Go back one page
  scroll down     Scroll down  (default)
  scroll up       Scroll up
  type <text>     Type text at current focus
  enter           Press Enter
  screenshot      Save a screenshot
  url             Print current URL
  help            Show this message
  quit / exit     Close and exit
"""

def _repl(agent: WebAgent):
    print(HELP)
    while True:
        try:
            raw = input("agent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not raw:
            continue

        low = raw.lower()

        if low in ("quit", "exit"):
            print("Bye.")
            break
        elif low == "help":
            print(HELP)
        elif low == "url":
            print(f"  {agent.browser.current_url()}")
        elif low == "screenshot":
            agent.save_screenshot()
        elif low in ("back", "go back"):
            agent.back()
        elif low.startswith("search "):
            agent.search(raw[7:].strip())
        elif low.startswith("go "):
            agent.navigate(raw[3:].strip())
        elif low.startswith("scroll"):
            parts = low.split()
            direction = parts[1] if len(parts) > 1 else "down"
            agent.scroll(direction)
        elif low.startswith("type "):
            agent.type_text(raw[5:])
        elif low == "enter":
            agent.press_enter()
        else:
            # treat as a grounding task
            agent.act(raw)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="magma-repro interactive web agent")
    parser.add_argument("--url",      default=None,
                        help="URL to open on start (optional)")
    parser.add_argument("--port",     type=int, default=8787,
                        help="Inference server port (default: 8787)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run browser headless (default: visible)")
    parser.add_argument("--no-save",  action="store_true",
                        help="Don't save screenshots to disk")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # suppress noisy PIL debug logs
    logging.getLogger("PIL").setLevel(logging.WARNING)

    server_url = f"http://127.0.0.1:{args.port}"

    print(f"\nmagma-repro web agent")
    print(f"  server : {server_url}")
    print(f"  browser: {'headless' if args.headless else 'visible'}\n")

    with WebAgent(
        server_url=server_url,
        headless=args.headless,
        save_screenshots=not args.no_save,
    ) as agent:
        if args.url:
            agent.navigate(args.url)
        _repl(agent)


if __name__ == "__main__":
    main()