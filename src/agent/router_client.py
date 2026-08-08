# src/agent/router_client.py
"""
Local router client — talks to a REMOTE inference_server (e.g. Kaggle + ngrok)
instead of loading the model on this machine. Mirrors web_agent.py's
InferenceClient pattern, extended to also cover the image-description agent.

Requires the inference server to already be running remotely with all three
endpoints (/act, /plan, /describe) - see the updated inference_server.py and
the Kaggle notebook that runs it behind ngrok.

Usage:
    # image agent - single question against a local image, image agent
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --image outputs/img_test/IMG_1.jpg --input "describe what's on this screen"

    # image agent - loop of questions (each one is a fresh, independent request -
    # the server has no memory between calls, so this is NOT a stateful multi-turn
    # chat like router_agent.py's --interactive; it just re-sends the same image
    # with a new question each time. Good enough for "ask a few different things
    # about one photo", not for follow-ups that depend on prior answers.)
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --image outputs/img_test/IMG_1.jpg --loop

    # web agent - browser runs HERE, locally; only model calls go to the server
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --url https://www.amazon.in --input "search for a wireless mouse and click the first result"
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import time

sys.path.insert(0, ".")

import requests
from PIL import Image

from src.agent.browser_env import BrowserEnv

MAX_STEPS = 6

# ══════════════════════════════════════════════════════════════════════════
# Intent classification (identical to router_agent.py's, kept local since
# routing is cheap and shouldn't need a round-trip to decide)
# ══════════════════════════════════════════════════════════════════════════

WEB_AGENT_KEYWORDS = re.compile(
    r"\b(click|search|navigate|go to|open|buy|add to cart|scroll|log ?in|"
    r"sign ?in|submit|type|fill in|checkout|purchase)\b",
    re.IGNORECASE,
)

IMAGE_AGENT_KEYWORDS = re.compile(
    r"\b(describe|caption|what.?s (in|on)|what is (in|on)|what does .* show|"
    r"summar(y|ize)|explain (this|the) (image|screenshot|picture)|"
    r"contents? of (this|the) (image|screenshot|picture)|"
    r"what('?s| is) (happening|going on))\b",
    re.IGNORECASE,
)

QUESTION_RE = re.compile(r"^\s*(what|where|who|how many|is there|does|are there)\b", re.IGNORECASE)


def classify_intent(user_input: str, has_url: bool, has_image: bool, forced_mode: str | None) -> str:
    if forced_mode:
        return forced_mode
    web_hit = bool(WEB_AGENT_KEYWORDS.search(user_input))
    img_hit = bool(IMAGE_AGENT_KEYWORDS.search(user_input)) or bool(QUESTION_RE.match(user_input))
    if web_hit and not img_hit:
        return "web"
    if img_hit and not web_hit:
        return "image"
    if has_url and not has_image:
        return "web"
    if has_image and not has_url:
        return "image"
    return "image" if has_image else "web"


# ══════════════════════════════════════════════════════════════════════════
# Remote client
# ══════════════════════════════════════════════════════════════════════════

class RemoteInferenceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"  [server] mode={data.get('mode')}  status={data.get('status')}")
        except Exception as exc:
            raise RuntimeError(f"Cannot reach inference server at {self.base_url}. Is it running?\n  Error: {exc}")

    @staticmethod
    def _image_to_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _png_bytes_to_b64(png_bytes: bytes) -> str:
        return base64.b64encode(png_bytes).decode("utf-8")

    def describe(self, image_path: str, question: str | None = None) -> dict:
        payload = {"image_b64": self._image_to_b64(image_path), "question": question}
        try:
            r = requests.post(f"{self.base_url}/describe", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"response": "", "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"response": "", "error": str(exc)}

    def plan(self, png_bytes: bytes, goal: str, history: list[str]) -> dict:
        payload = {"image_b64": self._png_bytes_to_b64(png_bytes), "goal": goal, "history": history}
        try:
            r = requests.post(f"{self.base_url}/plan", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"action": None, "arg": None, "raw_response": "", "error": "Request timed out."}
        except Exception as exc:
            return {"action": None, "arg": None, "raw_response": "", "error": str(exc)}

    def act(self, png_bytes: bytes, task: str, dom_elements: list | None = None) -> dict:
        payload = {"image_b64": self._png_bytes_to_b64(png_bytes), "task": task}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(f"{self.base_url}/act", json=payload, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "mark_id": None, "raw_response": "", "elements": [],
                    "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"click_norm": None, "mark_id": None, "raw_response": "", "elements": [], "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Image agent (remote)
# ══════════════════════════════════════════════════════════════════════════

def run_image_agent_remote(client: RemoteInferenceClient, image_path: str, user_input: str) -> str:
    question = user_input.strip() if QUESTION_RE.match(user_input) else None
    result = client.describe(image_path, question=question)
    if result.get("error"):
        return f"error: {result['error']}"
    return result.get("response", "")


def run_image_loop_remote(client: RemoteInferenceClient, image_path: str) -> None:
    print("\nImage loaded - ask anything about it. Each question is independent "
          "(no memory between turns). Type 'exit' to quit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "q"):
            break
        result = client.describe(image_path, question=user_text if QUESTION_RE.match(user_text) else None)
        if result.get("error"):
            print(f"Assistant: [error: {result['error']}]\n")
        else:
            print(f"Assistant: {result.get('response', '')}\n")


# ══════════════════════════════════════════════════════════════════════════
# Web agent (browser local, model remote)
# ══════════════════════════════════════════════════════════════════════════

def run_web_agent_remote(client: RemoteInferenceClient, goal: str, start_url: str, headless: bool = False) -> str:
    with BrowserEnv(headless=headless, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history: list[str] = []

        for step in range(1, MAX_STEPS + 1):
            print(f"\n{'=' * 60}\nSTEP {step}\n{'=' * 60}")
            ss = browser.screenshot(wait_stable=True)
            png_bytes = ss.image_bytes if hasattr(ss, "image_bytes") else _pil_to_png_bytes(ss.image)

            t0 = time.time()
            plan_result = client.plan(png_bytes, goal, history)
            print(f"  ({time.time() - t0:.1f}s) planner said: {plan_result.get('raw_response')!r}")

            if plan_result.get("error"):
                return f"stopped: /plan error: {plan_result['error']}"

            action, arg = plan_result.get("action"), plan_result.get("arg")
            if action is None:
                return f"stopped: could not parse an action from {plan_result.get('raw_response')!r}"
            if action == "DONE":
                return f"done: {arg}"

            if action == "SEARCH":
                browser.search(arg)
                history.append(f"SEARCH({arg!r})")
            elif action == "CLICK":
                dom_elements = browser.get_interactive_elements()
                act_result = client.act(png_bytes, arg, dom_elements=dom_elements)
                print(f"    grounding response: {act_result.get('raw_response')!r} -> click_norm={act_result.get('click_norm')}")
                point = act_result.get("click_norm")
                if point:
                    browser.click(*point)
                history.append(f"CLICK({arg!r})")
            elif action == "SCROLL":
                browser.scroll(arg)
                history.append(f"SCROLL({arg!r})")

        return "stopped: MAX_STEPS reached"


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Local router client for a remote inference_server (Kaggle + ngrok).")
    parser.add_argument("--server-url", required=True, help="e.g. https://xxxx.ngrok-free.dev")
    parser.add_argument("--input", default=None, help="Instruction/question (single-shot mode)")
    parser.add_argument("--image", default=None, help="Path to a local image (image agent)")
    parser.add_argument("--url", default=None, help="Starting URL (web agent) - browser runs locally")
    parser.add_argument("--mode", choices=["web", "image"], default=None, help="Force a mode instead of classifying")
    parser.add_argument("--headless", action="store_true", help="Run the local browser headless")
    parser.add_argument("--loop", action="store_true", help="Image agent: repeated independent questions about --image")
    args = parser.parse_args()

    client = RemoteInferenceClient(args.server_url)

    if args.loop:
        if not args.image:
            parser.error("--loop requires --image")
        run_image_loop_remote(client, args.image)
        return

    if not args.input:
        parser.error("Provide --input '...' or use --loop with --image")

    mode = classify_intent(args.input, has_url=bool(args.url), has_image=bool(args.image), forced_mode=args.mode)
    print(f"[router] mode = {mode}", flush=True)

    if mode == "image":
        if not args.image:
            print("error: image mode selected but no --image was provided")
            return
        print(run_image_agent_remote(client, args.image, args.input))
        return

    if not args.url:
        print("error: web mode selected but no --url was provided")
        return
    print(run_web_agent_remote(client, args.input, args.url, headless=args.headless))


if __name__ == "__main__":
    main()