# src/agent/router_client.py
"""
Local router client — talks to a REMOTE inference_server (e.g. Kaggle + ngrok)
instead of loading the model on this machine. Mirrors web_agent.py's
InferenceClient pattern, extended to also cover the image-description agent.

Requires the inference server to already be running remotely with all three
endpoints (/act, /plan, /describe) - see the updated inference_server.py and
the Kaggle notebook that runs it behind ngrok.

Usage:
    # image agent - just open the image and start asking questions
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --image outputs/img_test/IMG_1.jpg

    # image agent - opening question, then continues into the same session
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --image outputs/img_test/IMG_1.jpg --input "describe what's on this screen"

    # web agent - browser runs HERE, locally; only model calls go to the server
    python -m src.agent.router_client --server-url https://xxxx.ngrok-free.dev \\
        --url https://www.amazon.in --input "search for a wireless mouse and click the first result"
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, ".")

import requests
from PIL import Image

from src.agent.browser_env import BrowserEnv

# Diagnostic detail (server health checks, routed mode, etc.) goes to a log
# file only -- the console stays clean for a presentation. Nothing is lost,
# just moved out of the way; check outputs/router_client.log if you need it.
_LOG_DIR = Path("outputs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("router_client")
log.setLevel(logging.INFO)
_file_handler = logging.FileHandler(_LOG_DIR / "router_client.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
log.addHandler(_file_handler)
log.propagate = False  # don't also send to root logger / console

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

# Confirmed live: the base model (adapter disabled, /describe's mode) confidently
# added "Tamil script" to a description of a sign that had no Tamil text at all --
# a plausible-sounding but false detail, not a random-sampling fluke (do_sample=False
# is already used server-side). This instruction is a mitigation, not a fix -- 3B
# VLMs confabulate even when explicitly told not to -- but it's cheap and worth
# always including rather than relying on the server's generic fallback prompt.
ANTI_HALLUCINATION_SUFFIX = (
    " Describe only what is clearly and directly visible. Be specific about visible "
    "text, colors, and objects, but do NOT guess, infer, or add plausible-sounding "
    "details you cannot directly confirm from the image itself -- for example, do not "
    "claim additional languages, text, or objects are present unless you can actually "
    "see them. If you're not sure about something, leave it out rather than guessing. "
    "Do NOT name or identify any specific language or script (e.g. do not say 'Tamil', "
    "'Chinese', 'Hindi', 'Arabic', etc.) unless the question specifically asks about the "
    "language or script of text in the image -- if text is visible but you're not asked "
    "about its language, just say the text is visible without naming a language."
)


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
            log.info("Server health check OK -- mode=%s status=%s", data.get('mode'), data.get('status'))
        except Exception as exc:
            raise RuntimeError(f"Cannot reach inference server at {self.base_url}. Is it running?\n  Error: {exc}")

    @staticmethod
    def _image_to_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _png_bytes_to_b64(png_bytes: bytes) -> str:
        return base64.b64encode(png_bytes).decode("utf-8")

    def describe(self, image_path: str, question: str | None = None, max_new_tokens: int = 200) -> dict:
        payload = {"image_b64": self._image_to_b64(image_path), "question": question,
                   "max_new_tokens": max_new_tokens}
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

# ══════════════════════════════════════════════════════════════════════════
# Image agent (remote) — WITH real conversation history
# ══════════════════════════════════════════════════════════════════════════
# /describe is stateless server-side (no session state, one image+prompt in,
# one answer out) -- so history is carried entirely client-side here: each
# turn's prompt includes the full prior Q&A transcript, and the image is
# just re-sent fresh every call (cheap; no server change needed for this).
# This means the model genuinely has access to what was asked/answered
# before, not just independent single-shot calls -- "what color is that?"
# after "what's on the desk?" will actually work.
#
# Honest cost: the prompt grows every turn (full transcript re-sent each
# time), so later questions in a long session will be slower to answer than
# the first one -- same tradeoff documented in router_agent.py's local
# run_image_chat, just here it's network+prompt-length instead of KV-cache.

# Default answer is short and structurally capped (low max_new_tokens), not
# just instructed to be brief -- confirmed live that instruction alone
# ("be specific, don't guess") did NOT stop confabulation: it hallucinated
# "Tamil script" once, then "Chinese characters" the next run, on a sign
# with no non-English text at all. A short, tightly-capped answer gives the
# model much less room to wander into confident fabrication. Full detail is
# still available, just opt-in.
SHORT_DESCRIBE_PROMPT = (
    "In ONE short sentence, state the single most obvious and certain fact about "
    "what this image shows -- e.g. what place, building, object, or scene it is, or "
    "what text is clearly and directly readable in it. Only state something you are "
    "certain of. Do not add extra descriptive detail, colors, or lists -- one "
    "concise, confident sentence only."
)
DETAILED_DESCRIBE_PROMPT = "Describe what you see in this image in detail."

SHORT_MAX_TOKENS = 60
DETAILED_MAX_TOKENS = 500

MORE_DETAIL_RE = re.compile(
    r"\b(more detail|in detail|in depth|in more detail|elaborate|expand|tell me more|"
    r"go deeper|describe more|more info|full description|detailed|fully|thoroughly)\b",
    re.IGNORECASE,
)

# Only a truly generic "describe the image" (no specific topic) should get
# the fully generic SHORT_DESCRIBE_PROMPT. Confirmed live: "Describe the
# surrounding" was being silently discarded and answered with the SAME
# canned generic answer as a plain "describe the image" -- the user's actual
# topic was thrown away rather than being kept and just made concise.
GENERIC_DESCRIBE_RE = re.compile(
    r"^\s*(describe|caption)\s+(this|the)?\s*(image|page|screen|picture)?\s*\.?\s*$",
    re.IGNORECASE,
)


def _build_prompt_with_history(user_text: str, qa_history: list[tuple[str, str]]) -> tuple[str, int]:
    """Returns (prompt, max_new_tokens). A genuine question (QUESTION_RE) is
    answered directly and concisely (short budget, explicit brevity
    instruction -- see fix note below); an explicit 'more detail' request
    gets the long detailed prompt; a truly generic "describe the image" gets
    the short generic baseline; anything else specific-but-short (e.g.
    "describe the surrounding") keeps the user's own topic, just wrapped to
    stay short and confident rather than being discarded."""
    if QUESTION_RE.match(user_text):
        # Confirmed live: sending the raw question with no brevity instruction
        # (and a 200-token budget) produced a full re-description of the whole
        # image plus the answer buried at the end, instead of a direct answer
        # -- got worse once conversation history was added, since "using the
        # conversation above" seems to nudge the model toward restating it.
        base_prompt = (
            f"{user_text} Answer directly and concisely -- a short, direct answer is "
            f"enough. Do NOT repeat the full earlier description unless specifically asked to."
        )
        max_tokens = SHORT_MAX_TOKENS
    elif MORE_DETAIL_RE.search(user_text):
        base_prompt, max_tokens = DETAILED_DESCRIBE_PROMPT, DETAILED_MAX_TOKENS
    elif GENERIC_DESCRIBE_RE.match(user_text):
        base_prompt, max_tokens = SHORT_DESCRIBE_PROMPT, SHORT_MAX_TOKENS
    else:
        base_prompt = (
            f"In ONE short, confident sentence, answer this about the image: {user_text}. "
            f"Only state something you are certain of; don't add extra detail beyond "
            f"what's asked."
        )
        max_tokens = SHORT_MAX_TOKENS

    if qa_history:
        transcript = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_history)
        base_prompt = (
            "Here is our conversation so far about this image:\n"
            f"{transcript}\n\n"
            f"Now answer this new question, using the conversation above AND the image "
            f"itself: {base_prompt}"
        )
    return base_prompt + ANTI_HALLUCINATION_SUFFIX, max_tokens


def _format_structured_description(text: str) -> str:
    """The base model often answers detailed descriptions in markdown --
    "1. **Header**:    - point one.    - point two.  2. **Next**: ..." -- which
    reads as one giant run-on paragraph with literal ** characters in a plain
    terminal. Reformat: strip bold markers, break each numbered header onto
    its own line, break each "- " sub-bullet onto its own line. Heuristic,
    not a real markdown parser -- good enough for how consistently this
    model formats these responses, not guaranteed for every possible shape."""
    text = text.replace("**", "")
    text = re.sub(r'\s+(\d{1,2})\.\s+(?=[A-Z])', r'\n\n\1. ', text)
    text = re.sub(r'\s{2,}-\s+', r'\n   - ', text)
    return text.strip()


def _print_assistant_reply(elapsed: float, answer: str, width: int = 88) -> None:
    """Word-wrapped, markdown-structure-aware print instead of relying on raw
    terminal wrap (breaks mid-word) or dumping numbered lists/bold markers as
    one flat paragraph."""
    prefix = f"({elapsed:.1f}s) Assistant: "
    formatted = _format_structured_description(answer)
    lines = formatted.split("\n")
    out_lines = []
    first = True
    for line in lines:
        if not line.strip():
            out_lines.append("")
            continue
        if line.startswith("   - "):
            indent = "   - "
            content = line[len(indent):]
        else:
            indent = ""
            content = line
        if first:
            wrapped = textwrap.fill(content, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))
            first = False
        else:
            cont_indent = indent if indent else "   "
            wrapped = textwrap.fill(content, width=width, initial_indent=indent, subsequent_indent=cont_indent)
        out_lines.append(wrapped)
    print("\n" + "\n".join(out_lines) + "\n")


def run_image_session(client: RemoteInferenceClient, image_path: str, first_input: str | None = None) -> None:
    """Continuous multi-turn Q&A about ONE image, with real history (see module
    note above). Default answers are short and to-the-point; say "more detail"
    / "elaborate" / "describe more" to get the fuller version. If first_input
    is given, it's asked immediately as the first turn; either way, drops into
    an interactive loop afterward."""
    qa_history: list[tuple[str, str]] = []

    def ask(user_text: str) -> str:
        prompt, max_tokens = _build_prompt_with_history(user_text, qa_history)
        result = client.describe(image_path, question=prompt, max_new_tokens=max_tokens)
        answer = f"[error: {result['error']}]" if result.get("error") else result.get("response", "")
        qa_history.append((user_text, answer))
        return answer

    log.info("Session started. Short answers by default; 'more detail'/'describe more' "
              "for the fuller version; 'exit' to quit.")
    print("Image loaded\n")

    if first_input:
        print(f"You: {first_input}\n")
        t0 = time.time()
        answer = ask(first_input)
        _print_assistant_reply(time.time() - t0, answer)

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
        t0 = time.time()
        answer = ask(user_text)
        _print_assistant_reply(time.time() - t0, answer)


# ══════════════════════════════════════════════════════════════════════════
# Web agent (browser local, model remote)
# ══════════════════════════════════════════════════════════════════════════

def _dom_search(browser: BrowserEnv, query: str) -> bool:
    """
    Compound DOM-direct search action, mirroring tests/planner_agent.py and
    web_agent.py's WebAgent.search() - there's no browser.search() method on
    BrowserEnv itself, this logic lives at the call-site in every script that
    uses it. Skips VLM inference entirely (the search bar is already reliably
    found via DOM query, so running OmniParser+Qwen for it is pure waste).
    """
    elems = browser.get_interactive_elements()
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
        (e for e in elems if e["tag"] == "button" or e.get("type") in ("submit", "button")),
        None,
    )

    bbox = input_elem["bbox_norm"]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    browser.click(cx, cy)
    time.sleep(0.15)
    browser.press_key("Control+a")   # select-all to clear existing text
    browser.type_text(query)
    time.sleep(0.15)
    browser.press_key("Escape")       # dismiss autocomplete (prevents Enter interception)
    time.sleep(0.1)

    if submit_elem is not None:
        sb = submit_elem["bbox_norm"]
        sx, sy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        browser.click(sx, sy)
    else:
        browser.press_key("Enter")

    try:
        browser.wait_for_load(timeout_ms=10_000)
    except Exception:
        pass
    return True


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
                _dom_search(browser, arg)
                history.append(f"SEARCH({arg!r})")
            elif action == "CLICK":
                dom_elements = browser.get_interactive_elements()
                act_result = client.act(png_bytes, arg, dom_elements=dom_elements)
                print(f"    grounding response: {act_result.get('raw_response')!r} -> click_norm={act_result.get('click_norm')}")
                point = act_result.get("click_norm")
                if point:
                    browser.click(point[0], point[1], verify_change=False)
                history.append(f"CLICK({arg!r})")
            elif action == "SCROLL":
                direction = (arg or "down").strip().lower()
                browser.scroll(delta_y=400 if direction == "down" else -400)
                history.append(f"SCROLL({direction!r})")

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

    os.system("cls" if os.name == "nt" else "clear")
    client = RemoteInferenceClient(args.server_url)

    # --image alone, no --input, goes straight into an interactive session --
    # no need for a separate --loop flag for this anymore. --loop is kept as
    # an explicit synonym for backward compatibility.
    if args.image and not args.input:
        run_image_session(client, args.image)
        return

    if args.loop:
        if not args.image:
            parser.error("--loop requires --image")
        run_image_session(client, args.image)
        return

    if not args.input:
        parser.error("Provide --input '...', or just --image to start an interactive image session")

    mode = classify_intent(args.input, has_url=bool(args.url), has_image=bool(args.image), forced_mode=args.mode)
    log.info("Routed to mode=%s for input=%r", mode, args.input)

    if mode == "image":
        if not args.image:
            print("error: image mode selected but no --image was provided")
            return
        run_image_session(client, args.image, first_input=args.input)
        return

    if not args.url:
        print("error: web mode selected but no --url was provided")
        return
    print(run_web_agent_remote(client, args.input, args.url, headless=args.headless))


if __name__ == "__main__":
    main()