# tests/planner_agent_gemini.py
"""
Continuous browser session, split-model variant using Google's free Gemini
API for planning instead of a paid model:

Planner  -> Gemini API (gemini-2.5-flash), free tier, no credit card needed
Executor -> POST /act (click grounding) and POST /describe (image
            description) on your existing inference server (Qwen + LoRA +
            OmniParser, local or Kaggle+ngrok, unchanged from before)

This is a SESSION, not a one-shot script: the browser stays open across
multiple directions you give it. Each new direction gets its own fresh
MAX_STEPS budget (6 by default), but `history` is a single list that is
NEVER reset for the whole session -- the agent keeps full context of
everything it's done, all the way back to the first direction you gave it.

Three kinds of input at the prompt:
  1. Direct browser commands -- "go to <url>" navigates immediately, no
     planner call at all.
  2. Describe requests -- "describe this page" / "what's on this page" /
     any question ("what does the sidebar say?") goes straight to
     /describe on the CURRENT screenshot, skipping the SEARCH/CLICK/SCROLL
     planning loop entirely (there's no browsing decision to make).
  3. Anything else -- treated as a new goal for the planner+executor loop
     (SEARCH/CLICK/SCROLL/DONE), same mechanism as before, just with a
     persistent history now instead of a fresh one each run.

CLICK phrasing: confirmed via train_smoke.jsonl inspection that this
checkpoint's training data is literal text-localization, not intent-based
grounding -- every training instruction is the VERBATIM on-page
text/caption of the target element itself, never a paraphrased
description or task-style phrase. Prompt instructs Gemini to phrase CLICK
arguments as the exact visible text on the element, not a
short-but-still-paraphrased description.

DONE verification: DONE previously fired on Gemini's own narrated intent
rather than checking the actual resulting screenshot -- confirmed live: a
run declared DONE while the cart icon still read 0 and the URL had never
left search results. Prompt requires DONE to cite specific visual
evidence in the CURRENT screenshot.

Sign-in wall (2026-08-09): confirmed live -- "Buy Now" routes through
Amazon's actual auth flow (amazon.in/ap/signin?...), which the agent
obviously can't and shouldn't try to click through. Explicit rule to
recognize the URL pattern and respond DONE with a clear reason instead
of idling out at MAX_STEPS.

Self-evaluation + URL-as-text (2026-08-09): ported two patterns from
browser-use/browser-use's system_prompt.md structure --
  1. The model itself judges whether its last action worked, from the
     fresh screenshot, rather than the calling code asserting success
     unconditionally (our old hardcoded "results are now showing" fed
     misleading context back into planning after silent failures).
  2. Current URL fed as text alongside the screenshot -- confirmed in
     tests/planner_agent.py that inferring page-type from pixels alone
     was too hard a sub-task for reliable planning; reading a URL as
     text was much easier.

use_adapter on /describe (2026-08-09): tested WITH the LoRA adapter
enabled (expected degenerate output, since it was only ever trained on
click-grounding, never captioning) -- surprising result, it produced a
coherent, accurate description anyway. Best explanation: the adapter's
learned behavior is conditional on the training prompt SHAPE ("where do
I direct my attention?" style), which /describe's prompt doesn't
resemble, so it likely never "triggers" and the base model's general
captioning ability runs through mostly undisturbed. Consistent with
lora_adapter_smoke being a light, 25-step checkpoint. Defaulted back to
adapter-DISABLED here for everyday use (the safe, validated default) --
pass use_adapter=True explicitly if you want to re-run that experiment.

Free tier limits (subject to change, check https://ai.google.dev for current
numbers): roughly 15 requests/minute and up to 1,000 requests/day for
gemini-2.5-flash.

Requires: pip install google-genai python-dotenv
          Get a free key from https://aistudio.google.com/apikey (no card)
          Put it in .env as GEMINI_API_KEY=AIza...

Usage:
    python tests/planner_agent_gemini.py
    python tests/planner_agent_gemini.py --url https://www.amazon.in
    python tests/planner_agent_gemini.py --server-url https://xxxx.ngrok-free.dev

Then at the "You:" prompt, give it directions one at a time:
    You: search for a wireless mouse and click the first result
    You: describe this page
    You: go to https://www.youtube.com
    You: search for lofi hip hop radio and open it
    You: exit
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
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory if present

from google import genai
from google.genai import types

from src.agent.browser_env import BrowserEnv

# Step-by-step detail (raw planner text, self-eval text, grounding responses,
# full URLs, step headers) goes to a log file only -- the console stays
# minimal for a demo: just the action taken, timing, and a page-loaded
# confirmation. Nothing is lost, just moved out of the way; check
# outputs/planner_agent_gemini.log if you need the full trace.
_LOG_DIR = Path("outputs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("planner_agent_gemini")
log.setLevel(logging.INFO)
_file_handler = logging.FileHandler(_LOG_DIR / "planner_agent_gemini.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
log.addHandler(_file_handler)
log.propagate = False  # don't also send to root logger / console

# ══════════════════════════════════════════════════════════════════════════════
# Server location
# ══════════════════════════════════════════════════════════════════════════════
SERVER_URL = "http://127.0.0.1:8787"
# SERVER_URL = "https://plop-isolated-blinked.ngrok-free.dev"

# ══════════════════════════════════════════════════════════════════════════════
# Planner model
# ══════════════════════════════════════════════════════════════════════════════
PLANNER_MODEL = "gemini-3.5-flash-lite"

ACTION_RE = re.compile(
    r'(?:ACTION:\s*)?(SEARCH|CLICK|SCROLL|DONE)\s*\(\s*"?([^")]*)"?\s*\)',
    re.IGNORECASE,
)
EVALUATION_RE = re.compile(r'EVALUATION:\s*(.+?)(?:\n|$)', re.IGNORECASE)

PLANNING_PROMPT_TEMPLATE = """You control a web browser one step at a time. You are shown the CURRENT screenshot.

Current URL: {current_url}
(URLs often hint at page type as text, which is more reliable than trying to judge it from
the screenshot alone -- e.g. a URL containing a search query parameter usually means you're
on a results/listing page; a URL for one specific item (a product, video, article) usually
means you're on that item's detail page.)

First, evaluate whether your PREVIOUS action actually worked, based on what the CURRENT
screenshot actually shows (not what you intended or expected) -- one short sentence citing
specific visual evidence. If this is the first step of this direction, just say so.

Then decide the single next action needed to make progress toward the CURRENT goal below.
Earlier steps from previous directions this session are shown for context only -- they may
be about a different goal entirely.

The ONLY four valid function calls are exactly these — never invent other names:
SEARCH("query")
CLICK("description of element")
SCROLL("down")
DONE("summary")

Respond in EXACTLY this two-line format, nothing else:
EVALUATION: <one sentence assessment of the previous action, grounded in the current screenshot>
ACTION: <the function call>

Reasoning steps for choosing the ACTION, in order:
1. Identify the page type from the URL and screenshot together (results/listing page, item detail page, a confirmation/login/other special page).
2. Never repeat the exact action you just took if your own EVALUATION above concluded it didn't change anything.
3. If the current URL contains "signin", "ap/signin", or the screenshot shows a sign-in / login page, respond DONE("reached a sign-in page - cannot proceed without account credentials") immediately. Do not attempt to click "Continue", enter credentials, or otherwise navigate a login flow.
4. Respond DONE ONLY if you can point to explicit visual evidence in the CURRENT screenshot that the CURRENT goal is achieved. Do NOT respond DONE just because you attempted an action that seemed like it should have worked — verify the result is actually visible. (The sign-in rule above is the one exception.)
5. Phrase the CLICK argument as the EXACT, VERBATIM text visible on the target element — copy it character-for-character from what's written on the button/link/title in the screenshot. Do NOT paraphrase, describe, summarize, or shorten it into your own words, and never use a full title or sentence — only the literal text string printed on the element itself, or the first few words of it if it's long. If the element you want has no visible text at all, pick a different, text-labeled path toward the goal instead if one exists.

Current goal: {goal}

Session history so far (may include earlier, different directions):
{history}

Respond now with the two-line EVALUATION / ACTION format:"""

MAX_STEPS_PER_DIRECTION = 6

# ══════════════════════════════════════════════════════════════════════════════
# Intent detection for direct/describe fast-paths (same style of classifier
# as src/agent/router_client.py's classify_intent, reused here so a
# "describe this page" direction doesn't burn a full planning round-trip
# through SEARCH/CLICK/SCROLL/DONE for a question that isn't a browsing
# decision at all).
# ══════════════════════════════════════════════════════════════════════════════

DESCRIBE_KEYWORDS = re.compile(
    r"\b(describe|caption|what.?s (in|on)|what is (in|on)|what does .* show|"
    r"summar(y|ize)|explain (this|the) (page|screen|image)|"
    r"contents? of (this|the) (page|screen|image)|"
    r"what('?s| is) (happening|going on))\b",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(r"^\s*(what|where|who|how many|is there|does|are there)\b", re.IGNORECASE)

GO_TO_RE = re.compile(r"^\s*go\s+to\s+(\S+)\s*$", re.IGNORECASE)


def _looks_like_url_token(token: str) -> bool:
    """True only for a single word that's actually a URL/domain shape --
    e.g. 'youtube.com' or 'https://amazon.in'. Guards against 'go to
    wikipedia page for vlm' being parsed as if 'wikipedia' were a literal
    hostname (confirmed live: that produced https://wikipedia and an
    NS_ERROR_UNKNOWN_HOST crash). Multi-word or dot-less destinations fall
    through to the planner instead, which already handles "go find X"
    style natural-language goals correctly via SEARCH+CLICK."""
    return token.lower().startswith("http") or "." in token


def is_describe_request(raw: str) -> bool:
    return bool(DESCRIBE_KEYWORDS.search(raw)) or bool(QUESTION_RE.match(raw))


# ══════════════════════════════════════════════════════════════════════════════
# Planner — Gemini API
# ══════════════════════════════════════════════════════════════════════════════

class GeminiPlanner:
    def __init__(self, model: str = PLANNER_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key (no card required) from "
                "https://aistudio.google.com/apikey and put it in .env as "
                "GEMINI_API_KEY=AIza..."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def plan(self, png_bytes: bytes, goal: str, history: list[dict], current_url: str) -> str:
        hist_str = _format_history(history)
        prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str, current_url=current_url)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=30_000),  # fail fast, don't hang
                ),
            )
            return (response.text or "").strip()
        except Exception as exc:
            print(f"  ✗ Gemini planner error: {exc}\n")
            return ""


def _format_history(history: list[dict]) -> str:
    """
    Each entry: {"action_desc": str, "evaluation": str | None}
    Covers SEARCH/CLICK/SCROLL/DONE entries (evaluation filled in one turn
    later, once the model has seen the result) AND describe entries (the
    description text itself lives in "evaluation" so it naturally shows up
    as context for later directions too -- e.g. asking to click something
    that was only ever mentioned in an earlier description).
    """
    if not history:
        return "(none - this is the first direction of the session)"
    lines = []
    for h in history:
        ev = h["evaluation"] or "(not yet evaluated)"
        lines.append(f'- {h["action_desc"]}  [result: {ev}]')
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Server client — /act and /describe
# ══════════════════════════════════════════════════════════════════════════════

class GroundingClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            log.info("Grounding server health check OK -- mode=%s status=%s", data.get('mode'), data.get('status'))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach grounding server at {self.base_url}. Is it running?\n  Error: {exc}"
            )

    def act(self, png_bytes: bytes, task: str, dom_elements: list | None = None) -> dict:
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "task": task}
        if dom_elements:
            payload["dom_elements"] = dom_elements
        try:
            r = requests.post(f"{self.base_url}/act", json=payload, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"click_norm": None, "mark_id": None, "raw_response": "",
                    "elements": [], "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"click_norm": None, "mark_id": None, "raw_response": "",
                    "elements": [], "error": str(exc)}

    def describe(self, png_bytes: bytes, question: str | None = None, use_adapter: bool = False,
                 max_new_tokens: int = 500) -> dict:
        """POST /describe. use_adapter=False is the safe, validated default --
        see the use_adapter module docstring note above for why. max_new_tokens
        default raised to 500 (server default is 200) since 200 was confirmed
        live to cut off in-depth descriptions of content-heavy pages mid-sentence."""
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {"image_b64": image_b64, "use_adapter": use_adapter, "max_new_tokens": max_new_tokens}
        if question:
            payload["question"] = question
        try:
            r = requests.post(f"{self.base_url}/describe", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"response": "", "error": "Request timed out — server may be busy."}
        except Exception as exc:
            return {"response": "", "error": str(exc)}


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def parse_evaluation(response: str) -> str | None:
    m = EVALUATION_RE.search(response)
    return m.group(1).strip() if m else None


def ground_click(client: GroundingClient, browser: BrowserEnv, task_desc: str):
    ss = browser.screenshot(wait_stable=True, label=f"click_{task_desc}")
    dom_elements = browser.get_interactive_elements()
    result = client.act(ss.png_bytes, task_desc, dom_elements=dom_elements)
    log.info("Grounding response for %r: %r", task_desc, result.get("raw_response", ""))
    if result.get("error"):
        log.info("Grounding server error: %s", result['error'])
        return None
    click_norm = result.get("click_norm")
    return tuple(click_norm) if click_norm else None


# ══════════════════════════════════════════════════════════════════════════════
# One direction's worth of planning+execution -- MAX_STEPS_PER_DIRECTION
# fresh steps, but `history` is the SAME list passed in from the session
# and is appended to, never replaced or cleared.
# ══════════════════════════════════════════════════════════════════════════════

def run_direction(planner: GeminiPlanner, executor: GroundingClient, browser: BrowserEnv,
                   goal: str, history: list[dict]) -> None:
    for step in range(1, MAX_STEPS_PER_DIRECTION + 1):
        log.info("--- Step %d/%d for goal=%r ---", step, MAX_STEPS_PER_DIRECTION, goal)

        ss = browser.screenshot(wait_stable=True, label=goal)
        current_url = browser.current_url()
        log.info("current url: %s", current_url)

        t0 = time.time()
        response = planner.plan(ss.png_bytes, goal, history, current_url)
        action, arg = parse_action(response)
        evaluation = parse_evaluation(response)
        plan_elapsed = time.time() - t0
        log.info("planner said: %r", response)

        if history and evaluation:
            history[-1]["evaluation"] = evaluation
            log.info("self-eval of prior step: %s", evaluation)

        if action is None:
            print("  ✗ couldn't parse a valid action from the planner -- stopping this direction.\n")
            log.info("Unparseable response, stopping: %r", response)
            return

        if action == "DONE":
            print(f"({plan_elapsed:.1f}s) DONE: {arg}\n")
            history.append({"action_desc": f'DONE("{arg}")', "evaluation": None})
            return

        elif action == "SEARCH":
            elems = browser.get_interactive_elements()
            input_elem = next(
                (e for e in elems if e["tag"] in ("input", "textarea")
                 and e["type"] not in ("submit", "button", "checkbox", "radio", "image", "hidden")),
                None,
            )
            if input_elem is None:
                print("  ✗ no search input found on this page -- stopping this direction.\n")
                return
            bbox = input_elem["bbox_norm"]
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            browser.click(cx, cy)
            time.sleep(0.15)
            browser.press_key("Control+a")
            browser.type_text(arg)
            time.sleep(0.15)
            browser.press_key("Enter")
            try:
                browser.wait_for_load(timeout_ms=10000)
            except Exception:
                pass
            history.append({"action_desc": f'SEARCH("{arg}")', "evaluation": None})
            print(f'({plan_elapsed:.1f}s planning) SEARCH("{arg}") \u2192 Webpage loaded\n')

        elif action == "CLICK":
            t0 = time.time()
            point = ground_click(executor, browser, arg)
            ground_elapsed = time.time() - t0
            if point is None:
                print(f"  ✗ grounding couldn't find {arg!r} on the page -- stopping this direction.\n")
                return
            browser.click(point[0], point[1], verify_change=False)
            try:
                browser.wait_for_load(timeout_ms=5000)
            except Exception:
                pass
            history.append({"action_desc": f'CLICK("{arg}")', "evaluation": None})
            print(f'({plan_elapsed:.1f}s planning, {ground_elapsed:.1f}s grounding) CLICK("{arg}") \u2192 Webpage loaded\n')

        elif action == "SCROLL":
            direction = arg.strip().lower() if arg else "down"
            browser.scroll(delta_y=400 if direction == "down" else -400)
            history.append({"action_desc": f'SCROLL("{direction}")', "evaluation": None})
            print(f'({plan_elapsed:.1f}s planning) SCROLL("{direction}")\n')

        log.info("url now: %s", browser.current_url())

    print(f"  Stopped after {MAX_STEPS_PER_DIRECTION} steps without finishing this direction.\n")
    log.info("Hit MAX_STEPS_PER_DIRECTION=%d without DONE for goal=%r", MAX_STEPS_PER_DIRECTION, goal)


def _print_wrapped(prefix: str, text: str, width: int = 88) -> None:
    """Clean word-wrapped print with a hanging indent under prefix, instead of
    relying on raw terminal wrap (which breaks mid-word at the column edge)."""
    wrapped = textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))
    print(f"{wrapped}\n")


def _format_structured_description(text: str) -> str:
    """The base model often answers detailed descriptions in markdown --
    "1. **Header**:    - point one.    - point two.  2. **Next**: ..." -- which
    reads as one giant run-on paragraph with literal ** characters in a plain
    terminal (confirmed live, that's exactly what a 700-token Wikipedia
    description looked like). Reformat: strip bold markers, break each
    numbered header onto its own line, break each "- " sub-bullet onto its
    own line. Heuristic, not a real markdown parser -- good enough for how
    consistently this model formats these responses, not guaranteed for
    every possible response shape."""
    text = text.replace("**", "")
    text = re.sub(r'\s+(\d{1,2})\.\s+(?=[A-Z])', r'\n\n\1. ', text)
    text = re.sub(r'\s{2,}-\s+', r'\n   - ', text)
    return text.strip()


def _print_description(prefix: str, text: str, width: int = 88) -> None:
    """Like _print_wrapped, but parses markdown structure first (see
    _format_structured_description) so numbered headers and sub-bullets each
    get their own line and indent, instead of one flat wrapped paragraph."""
    formatted = _format_structured_description(text)
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
    print("\n".join(out_lines) + "\n")


def run_describe(executor: GroundingClient, browser: BrowserEnv, raw: str, history: list[dict]) -> None:
    """Fast-path: skip the planner entirely, describe the CURRENT screenshot directly."""
    log.info("Describing current page for request: %r", raw)
    ss = browser.screenshot(wait_stable=True, label=f"describe_{raw}")
    question = raw.strip() if QUESTION_RE.match(raw) else None
    t0 = time.time()
    result = executor.describe(ss.png_bytes, question=question, use_adapter=False, max_new_tokens=700)
    elapsed = time.time() - t0
    if result.get("error"):
        print(f"  ✗ describe error: {result['error']}\n")
        history.append({"action_desc": f'DESCRIBE("{raw}")', "evaluation": f"error: {result['error']}"})
        return
    description = result.get("response", "")
    _print_description(f"({elapsed:.1f}s) Assistant: ", description)
    # Description becomes part of session history -- so a LATER direction
    # like "click the thing you just mentioned" has it as context.
    history.append({"action_desc": f'DESCRIBE("{raw}")', "evaluation": description})


def run_goto(browser: BrowserEnv, url: str, history: list[dict]) -> None:
    if not url.startswith("http"):
        url = "https://" + url
    log.info("Navigating directly to %s", url)
    try:
        browser.navigate(url)
        print("Webpage loaded\n")
        history.append({"action_desc": f'GOTO("{url}")', "evaluation": f"navigated to {browser.current_url()}"})
    except Exception as exc:
        print(f"  ✗ navigation error: {exc}\n")
        history.append({"action_desc": f'GOTO("{url}")', "evaluation": f"error: {exc}"})


HELP = """
Give directions one at a time. History carries across the whole session --
the agent remembers everything done so far, even across different goals.

  <a task>            e.g. "search for a wireless mouse and click the first result"
                       Runs the planner+executor loop, up to 6 fresh steps.
  describe...          e.g. "describe this page" / "what does the sidebar say?"
                       Skips planning, describes the CURRENT screenshot directly.
  go to <url>          Navigate directly to a literal URL/domain (e.g. "go to youtube.com"),
                       no planning involved. A natural-language destination (e.g. "go to the
                       wikipedia page for VLM") is NOT a literal URL, so it's treated as a
                       normal task instead -- the planner will search for it and click through.
  help                 Show this message.
  exit / quit          Close the browser and end the session.

Full step-by-step detail (raw model text, URLs, grounding responses) is
logged to outputs/planner_agent_gemini.log -- console stays minimal.
"""


def run_session(start_url: str, server_url: str) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    planner = GeminiPlanner()
    executor = GroundingClient(server_url)
    history: list[dict] = []   # persists for the WHOLE session, never reset

    with BrowserEnv(headless=False, save_screenshots=True) as browser:
        browser.navigate(start_url)
        print("Browser loaded\n")
        print("Type 'help' for available commands.")

        while True:
            try:
                raw = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print("Bye.")
                break
            if raw.lower() == "help":
                print(HELP)
                continue

            print()  # gap between "You: ..." and the first line of output

            goto_match = GO_TO_RE.match(raw)
            if goto_match and _looks_like_url_token(goto_match.group(1)):
                run_goto(browser, goto_match.group(1), history)
                continue

            if is_describe_request(raw):
                run_describe(executor, browser, raw, history)
                continue

            # Anything else: a new goal for the planner+executor loop.
            # Fresh MAX_STEPS_PER_DIRECTION budget, but `history` is the
            # SAME list -- never cleared, never replaced.
            run_direction(planner, executor, browser, raw, history)

        print("\nSession ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.duckduckgo.com", help="Starting URL")
    parser.add_argument("--server-url", default=None,
                        help="Grounding server URL — overrides SERVER_URL constant above")
    args = parser.parse_args()

    server_url = args.server_url if args.server_url else SERVER_URL
    run_session(args.url, server_url)