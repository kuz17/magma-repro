# src/agent/router_agent.py
"""
Multi-agent router.

Single entrypoint that looks at the user's input (plus whether a --url or
--image was supplied) and dispatches to ONE of two agents. Each agent loads
only what it needs:

  - "web"   -> planner+executor browser loop (src/agent/browser_env.py +
               click_visualizer.DemoRunner), same mechanism as
               tests/planner_agent.py:
                 planner  = base Qwen   (adapter disabled) picks next action
                 executor = fine-tuned Qwen (adapter enabled) grounds CLICK
                            targets to coordinates

  - "image" -> plain description / VQA on a single image, using the BASE
               model (adapter disabled). The LoRA adapter was trained only
               on click-to-coordinate grounding (Mark: N / Coordinate: (x,y)
               style targets) - it has never seen free-form captioning
               data, so asking it a "describe this" question with the
               adapter ON is out-of-distribution for it. disable_adapter()
               is the same mechanism already validated in
               tests/smoke_disable_adapter.py, just reused for a different
               prompt.

Usage:
    python -m src.agent.router_agent --image path/to/shot.png \\
        --input "describe what's happening on this screen"

    python -m src.agent.router_agent --url https://www.amazon.in \\
        --input "search for a wireless mouse and click the first result"

    # force the route explicitly instead of relying on the classifier
    python -m src.agent.router_agent --image shot.png --input "..." --mode image
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from PIL import Image

from src.agent.click_visualizer import DemoRunner
from src.agent.browser_env import BrowserEnv

QWEN_PATH = "models/qwen2_5_vl_3b"

# Cap the longest side of any image fed to the image agent. Qwen2.5-VL's
# vision tower scales attention memory with patch count, so an un-resized
# phone photo (thousands of px/side) can OOM a 4GB card even though a
# browser screenshot at similar "1 image" cost works fine. Screenshots are
# already reasonably sized so this is mostly a no-op for the web agent's
# own image path - it only bites for arbitrary user photos.
MAX_IMAGE_SIDE = 1024

# ══════════════════════════════════════════════════════════════════════════
# Intent classification
# ══════════════════════════════════════════════════════════════════════════
# Deliberately a cheap regex classifier, not a model call: this decision has
# to happen before we know which mode (and therefore which adapter state)
# to load the model in, and it needs to be fast/deterministic. If this
# proves too brittle in practice, the fallback below (has_url / has_image)
# is the safety net, and --mode always lets the user override outright.

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

# Bare question words with no image/click context -> treat as VQA on the
# supplied image, not a browser command ("what color is the button" should
# go to the image agent, not be mistaken for a click instruction).
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

    # Ambiguous or no keyword hit at all -> fall back to what was provided.
    if has_url and not has_image:
        return "web"
    if has_image and not has_url:
        return "image"

    # Both or neither provided and still ambiguous: prefer image if we have
    # one (safer default - describing is non-destructive; clicking isn't).
    return "image" if has_image else "web"


# ══════════════════════════════════════════════════════════════════════════
# Image agent (base model, adapter disabled, NO OmniParser)
# ══════════════════════════════════════════════════════════════════════════
# Deliberately does NOT go through DemoRunner: DemoRunner's __init__
# unconditionally loads OmniParser (YOLO + Florence-2) too, which the image
# agent never touches. On a 4GB card with ~0.39GB typical free headroom
# (per devlog), paying for OmniParser's VRAM footprint just to describe an
# image is pure waste - so this path loads only Qwen + the adapter, mirroring
# tests/smoke_disable_adapter.py's loader rather than click_visualizer.py's.

_qwen_only_cache: dict = {}


def load_qwen_for_description(lora_path: str):
    """Load Qwen2.5-VL-3B (4-bit) + LoRA, no OmniParser. Cached per lora_path
    so repeated calls in one process don't reload."""
    if lora_path in _qwen_only_cache:
        return _qwen_only_cache[lora_path]

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("Loading Qwen2.5-VL-3B (4-bit) - image agent, OmniParser skipped...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, quantization_config=bnb, device_map="auto",
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    print("Qwen ready (image agent).")

    _qwen_only_cache[lora_path] = (model, processor)
    return model, processor


def _resize_for_vram(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    print(f"  [image agent] resizing {w}x{h} -> {new_size[0]}x{new_size[1]} to fit VRAM", flush=True)
    return image.resize(new_size, Image.LANCZOS)


def run_qwen_describe(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 200) -> str:
    """
    Token budget suited to prose description/VQA, unlike DemoRunner._run_qwen's
    20-token budget (that one's tuned for coordinate/mark output).
    """
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def run_image_agent(image_path: str, user_input: str, lora_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    image = _resize_for_vram(image)

    if QUESTION_RE.match(user_input):
        prompt = user_input.strip()  # pass the question through as-is (VQA)
    else:
        prompt = "Describe what you see in this image in detail."

    print(f"  [image agent] prompt: {prompt!r}", flush=True)
    model, processor = load_qwen_for_description(lora_path)
    with model.disable_adapter():
        response = run_qwen_describe(model, processor, image, prompt)
    return response


def _generate_from_messages(model, processor, messages: list, max_new_tokens: int = 200) -> str:
    """Same generation path as run_qwen_describe, but takes a full running
    messages list (multi-turn) instead of building a single-turn one."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def run_image_chat(image_path: str, lora_path: str) -> None:
    """
    Multi-turn REPL over one image, like a normal chat client: the image is
    sent once on the first turn, then each further question is appended to
    a running `messages` history so follow-ups ("what color is it?" after
    "what's on the desk?") have context. Model stays loaded across turns -
    no reload per question.

    Note: each turn re-runs the full history through generate() (no KV
    cache reuse across turns), which is fine for a handful of short-image
    exchanges but will get slower turn-over-turn as history grows - if that
    becomes noticeable, trimming history to the last N turns is the next
    step, not a full rewrite.
    """
    image = Image.open(image_path).convert("RGB")
    image = _resize_for_vram(image)
    model, processor = load_qwen_for_description(lora_path)

    messages: list = []
    first_turn = True
    print("\nImage loaded - ask anything about it. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        if first_turn:
            content = [{"type": "image", "image": image}, {"type": "text", "text": user_text}]
            first_turn = False
        else:
            content = [{"type": "text", "text": user_text}]
        messages.append({"role": "user", "content": content})

        with model.disable_adapter():
            response = _generate_from_messages(model, processor, messages)

        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})


# ══════════════════════════════════════════════════════════════════════════
# Web agent (planner+executor loop - mirrors tests/planner_agent.py)
# ══════════════════════════════════════════════════════════════════════════

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

Examples of correct responses:
CLICK("the blue Add to cart button")
SEARCH("wireless mouse")
SCROLL("down")

Goal: {goal}

Already completed:
{history}

Look at the screenshot. Respond with ONLY one function call for the next step:"""

MAX_STEPS = 6


def parse_action(response: str):
    m = ACTION_RE.search(response)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).strip()


def plan_next_action(runner: DemoRunner, image: Image.Image, goal: str, history: list[str]):
    hist_str = "\n".join(f"- {h}" for h in history) if history else "(none - this is the first step)"
    prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal, history=hist_str)
    with runner._qwen.disable_adapter():
        response = runner._run_qwen(image, prompt)
    action, arg = parse_action(response)
    return response, action, arg


def run_web_agent(runner: DemoRunner, goal: str, start_url: str, headless: bool = False) -> str:
    with BrowserEnv(headless=headless, save_screenshots=True) as browser:
        browser.navigate(start_url)
        history: list[str] = []

        for step in range(1, MAX_STEPS + 1):
            print(f"\n{'=' * 60}\nSTEP {step}\n{'=' * 60}")
            ss = browser.screenshot(wait_stable=True)

            t0 = time.time()
            response, action, arg = plan_next_action(runner, ss.image, goal, history)
            print(f"  ({time.time() - t0:.1f}s) planner said: {response!r}")

            if action is None:
                return f"stopped: could not parse an action from {response!r}"

            if action == "DONE":
                return f"done: {arg}"

            if action == "SEARCH":
                browser.search(arg)
                history.append(f"SEARCH({arg!r})")
            elif action == "CLICK":
                tmp_path = "/tmp/router_agent_step.png"
                ss.image.save(tmp_path)
                dom_elements = browser.get_interactive_elements()
                response, point = runner.act(tmp_path, arg, dom_elements=dom_elements)
                print(f"    grounding response: {response!r} -> point={point}")
                if point:
                    browser.click(*point)
                history.append(f"CLICK({arg!r})")
            elif action == "SCROLL":
                browser.scroll(arg)
                history.append(f"SCROLL({arg!r})")

        return "stopped: MAX_STEPS reached"


# ══════════════════════════════════════════════════════════════════════════
# Router entrypoint
# ══════════════════════════════════════════════════════════════════════════

def route(user_input: str, image_path: str | None, url: str | None, forced_mode: str | None,
          lora_path: str = "models/lora_adapter_smoke", headless: bool = False) -> str:
    mode = classify_intent(user_input, has_url=bool(url), has_image=bool(image_path), forced_mode=forced_mode)
    print(f"[router] mode = {mode}", flush=True)

    if mode == "image":
        if not image_path:
            return "error: image mode selected but no --image was provided"
        # No DemoRunner here on purpose - image mode never needs OmniParser,
        # so skip paying for its VRAM. See load_qwen_for_description().
        return run_image_agent(image_path, user_input, lora_path)

    if not url:
        return "error: web mode selected but no --url was provided"
    # Web mode DOES need OmniParser (grounding uses it), so DemoRunner is
    # the right loader here.
    runner = DemoRunner(lora_path=lora_path, tag="router_agent", training_style=True)
    return run_web_agent(runner, user_input, url, headless=headless)


def main():
    parser = argparse.ArgumentParser(description="Route a user request to the web agent or image-description agent.")
    parser.add_argument("--input", default=None, help="Natural language instruction / question (single-shot mode)")
    parser.add_argument("--image", default=None, help="Path to an image (for the image agent)")
    parser.add_argument("--url", default=None, help="Starting URL (for the web agent)")
    parser.add_argument("--mode", choices=["web", "image"], default=None, help="Force a mode instead of classifying")
    parser.add_argument("--lora", default="models/lora_adapter_smoke", help="LoRA adapter dir")
    parser.add_argument("--headless", action="store_true",
                         help="Run the browser headless (required on Kaggle/servers with no display)")
    parser.add_argument("--interactive", action="store_true",
                         help="Multi-turn chat about --image instead of a single-shot answer. Web mode doesn't support this yet.")
    args = parser.parse_args()

    if args.interactive:
        if not args.image:
            parser.error("--interactive currently only supports the image agent - pass --image")
        run_image_chat(args.image, args.lora)
        return

    if not args.input:
        parser.error("Provide --input '...' for single-shot mode, or --interactive --image ... for a chat loop")

    result = route(args.input, args.image, args.url, args.mode, lora_path=args.lora, headless=args.headless)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()