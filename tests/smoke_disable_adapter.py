# /tests/smoke_disable_adapter.py
"""
Standalone smoke test: does PEFT's disable_adapter() context manager work
cleanly on a 4-bit quantized Qwen2.5-VL adapter, without touching
load_adapter/set_adapter/unload (the confirmed-buggy path)?

Run directly — does not import anything from src/agent/.
"""
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image

QWEN_PATH = "models/qwen2_5_vl_3b"
LORA_PATH = "models/lora_adapter_smoke"
TEST_IMAGE = "outputs/browser/ss_0000.png"  # swap for any screenshot you have

print("Loading base Qwen2.5-VL-3B (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_PATH, quantization_config=bnb, device_map="auto",
)

print(f"Loading LoRA adapter: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)
processor = AutoProcessor.from_pretrained(QWEN_PATH)
print("Adapter loaded.\n")


def run_once(image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


image = Image.open(TEST_IMAGE).convert("RGB")
prompt = "Describe what you see on this screen in one sentence."

print("── WITH adapter (default) ──")
with_adapter = run_once(image, prompt)
print(f"  {with_adapter!r}\n")

print("── WITH disable_adapter() (should behave like base model) ──")
try:
    with model.disable_adapter():
        without_adapter = run_once(image, prompt)
    print(f"  {without_adapter!r}\n")
except Exception as exc:
    print(f"  ✗ disable_adapter() THREW: {exc}\n")
    without_adapter = None

print("── back to WITH adapter (confirm it re-enables cleanly) ──")
with_adapter_again = run_once(image, prompt)
print(f"  {with_adapter_again!r}\n")

print("═" * 60)
print("RESULTS")
print("═" * 60)
if without_adapter is None:
    print("✗ disable_adapter() failed outright — do not build on this.")
elif with_adapter == without_adapter:
    print("⚠ Outputs are IDENTICAL with/without adapter.")
    print("  Either disable_adapter() silently did nothing (same failure")
    print("  mode as the load_adapter/set_adapter bug), OR this prompt")
    print("  happens to produce the same output either way — try a more")
    print("  distinctive prompt/image before concluding it's broken.")
elif with_adapter == with_adapter_again:
    print("✓ disable_adapter() produced a DIFFERENT output, and the")
    print("  adapter re-enabled cleanly afterward (first and third runs")
    print("  match). This looks like a reliable toggle — safe to build on.")
else:
    print("⚠ with_adapter != with_adapter_again — outputs aren't even")
    print("  stable across repeated calls with the SAME adapter state.")
    print("  Check generation is deterministic (do_sample=False expected")
    print("  by default, but confirm) before trusting any of this.")