# src/train/finetune.py
"""
QLoRA fine-tuning of Qwen2.5-VL-3B-Instruct on SoM grounding conversations.
Designed to run on Google Colab free T4 (16GB VRAM).

─── Setup (run once in Colab) ────────────────────────────────────────────────
!pip install transformers accelerate bitsandbytes peft trl qwen-vl-utils torchvision

─── Data upload options ──────────────────────────────────────────────────────
Option A (recommended): Mount Google Drive, copy data there once.
    from google.colab import drive
    drive.mount('/content/drive')
    # set TRAIN_JSONL and RENDERS_DIR to paths under /content/drive/MyDrive/

Option B: Upload train.jsonl + renders zip directly to Colab session.
    # Will be lost on session reset. Fine for a single run.

─── Run ──────────────────────────────────────────────────────────────────────
    python finetune.py

─── Output ───────────────────────────────────────────────────────────────────
    lora_adapter/   ← download this (~100MB), copy to models/lora_adapter/
                       then run: python -m src.eval.eval --adapter models/lora_adapter
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── qwen_vl_utils ─────────────────────────────────────────────────────────
from qwen_vl_utils import process_vision_info

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as _VLModel  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths for your Colab environment
# ══════════════════════════════════════════════════════════════════════════════

# ── data paths ────────────────────────────────────────────────────────────
# Colab: upload train.jsonl and the renders/ directory, then set these.
# Drive example: "/content/drive/MyDrive/magma-repro/data/processed/seeclick_web/train.jsonl"
TRAIN_JSONL = "train.jsonl"
RENDERS_DIR = "renders/"          # directory containing *_som.png files

# ── model ─────────────────────────────────────────────────────────────────
# On Colab: use HF ID (downloads automatically).
# Locally:  use "/data/Magma/magma-repro/models/qwen2_5_vl_3b"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── output ────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "checkpoints"
ADAPTER_DIR = "lora_adapter"

# ── training hyperparams ──────────────────────────────────────────────────
EPOCHS         = 3
BATCH_SIZE     = 1        # T4 16GB: 1 is safe with image tokens
GRAD_ACCUM     = 8        # effective batch size = 8
LR             = 2e-4
WARMUP_RATIO   = 0.03
MAX_SEQ_LEN    = 512      # output is short; input image handles the rest
SAVE_STEPS     = 200

# ── LoRA ──────────────────────────────────────────────────────────────────
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
# Target all linear projection layers in the language model
LORA_TARGETS   = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = "You are agent that can see, talk and act."


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SoMDataset(Dataset):
    """
    Loads train.jsonl and serves (image, user_text, assistant_text) triples.

    Training signal: first grounding turn only (user→assistant).
    This directly matches what the eval harness measures.

    Skips samples where the SoM image is missing.
    """

    def __init__(self, jsonl_path: str, renders_dir: str):
        self.renders_dir = Path(renders_dir)
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                entry = self._parse(sample)
                if entry is not None:
                    self.samples.append(entry)

        print(f"Dataset: {len(self.samples):,} valid samples "
              f"(from {jsonl_path})")

    def _parse(self, sample: dict) -> Optional[dict]:
        convs = sample.get("conversations", [])
        if len(convs) < 2:
            return None

        # SoM image path — try som_image field first, then derive from stem
        som_path = None
        if sample.get("som_image"):
            p = Path(sample["som_image"])
            if p.exists():
                som_path = p
        if som_path is None:
            stem = Path(sample["image"]).stem
            p = self.renders_dir / f"{stem}_som.png"
            if p.exists():
                som_path = p
        if som_path is None:
            return None   # skip — no SoM image

        # First grounding turn
        user_raw  = convs[0]["value"]
        asst_text = convs[1]["value"]

        # Strip "<image>\n" prefix from user turn
        user_text = re.sub(r"^<image>\n?", "", user_raw).strip()

        # Skip non-grounding turns (point→text, bbox→text produce no coords)
        if not any(kw in asst_text for kw in ("Coordinate:", "coordinate:")):
            return None

        return {
            "som_path":  som_path,
            "user":      user_text,
            "assistant": asst_text,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "image":     Image.open(s["som_path"]).convert("RGB"),
            "user":      s["user"],
            "assistant": s["assistant"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Data collator
# ══════════════════════════════════════════════════════════════════════════════

class SoMCollator:
    """
    Converts a batch of (image, user, assistant) dicts into model inputs
    with labels masked on the user/image tokens (loss only on assistant).
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        all_inputs = []

        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text",  "text":  item["user"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": item["assistant"],
                },
            ]

            # Full sequence (with assistant response, no generation prompt)
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Prompt only (to find where assistant response starts)
            prompt_messages = messages[:-1]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, _ = process_vision_info(messages)

            # Tokenize full sequence
            enc = self.processor(
                text=[full_text],
                images=image_inputs,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize prompt to find mask boundary
            prompt_enc = self.processor(
                text=[prompt_text],
                images=image_inputs,
                return_tensors="pt",
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Labels: -100 for prompt tokens, real ids for assistant tokens
            labels = enc["input_ids"].clone()
            labels[0, :prompt_len] = -100
            # Also mask padding
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            enc["labels"] = labels
            all_inputs.append(enc)

        # Stack batch
        return {
            k: torch.cat([x[k] for x in all_inputs], dim=0)
            for k in all_inputs[0]
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("QLoRA fine-tuning: Qwen2.5-VL-3B-Instruct on SoM data")
    print("=" * 60)
    print(f"  Train JSONL  : {TRAIN_JSONL}")
    print(f"  Renders dir  : {RENDERS_DIR}")
    print(f"  Model        : {MODEL_ID}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Effective bs : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  LoRA rank    : {LORA_RANK}")
    print()

    # ── load dataset ─────────────────────────────────────────────────
    dataset = SoMDataset(TRAIN_JSONL, RENDERS_DIR)
    if len(dataset) == 0:
        raise RuntimeError(
            "No valid training samples found. "
            "Check TRAIN_JSONL and RENDERS_DIR paths."
        )

    # ── load model in 4-bit ───────────────────────────────────────────
    print("Loading model in 4-bit ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = _VLModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.enable_input_require_grads()      # required for QLoRA

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # ── apply LoRA ────────────────────────────────────────────────────
    print("Applying LoRA ...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── training args ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,   # critical — we handle collation ourselves
        dataloader_num_workers=2,
        report_to="none",              # disable wandb/tensorboard in Colab
        label_names=["labels"],
    )

    # ── trainer ───────────────────────────────────────────────────────
    collator = SoMCollator(processor)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        # SFTTrainer: disable its own tokenization — we handle it in collator
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LEN,
    )

    # ── train ─────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer.train()

    # ── save adapter ──────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter to {ADAPTER_DIR} ...")
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    print("Done.")
    print(f"\nDownload '{ADAPTER_DIR}/' and copy to your local machine at:")
    print("  models/lora_adapter/")
    print("\nThen run eval:")
    print("  python -m src.eval.eval --adapter models/lora_adapter --name finetuned")


if __name__ == "__main__":
    main()