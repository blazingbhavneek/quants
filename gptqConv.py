#!/usr/bin/env python3

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "inclusionAI/LLaDA2.1-mini"
OUTPUT_DIR = "./inclusionAI/LLaDA2.1-mini-GPTQ-4bit"

# ============ Configuration ============
BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 2048  # Increased for better quality
SEQ_LEN = 512
BATCH_SIZE = 1

# ============ Quantization Config ============
# For Marlin compatibility in vLLM/SGLang:
# - bits must be 4 (Marlin is 4-bit only)
# - sym=True (symmetric quantization required for Marlin)
# - desc_act=False recommended for better Marlin performance
quant_config = QuantizeConfig(
    bits=BITS,
    group_size=GROUP_SIZE,
    sym=True,           # REQUIRED for Marlin kernel compatibility
    desc_act=False,     # Recommended: False for better Marlin perf, True for slightly better quality
)

# ============ Load Tokenizer ============
print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============ Load Calibration Dataset ============
print(f"📥 Loading calibration dataset...")

# Option 1: WikiText-2 (small, fast download)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_column = "text"

# Concatenate texts and tokenize
all_text = "\n\n".join([t for t in dataset[text_column] if t.strip()])
all_tokens = tokenizer(all_text, return_tensors="pt")["input_ids"].squeeze(0)

# Create calibration samples
calibration_dataset = []
for i in range(0, len(all_tokens) - SEQ_LEN, SEQ_LEN):
    if len(calibration_dataset) >= NUM_CALIBRATION_SAMPLES:
        break
    
    input_ids = all_tokens[i:i + SEQ_LEN]
    attention_mask = torch.ones_like(input_ids)
    
    calibration_dataset.append({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

print(f"✅ Prepared {len(calibration_dataset)} calibration samples")

# ============ Load Model ============
print("📥 Loading model...")
model = GPTQModel.load(MODEL_ID, quant_config, trust_remote_code=True)

# ============ Quantize ============
print(f"⚙️ Quantizing with batch_size={BATCH_SIZE}...")
model.quantize(calibration_dataset, batch_size=BATCH_SIZE)

# ============ Save ============
# Model is saved in GPTQ format; vLLM/SGLang will auto-convert to Marlin at load time
print(f"💾 Saving to {OUTPUT_DIR}...")
model.save(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Done!")