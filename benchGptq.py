#!/usr/bin/env python3
import torch, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

correct = 0
MODEL_DIR = "/run/media/blazingbhavneek/Common/Code/sglangServer/inclusionAI/LLaDA2.1-mini-GPTQ-4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
device = next(model.parameters()).device

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
ds = ds.select(range(20))  # small test

CHOICES = "ABCDEFGHIJ"

from collections import defaultdict

def build_fewshot(ds_train, category, n=5):
    examples = [ex for ex in ds_train if ex["category"] == category][:n]
    shots = ""
    for ex in examples:
        opts = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(ex["options"]))
        gold = CHOICES[ex["answer_index"]]
        shots += f"Question: {ex['question']}\n{opts}\nAnswer: {gold}\n\n"
    return shots

ds_train = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

def format_prompt(ex, fewshot=""):
    q = ex["question"]
    opts = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(ex["options"]))
    return f"{fewshot}Question: {q}\n{opts}\nAnswer:"

for i, ex in enumerate(ds):
    fewshot = build_fewshot(ds_train, ex["category"], n=5)
    prompt = format_prompt(ex, fewshot)
    ...
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs=inputs["input_ids"],
        gen_length=64,
        steps=16,
        block_length=32,
        temperature=0.0,
        eos_early_stop=True,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # extract first A-J letter
    import re
    match = re.search(r'\b([A-J])\b', text)
    pred = match.group(1) if match else ""
    gold = CHOICES[ex["answer_index"]]
    correct += pred == gold
    print(f"[{i+1}/20] pred={pred} gold={gold} | {text[:80]!r}")

print(f"\nAccuracy: {correct}/20 = {correct/20:.1%}")