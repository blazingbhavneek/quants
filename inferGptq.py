#!/usr/bin/env python3
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/run/media/blazingbhavneek/Common/Code/sglangServer/inclusionAI/LLaDA2.1-mini-GPTQ-4bit"

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
print("📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

PROMPT = "Explain the theory of relativity in simple terms."
print(f"\n💬 Prompt: {PROMPT}\n")

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)


def generate_streaming(model, tokenizer, input_ids, gen_length=512, block_length=32,
                       steps=32, temperature=0.0, eos_early_stop=True,
                       mask_id=156895, eos_id=156892, threshold=0.95):

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (
        block_mask
        .repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0).unsqueeze(0)
        .to(torch.bfloat16)
    )

    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    buffer = ""

    print("🤖 Response: ", end="", flush=True)

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length
        cur_x = x[:, :current_window_end]
        cur_attn = attn_mask[:, :, :current_window_end, :current_window_end]
        cur_pos = position_ids[:, :current_window_end]

        for _ in range(steps):
            active_mask = cur_x[:, -block_length:] == mask_id
            if not torch.any(active_mask):
                break

            with torch.no_grad():
                logits = model(cur_x, attention_mask=cur_attn, position_ids=cur_pos).logits

            probs = torch.softmax(logits[:, -block_length:, :], dim=-1)
            x0_p, x0 = probs.max(dim=-1)

            mask_conf = torch.where(active_mask, x0_p, torch.tensor(-torch.inf, device=device))
            if active_mask.sum() > 0:
                high_conf = mask_conf[0] > threshold
                if high_conf.any():
                    cur_x[:, -block_length:][0][high_conf] = x0[0][high_conf]
                else:
                    idx = mask_conf[0].argmax()
                    cur_x[:, -block_length:][0][idx] = x0[0][idx]

        x[:, :current_window_end] = cur_x

        new_text = tokenizer.decode(x[0, prompt_length:current_window_end], skip_special_tokens=True)
        if new_text != buffer:
            sys.stdout.write(new_text[len(buffer):])
            sys.stdout.flush()
            buffer = new_text

        if eos_early_stop:
            generated = x[0, prompt_length:current_window_end]
            if (generated == mask_id).sum() == 0 and (generated == eos_id).any():
                break

    print()
    return x[:, prompt_length:prompt_length + gen_length]


generate_streaming(model, tokenizer, input_ids)
