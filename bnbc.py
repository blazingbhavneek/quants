#!/usr/bin/env python3
"""
Convert HF model to BitsAndBytes 4-bit format (RAM optimized).
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import argparse
from safetensors.torch import save_file


def convert_model_to_bnb_4bit(
    model_name: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_id: str = None
):
    """Convert model to BnB 4-bit with minimal RAM usage."""
    
    print(f"Converting {model_name} to BnB 4-bit format")
    print("=" * 80)
    
    # BnB config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except:
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving model and tokenizer...")
    # Use HuggingFace's native save - it handles BnB state correctly
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Saved to {output_dir}")
    
    # Clear model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if push_to_hub and hub_model_id:
        print(f"Pushing to {hub_model_id}...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id=hub_model_id,
            repo_type="model",
        )
        print(f"✓ Pushed to Hub")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="inclusionAI/LLaDA2.1-mini")
    parser.add_argument("--output-dir", type=str, default="./inclusionAI/LLaDA2.1-mini-bnb")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default="inclusionAI/LLaDA2.1-mini-bnb")
    
    args = parser.parse_args()
    
    convert_model_to_bnb_4bit(
        model_name=args.model_name,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )