import argparse
import gc
import json
import os
import shutil
import torch
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Shard streaming + key remap  (peak RAM = 1 shard, never the full model)
# ──────────────────────────────────────────────────────────────────────────────

def remap_and_save_shards(model_path: str, staging_dir: Path):
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = list(dict.fromkeys(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    new_weight_map = {}
    out_shard_idx  = 0
    total_dropped  = 0

    for shard in shard_files:
        src = os.path.join(model_path, shard)
        print(f"  {shard} ... loading", end="", flush=True)

        raw = load_file(src, device="cpu")
        print(f" ({len(raw)} keys) remapping ...", end="", flush=True)

        remapped = {}
        for k, v in raw.items():
            if k.startswith("language_model."):
                new_k = "model." + k[len("language_model."):]
                remapped[new_k] = v.to(torch.float16) if v.dtype != torch.float16 else v
            elif k == "lm_heads.0.weight":
                remapped["lm_head.weight"] = v.to(torch.float16) if v.dtype != torch.float16 else v
            else:
                total_dropped += 1

        del raw
        gc.collect()

        if not remapped:
            print(" skipped (no backbone keys)")
            continue

        out_shard_idx += 1
        out_name = f"model-{out_shard_idx:05d}-of-XXXX.safetensors"
        save_file(remapped, str(staging_dir / out_name))
        for k in remapped:
            new_weight_map[k] = out_name

        print(f" saved {len(remapped)} keys → {out_name}")
        del remapped
        gc.collect()

    # Rename placeholders now we know total count
    total = out_shard_idx
    final_weight_map = {}
    for old_name in sorted(set(new_weight_map.values())):
        idx   = int(old_name.split("-")[1])
        final = f"model-{idx:05d}-of-{total:05d}.safetensors"
        (staging_dir / old_name).rename(staging_dir / final)
        for k, v in new_weight_map.items():
            if v == old_name:
                final_weight_map[k] = final

    total_bytes = sum(
        (staging_dir / f).stat().st_size
        for f in set(final_weight_map.values())
    )
    with open(staging_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_bytes},
                   "weight_map": final_weight_map}, f, indent=2)

    print(f"\n  Wrote {total} shard(s), dropped {total_dropped} non-backbone keys.")


def save_qwen3_config(model_path: str, staging_dir: Path):
    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)

    lang_cfg = raw.get("language_config", raw)
    lang_cfg["model_type"]    = "qwen3"
    lang_cfg["architectures"] = ["Qwen3ForCausalLM"]
    # Remove auto_map so transformers never tries to load MossTTS custom classes
    lang_cfg.pop("auto_map", None)

    with open(staging_dir / "config.json", "w") as f:
        json.dump(lang_cfg, f, indent=2)

    print(f"  config.json written — "
          f"{lang_cfg.get('num_hidden_layers')} layers, "
          f"hidden={lang_cfg.get('hidden_size')}, "
          f"vocab={lang_cfg.get('vocab_size')}")


def fix_tokenizer_config(staging_dir: Path):
    """Strip any MossTTS model_type references from the saved tokenizer config."""
    tc_path = staging_dir / "tokenizer_config.json"
    if not tc_path.exists():
        return
    with open(tc_path) as f:
        tc = json.load(f)
    # These keys tie the tokenizer back to the custom MossTTS config class
    for key in ("model_type", "auto_map", "tokenizer_class"):
        tc.pop(key, None)
    with open(tc_path, "w") as f:
        json.dump(tc, f, indent=2)
    print("  tokenizer_config.json cleaned (removed moss model_type refs)")

# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def get_calibration_data(tokenizer, dataset_name, num_samples, seq_len):
    if os.path.isfile(dataset_name):
        with open(dataset_name) as f:
            texts = [l.strip() for l in f if l.strip()]
    elif dataset_name == "wikitext2":
        from datasets import load_dataset
        data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in data["text"] if t.strip()]
    elif dataset_name == "c4":
        from datasets import load_dataset
        data  = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = [s["text"] for s in data.take(num_samples * 8)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    all_ids = tokenizer(
        " ".join(texts),
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    samples = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        samples.append({"input_ids": all_ids[i : i + seq_len].unsqueeze(0)})
        if len(samples) >= num_samples:
            break

    if not samples:
        raise RuntimeError(
            f"0 calibration samples produced. "
            f"Try --seq_len 512 or a larger dataset."
        )

    print(f"  {len(samples)} samples × {seq_len} tokens")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--staging_dir", default=None,
                        help="Cached remapped fp16 shards. "
                             "Defaults to <script_dir>/moss_qwen3_staging. "
                             "Delete folder (or just .staging_complete) to redo.")
    parser.add_argument("--bits",         type=int,   default=4, choices=[2,3,4,8])
    parser.add_argument("--group_size",   type=int,   default=128)
    parser.add_argument("--desc_act",     action="store_true", default=False)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--calibration_dataset",     default="wikitext2")
    parser.add_argument("--num_calibration_samples", type=int, default=1024)
    parser.add_argument("--seq_len",      type=int,   default=512)
    args = parser.parse_args()

    SCRIPT_DIR  = Path(__file__).resolve().parent
    staging_dir = Path(args.staging_dir) if args.staging_dir else SCRIPT_DIR / "moss_qwen3_staging"
    sentinel    = staging_dir / ".staging_complete"

    print(f"\n[Config] bits={args.bits}  group_size={args.group_size}  "
          f"desc_act={args.desc_act}  damp={args.damp_percent}")
    print(f"[Config] model_path   = {args.model_path}")
    print(f"[Config] output_path  = {args.output_path}")
    print(f"[Config] staging_dir  = {staging_dir}\n")

    # ── Staging (skipped on re-runs) ──────────────────────────────────────────
    if sentinel.exists():
        print(f"[CACHE HIT] {staging_dir} already staged — skipping remap.\n"
              f"            Delete {sentinel} to force redo.\n")
    else:
        staging_dir.mkdir(parents=True, exist_ok=True)

        print("[1/3] Streaming + remapping safetensor shards...")
        remap_and_save_shards(args.model_path, staging_dir)

        print("\n[2/3] Writing Qwen3 config.json...")
        save_qwen3_config(args.model_path, staging_dir)

        print("\n[3/3] Saving tokenizer...")
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        except Exception as e:
            print(f"  Falling back to Qwen/Qwen3-8B tokenizer ({e})")
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        tok.pad_token = tok.eos_token
        tok.save_pretrained(str(staging_dir))
        fix_tokenizer_config(staging_dir)          # ← add this line
        print(f"  Tokenizer saved. Vocab: {tok.vocab_size}")
        del tok
        gc.collect()

        sentinel.touch()
        print(f"\n  ✓ Staging done. ({sentinel})\n")

    # ── Tokenizer for calibration ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(staging_dir), trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Calibration data ──────────────────────────────────────────────────────
    print("[4/5] Building calibration dataset...")
    calibration_data = get_calibration_data(
        tokenizer=tokenizer,
        dataset_name=args.calibration_dataset,
        num_samples=args.num_calibration_samples,
        seq_len=args.seq_len,
    )

    # ── GPTQ Quantization ─────────────────────────────────────────────────────
    print("\n[5/5] Running GPTQ quantization...")
    from gptqmodel import GPTQModel, QuantizeConfig

    # from gptqmodel import HessianConfig

    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        damp_percent=args.damp_percent,
        sym=True,
        true_sequential=True,
        # hessian=HessianConfig(chunk_bytes=512*1024*1024),  # ← 512MB chunks instead of loading full layer at once
        offload_to_disk=True,
        vram_strategy="balanced",
    )
    print(f"  {quant_config}")

    gptq_model = GPTQModel.from_pretrained(
        str(staging_dir),
        quantize_config=quant_config,
    )

    gptq_model.quantize(
        calibration_data,
        batch_size=1,
        # tokenizer intentionally omitted — calibration data is pre-tokenized
        # input_ids tensors; passing the tokenizer here triggers Tokenicer.load()
        # which chokes on MossTTSDelayConfig.bos_token_id not existing
    )

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    print(f"\nSaving quantized model to {args.output_path} ...")
    gptq_model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    for fname in ("config.json", "configuration_moss_tts.py",
                  "modeling_moss_tts.py", "processing_moss_tts.py",
                  "inference_utils.py"):
        src = os.path.join(args.model_path, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.output_path, "moss_" + fname))

    print(f"\n✅  Done! → {args.output_path}")


if __name__ == "__main__":
    main()
