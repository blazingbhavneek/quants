"""
merge_moss_gptq.py
==================
Takes an already-quantized GPTQ backbone repo + the original MOSS-TTS repo,
and produces ONE unified HF repo that MossTTSDelayModel.from_pretrained() can
load directly (it auto-detects the GPTQ backbone via quantize_config.json).

Usage
-----
python merge_moss_gptq.py \
    --moss_path   /path/to/OpenMOSS-Team/MOSS-TTS \
    --gptq_path   /path/to/OpenMOSS-Team/MOSS-TTS-GPTQ \
    --output_path /path/to/MOSS-TTS-GPTQ-MERGED

What the script does
--------------------
1. Copies the entire GPTQ repo to the output directory (gives us the quantized
   backbone safetensors + quantize_config.json + gptqmodel config).
2. Extracts the NON-backbone weights from the original MOSS-TTS shards:
       emb_ext.*.weight
       lm_heads.*.weight
       language_model.norm.weight
3. Saves those tensors as a single extra safetensors shard
   (moss_extras.safetensors) in the output directory.
4. Patches the output model.safetensors.index.json to reference the extras
   shard, so all keys are discoverable in one place.
5. Copies the MOSS-TTS config.json / custom Python files so
   trust_remote_code=True works.

After running, inference becomes simply:
    model = MossTTSDelayModel.from_pretrained(output_path, trust_remote_code=True)
No gptq_backbone_path needed — the model detects it from quantize_config.json.
"""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ── Keys we want to pull from the original MOSS-TTS safetensors ──────────────

def _is_extra_key(key: str) -> bool:
    """Return True for weights we must copy from the original repo."""
    return (
        key.startswith("emb_ext.")
        or key.startswith("lm_heads.")
        or key == "language_model.norm.weight"
    )


def extract_extras(moss_path: str) -> dict:
    """Stream original MOSS-TTS shards and collect extra (non-backbone) tensors."""
    index_path = os.path.join(moss_path, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # Build shard → list-of-keys map to avoid loading a shard multiple times
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if _is_extra_key(key):
                shard_to_keys.setdefault(shard, []).append(key)
    else:
        # Single-file model
        shard_to_keys = {"model.safetensors": None}  # None = load all, filter later

    extras: dict[str, torch.Tensor] = {}

    for shard, wanted_keys in shard_to_keys.items():
        src = os.path.join(moss_path, shard)
        print(f"  Loading {shard} ...", end="", flush=True)
        raw = load_file(src, device="cpu")
        print(f" ({len(raw)} keys)")

        if wanted_keys is None:
            # Single-shard fallback: filter inline
            wanted_keys = [k for k in raw if _is_extra_key(k)]

        for k in wanted_keys:
            if k in raw:
                t = raw[k]
                extras[k] = t.to(torch.float16) if t.dtype == torch.float32 else t
                print(f"    ✓  {k}  {tuple(t.shape)}  {t.dtype}")
            else:
                print(f"    ✗  {k} NOT FOUND in {shard}")

        del raw
        gc.collect()

    return extras


# ── Patch the GPTQ index to include the extras shard ─────────────────────────

EXTRAS_SHARD_NAME = "moss_extras.safetensors"


def patch_index(output_path: str, extra_keys: list[str]):
    index_path = os.path.join(output_path, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
    else:
        # GPTQ repos sometimes use a flat single safetensors file
        index = {"metadata": {}, "weight_map": {}}
        # Populate from existing safetensors files
        for f in Path(output_path).glob("*.safetensors"):
            if f.name == EXTRAS_SHARD_NAME:
                continue
            partial = load_file(str(f), device="cpu")
            for k in partial:
                index["weight_map"][k] = f.name
            del partial
            gc.collect()

    # Add/overwrite entries for the extras shard
    for k in extra_keys:
        index["weight_map"][k] = EXTRAS_SHARD_NAME

    # Recompute total_size
    all_shards = set(index["weight_map"].values())
    total_bytes = sum(
        (Path(output_path) / s).stat().st_size
        for s in all_shards
        if (Path(output_path) / s).exists()
    )
    index.setdefault("metadata", {})["total_size"] = total_bytes

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Patched index: {len(index['weight_map'])} total keys across {len(all_shards)} shard(s).")


# ── Copy MOSS-TTS config + custom files ──────────────────────────────────────

MOSS_FILES_TO_COPY = [
    "config.json",
    "configuration_moss_tts.py",
    "modeling_moss_tts.py",
    "processing_moss_tts.py",
    "inference_utils.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def copy_moss_files(moss_path: str, output_path: str):
    copied = []
    for fname in MOSS_FILES_TO_COPY:
        src = os.path.join(moss_path, fname)
        if os.path.isfile(src):
            dst = os.path.join(output_path, fname)
            shutil.copy2(src, dst)
            copied.append(fname)
    print(f"  Copied from MOSS-TTS: {', '.join(copied)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge MOSS-TTS extras (emb_ext / lm_heads / norm) into an "
                    "existing GPTQ backbone repo to produce a single self-contained repo."
    )
    parser.add_argument("--moss_path",   required=True,
                        help="Path to original MOSS-TTS repo (fp32 / bf16).")
    parser.add_argument("--gptq_path",   required=True,
                        help="Path to already-quantized GPTQ backbone repo.")
    parser.add_argument("--output_path", required=True,
                        help="Destination path for the merged repo.")
    args = parser.parse_args()

    moss_path   = args.moss_path
    gptq_path   = args.gptq_path
    output_path = args.output_path

    print(f"\n[merge_moss_gptq]")
    print(f"  MOSS-TTS  : {moss_path}")
    print(f"  GPTQ repo : {gptq_path}")
    print(f"  Output    : {output_path}\n")

    # ── Step 1: Copy GPTQ repo to output ──────────────────────────────────────
    if Path(output_path).resolve() != Path(gptq_path).resolve():
        print("[1/4] Copying GPTQ repo to output directory...")
        if os.path.exists(output_path):
            print(f"  WARNING: {output_path} already exists — merging into it.")
        else:
            shutil.copytree(gptq_path, output_path)
            print(f"  Copied {gptq_path} → {output_path}")
    else:
        print("[1/4] output_path == gptq_path, editing in-place.")

    # ── Step 2: Extract extras from MOSS-TTS ──────────────────────────────────
    print("\n[2/4] Extracting extra weights from MOSS-TTS shards...")
    extras = extract_extras(moss_path)

    if not extras:
        raise RuntimeError(
            "No extra keys found in MOSS-TTS shards. "
            "Check that --moss_path points to the original (non-GPTQ) repo."
        )

    # ── Step 3: Save extras shard ──────────────────────────────────────────────
    extras_path = os.path.join(output_path, EXTRAS_SHARD_NAME)
    print(f"\n[3/4] Saving {len(extras)} tensors → {EXTRAS_SHARD_NAME} ...")
    save_file(extras, extras_path)
    size_mb = Path(extras_path).stat().st_size / 1e6
    print(f"  Saved {extras_path}  ({size_mb:.1f} MB)")
    del extras
    gc.collect()

    # ── Step 4: Patch the index + copy MOSS config files ──────────────────────
    print("\n[4/4] Patching model.safetensors.index.json ...")

    # Reload extras key list without the tensors
    index_path = os.path.join(moss_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            moss_index = json.load(f)
        extra_keys = [k for k in moss_index["weight_map"] if _is_extra_key(k)]
    else:
        # Fallback: scan the extras shard we just saved
        tmp = load_file(extras_path, device="cpu")
        extra_keys = list(tmp.keys())
        del tmp

    patch_index(output_path, extra_keys)

    print("\n  Copying MOSS-TTS config and custom files...")
    copy_moss_files(moss_path, output_path)

    # Verify quantize_config.json is present (needed for auto-detection)
    qc_out = os.path.join(output_path, "quantize_config.json")
    if os.path.isfile(qc_out):
        print(f"\n  ✓ quantize_config.json present — auto-detection will work.")
    else:
        qc_src = os.path.join(gptq_path, "quantize_config.json")
        if os.path.isfile(qc_src):
            shutil.copy2(qc_src, qc_out)
            print(f"\n  ✓ quantize_config.json copied from GPTQ repo — auto-detection will work.")
        else:
            print(f"\n  ⚠ quantize_config.json not found in GPTQ repo either. "
                  f"Auto-detection may fail; pass gptq_backbone_path explicitly.")

    print(f"\n✅  Done!  Merged repo → {output_path}")
    print("""
Usage after merging
-------------------
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

model = MossTTSDelayModel.from_pretrained(
    "{output}",
    trust_remote_code=True,
)  # gptq_backbone_path is NOT needed — auto-detected from quantize_config.json
""".format(output=output_path))


if __name__ == "__main__":
    main()
