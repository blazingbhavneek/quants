from pathlib import Path
import importlib.util
import gc
import json
import torch
import torchaudio
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from gptqmodel import GPTQModel
from scipy.io import wavfile

from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

WEIGHTS_PATH       = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-TTS"
GPTQ_PATH          = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-TTS-GPTQ"
AUDIO_TOK_NF4_PATH = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-Audio-Tokenizer-NF4"

device = "cuda" if torch.cuda.is_available() else "cpu"

def resolve_attn_implementation() -> str:
    if device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        if torch.cuda.get_device_capability()[0] >= 8:
            return "flash_attention_2"
    return "sdpa" if device == "cuda" else "eager"

attn_implementation = resolve_attn_implementation()
print(f"[INFO] attn_implementation={attn_implementation}")

# ── Step 1: Audio tokenizer ────────────────────────────────────────────────────
print("[1/4] Loading audio tokenizer (float32)...")
audio_tokenizer = AutoModel.from_pretrained(
    "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-Audio-Tokenizer",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cuda",
)
audio_tokenizer.eval()
print(f"[INFO] VRAM after audio tok: {torch.cuda.memory_allocated()/1e9:.2f} GB")
gc.collect()
torch.cuda.empty_cache()

# ── Step 2: GPTQ backbone ─────────────────────────────────────────────────────
print("[2/4] Loading GPTQ backbone...")
gptq_causal_lm = GPTQModel.from_quantized(
    GPTQ_PATH,
    device=device,
    dtype=torch.bfloat16,
    backend="torch",
)
print(f"[INFO] VRAM after GPTQ: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Step 3: Build model shell + inject backbone ───────────────────────────────
print("[3/4] Building model shell + injecting backbone...")
config = AutoConfig.from_pretrained(WEIGHTS_PATH, trust_remote_code=True)
with torch.device("meta"):
    model = MossTTSDelayModel(config)

# ── Inline wrapper — keeps full GPTQ object alive, never unwraps inner model ──
import torch.nn as nn

class _GPTQBackboneWrapper(nn.Module):
    def __init__(self, gptq_causal_lm):
        super().__init__()
        self.gptq_causal_lm = gptq_causal_lm

    def get_input_embeddings(self):
        # gptq_causal_lm.model = Qwen3ForCausalLM
        # gptq_causal_lm.model.model = Qwen3Model
        return self.gptq_causal_lm.model.model.embed_tokens

    def forward(self, **kwargs):
        # Call Qwen3Model directly (inside ForCausalLM) — but keep gptq_causal_lm alive
        # so ExllamaV2/Torch quantized layers retain their parent references
        return self.gptq_causal_lm.model.model(**kwargs)

model.language_model = _GPTQBackboneWrapper(gptq_causal_lm)
# do NOT del gptq_causal_lm — the wrapper holds a reference, it won't be freed
gc.collect()
torch.cuda.empty_cache()

# ── Step 4: Load emb_ext + lm_heads ──────────────────────────────────────────
print("[4/4] Loading non-backbone weights (emb_ext + lm_heads)...")
with open(Path(WEIGHTS_PATH) / "model.safetensors.index.json") as f:
    index = json.load(f)

needed_shards = set(
    v for k, v in index["weight_map"].items()
    if k.startswith("emb_ext.") or k.startswith("lm_heads.")
)
non_backbone = {}
for shard in needed_shards:
    tensors = load_file(str(Path(WEIGHTS_PATH) / shard), device="cpu")
    for k, v in tensors.items():
        if k.startswith("emb_ext.") or k.startswith("lm_heads."):
            non_backbone[k] = v.to(torch.bfloat16)
    del tensors
    gc.collect()

model.emb_ext  = model.emb_ext.to_empty(device="cpu")
model.lm_heads = model.lm_heads.to_empty(device="cpu")
model.load_state_dict(non_backbone, strict=False)
model.emb_ext  = model.emb_ext.to(device=device, dtype=torch.bfloat16)
model.lm_heads = model.lm_heads.to(device=device, dtype=torch.bfloat16)
model.eval()

print(f"[INFO] VRAM after full model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Processor ─────────────────────────────────────────────────────────────────
processor = MossTTSDelayProcessor.from_pretrained(
    WEIGHTS_PATH,
    _preloaded_audio_tokenizer=audio_tokenizer,
)
processor.audio_tokenizer = audio_tokenizer

# ── Debug forward pass ────────────────────────────────────────────────────────
with torch.no_grad():
    batch = processor([[processor.build_user_message(text="test")]], mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Check audio token index range vs emb_ext vocab size
    audio_ids = input_ids[..., 1:]
    print(f"audio_ids range: {audio_ids.min().item()} - {audio_ids.max().item()}")
    print(f"emb_ext vocab size: {model.emb_ext[0].num_embeddings}")

    embeds = model.get_input_embeddings(input_ids)
    print(f"embeds nan: {embeds.isnan().any()}, inf: {embeds.isinf().any()}")

    # Find which layer first produces NaN
    backbone_out = model.language_model.gptq_causal_lm.model.model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    nan_layer = None
    for i, hs in enumerate(backbone_out.hidden_states):
        if hs.isnan().any() or hs.isinf().any():
            nan_layer = i
            print(f"NaN/Inf first appears at hidden_state[{i}]")
            break
    if nan_layer is None:
        print("All hidden states clean — NaN is in lm_head projection")

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"logit[0] nan: {out.logits[0].isnan().any()}, inf: {out.logits[0].isinf().any()}")

# ── Conversations ─────────────────────────────────────────────────────────────
text_1 = "亲爱的你，\n你好呀。\n\n今天，我想用最认真、最温柔的声音，对你说一些重要的话。\n这些话，像一颗小小的星星，希望能在你的心里慢慢发光。"
text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"

ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    [processor.build_user_message(text=text_1)],
    [processor.build_user_message(text=text_2)],
    [processor.build_user_message(text=text_3)],
    [processor.build_user_message(text=text_4)],
    [processor.build_user_message(text=text_5)],
    [processor.build_user_message(text=text_6)],
    [processor.build_user_message(text=text_1, reference=[ref_audio_1])],
    [processor.build_user_message(text=text_2, reference=[ref_audio_2])],
    [processor.build_user_message(text=text_2, tokens=325)],
    [processor.build_user_message(text=text_2, tokens=600)],
]

# ── Inference ─────────────────────────────────────────────────────────────────
batch_size = 1
save_dir = Path("inference_root")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0

with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        # Cast to int — generation_ids are integer token indices, not floats
        outputs_fixed = [(start_len, ids.long()) for start_len, ids in outputs]

        for message in processor.decode(outputs_fixed):
            audio    = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
            print(f"  Saved {out_path}")
            