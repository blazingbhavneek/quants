from pathlib import Path
import torch
import torchaudio
from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Single merged repo — no separate GPTQ_PATH needed ─────────────────────────
MERGED_PATH    = "OpenMOSS-Team/MOSS-TTS-GPTQ-Merged"
AUDIO_TOK_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-FP16"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────
# quantize_config.json in the merged repo triggers auto-detection of the
# GPTQ backbone — no gptq_backbone_path needed
model = MossTTSDelayModel.from_pretrained(
    MERGED_PATH,
    gptq_device=device,
    trust_remote_code=True,
).eval()
print(f"[INFO] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Processor ─────────────────────────────────────────────────────────────────
processor = MossTTSDelayProcessor.from_pretrained(
    MERGED_PATH,
    codec_path=AUDIO_TOK_PATH,
    trust_remote_code=True,
)

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
save_dir = Path("inference_root")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0

with torch.no_grad():
    for start in range(0, len(conversations), 1):
        batch = processor([conversations[start]], mode="generation")
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        for message in processor.decode([(sl, ids.long()) for sl, ids in outputs]):
            audio    = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
            print(f"  Saved {out_path}")
