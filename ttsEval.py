import json
from pathlib import Path
import torch
import torchaudio
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

MERGED_PATH    = "OpenMOSS-Team/MOSS-TTS-GPTQ-Merged"
AUDIO_TOK_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-FP16"
EVAL_META      = "seed-tts-eval/data/zh/meta.lst"
EVAL_DATA_DIR  = Path("seed-tts-eval/data/zh")
OUTPUT_DIR     = Path("eval_outputs/zh")

device = "cuda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = MossTTSDelayModel.from_pretrained(
    MERGED_PATH, gptq_device=device, trust_remote_code=True
).eval()

processor = MossTTSDelayProcessor.from_pretrained(
    MERGED_PATH, codec_path=AUDIO_TOK_PATH, trust_remote_code=True
)

with open(EVAL_META) as f:
    meta = [line.strip().split("|") for line in f if line.strip()]

with torch.no_grad():
    for uid, ref_text, ref_audio_rel, target_text in meta:
        ref_audio = str(EVAL_DATA_DIR / ref_audio_rel)
        text      = target_text

        batch = processor(
            [processor.build_user_message(text=text, reference=[ref_audio])],
            mode="generation"
        )
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=4096,
        )
        for msg in processor.decode([(sl, ids.long()) for sl, ids in outputs]):
            audio = msg.audio_codes_list[0]
            torchaudio.save(
                OUTPUT_DIR / f"{uid}.wav",
                audio.unsqueeze(0),
                processor.model_config.sampling_rate
            )
            print(f"  {uid}.wav")
