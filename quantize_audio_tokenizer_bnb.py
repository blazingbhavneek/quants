# quantize_audio_tokenizer_bnb.py
from transformers import AutoModel, BitsAndBytesConfig
import torch

AUDIO_TOK_PATH     = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-Audio-Tokenizer"
AUDIO_TOK_NF4_PATH = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-Audio-Tokenizer-NF4"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # extra ~0.1 bpw saving
)

model = AutoModel.from_pretrained(
    AUDIO_TOK_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="cuda",
)

model.save_pretrained(AUDIO_TOK_NF4_PATH)
print("Saved NF4 audio tokenizer")