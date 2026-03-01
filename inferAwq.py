import torch
import llmcompressor.modeling.patch.llada2_patch  # apply monkey-patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import dispatch_for_generation

SAVE_DIR = "LLaDA2.1-mini-AWQ-4bit"

# Load the compressed model — weights stay packed/quantized, no re-compression
model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",  # or "cuda" if you have VRAM
)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)

# Dispatch for generation: sets up quantized linear kernels properly
dispatch_for_generation(model)

# Tokenize prompt
prompt = "The history of artificial intelligence began"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

# Use LLaDA2's custom generate (NOT hf .generate())
output = model.generate(
    input_ids,
    gen_length=100,       # tokens to generate
    steps=32,             # denoising steps per block
    block_length=32,      # parallel decoding block size
    temperature=0.0,      # greedy
    mask_id=156895,
    eos_id=156892,
    eos_early_stop=True,
)

# output starts after the prompt
generated_ids = output[0][input_ids.shape[1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))