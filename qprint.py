import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Original bf16
model_orig = AutoModelForCausalLM.from_pretrained(
    "JetLM/SDAR-8B-Chat", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
with open("keys_original.txt", "w") as f:
    for key in sorted(model_orig.state_dict().keys()):
        f.write(key + "\n")
del model_orig

# BNB int8 from local
model_bnb = AutoModelForCausalLM.from_pretrained(
    "JetLM/SDAR-8B-Chat-int8",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
    trust_remote_code=True,
)
with open("keys_bnb.txt", "w") as f:
    for key in sorted(model_bnb.state_dict().keys()):
        f.write(key + "\n")

print("Done!")


# BNB nf4 from local
model_bnb = AutoModelForCausalLM.from_pretrained(
    "JetLM/SDAR-8B-Chat-nf4",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
    device_map="auto",
    trust_remote_code=True,
)
with open("keys_bnb_nf4.txt", "w") as f:
    for key in sorted(model_bnb.state_dict().keys()):
        f.write(key + "\n")

print("Done!")


