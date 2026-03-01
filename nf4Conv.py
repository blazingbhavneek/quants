from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "JetLM/SDAR-8B-Chat",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    ),
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("JetLM/SDAR-8B-Chat", trust_remote_code=True)

model.save_pretrained("JetLM/SDAR-8B-Chat-nf4")
tokenizer.save_pretrained("JetLM/SDAR-8B-Chat-nf4")