from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load your existing GPTQ model with Marlin conversion
marlin_model = AutoGPTQForCausalLM.from_quantized(
    "./inclusionAI/LLaDA2.1-mini-GPTQ-4bit",
    use_marlin=True,
    device_map="auto",
    trust_remote_code=True
)
marlin_model.save_pretrained("./inclusionAI/LLaDA2.1-mini-GPTQ-Marlin")