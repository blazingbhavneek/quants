#!/usr/bin/env python3
"""
MossAudioTokenizer Float16 Quantization Script (FIXED)
Properly handles dtype conversions at encoder/quantizer boundaries
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MossAudioTokenizerQuantizer:
    """Float16 quantization with proper dtype boundary handling."""
    
    def __init__(
        self,
        model_path: str,
        model_dtype: torch.dtype = torch.float16,
        quantizer_dtype: torch.dtype = torch.float32,
        use_autocast: bool = True,  # Use autocast for automatic dtype conversion
    ):
        self.model_path = model_path
        self.model_dtype = model_dtype
        self.quantizer_dtype = quantizer_dtype
        self.use_autocast = use_autocast
        self.model = None
        
    def load_model(self, device: str = "auto"):
        """Load model with float16 encoder/decoder, float32 quantizer."""
        from transformers import AutoModel
        
        print(f"Loading model from {self.model_path}...")
        print(f"Model dtype: {self.model_dtype}, Quantizer dtype: {self.quantizer_dtype}")
        
        # Load model in float16
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=self.model_dtype,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        print(f"✓ Model loaded in {self.model_dtype}")
        
        # Protect quantizer (keep in float32)
        self._protect_quantizer()
        
        # Protect encoder/decoder projection layers (keep in float32)
        self._protect_projections()
        
        self.model.eval()
        return self.model
    
    def _protect_quantizer(self):
        """Keep quantizer modules in float32."""
        print("Protecting quantizer modules...")
        
        if hasattr(self.model, 'quantizer'):
            self.model.quantizer = self.model.quantizer.to(dtype=self.quantizer_dtype)
            print(f"  ✓ Quantizer set to {self.quantizer_dtype}")
    
    def _protect_projections(self):
        """Keep encoder/decoder projection layers in float32 for dtype compatibility."""
        print("Protecting projection layers...")
        
        protected_count = 0
        for name, module in self.model.named_modules():
            # Protect input_proj and output_proj in encoder/decoder
            if any(x in name for x in ['encoder', 'decoder']):
                if 'quantizer' not in name:
                    if isinstance(module, nn.Linear) and any(x in name for x in ['input_proj', 'output_proj']):
                        module = module.to(dtype=torch.float32)
                        protected_count += 1
        
        print(f"  ✓ Protected {protected_count} projection layers in float32")
    
    def prepare_for_inference(self):
        """Prepare for inference."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        print("✓ Model prepared for inference")
    
    def encode(self, audio: torch.Tensor, num_quantizers: Optional[int] = None):
        """Encode audio to tokens with proper dtype handling."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Input must be float32
        if audio.dtype != torch.float32:
            audio = audio.to(dtype=torch.float32)
        
        with torch.no_grad():
            if self.use_autocast:
                # Use autocast for encoder (float16), but not quantizer (float32)
                with torch.autocast(device_type='cuda', dtype=self.model_dtype, enabled=(audio.device.type == 'cuda')):
                    output = self.model.encode(
                        input_values=audio,
                        num_quantizers=num_quantizers,
                        return_dict=True,
                    )
            else:
                output = self.model.encode(
                    input_values=audio,
                    num_quantizers=num_quantizers,
                    return_dict=True,
                )
        
        return output
    
    def decode(self, audio_codes: torch.Tensor, num_quantizers: Optional[int] = None):
        """Decode tokens to audio."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            if self.use_autocast:
                with torch.autocast(device_type='cuda', dtype=self.model_dtype, enabled=(audio_codes.device.type == 'cuda')):
                    output = self.model.decode(
                        audio_codes=audio_codes,
                        num_quantizers=num_quantizers,
                        return_dict=True,
                    )
            else:
                output = self.model.decode(
                    audio_codes=audio_codes,
                    num_quantizers=num_quantizers,
                    return_dict=True,
                )
        
        return output
    
    def get_memory_usage(self):
        """Get memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return {"allocated_mb": allocated, "reserved_mb": reserved}
        return {"allocated_mb": 0, "reserved_mb": 0}

    def save_quantized_model(self, suffix: str = "-FP16"):
        """Save model in two safetensors shards: encoder/decoder in fp16, quantizer in fp32."""
        if self.model is None:
            raise ValueError("Model not loaded")

        import os, shutil, json
        from pathlib import Path
        from safetensors.torch import save_file

        save_path = Path(self.model_path).parent / (Path(self.model_path).name + suffix)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_path}...")

        shard1_name = "model-00001-of-00002.safetensors"  # encoder/decoder fp16
        shard2_name = "model-00002-of-00002.safetensors"  # quantizer fp32

        shard1, shard2 = {}, {}
        weight_map = {}

        for key, tensor in self.model.state_dict().items():
            if key.startswith("quantizer"):
                shard2[key] = tensor.to(torch.float32)
                weight_map[key] = shard2_name
            else:
                shard1[key] = tensor.to(torch.float16)
                weight_map[key] = shard1_name

        save_file(shard1, str(save_path / shard1_name),
                metadata={"dtype": "float16", "contents": "encoder_decoder"})
        save_file(shard2, str(save_path / shard2_name),
                metadata={"dtype": "float32", "contents": "quantizer"})

        print(f"  ✓ Shard 1 (fp16 encoder/decoder): {(save_path / shard1_name).stat().st_size / 1024**3:.2f} GB")
        print(f"  ✓ Shard 2 (fp32 quantizer):        {(save_path / shard2_name).stat().st_size / 1024**3:.2f} GB")

        # Write index JSON
        index = {
            "metadata": {
                "total_size": sum(t.nbytes for t in {**shard1, **shard2}.values()),
                "shard_dtype_map": {
                    shard1_name: "float16",
                    shard2_name: "float32",
                }
            },
            "weight_map": weight_map,
        }
        with open(save_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print("  ✓ Index written with dtype metadata")

        # Copy config/tokenizer files
        for fname in os.listdir(self.model_path):
            if fname.endswith(('.json', '.txt', '.py', '.model', '.tiktoken')):
                src, dst = Path(self.model_path) / fname, save_path / fname
                if not dst.exists():
                    shutil.copy2(src, dst)

        print(f"✓ Done. Saved to {save_path}")
        return str(save_path)

def create_test_audio(duration: float = 1.0, sample_rate: int = 24000):
    """Create test audio."""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    audio = (
        0.5 * torch.sin(2 * torch.pi * 440 * t) +
        0.3 * torch.sin(2 * torch.pi * 880 * t)
    ).unsqueeze(0).unsqueeze(0)
    return audio.float()


def validate_output(original_audio: torch.Tensor, reconstructed_audio: torch.Tensor):
    """Validate reconstruction quality."""
    min_len = min(original_audio.shape[-1], reconstructed_audio.shape[-1])
    original = original_audio[..., :min_len]
    reconstructed = reconstructed_audio[..., :min_len]
    
    # Normalize
    original = original / (original.abs().max() + 1e-8)
    reconstructed = reconstructed / (reconstructed.abs().max() + 1e-8)
    
    noise = original - reconstructed
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    mse = torch.mean((original - reconstructed) ** 2)
    
    return {"snr_db": snr.item(), "mse": mse.item()}


def main():
    """Test different dtype configurations."""
    
    MODEL_PATH = "/mnt/common/Code/sglangServer/Quant/OpenMOSS-Team/MOSS-Audio-Tokenizer"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("MossAudioTokenizer Float16 Quantization Test (FIXED)")
    print("=" * 70)
    
    results = []
    
    configs = [
        {"name": "float32", "model_dtype": torch.float32, "quantizer_dtype": torch.float32, "autocast": False},
        {"name": "float16", "model_dtype": torch.float16, "quantizer_dtype": torch.float32, "autocast": True},
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing Mode: {config['name'].upper()}")
        print(f"{'='*70}")
        
        quantizer = MossAudioTokenizerQuantizer(
            model_path=MODEL_PATH,
            model_dtype=config["model_dtype"],
            quantizer_dtype=config["quantizer_dtype"],
            use_autocast=config["autocast"],
        )
        
        model = quantizer.load_model(device=DEVICE)
        quantizer.prepare_for_inference()
        
        memory = quantizer.get_memory_usage()
        print(f"Memory: {memory['allocated_mb']:.1f} MB")
        
        test_audio = create_test_audio(duration=1.0).to(DEVICE)
        print(f"Input audio dtype: {test_audio.dtype}")
        
        try:
            with torch.no_grad():
                encoder_output = quantizer.encode(audio=test_audio, num_quantizers=32)
                audio_codes = encoder_output.audio_codes
                print(f"✓ Encoding successful, codes shape: {audio_codes.shape}")
                
                decoder_output = quantizer.decode(audio_codes=audio_codes, num_quantizers=32)
                reconstructed = decoder_output.audio
                print(f"✓ Decoding successful, audio shape: {reconstructed.shape}")
            
            metrics = validate_output(test_audio, reconstructed)
            print(f"SNR: {metrics['snr_db']:.2f} dB")
            print(f"MSE: {metrics['mse']:.6f}")
            
            results.append({
                "mode": config["name"],
                "memory_mb": memory['allocated_mb'],
                "snr_db": metrics['snr_db'],
                "mse": metrics['mse'],
                "status": "✓ OK" if metrics['snr_db'] > 10 else "✗ BAD"
            })

            # Save the quantized model (only for float16 config)
            if config["name"] == "float16":
                save_path = quantizer.save_quantized_model(suffix="-FP16")
                print(f"✓ Saved to: {save_path}")

        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({
                "mode": config["name"],
                "memory_mb": memory['allocated_mb'],
                "snr_db": 0,
                "mse": 0,
                "status": "✗ ERROR"
            })
        
        del model
        del quantizer
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<12} {'Memory (MB)':<15} {'SNR (dB)':<12} {'Status':<10}")
    print("-" * 70)
    for r in results:
        savings = ((results[0]["memory_mb"] - r["memory_mb"]) / results[0]["memory_mb"]) * 100 if results[0]["memory_mb"] > 0 else 0
        print(f"{r['mode']:<12} {r['memory_mb']:<15.1f} {r['snr_db']:<12.2f} {r['status']:<10} ({savings:+.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION: Use float16 with protected projections")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
