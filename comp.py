#!/usr/bin/env python3
"""
Script to compare weight mappings between original and BnB quantized models.
This helps create the bitsandbytes_stacked_params_mapping for SGLang.

Loads models sequentially to conserve VRAM.

Usage: python compare_weight_mapping_sequential.py
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from collections import defaultdict
import re
import gc


def analyze_weight_mapping(original_model_name: str, bnb_model_name: str):
    """
    Compare weight mappings between original and BnB models.
    Loads models sequentially to save VRAM.
    
    Args:
        original_model_name: Original model name/path
        bnb_model_name: BnB quantized model name/path
    """
    
    print("=" * 100)
    print("WEIGHT MAPPING ANALYSIS: Original vs BitsAndBytes Model (Sequential Loading)")
    print("=" * 100)
    print("\nNote: Loading models one at a time to conserve VRAM")
    
    # ========== STEP 1: Load original model and extract parameter info ==========
    print(f"\n[1/5] Loading original model: {original_model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  ✓ Loaded: {type(original_model).__name__}")
    
    print(f"\n[2/5] Extracting original model parameters...")
    original_params_info = {}
    for name, param in original_model.named_parameters():
        original_params_info[name] = {
            'shape': tuple(param.shape),
            'dtype': str(param.dtype),
        }
    
    print(f"  ✓ Extracted {len(original_params_info)} parameters")
    
    # Print original parameters
    print(f"\n{'─' * 100}")
    print(f"ORIGINAL MODEL PARAMETERS ({len(original_params_info)} total)")
    print(f"{'─' * 100}")
    
    # Filter out duplicate expert patterns - only show expert.0 if multiple experts exist
    def should_print_param(param_name, all_params):
        """Filter to show only expert.0 when multiple experts exist"""
        # Check if this is an expert parameter
        expert_match = re.search(r'\.experts\.(\d+)\.', param_name)
        if expert_match:
            expert_num = int(expert_match.group(1))
            # Only print expert.0, skip all others
            if expert_num > 0:
                # Check if expert.0 exists for this layer
                expert_0_version = param_name.replace(f'.experts.{expert_num}.', '.experts.0.')
                if expert_0_version in all_params:
                    return False  # Skip, we'll show expert.0 instead
            # If this is expert.0, print it with a note about total experts
            elif expert_num == 0:
                # Count how many experts exist
                base_pattern = re.sub(r'\.experts\.\d+\.', '.experts.', param_name)
                expert_count = sum(1 for p in all_params if re.search(r'\.experts\.\d+\.', p) and 
                                 re.sub(r'\.experts\.\d+\.', '.experts.', p) == base_pattern)
                return True, expert_count
        return True, None
    
    printed_count = 0
    for name in sorted(original_params_info.keys()):
        result = should_print_param(name, original_params_info.keys())
        if isinstance(result, tuple):
            should_print, expert_count = result
            if should_print:
                info = original_params_info[name]
                if expert_count and expert_count > 1:
                    print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']} [+{expert_count-1} more experts]")
                else:
                    print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']}")
                printed_count += 1
        elif result:
            info = original_params_info[name]
            print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']}")
            printed_count += 1
    
    if printed_count < len(original_params_info):
        print(f"  ... ({len(original_params_info) - printed_count} expert parameters hidden, showing only expert.0)")

    
    # Delete original model to free VRAM
    print(f"\n[3/5] Freeing VRAM (deleting original model)...")
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
    gc.collect()
    
    # Additional aggressive cleanup
    import time
    time.sleep(2)  # Give system time to release memory
    
    if torch.cuda.is_available():
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    print(f"  ✓ VRAM freed")
    
    # ========== STEP 2: Load BnB model on CPU ==========
    print(f"\n[4/5] Loading BnB model ON CPU: {bnb_model_name}")
    print(f"  Note: Loading on CPU to avoid VRAM issues")
    
    # Force CPU loading - no GPU at all
    try:
        bnb_model = AutoModelForCausalLM.from_pretrained(
            bnb_model_name,
            device_map="cpu",  # Force CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  ✓ Loaded: {type(bnb_model).__name__} (on CPU)")
    except Exception as e:
        print(f"  ✗ Could not load BnB model: {e}")
        print(f"\n  Note: Make sure you've run convert_to_bnb.py first to create the BnB model.")
        return
    
    print(f"\n[5/5] Extracting BnB model parameters...")
    bnb_params_info = {}
    for name, param in bnb_model.named_parameters():
        param_type = type(param).__name__
        bnb_params_info[name] = {
            'shape': tuple(param.shape) if hasattr(param, 'shape') else 'N/A',
            'dtype': str(param.dtype) if hasattr(param, 'dtype') else 'N/A',
            'type': param_type,
        }
    
    print(f"  ✓ Extracted {len(bnb_params_info)} parameters")
    
    # Print BnB parameters
    print(f"\n{'─' * 100}")
    print(f"BNB MODEL PARAMETERS ({len(bnb_params_info)} total)")
    print(f"{'─' * 100}")
    
    printed_count = 0
    for name in sorted(bnb_params_info.keys()):
        result = should_print_param(name, bnb_params_info.keys())
        if isinstance(result, tuple):
            should_print, expert_count = result
            if should_print:
                info = bnb_params_info[name]
                if expert_count and expert_count > 1:
                    print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']} | type: {info['type']} [+{expert_count-1} more experts]")
                else:
                    print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']} | type: {info['type']}")
                printed_count += 1
        elif result:
            info = bnb_params_info[name]
            print(f"  {name:80s} | shape: {str(info['shape']):30s} | dtype: {info['dtype']} | type: {info['type']}")
            printed_count += 1
    
    if printed_count < len(bnb_params_info):
        print(f"  ... ({len(bnb_params_info) - printed_count} expert parameters hidden, showing only expert.0)")

    
    # Delete BnB model to free VRAM
    del bnb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # ========== STEP 3: Analyze the extracted information ==========
    original_params = set(original_params_info.keys())
    bnb_params = set(bnb_params_info.keys())
    
    # Parameters only in original (these might be stacked in BnB)
    only_in_original = original_params - bnb_params
    # Parameters only in BnB (these might be stacked versions)
    only_in_bnb = bnb_params - original_params
    # Common parameters
    common_params = original_params & bnb_params
    
    print(f"\n{'=' * 100}")
    print(f"MAPPING ANALYSIS")
    print(f"{'=' * 100}")
    print(f"  Common parameters: {len(common_params)}")
    print(f"  Only in original: {len(only_in_original)}")
    print(f"  Only in BnB: {len(only_in_bnb)}")
    
    # Group parameters by layer to identify stacking patterns
    print(f"\n{'─' * 100}")
    print(f"PARAMETERS ONLY IN ORIGINAL (potentially stacked in BnB):")
    print(f"{'─' * 100}")
    
    # Group by layer
    layer_groups = defaultdict(list)
    for param_name in sorted(only_in_original):
        # Extract layer number if present
        match = re.search(r'layers?\.(\d+)', param_name)
        if match:
            layer_num = match.group(1)
            layer_groups[layer_num].append(param_name)
        else:
            layer_groups['other'].append(param_name)
    
    for layer_num in sorted(layer_groups.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        print(f"\n  Layer {layer_num}:")
        layer_params = layer_groups[layer_num]
        
        # Filter experts - only show expert.0
        filtered_params = []
        for param_name in layer_params:
            result = should_print_param(param_name, layer_params)
            if isinstance(result, tuple):
                should_print, expert_count = result
                if should_print:
                    filtered_params.append((param_name, expert_count))
            elif result:
                filtered_params.append((param_name, None))
        
        for item in filtered_params:
            if isinstance(item, tuple):
                param_name, expert_count = item
                info = original_params_info[param_name]
                if expert_count and expert_count > 1:
                    print(f"    {param_name:80s} | shape: {info['shape']} [+{expert_count-1} more experts]")
                else:
                    print(f"    {param_name:80s} | shape: {info['shape']}")
        
        if len(filtered_params) < len(layer_params):
            print(f"    ... ({len(layer_params) - len(filtered_params)} expert parameters hidden)")
    
    print(f"\n{'─' * 100}")
    print(f"PARAMETERS ONLY IN BNB (potentially stacked versions):")
    print(f"{'─' * 100}")
    
    bnb_layer_groups = defaultdict(list)
    for param_name in sorted(only_in_bnb):
        match = re.search(r'layers?\.(\d+)', param_name)
        if match:
            layer_num = match.group(1)
            bnb_layer_groups[layer_num].append(param_name)
        else:
            bnb_layer_groups['other'].append(param_name)
    
    for layer_num in sorted(bnb_layer_groups.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        print(f"\n  Layer {layer_num}:")
        layer_params = bnb_layer_groups[layer_num]
        
        # Filter experts - only show expert.0
        filtered_params = []
        for param_name in layer_params:
            result = should_print_param(param_name, layer_params)
            if isinstance(result, tuple):
                should_print, expert_count = result
                if should_print:
                    filtered_params.append((param_name, expert_count))
            elif result:
                filtered_params.append((param_name, None))
        
        for item in filtered_params:
            if isinstance(item, tuple):
                param_name, expert_count = item
                info = bnb_params_info[param_name]
                if expert_count and expert_count > 1:
                    print(f"    {param_name:80s} | shape: {info['shape']} [+{expert_count-1} more experts]")
                else:
                    print(f"    {param_name:80s} | shape: {info['shape']}")
        
        if len(filtered_params) < len(layer_params):
            print(f"    ... ({len(layer_params) - len(filtered_params)} expert parameters hidden)")
    
    # Suggest stacked parameter mappings
    print(f"\n{'=' * 100}")
    print(f"SUGGESTED STACKED PARAMETER MAPPING")
    print(f"{'=' * 100}")
    print(f"\nBased on the analysis, here's a suggested mapping for SGLang:")
    print(f"\nbitsandbytes_stacked_params_mapping = {{")
    
    # Try to detect common stacking patterns
    detected_mappings = detect_stacking_patterns(only_in_original, only_in_bnb, original_params_info, bnb_params_info)
    
    if detected_mappings:
        for orig_suffix, (stacked_suffix, index) in detected_mappings.items():
            print(f'    "{orig_suffix}": ("{stacked_suffix}", {index}),')
    else:
        print("    # No automatic detection - manual analysis required")
        print("    # Format: original_suffix: (stacked_suffix, index)")
        print("    # Example from llama:")
        print('    # ".q_proj": (".qkv_proj", 0),')
        print('    # ".k_proj": (".qkv_proj", 1),')
        print('    # ".v_proj": (".qkv_proj", 2),')
    
    print(f"}}")
    
    # Print detailed comparison for manual analysis
    print(f"\n{'=' * 100}")
    print(f"DETAILED COMPARISON FOR MANUAL MAPPING")
    print(f"{'=' * 100}")
    print(f"\nCompare these groups to identify stacking patterns:\n")
    
    # Look for QKV-like patterns
    qkv_pattern = re.compile(r'(.*)\.(q_proj|k_proj|v_proj|query|key|value|wq|wk|wv)')
    gate_up_pattern = re.compile(r'(.*)\.(gate_proj|up_proj|w1|w3)')
    
    qkv_groups = defaultdict(list)
    gate_up_groups = defaultdict(list)
    
    for param_name in only_in_original:
        match = qkv_pattern.match(param_name)
        if match:
            base = match.group(1)
            proj_type = match.group(2)
            qkv_groups[base].append((proj_type, param_name))
        
        match = gate_up_pattern.match(param_name)
        if match:
            base = match.group(1)
            proj_type = match.group(2)
            gate_up_groups[base].append((proj_type, param_name))
    
    if qkv_groups:
        print(f"QKV-like patterns found:")
        for base, params in sorted(qkv_groups.items())[:3]:  # Show first 3 examples
            print(f"\n  Base: {base}")
            for proj_type, param_name in sorted(params):
                info = original_params_info[param_name]
                print(f"    {proj_type:15s} | {param_name:60s} | shape: {info['shape']}")
            
            # Look for corresponding stacked param in BnB
            possible_stacked = [p for p in only_in_bnb if base in p]
            if possible_stacked:
                print(f"  Possible stacked version in BnB:")
                for bnb_param in possible_stacked:
                    info = bnb_params_info[bnb_param]
                    print(f"    {bnb_param:60s} | shape: {info['shape']}")
    
    if gate_up_groups:
        print(f"\n\nGate/Up-like patterns found:")
        for base, params in sorted(gate_up_groups.items())[:3]:  # Show first 3 examples
            print(f"\n  Base: {base}")
            for proj_type, param_name in sorted(params):
                info = original_params_info[param_name]
                print(f"    {proj_type:15s} | {param_name:60s} | shape: {info['shape']}")
            
            # Look for corresponding stacked param in BnB
            possible_stacked = [p for p in only_in_bnb if base in p]
            if possible_stacked:
                print(f"  Possible stacked version in BnB:")
                for bnb_param in possible_stacked:
                    info = bnb_params_info[bnb_param]
                    print(f"    {bnb_param:60s} | shape: {info['shape']}")
    
    print(f"\n{'=' * 100}")
    print(f"Analysis complete! Both models have been unloaded from memory.")
    print(f"{'=' * 100}")


def detect_stacking_patterns(only_in_original, only_in_bnb, original_params_info, bnb_params_info):
    """
    Attempt to automatically detect stacking patterns by comparing shapes.
    
    Returns:
        dict: Mapping of original suffixes to (stacked suffix, index)
    """
    mappings = {}
    
    # Common patterns to check - expanded for different naming conventions
    qkv_variants = [
        (['q_proj', 'k_proj', 'v_proj'], ['qkv_proj', 'qkv']),
        (['wq', 'wk', 'wv'], ['wqkv', 'qkv_proj']),
        (['query', 'key', 'value'], ['qkv', 'query_key_value']),
    ]
    
    gate_up_variants = [
        (['gate_proj', 'up_proj'], ['gate_up_proj', 'gate_up']),
        (['w1', 'w3'], ['w13', 'gate_up_proj']),
    ]
    
    all_patterns = qkv_variants + gate_up_variants
    
    for components, stacked_names in all_patterns:
        # Check if this pattern exists by looking at actual parameter names
        for stacked_name in stacked_names:
            # Find if we have all components
            component_params = {}
            for component in components:
                matching = [p for p in only_in_original if p.endswith(f'.{component}.weight')]
                if matching:
                    component_params[component] = matching[0]
            
            # If we found all components, look for stacked version
            if len(component_params) == len(components):
                # Extract the base name from the first component
                first_param = list(component_params.values())[0]
                base = first_param.rsplit('.', 2)[0]  # Remove .component.weight
                
                # Look for stacked version
                stacked_param = f"{base}.{stacked_name}.weight"
                if stacked_param in only_in_bnb:
                    # Verify by shape - stacked should have sum of dimensions
                    component_shapes = [original_params_info[p]['shape'] for p in component_params.values()]
                    stacked_shape = bnb_params_info[stacked_param]['shape']
                    
                    # For linear layers, stacked dimension should match sum
                    if len(component_shapes[0]) >= 2 and stacked_shape != 'N/A':
                        expected_dim = sum(s[0] for s in component_shapes)
                        if stacked_shape[0] == expected_dim:
                            # Found a match!
                            for idx, component in enumerate(components):
                                mappings[f'.{component}'] = (f'.{stacked_name}', idx)
                            break
    
    return mappings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare weight mappings (sequential loading)")
    parser.add_argument(
        "--original-model",
        type=str,
        default="inclusionAI/LLaDA2.1-mini",
        help="Original model name or path"
    )
    parser.add_argument(
        "--bnb-model",
        type=str,
        default="inclusionAI/LLaDA2.1-mini-bnb",
        help="BnB quantized model name or path"
    )
    
    args = parser.parse_args()
    
    analyze_weight_mapping(
        original_model_name=args.original_model,
        bnb_model_name=args.bnb_model
    )
