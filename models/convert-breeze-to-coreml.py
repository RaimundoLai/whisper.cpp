#!/usr/bin/env python3
# Convert Breeze-ASR-25 (MediaTek) HuggingFace model to CoreML format for whisper.cpp
#
# Usage: python convert-breeze-to-coreml.py [--output-dir DIR] [--quantize] [--optimize-ane]
#
# This script converts the Breeze-ASR-25 model from HuggingFace to CoreML format
# compatible with whisper.cpp's CoreML encoder path expectations.
#
# Requirements:
#   pip install torch coremltools transformers ane-transformers
#
# The output .mlmodelc file should be placed alongside your GGML model file.
# For example, if your GGML model is:
#   /path/to/ggml-breeze-asr.bin
# Then the CoreML encoder should be:
#   /path/to/ggml-breeze-asr-encoder.mlmodelc

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Try to import coremltools
try:
    import coremltools as ct
    from coremltools.models.neural_network.quantization_utils import quantize_weights
except ImportError:
    print("Error: coremltools not found. Install with: pip install coremltools")
    sys.exit(1)

# Try to import transformers
try:
    from transformers import WhisperModel, WhisperConfig, WhisperProcessor
except ImportError:
    print("Error: transformers not found. Install with: pip install transformers")
    sys.exit(1)

# Try to import ane_transformers for ANE-optimized LayerNorm
try:
    from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
    HAS_ANE_TRANSFORMERS = True
except ImportError:
    HAS_ANE_TRANSFORMERS = False
    print("Warning: ane_transformers not found. ANE optimization will be disabled.")
    print("Install with: pip install ane-transformers")


# ============================================================================
# HuggingFace to OpenAI Whisper key mapping
# ============================================================================

WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}


def rename_keys(s_dict):
    """Rename HuggingFace keys to OpenAI Whisper format."""
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)
        if new_key != key:
            s_dict[new_key] = s_dict.pop(key)
    return s_dict


# ============================================================================
# ANE-optimized modules (from convert-whisper-to-coreml.py)
# ============================================================================

def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights"""
    for k in state_dict:
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])

        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict


if HAS_ANE_TRANSFORMERS:
    class LayerNormANE(LayerNormANEBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._register_load_state_dict_pre_hook(
                correct_for_bias_scale_order_inversion)


class MultiHeadAttentionANE(nn.Module):
    """ANE-optimized Multi-Head Attention using Conv2d instead of Linear."""
    
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention_ane(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention_ane(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        _, dim, _, seqlen = q.size()
        dim_per_head = dim // self.n_head
        scale = float(dim_per_head)**-0.5

        q = q * scale
        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(dim_per_head, dim=3)
        mh_v = v.split(dim_per_head, dim=1)

        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]

        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask[:, :seqlen, :, :seqlen]

        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]
        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_weights, mh_v)]
        attn = torch.cat(attn, dim=1)

        return attn, torch.cat(mh_qk, dim=1).float().detach()


class ResidualAttentionBlockANE(nn.Module):
    """ANE-optimized Residual Attention Block."""
    
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttentionANE(n_state, n_head)
        if HAS_ANE_TRANSFORMERS:
            self.attn_ln = LayerNormANE(n_state)
        else:
            self.attn_ln = nn.LayerNorm(n_state)
        
        self.cross_attn = MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
        if HAS_ANE_TRANSFORMERS:
            self.cross_attn_ln = LayerNormANE(n_state) if cross_attention else None
        else:
            self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        if HAS_ANE_TRANSFORMERS:
            self.mlp_ln = LayerNormANE(n_state)
        else:
            self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderANE(nn.Module):
    """ANE-optimized Audio Encoder for Whisper."""
    
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)]
        )
        if HAS_ANE_TRANSFORMERS:
            self.ln_post = LayerNormANE(n_state)
        else:
            self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        assert x.shape[1:] == self.positional_embedding.shape[::-1], "incorrect audio shape"

        # Add positional embedding and add dummy dim for ANE
        x = (x + self.positional_embedding.transpose(0, 1)).to(x.dtype).unsqueeze(2)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        x = x.squeeze(2).transpose(1, 2)

        return x


# ============================================================================
# Simple Encoder Wrapper (non-ANE, for basic conversion)
# ============================================================================

class SimpleEncoderWrapper(nn.Module):
    """Simple wrapper to return tensor instead of dict from HuggingFace encoder."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        outputs = self.encoder(input_features)
        return outputs.last_hidden_state


# ============================================================================
# Conversion functions
# ============================================================================

def convert_hf_to_openai_state_dict(hf_state_dict: dict) -> dict:
    """Convert HuggingFace Whisper state dict to OpenAI format."""
    converted = {}
    
    for key, value in hf_state_dict.items():
        new_key = key
        
        # Apply renaming
        for k, v in WHISPER_MAPPING.items():
            if k in new_key:
                new_key = new_key.replace(k, v)
        
        converted[new_key] = value
    
    return converted


def load_breeze_model(model_id: str = "MediaTek-Research/Breeze-ASR-25"):
    """Load Breeze-ASR-25 model from HuggingFace."""
    print(f"Loading model from HuggingFace: {model_id}")
    
    model = WhisperModel.from_pretrained(model_id)
    config = WhisperConfig.from_pretrained(model_id)
    
    print(f"  d_model: {config.d_model}")
    print(f"  encoder_layers: {config.encoder_layers}")
    print(f"  decoder_layers: {config.decoder_layers}")
    print(f"  encoder_attention_heads: {config.encoder_attention_heads}")
    print(f"  num_mel_bins: {config.num_mel_bins}")
    print(f"  max_source_positions: {config.max_source_positions}")
    
    return model, config


def build_ane_encoder(config: WhisperConfig, hf_encoder_state_dict: dict):
    """Build ANE-optimized encoder and load weights from HuggingFace format."""
    
    # Create ANE encoder with matching dimensions
    encoder = AudioEncoderANE(
        n_mels=config.num_mel_bins,
        n_ctx=config.max_source_positions,
        n_state=config.d_model,
        n_head=config.encoder_attention_heads,
        n_layer=config.encoder_layers,
    )
    
    # Register hook to convert Linear weights to Conv2d format
    encoder._register_load_state_dict_pre_hook(linear_to_conv2d_map)
    
    # Convert HuggingFace keys to OpenAI format
    converted_state_dict = convert_hf_to_openai_state_dict(hf_encoder_state_dict)
    
    # Remove 'encoder.' prefix
    encoder_state_dict = {}
    for key, value in converted_state_dict.items():
        if key.startswith("encoder."):
            new_key = key[8:]  # Remove 'encoder.' prefix
            encoder_state_dict[new_key] = value
    
    # Load weights
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")
    
    return encoder


def convert_encoder_to_coreml(encoder: nn.Module, n_mels: int, quantize: bool = False, deployment_target: str = "macOS14"):
    """Convert encoder to CoreML format."""
    encoder.eval()
    
    # Standard Whisper input: 30s audio = 3000 mel frames
    input_shape = (1, n_mels, 3000)
    input_data = torch.randn(input_shape)
    
    print(f"  Tracing encoder with input shape: {input_shape}")
    traced_model = torch.jit.trace(encoder, input_data)
    
    print("  Converting to CoreML...")
    target = getattr(ct.target, deployment_target, None)
    if target is None:
        avail = sorted([n for n in dir(ct.target) if n.startswith(("macOS", "iOS"))])
        raise SystemExit(f"Unknown deployment target: {deployment_target}. Available: {', '.join(avail)}")

    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
    )
    
    if quantize:
        print("  Quantizing to F16...")
        coreml_model = quantize_weights(coreml_model, nbits=16)
    
    return coreml_model


def compile_mlpackage(mlpackage_path: Path, output_dir: Path, output_name: str) -> Path:
    """Compile .mlpackage to .mlmodelc using coremlcompiler."""
    print(f"\n  Compiling to .mlmodelc...")
    
    try:
        # Run without capture_output to see logs, check=True to raise error on failure
        subprocess.run(
            ['xcrun', 'coremlcompiler', 'compile', str(mlpackage_path), str(output_dir)],
            check=True
        )
        
        # Find the compiled output (usually same name with .mlmodelc extension)
        default_output = output_dir / f"{mlpackage_path.stem}.mlmodelc"
        target_output = output_dir / f"{output_name}-encoder.mlmodelc"
        
        if default_output.exists():
            if target_output.exists() and default_output != target_output:
                shutil.rmtree(target_output)
            
            if default_output != target_output:
                shutil.move(str(default_output), str(target_output))
                
            print(f"  Compiled model: {target_output}")
            return target_output
        else:
            print(f"  Warning: Compiled model not found at expected path: {default_output}")
            return None
            
    except Exception as e:
        print(f"  Compilation failed: {e}")
        return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Breeze-ASR-25 model to CoreML format for whisper.cpp"
    )
    parser.add_argument(
        "--model-id", type=str, default="MediaTek-Research/Breeze-ASR-25",
        help="HuggingFace model ID (default: MediaTek-Research/Breeze-ASR-25)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Output directory for converted models (default: current directory)"
    )
    parser.add_argument(
        "--output-name", type=str, default="ggml-breeze-asr",
        help="Base name for output files (default: ggml-breeze-asr)"
    )
    parser.add_argument(
        "--quantize", action="store_true", default=False,
        help="Quantize weights to F16"
    )
    parser.add_argument(
        "--optimize-ane", action="store_true", default=True,
        help="Optimize for ANE execution (default: True)"
    )
    parser.add_argument(
        "--no-optimize-ane", action="store_true", default=False,
        help="Disable ANE optimization (use simple wrapper)"
    )
    parser.add_argument(
        "--skip-compile", action="store_true", default=False,
        help="Skip compilation to .mlmodelc (only save .mlpackage)"
    )
    parser.add_argument(
        "--deployment-target", type=str, default="macOS14",
        help="Minimum deployment target (e.g. macOS14, macOS15)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_ane = args.optimize_ane and not args.no_optimize_ane
    
    if use_ane and not HAS_ANE_TRANSFORMERS:
        print("Warning: ane_transformers not available, falling back to simple wrapper")
        use_ane = False
    
    # Load model
    model, config = load_breeze_model(args.model_id)
    model.eval()
    
    # Build encoder
    print("\nBuilding encoder...")
    if use_ane:
        print("  Using ANE-optimized architecture")
        encoder_state_dict = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
        encoder = build_ane_encoder(config, encoder_state_dict)
    else:
        print("  Using simple wrapper (no ANE optimization)")
        encoder = SimpleEncoderWrapper(model.encoder)
    
    encoder.eval()
    
    # Convert to CoreML
    print("\nConverting encoder to CoreML...")
    coreml_encoder = convert_encoder_to_coreml(
        encoder, 
        n_mels=config.num_mel_bins,
        quantize=args.quantize,
        deployment_target=args.deployment_target
    )
    
    # Save .mlpackage
    mlpackage_path = output_dir / f"{args.output_name}-encoder.mlpackage"
    print(f"\nSaving mlpackage to: {mlpackage_path}")
    coreml_encoder.save(str(mlpackage_path))
    
    # Compile to .mlmodelc
    if not args.skip_compile:
        compiled_path = compile_mlpackage(mlpackage_path, output_dir, args.output_name)
    else:
        compiled_path = None
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {mlpackage_path}")
    if compiled_path:
        print(f"  - {compiled_path}")
    
    print(f"\nTo use with whisper.cpp:")
    print(f"  1. Your GGML model should be named: {args.output_name}.bin")
    print(f"  2. Place the .mlmodelc file in the same directory as your .bin file")
    print(f"  3. whisper.cpp will automatically load: {args.output_name}-encoder.mlmodelc")
    
    if compiled_path:
        print(f"\nExample:")
        print(f"  cp -r {compiled_path} /path/to/models/")
        print(f"  # Ensure /path/to/models/{args.output_name}.bin exists")
