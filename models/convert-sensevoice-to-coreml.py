import argparse
import torch
import coremltools as ct
from typing import Optional, Tuple

try:
    from funasr import AutoModel
except ImportError:
    print("Please install funasr: pip install funasr")
    exit(1)

def convert_sensevoice(model_name_or_path: str, quantize_type: str = "fp16", output_dir: str = "models", deployment_target: str = "macOS14"):
    """
    Convert SenseVoice model to CoreML.

    Usage Examples:
    ---------------
    # 1. Default (FP16 quantization):
    # python sensevoice-encode.py --model-path iic/SenseVoiceSmall

    # 2. FP16 explicit:
    # python sensevoice-encode.py --model-path iic/SenseVoiceSmall --quantize fp16

    # 3. Int8 (q8_0) quantization:
    # python sensevoice-encode.py --model-path iic/SenseVoiceSmall --quantize q8_0
    # OR
    # python sensevoice-encode.py --model-path iic/SenseVoiceSmall --quantize int8

    # 4. FP32 (no quantization):
    # python sensevoice-encode.py --model-path iic/SenseVoiceSmall --quantize float32
    
    Args:
        model_name_or_path: Path or HF ID of the model.
        quantize_type: Quantization type ('fp16', 'q8_0', 'int8', 'float32'). Default is 'fp16'.
        output_dir: Output directory for the model.
    """
    print(f"Loading model: {model_name_or_path}")
    model = AutoModel(model=model_name_or_path, device="cpu", disable_update=True)
    
    sense_voice_model = model.model
    sense_voice_model.eval()

    print("Model structure:")
    for name, module in sense_voice_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            print(f"{name}: {module}")
    if hasattr(sense_voice_model, 'embed'):
        print(f"model.embed: {sense_voice_model.embed}")

    # Wrapper to simplify encoder input
    class SenseVoiceEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
            # speech is [B, T, 560] (LFR features)
            res = self.model.encoder(speech, speech_lengths)
            
            # If encoder returns a tuple (output, lengths), take output.
            if isinstance(res, tuple):
                return res[0]
            return res

    encoder_wrapper = SenseVoiceEncoderWrapper(sense_voice_model)
    encoder_wrapper.eval()

    print("Tracing Encoder...")
    
    input_dim = 560
    dummy_input = torch.randn(1, 200, input_dim)
    dummy_lengths = torch.tensor([200], dtype=torch.int32)
    
    # Trace
    traced_encoder = torch.jit.trace(encoder_wrapper, (dummy_input, dummy_lengths))
    
    print("Converting to CoreML...")
    
    # Determine precision based on quantize_type
    precision = ct.precision.FLOAT32
    if quantize_type == "fp16":
        precision = ct.precision.FLOAT16
        print("Using precision: FLOAT16")
    else:
        print("Using precision: FLOAT32 (will quantize later if needed)")

    target = getattr(ct.target, deployment_target, None)
    if target is None:
        avail = sorted([n for n in dir(ct.target) if n.startswith(("macOS", "iOS"))])
        raise SystemExit(f"Unknown deployment target: {deployment_target}. Available: {', '.join(avail)}")

    mlmodel = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(name="speech", shape=(1, ct.RangeDim(1, 4000), input_dim)),
            ct.TensorType(name="speech_lengths", shape=(1,), dtype=int)
        ],
        outputs=[ct.TensorType(name="encoder_out")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision
    )

    quant_suffix = ""
    if quantize_type == "q8_0" or quantize_type == "int8":
        print("Quantizing weights (8-bit) using coremltools.optimize.coreml...")
        try:
            import coremltools.optimize.coreml as cto
            op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_tensor")
            config = cto.OptimizationConfig(global_config=op_config)
            mlmodel = cto.linear_quantize_weights(mlmodel, config=config)
            quant_suffix = "-q8_0"
        except ImportError:
            print("Error: coremltools.optimize.coreml not found. Please upgrade coremltools (>=7.0).")
            raise
        except Exception as e:
            print(f"Error during int8 quantization: {e}")
            raise
    elif quantize_type == "fp16":
        # Handled at conversion time via compute_precision
        pass
    elif quantize_type == "float32":
        print("Keeping weights as float32...")
    else:
        print(f"Unknown quantization type: {quantize_type}, defaulting to fp16 logic (if implicit)")

    save_name = f"SenseVoiceSmall{quant_suffix}-Encoder"
    save_path = f"{output_dir}/{save_name}.mlpackage"
    mlmodel.save(save_path)
    print(f"Saved CoreML model to {save_path}")

    # Compile the model to .mlmodelc
    print("Compiling model to .mlmodelc...")
    import subprocess
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)
    mlpackage_path = Path(save_path)
    
    try:
        # xcrun coremlcompiler compile <source> <output_dir>
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(mlpackage_path), str(output_path)], check=True)
        
        # CoreML compiler usually keeps the base name. 
        # If input is SenseVoiceSmall-q8_0-Encoder.mlpackage, output folder is typically SenseVoiceSmall-q8_0-Encoder.mlmodelc
        
        compiled_name = mlpackage_path.stem + ".mlmodelc"
        default_output = output_path / compiled_name
        
        if default_output.exists():
            print(f"Compiled model: {default_output}")
        else:
            found = list(output_path.glob("*.mlmodelc"))
            
            # Try to match the most recent one or matching name
            matched_found = [f for f in found if mlpackage_path.stem in f.name]
             
            if matched_found:
                print(f"Compiled model found at: {matched_found[0]}")
            elif found:
                print(f"Compiled model found at (might need renaming): {found[-1]}")
            else:
                 print(f"Compiled model not found at expected path: {default_output}")
            
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("You may need to manually compile using: xcrun coremlcompiler compile ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="iic/SenseVoiceSmall", help="Model path or ModelScope ID")
    # quantize parameter: default is fp16. Acceptable: fp16, q8_0, float32
    parser.add_argument("--quantize", type=str, default="fp16", help="Quantization type (default: fp16). Options: fp16, q8_0, float32")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--deployment-target", type=str, default="macOS14", help="Minimum deployment target (e.g. macOS14, macOS15)")
    
    args = parser.parse_args()
    
    convert_sensevoice(args.model_path, args.quantize, args.output_dir, args.deployment_target)
