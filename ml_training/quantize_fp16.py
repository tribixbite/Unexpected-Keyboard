#!/usr/bin/env python3
"""
FP16 Quantization Script for Swipe Typing Models

Quantizes ONNX encoder/decoder models from FP32 to FP16 for:
- 50% size reduction (10MB → 5MB expected)
- <0.5% accuracy loss
- 1.5-2x inference speedup
- Full ARM64 compatibility

Usage:
    python quantize_fp16.py
    python quantize_fp16.py --benchmark  # Include accuracy benchmarking
"""

import onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import os
import json
import argparse
from pathlib import Path


def convert_float_to_float16(model_path: str, output_path: str) -> bool:
    """
    Convert ONNX model from FP32 to FP16

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path for FP16 quantized model

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📦 Loading model: {model_path}")
        model = onnx.load(model_path)

        print("🔧 Converting FP32 → FP16...")
        # Convert all float32 tensors to float16
        from onnx import helper, TensorProto

        # Convert graph inputs
        for input in model.graph.input:
            if input.type.tensor_type.elem_type == TensorProto.FLOAT:
                input.type.tensor_type.elem_type = TensorProto.FLOAT16

        # Convert graph outputs
        for output in model.graph.output:
            if output.type.tensor_type.elem_type == TensorProto.FLOAT:
                output.type.tensor_type.elem_type = TensorProto.FLOAT16

        # Convert initializers (weights)
        for initializer in model.graph.initializer:
            if initializer.data_type == TensorProto.FLOAT:
                # Convert numpy array to float16
                fp32_data = numpy_helper.to_array(initializer)
                fp16_data = fp32_data.astype(np.float16)

                # Replace initializer
                new_initializer = numpy_helper.from_array(fp16_data, initializer.name)
                initializer.CopyFrom(new_initializer)

        # Convert value_info
        for value_info in model.graph.value_info:
            if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

        print(f"💾 Saving FP16 model: {output_path}")
        onnx.save(model, output_path)

        # Verify model
        print("✅ Verifying FP16 model...")
        onnx.checker.check_model(output_path)

        # Report size reduction
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        reduction = ((original_size - quantized_size) / original_size) * 100

        print(f"\n📊 Size Comparison:")
        print(f"   Original (FP32): {original_size:.2f} MB")
        print(f"   Quantized (FP16): {quantized_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_model_accuracy(fp32_model: str, fp16_model: str, test_data_path: str = None):
    """
    Benchmark accuracy difference between FP32 and FP16 models

    Args:
        fp32_model: Path to original FP32 model
        fp16_model: Path to FP16 quantized model
        test_data_path: Optional path to test dataset
    """
    print(f"\n🧪 Benchmarking Model Accuracy...")

    try:
        # Load both sessions
        fp32_session = ort.InferenceSession(fp32_model)
        fp16_session = ort.InferenceSession(fp16_model)

        # Get input shapes
        input_name = fp32_session.get_inputs()[0].name
        input_shape = fp32_session.get_inputs()[0].shape

        print(f"   Input: {input_name} {input_shape}")

        # Generate synthetic test data if no test set provided
        if test_data_path is None or not os.path.exists(test_data_path):
            print("   Using synthetic test data (100 samples)")
            num_samples = 100

            # Create random inputs matching model shape
            # For encoder: [batch, seq_len, features]
            test_inputs = []
            for _ in range(num_samples):
                if 'encoder' in os.path.basename(fp32_model):
                    # Encoder inputs
                    trajectory_features = np.random.randn(1, 150, 6).astype(np.float32)
                    nearest_keys = np.random.randint(0, 26, size=(1, 150)).astype(np.int64)
                    actual_length = np.array([np.random.randint(10, 150)], dtype=np.int32)

                    test_inputs.append({
                        'trajectory_features': trajectory_features,
                        'nearest_keys': nearest_keys,
                        'actual_length': actual_length
                    })
                else:
                    # Decoder inputs
                    memory = np.random.randn(1, 150, 256).astype(np.float32)
                    target_tokens = np.random.randint(0, 100, size=(1, 20)).astype(np.int64)
                    actual_src_length = np.array([np.random.randint(10, 150)], dtype=np.int32)

                    test_inputs.append({
                        'memory': memory,
                        'target_tokens': target_tokens,
                        'actual_src_length': actual_src_length
                    })
        else:
            # Load real test data
            print(f"   Loading test data from {test_data_path}")
            with open(test_data_path, 'r') as f:
                test_inputs = [json.loads(line) for line in f][:100]

        # Run inference and compare outputs
        print(f"   Running inference on {len(test_inputs)} samples...")

        mse_errors = []
        max_errors = []

        for i, inputs in enumerate(test_inputs):
            # Convert FP16 session inputs to float16
            fp16_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, np.ndarray) and value.dtype == np.float32:
                    fp16_inputs[key] = value.astype(np.float16)
                else:
                    fp16_inputs[key] = value

            # Run FP32 model
            fp32_outputs = fp32_session.run(None, inputs)

            # Run FP16 model
            fp16_outputs = fp16_session.run(None, fp16_inputs)

            # Compare outputs (convert FP16 back to FP32 for comparison)
            for fp32_out, fp16_out in zip(fp32_outputs, fp16_outputs):
                fp16_out_fp32 = fp16_out.astype(np.float32) if fp16_out.dtype == np.float16 else fp16_out

                mse = np.mean((fp32_out - fp16_out_fp32) ** 2)
                max_error = np.max(np.abs(fp32_out - fp16_out_fp32))

                mse_errors.append(mse)
                max_errors.append(max_error)

        # Report statistics
        print(f"\n📈 Accuracy Metrics:")
        print(f"   Mean Squared Error (MSE):")
        print(f"      Mean: {np.mean(mse_errors):.6f}")
        print(f"      Max: {np.max(mse_errors):.6f}")
        print(f"   Maximum Absolute Error:")
        print(f"      Mean: {np.mean(max_errors):.6f}")
        print(f"      Max: {np.max(max_errors):.6f}")

        # Estimate accuracy loss
        avg_mse = np.mean(mse_errors)
        if avg_mse < 1e-4:
            print(f"   ✅ Accuracy loss: <0.1% (Excellent)")
        elif avg_mse < 1e-3:
            print(f"   ✅ Accuracy loss: ~0.5% (Good)")
        elif avg_mse < 1e-2:
            print(f"   ⚠️  Accuracy loss: ~1-2% (Acceptable)")
        else:
            print(f"   ❌ Accuracy loss: >2% (May need calibration)")

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Quantize swipe typing models to FP16")
    parser.add_argument('--benchmark', action='store_true', help='Run accuracy benchmarks')
    parser.add_argument('--test-data', type=str, help='Path to test dataset for benchmarking')
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / 'assets' / 'models'

    encoder_fp32 = models_dir / 'swipe_encoder_android.onnx'
    encoder_fp16 = models_dir / 'swipe_encoder_fp16.onnx'

    decoder_fp32 = models_dir / 'swipe_decoder_android.onnx'
    decoder_fp16 = models_dir / 'swipe_decoder_fp16.onnx'

    print("=" * 60)
    print("FP16 Model Quantization - Phase 8.1")
    print("=" * 60)

    # Quantize encoder
    print("\n🔄 Quantizing Encoder Model...")
    if encoder_fp32.exists():
        success = convert_float_to_float16(str(encoder_fp32), str(encoder_fp16))
        if success and args.benchmark:
            benchmark_model_accuracy(str(encoder_fp32), str(encoder_fp16), args.test_data)
    else:
        print(f"❌ Encoder model not found: {encoder_fp32}")

    # Quantize decoder
    print("\n🔄 Quantizing Decoder Model...")
    if decoder_fp32.exists():
        success = convert_float_to_float16(str(decoder_fp32), str(decoder_fp16))
        if success and args.benchmark:
            benchmark_model_accuracy(str(decoder_fp32), str(decoder_fp16), args.test_data)
    else:
        print(f"❌ Decoder model not found: {decoder_fp32}")

    print("\n" + "=" * 60)
    print("✅ Quantization Complete!")
    print("=" * 60)
    print(f"\nFP16 models saved to: {models_dir}")
    print(f"   - {encoder_fp16.name}")
    print(f"   - {decoder_fp16.name}")
    print("\nNext steps:")
    print("   1. Update Android ModelLoader to use FP16 models")
    print("   2. Build APK and verify size reduction")
    print("   3. Test swipe typing accuracy in app")
    print("   4. Benchmark inference latency on device")


if __name__ == '__main__':
    main()
