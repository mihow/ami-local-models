#!/usr/bin/env python3
"""
Debug script to test onnx2tf conversion step by step.
"""
import sys
from pathlib import Path

print("Python version:", sys.version)
print("Current working directory:", Path.cwd())

print("\n--- Testing imports ---")
try:
    import onnx2tf
    print("✓ onnx2tf import successful")
    print("onnx2tf version:", getattr(onnx2tf, '__version__', 'unknown'))
except ImportError as e:
    print("✗ onnx2tf import failed:", e)
    sys.exit(1)

try:
    import tensorflow as tf
    print("✓ tensorflow import successful")
    print("tensorflow version:", tf.__version__)
except ImportError as e:
    print("✗ tensorflow import failed:", e)
    sys.exit(1)

print("\n--- Testing ONNX file ---")
onnx_path = Path("converted_models/resnet50.onnx")
if onnx_path.exists():
    print(f"✓ ONNX file exists: {onnx_path}")
    print(f"File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
else:
    print(f"✗ ONNX file not found: {onnx_path}")
    sys.exit(1)

print("\n--- Testing onnx2tf conversion ---")
try:
    output_dir = Path("converted_models/debug_tf")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Converting {onnx_path} to {output_dir}")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        output_tflite=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False  # Enable verbose to see what's happening
    )
    print("✓ onnx2tf conversion completed")
    
    # Check what files were created
    created_files = list(output_dir.rglob("*"))
    print(f"Created {len(created_files)} files:")
    for file in created_files:
        if file.is_file():
            print(f"  {file}")
            
except Exception as e:
    print(f"✗ onnx2tf conversion failed: {e}")
    import traceback
    traceback.print_exc()
