#!/usr/bin/env python3
"""
Script to convert a ResNet50 TIMM model to TFLite format.
"""
import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import urllib.request

import numpy as np
import torch
import timm
import tensorflow as tf
import typer
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversionConfig(BaseModel):
    """Configuration for model conversion."""
    weights_path: str = Field(..., description="Path or URL to the PyTorch model weights")
    labels_path: Optional[str] = Field(None, description="Path or URL to the labels file")
    output_dir: Path = Field(Path("./converted_models"), description="Directory to save converted models")
    input_size: int = Field(300, description="Input image size")
    num_classes: Optional[int] = Field(None, description="Number of output classes")
    model_name: str = Field("resnet50", description="Model name for the output file")
    batch_size: int = Field(1, description="Batch size for conversion")

def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    return path.startswith("http://") or path.startswith("https://")

def download_file(path: str, output_path: Path) -> Path:
    """Download a file from a URL or return local path."""
    if not is_url(path):
        return Path(path)
        
    logger.info(f"Downloading file from {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    urllib.request.urlretrieve(path, output_path)
    logger.info(f"File downloaded to {output_path}")
    return output_path

def get_num_classes(labels_path: str) -> int:
    """Get the number of classes from the labels file."""
    local_path = Path("./temp_labels.json")
    
    file_path = download_file(labels_path, local_path)
    
    with open(file_path, "r") as f:
        category_map = json.load(f)
        
    num_classes = len(category_map)
    logger.info(f"Found {num_classes} classes in the labels file")
    
    return num_classes

def load_timm_model(config: ConversionConfig) -> torch.nn.Module:
    """Load a TIMM model with the specified weights."""
    # Get number of classes from labels if not provided
    if config.num_classes is None and config.labels_path:
        config.num_classes = get_num_classes(config.labels_path)
    
    if config.num_classes is None:
        raise ValueError("Number of classes must be provided either directly or via a labels file")
    
    logger.info(f"Loading TIMM model with {config.num_classes} classes")
    model = timm.create_model(
        "resnet50",
        weights=None,
        num_classes=config.num_classes,
    )
    
    # Download weights if needed
    local_weights_path = Path("./temp_weights.pth")
    weights_path = download_file(config.weights_path, local_weights_path)
    
    logger.info(f"Loading weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def export_to_onnx(model: torch.nn.Module, config: ConversionConfig) -> Path:
    """Export PyTorch model to ONNX format."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = config.output_dir / f"{config.model_name}.onnx"
    
    # Create dummy input for tracing
    dummy_input = torch.randn(
        config.batch_size, 3, config.input_size, config.input_size, 
        requires_grad=True
    )
    
    logger.info(f"Exporting model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    return onnx_path

def convert_onnx_to_tf(onnx_path: Path, config: ConversionConfig) -> Path:
    """
    Convert ONNX model to TensorFlow SavedModel format using onnx2tf.
    
    Note: onnx-tf has been deprecated and migrated to tf2onnx for TensorFlow→ONNX conversion.
    For ONNX→TensorFlow conversion, onnx2tf is the modern replacement.
    """
    try:
        import onnx2tf
        logger.info(f"Converting ONNX model to TensorFlow using onnx2tf: {onnx_path}")
        
        tf_model_path = config.output_dir / "tf_model"
        
        # Use onnx2tf for conversion (modern replacement for deprecated onnx-tf backend)
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_model_path),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        # onnx2tf creates a saved_model folder inside the output directory
        actual_tf_model_path = tf_model_path / "saved_model"
        if not actual_tf_model_path.exists():
            # Fallback: sometimes the model is saved directly in the output folder
            actual_tf_model_path = tf_model_path
            
        logger.info(f"TensorFlow model saved to {actual_tf_model_path}")
        return actual_tf_model_path
        
    except ImportError:
        logger.warning("onnx2tf not available, falling back to direct TFLite conversion")
        # If onnx2tf is not available, we'll skip TensorFlow SavedModel and convert directly
        return None

def convert_tf_to_tflite(tf_model_path: Path, config: ConversionConfig, onnx_path: Path = None) -> Path:
    """Convert TensorFlow model to TFLite format, with fallback to direct ONNX conversion."""
    tflite_path = config.output_dir / f"{config.model_name}.tflite"
    
    if tf_model_path and tf_model_path.exists():
        # Standard TensorFlow SavedModel to TFLite conversion
        logger.info(f"Loading TensorFlow model from {tf_model_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        
        # Set converter options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        logger.info("Converting TensorFlow model to TFLite")
        tflite_model = converter.convert()
        
    elif onnx_path and onnx_path.exists():
        # Direct ONNX to TFLite conversion using onnx2tf
        try:
            import onnx2tf
            logger.info("Converting ONNX directly to TFLite using onnx2tf")
            
            # Convert directly to TFLite
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(config.output_dir / "temp_tf"),
                output_tfjs=False,
                output_keras_v3=False,
                output_saved_model=False,
                output_tflite=True,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True
            )
            
            # Find the generated TFLite file
            temp_tflite_files = list((config.output_dir / "temp_tf").glob("*.tflite"))
            if temp_tflite_files:
                import shutil
                shutil.move(str(temp_tflite_files[0]), str(tflite_path))
                # Clean up temp directory
                shutil.rmtree(str(config.output_dir / "temp_tf"))
                logger.info(f"TFLite model saved to {tflite_path}")
                return tflite_path
            else:
                raise RuntimeError("No TFLite file generated by onnx2tf")
                
        except ImportError:
            raise RuntimeError("Cannot convert to TFLite: no TensorFlow SavedModel available and onnx2tf not installed")
    else:
        raise RuntimeError("Cannot convert to TFLite: no valid input model available")
    
    # Save the TFLite model (for standard TF SavedModel conversion)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved to {tflite_path}")
    return tflite_path

def convert_tf_to_onnx_optional(tf_model_path: Path, config: ConversionConfig) -> Optional[Path]:
    """
    Optional function to convert TensorFlow model back to ONNX using tf2onnx.
    This is useful for testing or validation purposes.
    
    Note: tf2onnx is the successor to onnx-tf frontend functionality.
    """
    try:
        import tf2onnx
        import onnx
        
        onnx_validation_path = config.output_dir / f"{config.model_name}_from_tf.onnx"
        
        logger.info(f"Converting TensorFlow model back to ONNX using tf2onnx: {tf_model_path}")
        
        # Convert TensorFlow SavedModel to ONNX
        model_proto, _ = tf2onnx.convert.from_saved_model(
            str(tf_model_path),
            opset=15,
            output_path=str(onnx_validation_path)
        )
        
        logger.info(f"Validation ONNX model saved to {onnx_validation_path}")
        return onnx_validation_path
        
    except ImportError:
        logger.info("tf2onnx not available, skipping TensorFlow to ONNX validation")
        return None
    except Exception as e:
        logger.warning(f"Could not convert TensorFlow back to ONNX: {e}")
        return None

def verify_tflite_model(tflite_path: Path, config: ConversionConfig) -> None:
    """Verify the TFLite model by running inference on a dummy input."""
    logger.info(f"Verifying TFLite model: {tflite_path}")
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_data = np.random.random(input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    logger.info(f"Output shape: {output_data.shape}")
    logger.info("TFLite model verification successful")

def main(
    weights_path: str = typer.Argument(..., help="Path or URL to the PyTorch model weights"),
    output_dir: Path = typer.Option(Path("./converted_models"), help="Directory to save converted models"),
    labels_path: Optional[str] = typer.Option(None, help="Path or URL to the labels file"),
    num_classes: Optional[int] = typer.Option(None, help="Number of output classes"),
    input_size: int = typer.Option(300, help="Input image size"),
    model_name: str = typer.Option("resnet50", help="Model name for the output file"),
    verify: bool = typer.Option(True, help="Verify the converted TFLite model")
) -> None:
    """Convert a ResNet50 TIMM model to TFLite format."""
    config = ConversionConfig(
        weights_path=weights_path,
        labels_path=labels_path,
        output_dir=output_dir,
        input_size=input_size,
        num_classes=num_classes,
        model_name=model_name
    )
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch model
    model = load_timm_model(config)
    
    # Convert to ONNX
    onnx_path = export_to_onnx(model, config)
    
    # Convert ONNX to TensorFlow
    tf_model_path = convert_onnx_to_tf(onnx_path, config)
    
    # Convert TensorFlow to TFLite (with fallback to direct ONNX conversion)
    tflite_path = convert_tf_to_tflite(tf_model_path, config, onnx_path)
    
    # Optional: Convert TensorFlow back to ONNX for validation (using tf2onnx)
    if tf_model_path and tf_model_path.exists():
        convert_tf_to_onnx_optional(tf_model_path, config)
    
    # Verify TFLite model
    if verify:
        verify_tflite_model(tflite_path, config)
    
    logger.info(f"Conversion complete. TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    typer.run(main)
