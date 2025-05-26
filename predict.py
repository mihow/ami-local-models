#!/usr/bin/env python3
"""
Simple script to load a TFLite model and perform inference on images from URLs.
"""
import json
import logging
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import typer
from PIL import Image
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceConfig(BaseModel):
    """Configuration for TFLite inference."""
    model_path: Path = Field(..., description="Path to TFLite model file")
    category_map_path: Optional[Path] = Field(None, description="Path to category map JSON file")
    input_size: int = Field(300, description="Input image size")
    top_k: int = Field(5, description="Number of top predictions to return")

def load_category_map(path: Path) -> Dict[str, int]:
    """Load category map from JSON file."""
    logger.info(f"Loading category map from {path}")
    with open(path, 'r') as f:
        category_map = json.load(f)
    
    num_classes = len(category_map)
    logger.info(f"Loaded category map with {num_classes} classes")
    return category_map

def get_inverse_category_map(category_map: Dict[str, int]) -> Dict[int, str]:
    """Create an inverse mapping from class indices to names."""
    inverse_map = {v: k for k, v in category_map.items()}
    return inverse_map

def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    logger.info(f"Downloading image from {url}")
    try:
        with urllib.request.urlopen(url) as response:
            img = Image.open(BytesIO(response.read()))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        logger.info(f"Downloaded image: {img.size} pixels, mode: {img.mode}")
        return img
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise

def preprocess_image(img: Image.Image, input_size: int) -> np.ndarray:
    """
    Preprocess image for model input.
    Applies the same preprocessing as the training pipeline.
    """
    logger.info(f"Preprocessing image: {img.size} -> {input_size}x{input_size}")
    
    # Pad to square (same as training preprocessing)
    width, height = img.size
    padding = [0, 0, 0, 0]  # left, top, right, bottom
    
    if height < width:
        padding[3] = width - height  # pad bottom
    elif height > width:
        padding[2] = height - width  # pad right
    
    # Apply padding if needed
    if any(padding):
        from PIL import ImageOps
        img = ImageOps.expand(img, tuple(padding), fill=0)
        logger.info(f"Applied padding: {padding}")
    
    # Resize to target size
    img = img.resize((input_size, input_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and ensure float32 throughout
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize using ImageNet statistics (ensure float32 precision)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    
    # Perform normalization and ensure result is float32
    img_array = (img_array - mean) / std
    img_array = img_array.astype(np.float32)  # Explicit cast to ensure float32
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

def load_tflite_model(model_path: Path) -> tf.lite.Interpreter:
    """Load TFLite model and allocate tensors."""
    logger.info(f"Loading TFLite model from {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Model input shape: {input_details[0]['shape']}")
    logger.info(f"Model output shape: {output_details[0]['shape']}")
    logger.info(f"Input dtype: {input_details[0]['dtype']}")
    logger.info(f"Output dtype: {output_details[0]['dtype']}")
    
    return interpreter

def run_inference(
    interpreter: tf.lite.Interpreter,
    input_data: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run inference on the TFLite model."""
    logger.info("Running TFLite inference")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Validate input data type and shape
    expected_dtype = input_details[0]['dtype']
    expected_shape = input_details[0]['shape']
    
    logger.info(f"Input data dtype: {input_data.dtype}, expected: {expected_dtype}")
    logger.info(f"Input data shape: {input_data.shape}, expected: {expected_shape}")
    
    # Ensure input data matches expected type
    if input_data.dtype != expected_dtype:
        logger.info(f"Converting input from {input_data.dtype} to {expected_dtype}")
        input_data = input_data.astype(expected_dtype)
    
    # Validate shape compatibility
    if list(input_data.shape) != list(expected_shape):
        logger.warning(f"Input shape mismatch: got {input_data.shape}, expected {expected_shape}")
    
    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference and measure time
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        logger.info(f"Output shape: {output_data.shape}, dtype: {output_data.dtype}")
        
        return output_data, inference_time
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        logger.error(f"Input details: {input_details[0]}")
        logger.error(f"Actual input - shape: {input_data.shape}, dtype: {input_data.dtype}")
        raise

def get_top_predictions(
    output_data: np.ndarray,
    category_map: Optional[Dict[str, int]],
    top_k: int = 5
) -> List[Tuple[str, float, int]]:
    """
    Get top-k predictions from model output.
    
    Returns:
        List of tuples: (class_name, probability, class_index)
    """
    # Apply softmax to get probabilities
    logits = output_data[0]  # Remove batch dimension
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get top k indices
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    # Convert to class names if category map is available
    results = []
    if category_map:
        inverse_map = get_inverse_category_map(category_map)
        for idx, prob in zip(top_k_indices, top_k_probs):
            class_name = inverse_map.get(idx, f"Unknown_Class_{idx}")
            results.append((class_name, float(prob), int(idx)))
    else:
        for idx, prob in zip(top_k_indices, top_k_probs):
            results.append((f"Class_{idx}", float(prob), int(idx)))
    
    return results

def print_predictions(
    predictions: List[Tuple[str, float, int]],
    inference_time: float
) -> None:
    """Print prediction results in a formatted way."""
    print("\n" + "="*60)
    print("🔍 INFERENCE RESULTS")
    print("="*60)
    
    print(f"⏱️  Inference time: {inference_time:.4f} seconds\n")
    
    print("🏆 Top Predictions:")
    for i, (class_name, prob, class_idx) in enumerate(predictions, 1):
        confidence_bar = "█" * int(prob * 20)  # Visual confidence bar
        print(f"  {i}. {class_name}")
        print(f"     Confidence: {prob:.4f} ({prob*100:.2f}%) {confidence_bar}")
        print(f"     Class Index: {class_idx}")
        print()
    
    # Highlight top prediction
    if predictions:
        top_class, top_prob, top_idx = predictions[0]
        print(f"🎯 **Top Prediction**: {top_class} ({top_prob*100:.1f}% confidence)")
    
    print("="*60)

def main(
    model_path: Path = typer.Argument(..., help="Path to TFLite model file"),
    image_url: str = typer.Argument(..., help="URL of the image to classify"),
    category_map_path: Optional[Path] = typer.Option(None, help="Path to category map JSON file"),
    input_size: int = typer.Option(300, help="Input image size"),
    top_k: int = typer.Option(5, help="Number of top predictions to return"),
    save_preprocessed: bool = typer.Option(False, help="Save preprocessed image for debugging"),
    debug: bool = typer.Option(False, help="Enable debug mode with detailed logging")
) -> None:
    """
    Load a TFLite model and perform inference on an image from a URL.
    
    Examples:
        python tflite_inference.py model.tflite "https://example.com/image.jpg"
        python tflite_inference.py model.tflite "https://example.com/image.jpg" --category-map-path labels.json
        python tflite_inference.py model.tflite "https://example.com/image.jpg" --debug
    """
    try:
        # Set debug logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        config = InferenceConfig(
            model_path=model_path,
            category_map_path=category_map_path,
            input_size=input_size,
            top_k=top_k
        )
        
        # Load category map if provided
        category_map = None
        if config.category_map_path:
            category_map = load_category_map(config.category_map_path)
        
        # Load TFLite model
        interpreter = load_tflite_model(config.model_path)
        
        # Download and preprocess image
        img = download_image(image_url)
        
        # Save original image info
        original_size = img.size
        
        # Preprocess image
        input_data = preprocess_image(img, config.input_size)
        
        # Debug: validate input data
        if debug:
            logger.debug(f"Input data statistics:")
            logger.debug(f"  Shape: {input_data.shape}")
            logger.debug(f"  Dtype: {input_data.dtype}")
            logger.debug(f"  Min: {input_data.min():.6f}")
            logger.debug(f"  Max: {input_data.max():.6f}")
            logger.debug(f"  Mean: {input_data.mean():.6f}")
            logger.debug(f"  Std: {input_data.std():.6f}")
        
        # Save preprocessed image if requested
        if save_preprocessed:
            preprocessed_img = input_data[0]  # Remove batch dimension
            # Denormalize for saving
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
            denorm_img = (preprocessed_img * std + mean).astype(np.uint8)
            denorm_img = np.clip(denorm_img, 0, 255)
            
            save_path = Path(f"preprocessed_image_{config.input_size}x{config.input_size}.jpg")
            Image.fromarray(denorm_img).save(save_path)
            logger.info(f"Saved preprocessed image to {save_path}")
        
        # Run inference
        output_data, inference_time = run_inference(interpreter, input_data)
        
        # Get top predictions
        predictions = get_top_predictions(output_data, category_map, config.top_k)
        
        # Print results
        print(f"\n📷 Image: {image_url}")
        print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
        print(f"🔄 Processed size: {config.input_size}x{config.input_size}")
        
        print_predictions(predictions, inference_time)
        
        # Log summary
        if predictions:
            top_class, top_prob, _ = predictions[0]
            logger.info(f"Classification complete: {top_class} ({top_prob*100:.1f}% confidence)")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that the TFLite model file exists and is valid")
        print("2. Verify the image URL is accessible")
        print("3. Ensure the category map file (if provided) is valid JSON")
        print("4. Check that input_size matches the model's expected input")
        print("5. Try running with --debug flag for detailed diagnostics")
        
        if debug:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
