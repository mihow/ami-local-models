#!/usr/bin/env python3
"""
Test script to compare inference results between original ResNet50 PyTorch model,
ONNX model, and TFLite model.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import argparse
import urllib.request
from io import BytesIO

import numpy as np
import torch
import torchvision
import onnxruntime
import tensorflow as tf
from PIL import Image
import timm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for model testing."""
    weights_path: str
    category_map_path: str
    model_name: str = "resnet50"
    input_size: int = 300
    onnx_path: Optional[str] = None
    tflite_path: Optional[str] = None
    device: str = "cpu"

def load_category_map(path: str) -> Dict[str, int]:
    """Load category map from JSON file."""
    logger.info(f"Loading category map from {path}")
    with open(path, 'r') as f:
        category_map = json.load(f)
    logger.info(f"Loaded category map with {len(category_map)} classes")
    return category_map

def get_inverse_category_map(category_map: Dict[str, int]) -> Dict[int, str]:
    """Create an inverse mapping from class indices to names."""
    return {v: k for k, v in category_map.items()}

def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    logger.info(f"Downloading image from {url}")
    with urllib.request.urlopen(url) as response:
        img = Image.open(BytesIO(response.read()))
    return img

def preprocess_image(img: Image.Image, config: TestConfig) -> torch.Tensor:
    """
    Preprocess image for model input.
    Returns both a PyTorch tensor and a numpy array.
    """
    # Get image size and create padding to make it square
    width, height = img.size
    padding = [0, 0, 0, 0]  # left, top, right, bottom
    
    if height < width:
        padding[3] = width - height  # pad bottom
    elif height > width:
        padding[2] = height - width  # pad right
    
    # Create transformations
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet normalization
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding=padding),
        torchvision.transforms.Resize((config.input_size, config.input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    
    # Apply transformations
    tensor = transform(img)
    
    return tensor

def load_pytorch_model(config: TestConfig) -> torch.nn.Module:
    """Load PyTorch model."""
    logger.info(f"Loading PyTorch model from {config.weights_path}")
    
    # Get number of classes from category map
    num_classes = len(load_category_map(config.category_map_path))
    
    # Create model
    model = timm.create_model(
        config.model_name,
        weights=None,
        num_classes=num_classes,
    )
    
    # Load weights
    checkpoint = torch.load(config.weights_path, map_location=config.device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.to(config.device)
    model.eval()
    
    return model

def run_pytorch_inference(
    model: torch.nn.Module, 
    image_tensor: torch.Tensor,
    config: TestConfig
) -> Tuple[List[str], List[float], float]:
    """
    Run inference with PyTorch model.
    Returns top-5 class names, probabilities, and inference time.
    """
    logger.info("Running PyTorch inference")
    
    # Add batch dimension
    batch = image_tensor.unsqueeze(0).to(config.device)
    
    # Run inference and measure time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(batch)
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    inference_time = time.time() - start_time
    
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Convert to numpy
    top5_prob = top5_prob.cpu().numpy()
    top5_indices = top5_indices.cpu().numpy()
    
    # Get class names
    inverse_category_map = get_inverse_category_map(
        load_category_map(config.category_map_path)
    )
    top5_classes = [inverse_category_map[idx] for idx in top5_indices]
    
    logger.info(f"PyTorch inference time: {inference_time:.4f} seconds")
    
    return top5_classes, top5_prob, inference_time

def load_onnx_model(config: TestConfig) -> onnxruntime.InferenceSession:
    """Load ONNX model."""
    if config.onnx_path is None:
        config.onnx_path = f"./converted_models/{config.model_name}.onnx"
    
    logger.info(f"Loading ONNX model from {config.onnx_path}")
    
    # Create ONNX inference session
    session = onnxruntime.InferenceSession(
        config.onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    return session

def run_onnx_inference(
    session: onnxruntime.InferenceSession,
    image_tensor: torch.Tensor,
    config: TestConfig
) -> Tuple[List[str], List[float], float]:
    """
    Run inference with ONNX model.
    Returns top-5 class names, probabilities, and inference time.
    """
    logger.info("Running ONNX inference")
    
    # Convert pytorch tensor to numpy array for ONNX
    input_array = image_tensor.unsqueeze(0).numpy()
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference and measure time
    start_time = time.time()
    outputs = session.run(None, {input_name: input_array})
    inference_time = time.time() - start_time
    
    # Apply softmax to get probabilities
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_prob = probabilities[top5_indices]
    
    # Get class names
    inverse_category_map = get_inverse_category_map(
        load_category_map(config.category_map_path)
    )
    top5_classes = [inverse_category_map[idx] for idx in top5_indices]
    
    logger.info(f"ONNX inference time: {inference_time:.4f} seconds")
    
    return top5_classes, top5_prob, inference_time

def load_tflite_model(config: TestConfig) -> tf.lite.Interpreter:
    """Load TFLite model."""
    if config.tflite_path is None:
        config.tflite_path = f"./converted_models/{config.model_name}.tflite"
    
    logger.info(f"Loading TFLite model from {config.tflite_path}")
    
    # Create TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=config.tflite_path)
    interpreter.allocate_tensors()
    
    return interpreter

def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    image_tensor: torch.Tensor,
    config: TestConfig
) -> Tuple[List[str], List[float], float]:
    """
    Run inference with TFLite model.
    Returns top-5 class names, probabilities, and inference time.
    """
    logger.info("Running TFLite inference")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input data
    input_data = image_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    # Run inference and measure time
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    inference_time = time.time() - start_time
    
    # Apply softmax to get probabilities
    logits = outputs[0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_prob = probabilities[top5_indices]
    
    # Get class names
    inverse_category_map = get_inverse_category_map(
        load_category_map(config.category_map_path)
    )
    top5_classes = [inverse_category_map[idx] for idx in top5_indices]
    
    logger.info(f"TFLite inference time: {inference_time:.4f} seconds")
    
    return top5_classes, top5_prob, inference_time

def calculate_similarity(
    pytorch_probs: np.ndarray,
    other_probs: np.ndarray,
    pytorch_classes: List[str],
    other_classes: List[str]
) -> Dict[str, float]:
    """
    Calculate similarity metrics between PyTorch and other model outputs.
    """
    # Calculate top-1 agreement (whether the top class matches)
    top1_agreement = 1.0 if pytorch_classes[0] == other_classes[0] else 0.0
    
    # Calculate top-5 overlap (percentage of classes that appear in both top-5 lists)
    overlap = len(set(pytorch_classes).intersection(set(other_classes)))
    top5_overlap = overlap / 5.0
    
    # Calculate probability difference for matching classes
    prob_diffs = []
    for idx, cls in enumerate(pytorch_classes):
        if cls in other_classes:
            other_idx = other_classes.index(cls)
            prob_diffs.append(abs(pytorch_probs[idx] - other_probs[other_idx]))
    
    avg_prob_diff = np.mean(prob_diffs) if prob_diffs else 1.0
    
    return {
        "top1_agreement": top1_agreement,
        "top5_overlap": top5_overlap,
        "avg_prob_diff": avg_prob_diff
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare model formats for inference")
    parser.add_argument("--image_url", type=str, required=True, 
                      help="URL of the image to test")
    parser.add_argument("--weights_path", type=str, required=True,
                      help="Path to PyTorch model weights")
    parser.add_argument("--category_map_path", type=str, required=True,
                      help="Path to category map JSON file")
    parser.add_argument("--input_size", type=int, default=300,
                      help="Input image size")
    parser.add_argument("--onnx_path", type=str, default=None,
                      help="Path to ONNX model")
    parser.add_argument("--tflite_path", type=str, default=None,
                      help="Path to TFLite model")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to run PyTorch inference on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Create config
    config = TestConfig(
        weights_path=args.weights_path,
        category_map_path=args.category_map_path,
        input_size=args.input_size,
        onnx_path=args.onnx_path,
        tflite_path=args.tflite_path,
        device=args.device
    )
    
    # Download and preprocess image
    img = download_image(args.image_url)
    image_tensor = preprocess_image(img, config)
    
    # Run PyTorch inference
    pytorch_model = load_pytorch_model(config)
    pytorch_classes, pytorch_probs, pytorch_time = run_pytorch_inference(
        pytorch_model, image_tensor, config
    )
    
    # Run ONNX inference
    try:
        onnx_session = load_onnx_model(config)
        onnx_classes, onnx_probs, onnx_time = run_onnx_inference(
            onnx_session, image_tensor, config
        )
        
        # Calculate similarity metrics
        onnx_similarity = calculate_similarity(
            pytorch_probs, onnx_probs, pytorch_classes, onnx_classes
        )
    except Exception as e:
        logger.error(f"Error running ONNX inference: {e}")
        onnx_classes, onnx_probs, onnx_time = [], [], 0.0
        onnx_similarity = {}
    
    # Run TFLite inference
    try:
        tflite_interpreter = load_tflite_model(config)
        tflite_classes, tflite_probs, tflite_time = run_tflite_inference(
            tflite_interpreter, image_tensor, config
        )
        
        # Calculate similarity metrics
        tflite_similarity = calculate_similarity(
            pytorch_probs, tflite_probs, pytorch_classes, tflite_classes
        )
    except Exception as e:
        logger.error(f"Error running TFLite inference: {e}")
        tflite_classes, tflite_probs, tflite_time = [], [], 0.0
        tflite_similarity = {}
    
    # Print results
    print("\n=== INFERENCE RESULTS ===\n")
    
    print("PyTorch Model Results:")
    for i, (cls, prob) in enumerate(zip(pytorch_classes, pytorch_probs)):
        print(f"  {i+1}. {cls}: {prob:.4f}")
    print(f"  Inference time: {pytorch_time:.4f} seconds\n")
    
    if onnx_classes:
        print("ONNX Model Results:")
        for i, (cls, prob) in enumerate(zip(onnx_classes, onnx_probs)):
            print(f"  {i+1}. {cls}: {prob:.4f}")
        print(f"  Inference time: {onnx_time:.4f} seconds")
        print(f"  Speedup vs PyTorch: {pytorch_time/onnx_time:.2f}x")
        print(f"  Top-1 agreement: {onnx_similarity['top1_agreement']:.2f}")
        print(f"  Top-5 overlap: {onnx_similarity['top5_overlap']:.2f}")
        print(f"  Avg probability difference: {onnx_similarity['avg_prob_diff']:.4f}\n")
    
    if tflite_classes:
        print("TFLite Model Results:")
        for i, (cls, prob) in enumerate(zip(tflite_classes, tflite_probs)):
            print(f"  {i+1}. {cls}: {prob:.4f}")
        print(f"  Inference time: {tflite_time:.4f} seconds")
        print(f"  Speedup vs PyTorch: {pytorch_time/tflite_time:.2f}x")
        print(f"  Top-1 agreement: {tflite_similarity['top1_agreement']:.2f}")
        print(f"  Top-5 overlap: {tflite_similarity['top5_overlap']:.2f}")
        print(f"  Avg probability difference: {tflite_similarity['avg_prob_diff']:.4f}\n")
    
    # Summary
    print("=== SUMMARY ===")
    print(f"PyTorch top prediction: {pytorch_classes[0]} ({pytorch_probs[0]:.4f})")
    
    if onnx_classes:
        print(f"ONNX top prediction: {onnx_classes[0]} ({onnx_probs[0]:.4f})")
        match_status = "✓ MATCH" if pytorch_classes[0] == onnx_classes[0] else "✗ MISMATCH"
        print(f"PyTorch vs ONNX: {match_status}")
    
    if tflite_classes:
        print(f"TFLite top prediction: {tflite_classes[0]} ({tflite_probs[0]:.4f})")
        match_status = "✓ MATCH" if pytorch_classes[0] == tflite_classes[0] else "✗ MISMATCH"
        print(f"PyTorch vs TFLite: {match_status}")
    
    # Overall assessment
    if onnx_classes and tflite_classes:
        all_match = (pytorch_classes[0] == onnx_classes[0] and 
                     pytorch_classes[0] == tflite_classes[0])
        
        if all_match:
            print("\nAll models agree on the top prediction.")
        else:
            print("\nWARNING: Models disagree on the top prediction.")
            print("This may indicate issues with the conversion process.")

if __name__ == "__main__":
    main()
