#!/usr/bin/env python3
"""
Gradio web app for testing the TFLite moth classification model.
"""
import json
import logging
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MothClassifier:
    """TFLite moth classification model wrapper."""
    
    def __init__(self, model_path: str, category_map_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.category_map_path = Path(category_map_path) if category_map_path else None
        self.interpreter = None
        self.category_map = None
        self.inverse_category_map = None
        self.input_size = 300  # Default input size
        
        self.load_model()
        if self.category_map_path:
            self.load_category_map()
    
    def load_model(self):
        """Load the TFLite model."""
        logger.info(f"Loading TFLite model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get model info
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.input_shape = input_details[0]['shape']
        self.output_shape = output_details[0]['shape']
        self.input_size = self.input_shape[1]  # Assuming square input
        
        logger.info(f"Model loaded - Input: {self.input_shape}, Output: {self.output_shape}")
    
    def load_category_map(self):
        """Load category mapping for class names."""
        logger.info(f"Loading category map from {self.category_map_path}")
        
        with open(self.category_map_path, 'r') as f:
            self.category_map = json.load(f)
        
        self.inverse_category_map = {v: k for k, v in self.category_map.items()}
        logger.info(f"Loaded {len(self.category_map)} species classes")
    
    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
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
        
        # Resize to target size
        img = img.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and ensure float32
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        
        img_array = (img_array - mean) / std
        img_array = img_array.astype(np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Union[Image.Image, str], top_k: int = 5) -> Tuple[List[Tuple[str, float]], float, Image.Image]:
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image or URL string
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (predictions, inference_time, processed_image)
        """
        # Handle image input
        if isinstance(image, str):
            # Download from URL
            with urllib.request.urlopen(image) as response:
                img = Image.open(BytesIO(response.read()))
        else:
            img = image
        
        # Store original for display
        original_img = img.copy()
        
        # Preprocess image
        input_data = self.preprocess_image(img)
        
        # Create processed image for display
        processed_img = Image.fromarray(
            ((input_data[0] * np.array([0.229, 0.224, 0.225])[None, None, :] + 
              np.array([0.485, 0.456, 0.406])[None, None, :]) * 255).astype(np.uint8)
        )
        
        # Run inference
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        
        # Get top predictions
        predictions = self.get_top_predictions(output_data, top_k)
        
        return predictions, inference_time, processed_img
    
    def get_top_predictions(self, output_data: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Get top-k predictions from model output."""
        # Apply softmax
        logits = output_data[0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # Get top k indices
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_k_probs = probabilities[top_k_indices]
        
        # Convert to class names
        predictions = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            if self.inverse_category_map:
                class_name = self.inverse_category_map.get(idx, f"Unknown_Class_{idx}")
            else:
                class_name = f"Class_{idx}"
            predictions.append((class_name, float(prob)))
        
        return predictions

# Global model instance
classifier = None

def load_model(model_path: str, category_map_path: str = None):
    """Load the model globally."""
    global classifier
    try:
        classifier = MothClassifier(model_path, category_map_path)
        return f"✅ Model loaded successfully! Input size: {classifier.input_size}x{classifier.input_size}, Classes: {len(classifier.category_map) if classifier.category_map else 'Unknown'}"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def classify_image(image, top_k=5):
    """Classify an uploaded image."""
    if classifier is None:
        return "❌ Please load a model first!", None, None, None
    
    if image is None:
        return "❌ Please upload an image!", None, None, None
    
    try:
        # Run prediction
        predictions, inference_time, processed_img = classifier.predict(image, top_k)
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                "Rank": i+1,
                "Species": pred[0],
                "Confidence": f"{pred[1]:.4f}",
                "Percentage": f"{pred[1]*100:.2f}%"
            }
            for i, pred in enumerate(predictions)
        ])
        
        # Create summary text
        if predictions:
            top_pred = predictions[0]
            summary = f"🎯 **Top Prediction**: {top_pred[0]} ({top_pred[1]*100:.1f}% confidence)\n"
            summary += f"⏱️ **Inference Time**: {inference_time:.4f} seconds\n"
            summary += f"📊 **Model**: {len(classifier.category_map) if classifier.category_map else 'Unknown'} species classifier"
        else:
            summary = "No predictions available"
        
        return summary, results_df, image, processed_img
        
    except Exception as e:
        return f"❌ Error during classification: {str(e)}", None, None, None

def classify_from_url(url, top_k=5):
    """Classify an image from URL."""
    if classifier is None:
        return "❌ Please load a model first!", None, None, None
    
    if not url or not url.strip():
        return "❌ Please enter a valid image URL!", None, None, None
    
    try:
        # Download and classify
        predictions, inference_time, processed_img = classifier.predict(url, top_k)
        
        # Download original image for display
        with urllib.request.urlopen(url) as response:
            original_img = Image.open(BytesIO(response.read()))
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                "Rank": i+1,
                "Species": pred[0],
                "Confidence": f"{pred[1]:.4f}",
                "Percentage": f"{pred[1]*100:.2f}%"
            }
            for i, pred in enumerate(predictions)
        ])
        
        # Create summary text
        if predictions:
            top_pred = predictions[0]
            summary = f"🎯 **Top Prediction**: {top_pred[0]} ({top_pred[1]*100:.1f}% confidence)\n"
            summary += f"⏱️ **Inference Time**: {inference_time:.4f} seconds\n"
            summary += f"🔗 **Source**: {url}\n"
            summary += f"📊 **Model**: {len(classifier.category_map) if classifier.category_map else 'Unknown'} species classifier"
        else:
            summary = "No predictions available"
        
        return summary, results_df, original_img, processed_img
        
    except Exception as e:
        return f"❌ Error during classification: {str(e)}", None, None, None

# Example URLs for testing
EXAMPLE_URLS = [
    "https://inaturalist-open-data.s3.amazonaws.com/photos/181855805/original.jpg",
    "https://static.inaturalist.org/photos/1234567/original.jpg",
    "http://production-chroma.s3.amazonaws.com/photos/5e7a7d5b3c12ef365be1d750/ouTcOUxZZC2a3wF.jpg"
]

def create_gradio_app():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="🦋 Moth Species Classifier",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .results-box {
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🦋 Moth Species Classifier
        
        Upload an image or provide a URL to classify moth species using a deep learning model.
        Trained on **2360 Central American moth species** using ResNet50 architecture.
        """)
        
        with gr.Tab("📁 Model Setup"):
            gr.Markdown("### Load your TFLite model and category map")
            
            with gr.Row():
                model_path_input = gr.Textbox(
                    label="TFLite Model Path",
                    placeholder="converted_models/tf_model/panama_resnet50_float32.tflite",
                    value="converted_models/tf_model/panama_resnet50_float32.tflite"
                )
                category_map_input = gr.Textbox(
                    label="Category Map Path (Optional)",
                    placeholder="data/panama_plus_category_map-with_names.json",
                    value="data/panama_plus_category_map-with_names.json"
                )
            
            load_btn = gr.Button("🔄 Load Model", variant="primary")
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            load_btn.click(
                fn=load_model,
                inputs=[model_path_input, category_map_input],
                outputs=[model_status]
            )
        
        with gr.Tab("📤 Upload Image"):
            gr.Markdown("### Upload an image file for classification")
            
            with gr.Row():
                with gr.Column(scale=1):
                    upload_image = gr.Image(
                        label="Upload Moth Image",
                        type="pil",
                        height=400
                    )
                    
                    upload_top_k = gr.Slider(
                        label="Number of Top Predictions",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    
                    upload_classify_btn = gr.Button("🔍 Classify Image", variant="primary")
                
                with gr.Column(scale=1):
                    upload_processed = gr.Image(
                        label="Preprocessed Image",
                        type="pil",
                        height=400
                    )
            
            with gr.Row():
                upload_summary = gr.Markdown(label="Results Summary")
            
            with gr.Row():
                upload_results = gr.DataFrame(
                    label="Detailed Predictions",
                    headers=["Rank", "Species", "Confidence", "Percentage"]
                )
            
            upload_classify_btn.click(
                fn=classify_image,
                inputs=[upload_image, upload_top_k],
                outputs=[upload_summary, upload_results, upload_image, upload_processed]
            )
        
        with gr.Tab("🔗 Image URL"):
            gr.Markdown("### Classify an image from a URL")
            
            with gr.Row():
                url_input = gr.Textbox(
                    label="Image URL",
                    placeholder="https://example.com/moth_image.jpg",
                    lines=2
                )
                
                url_top_k = gr.Slider(
                    label="Number of Top Predictions",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1
                )
            
            with gr.Row():
                url_classify_btn = gr.Button("🔍 Classify from URL", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Example URLs")
                    for i, example_url in enumerate(EXAMPLE_URLS):
                        gr.Button(
                            f"Example {i+1}",
                            size="sm"
                        ).click(
                            fn=lambda url=example_url: url,
                            outputs=[url_input]
                        )
            
            with gr.Row():
                with gr.Column(scale=1):
                    url_original = gr.Image(
                        label="Original Image",
                        type="pil",
                        height=400
                    )
                with gr.Column(scale=1):
                    url_processed = gr.Image(
                        label="Preprocessed Image", 
                        type="pil",
                        height=400
                    )
            
            with gr.Row():
                url_summary = gr.Markdown(label="Results Summary")
            
            with gr.Row():
                url_results = gr.DataFrame(
                    label="Detailed Predictions",
                    headers=["Rank", "Species", "Confidence", "Percentage"]
                )
            
            url_classify_btn.click(
                fn=classify_from_url,
                inputs=[url_input, url_top_k],
                outputs=[url_summary, url_results, url_original, url_processed]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Model
            
            This moth species classifier was trained on **2360 Central American moth species** using:
            
            - **Architecture**: ResNet50 (TIMM)
            - **Input Size**: 300×300 pixels
            - **Preprocessing**: ImageNet normalization with padding to square
            - **Model Format**: TensorFlow Lite (converted from PyTorch)
            
            ### How It Works
            
            1. **Image Preprocessing**: Images are padded to square, resized to 300×300, and normalized
            2. **Model Inference**: ResNet50 backbone extracts features and classifies species
            3. **Post-processing**: Softmax activation produces probability scores for each species
            
            ### Performance Notes
            
            - **CPU Inference**: Optimized for CPU deployment using TensorFlow Lite
            - **Accuracy**: Trained on high-quality iNaturalist moth observations
            - **Coverage**: Focused on Central American moth biodiversity
            
            ### Usage Tips
            
            - **Best Results**: Use clear, well-lit images of moths with minimal background
            - **Image Quality**: Higher resolution images generally produce better results
            - **Multiple Views**: Try different angles if the first prediction seems uncertain
            
            ---
            
            **Model Conversion Pipeline**: PyTorch → ONNX → TensorFlow → TFLite
            """)
        
        # Auto-load model on startup if files exist
        gr.Markdown("""
        ---
        💡 **Quick Start**: Load your model in the "Model Setup" tab, then test with images!
        """)
    
    return app

def main():
    """Main function to run the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Moth Species Classifier Web App")
    parser.add_argument("--model-path", type=str, default="converted_models/panama_resnet50.tflite",
                       help="Path to TFLite model")
    parser.add_argument("--category-map", type=str, default="data/03_moths_centralAmerica_category_map-202311110-with-names.json",
                       help="Path to category map JSON")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the app on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--auto-load", action="store_true", help="Auto-load model on startup")
    
    args = parser.parse_args()
    
    # Create the app
    app = create_gradio_app()
    
    # Auto-load model if requested and files exist
    if args.auto_load:
        if Path(args.model_path).exists():
            model_status = load_model(args.model_path, args.category_map)
            print(f"Auto-load: {model_status}")
        else:
            print(f"Auto-load failed: Model not found at {args.model_path}")
    
    # Launch the app
    print(f"🚀 Starting Moth Classifier Web App...")
    print(f"📂 Model Path: {args.model_path}")
    print(f"🏷️ Category Map: {args.category_map}")
    print(f"🌐 Host: {args.host}:{args.port}")
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()
