# ResNet50 to TFLite Converter

This project provides tools to convert ResNet50 TIMM models to TFLite format and compare inference results between PyTorch, ONNX, and TFLite model formats.

**Updated for 2025**: Uses the latest dependency versions including TensorFlow 2.19.0, PyTorch 2.7.0, and modern conversion tools:
- `tf2onnx` (successor to onnx-tf frontend) for TensorFlow → ONNX conversion
- `onnx2tf` (replacement for onnx-tf backend) for ONNX → TensorFlow/TFLite conversion

## Contents

1. `convert_to_tflite.py` - Script to convert PyTorch models to TFLite format via ONNX
2. `test_models.py` - Script to test and compare inference results from different model formats
3. `run_test.sh` - Helper script to run the conversion and testing workflow
4. `Dockerfile` - Creates a container with all dependencies installed

## Requirements

All requirements are installed in the Docker container, but if you want to run the scripts directly, you'll need:

- Python 3.11+
- PyTorch 2.7.0+
- TensorFlow 2.19.0+
- ONNX 1.18.0+
- tf2onnx 1.16.1+ (successor to onnx-tf frontend, for TensorFlow → ONNX)
- onnx2tf 1.20.0+ (replacement for onnx-tf backend, for ONNX → TensorFlow/TFLite)
- timm 1.0.0+
- typer 0.12.0+
- pydantic 2.7.0+
- onnxruntime 1.22.0+
- Pillow 10.3.0+

**Migration Notes**: 
- `tensorflow-addons` has been **deprecated** (stopped development May 2024)
- `onnx-tf` has been **deprecated** and migrated:
  - Frontend functionality → `tf2onnx` (TensorFlow to ONNX conversion)
  - Backend functionality → replaced by `onnx2tf` (ONNX to TensorFlow/TFLite conversion)

## Docker Setup

The easiest way to run the code is using the provided Docker container:

1. Build the Docker image:

```bash
docker build -t resnet50-tflite .
```

2. Run the container with your data:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/converted_models:/app/converted_models" \
  resnet50-tflite \
  /bin/bash
```

## Model Conversion

To convert a ResNet50 model to TFLite:

```bash
python convert_to_tflite.py /path/to/weights.pth \
  --labels-path /path/to/category_map.json \
  --input-size 300 \
  --model-name "resnet50_model" \
  --output-dir ./converted_models
```

### Conversion Process

The conversion uses a modern, robust pipeline that accounts for the recent deprecations and migrations:

1. **PyTorch → ONNX**: Standard PyTorch ONNX export
2. **ONNX → TensorFlow**: Uses `onnx2tf` (modern replacement for deprecated onnx-tf backend)
3. **TensorFlow → TFLite**: Standard TensorFlow Lite converter with optimizations

**Note on Tool Migration**: The original `onnx-tf` package has been deprecated:
- Frontend functionality migrated to `tf2onnx` (for TensorFlow → ONNX)
- Backend functionality replaced by `onnx2tf` (for ONNX → TensorFlow/TFLite)

The script automatically handles fallbacks if certain conversion tools are unavailable and provides detailed error messages for troubleshooting.

### Arguments

- `weights_path` (positional): Path or URL to PyTorch model weights
- `--labels-path`: Path or URL to category map JSON file (used to determine number of classes)
- `--input-size`: Input image size (default: 300)
- `--model-name`: Name for the output model files (default: "resnet50")
- `--output-dir`: Directory to save converted models (default: "./converted_models")
- `--verify`: Whether to verify the TFLite model after conversion (default: True)

## Model Testing

To compare inference results between different model formats:

```bash
python test_models.py \
  --image_url "https://example.com/path/to/moth/image.jpg" \
  --weights_path /path/to/weights.pth \
  --category_map_path /path/to/category_map.json \
  --input_size 300 \
  --onnx_path /path/to/model.onnx \
  --tflite_path /path/to/model.tflite
```

### Arguments

- `--image_url`: URL of the image to test
- `--weights_path`: Path to PyTorch model weights
- `--category_map_path`: Path to category map JSON file
- `--input_size`: Input image size (default: 300)
- `--onnx_path`: Path to ONNX model (optional)
- `--tflite_path`: Path to TFLite model (optional)
- `--device`: Device to run PyTorch inference on (default: "cpu")

## Using the Convenience Script

The `run_test.sh` script combines both conversion and testing in one step:

```bash
./run_test.sh \
  --image_url "https://example.com/path/to/moth/image.jpg" \
  --weights_path /path/to/weights.pth \
  --category_map_path /path/to/category_map.json \
  --input_size 300 \
  --model_name "resnet50_model"
```

If you only want to run testing without conversion, add the `--skip_conversion` flag.

## Example Usage with Panama Moth Model

```bash
# Download model weights and category map
mkdir -p data
wget https://object-arbutus.cloud.computecanada.ca/ami-models/moths/classification/panama_resetnet50_best_5aeb515a.pth -O data/panama_model.pth
# Category map is already available in the repository

# Run conversion and testing
./run_test.sh \
  --image_url "https://inaturalist-open-data.s3.amazonaws.com/photos/181855805/original.jpg" \
  --weights_path data/panama_model.pth \
  --category_map_path data/03_moths_centralAmerica_category_map-202311110-with-names.json \
  --input_size 300 \
  --model_name "panama_resnet50"
```

## Docker Example

To run the entire process with Docker:

```bash
# Build the Docker image
docker build -t resnet50-tflite .

# Run the container with the test script
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/converted_models:/app/converted_models" \
  resnet50-tflite \
  ./run_test.sh \
  --image_url "https://inaturalist-open-data.s3.amazonaws.com/photos/181855805/original.jpg" \
  --weights_path /app/data/panama_model.pth \
  --category_map_path /app/data/03_moths_centralAmerica_category_map-202311110-with-names.json \
  --input_size 300 \
  --model_name "panama_resnet50"
```

## Output

The testing script will show:
1. Top-5 predictions from each model format
2. Inference time for each model format
3. Similarity metrics (top-1 agreement, top-5 overlap, probability differences)
4. Summary of model agreement or disagreement

If all three models agree on the top prediction, it's a good sign that the conversion worked correctly.

## Notes

- The ONNX and TFLite models will be saved in the `converted_models` directory
- If the models disagree on predictions, it may indicate issues with the conversion process
- TFLite models typically have faster inference time but may have slightly different results due to quantization

### Library Migration Status (2025)

**Deprecated Libraries:**
- `tensorflow-addons`: Completely deprecated (May 2024)
- `onnx-tf`: Deprecated and functionality migrated

**Migration Path:**
- `onnx-tf` frontend → `tf2onnx` (TensorFlow → ONNX conversion)
- `onnx-tf` backend → `onnx2tf` (ONNX → TensorFlow/TFLite conversion)

**Modern Alternatives Used:**
- `tf2onnx` 1.16.1+: For any TensorFlow to ONNX conversion needs
- `onnx2tf` 1.20.0+: For ONNX to TensorFlow/TFLite conversion (our primary use case)
