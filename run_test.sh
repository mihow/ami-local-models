#!/bin/bash
set -e

# Default values
IMAGE_URL=""
WEIGHTS_PATH=""
CATEGORY_MAP_PATH=""
INPUT_SIZE=300
MODEL_NAME="resnet50"
CONVERT_MODEL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image_url)
      IMAGE_URL="$2"
      shift 2
      ;;
    --weights_path)
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    --category_map_path)
      CATEGORY_MAP_PATH="$2"
      shift 2
      ;;
    --input_size)
      INPUT_SIZE="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --skip_conversion)
      CONVERT_MODEL=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$IMAGE_URL" ] || [ -z "$WEIGHTS_PATH" ] || [ -z "$CATEGORY_MAP_PATH" ]; then
  echo "Usage: $0 --image_url URL --weights_path PATH --category_map_path PATH [--input_size SIZE] [--model_name NAME] [--skip_conversion]"
  exit 1
fi

# Get absolute paths
WEIGHTS_PATH=$(realpath "$WEIGHTS_PATH")
CATEGORY_MAP_PATH=$(realpath "$CATEGORY_MAP_PATH")

# Create output directory
OUTPUT_DIR="./converted_models"
mkdir -p "$OUTPUT_DIR"

# Convert model if needed
if [ "$CONVERT_MODEL" = true ]; then
  echo "Converting model to ONNX and TFLite formats..."
  python convert_to_tflite.py \
    "$WEIGHTS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --labels-path "$CATEGORY_MAP_PATH" \
    --input-size "$INPUT_SIZE" \
    --model-name "$MODEL_NAME"
fi

# Run inference test
echo "Running inference test with all model formats..."
python test_models.py \
  --image_url "$IMAGE_URL" \
  --weights_path "$WEIGHTS_PATH" \
  --category_map_path "$CATEGORY_MAP_PATH" \
  --input_size "$INPUT_SIZE" \
  --onnx_path "$OUTPUT_DIR/$MODEL_NAME.onnx" \
  --tflite_path "$OUTPUT_DIR/$MODEL_NAME.tflite"

echo "Testing complete!"
