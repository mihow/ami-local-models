FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install additional dependencies with latest 2025 versions
# Note: onnx-tf is deprecated and migrated to tf2onnx + onnx2tf
RUN pip install --no-cache-dir \
    tf2onnx>=1.16.1 \
    onnx2tf>=1.20.0 \
    timm>=1.0.0 \
    typer>=0.12.0 \
    pydantic>=2.7.0 \
    onnxruntime>=1.22.0 \
    pillow>=10.3.0 \
    tensorflow>=2.19.0

# Create app directory
WORKDIR /app

# Copy scripts
COPY convert_to_tflite.py /app/
COPY test_models.py /app/
COPY run_test.sh /app/

# Make scripts executable
RUN chmod +x /app/convert_to_tflite.py
RUN chmod +x /app/test_models.py
RUN chmod +x /app/run_test.sh

# Create directories
RUN mkdir -p /app/converted_models
RUN mkdir -p /app/data
