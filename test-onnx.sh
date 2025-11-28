#!/bin/bash
# test-onnx.sh
# Run the standalone ONNX test script in Termux environment

# Ensure python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing..."
    pkg install python
fi

# Install dependencies if needed
# Assuming onnxruntime and numpy are installed via pip
# If not, we can try to install them, but onnxruntime might be tricky on Termux ARM64
# Check if packages are installed
if ! python3 -c "import onnxruntime; import numpy" &> /dev/null; then
    echo "Installing python dependencies..."
    pip install onnxruntime numpy
fi

# Run the test
echo "Running test_onnx_cli.py..."
python3 test_onnx_cli.py
