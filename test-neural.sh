#!/bin/bash
# test-neural.sh
# Run the Python neural test iteration script

if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing..."
    pkg install python
fi

python3 test_neural.py
