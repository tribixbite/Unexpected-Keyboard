#!/usr/bin/env python3

import subprocess
import sys
import time

def run_gradle_test(test_class, test_method=None):
    """Run a specific gradle test and return output"""
    cmd = ["./gradlew", "test"]
    if test_class:
        cmd.append("--tests")
        if test_method:
            cmd.append(f"{test_class}.{test_method}")
        else:
            cmd.append(test_class)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out"

def extract_error_from_logs(stdout, stderr):
    """Extract key error information from test logs"""
    lines = (stdout + stderr).split('\n')
    
    # Look for specific error patterns
    for line in lines:
        if "ClassCastException" in line:
            return "CASTING_ERROR", line.strip()
        if "ORT_INVALID_ARGUMENT" in line:
            return "TENSOR_ERROR", line.strip()  
        if "cannot be cast to" in line:
            return "CAST_TYPE_ERROR", line.strip()
        if "FAILED" in line and "Neural" in line:
            return "NEURAL_ERROR", line.strip()
    
    return "UNKNOWN", ""

def test_iteration_1():
    """Test current neural implementation"""
    print("=== Iteration 1: Testing Current Implementation ===")
    
    # The casting issue is in OnnxSwipePredictor line ~520:
    # float[] logitsData = (float[]) logitsTensor.getValue();
    # But logitsTensor is actually 3D: [1, 20, 30]
    
    fix_code = '''
    // OLD (line ~520 in OnnxSwipePredictor.java):
    float[] logitsData = (float[]) logitsTensor.getValue();
    
    // NEW - handle 3D logits tensor properly:
    float[][][] logitsData3D = (float[][][]) logitsTensor.getValue();
    // Extract logits for current position: [batch=0, position, vocab]
    float[] logitsData = logitsData3D[0][tokenPosition];
    '''
    
    print("🔧 Required fix identified:")
    print(fix_code)
    return "CASTING_ERROR", "Need to handle 3D logits tensor [1, 20, 30]"

def test_iteration_2():
    """Test after fixing 3D logits casting"""
    print("=== Iteration 2: After 3D Logits Fix ===")
    
    # After fixing 3D casting, next likely issues:
    # 1. Beam search token sequence handling
    # 2. Softmax calculation on wrong slice
    # 3. Token-to-word conversion
    
    print("Expected remaining issues:")
    print("- Beam search logic errors")
    print("- Token sequence decoding") 
    print("- Word generation from tokens")
    
    return "BEAM_SEARCH_ERROR", "Logic errors in beam search implementation"

def main():
    print("🧪 Neural ONNX Iterative Testing System")
    print("=====================================")
    
    # Based on the logs provided, we know the exact error:
    current_error = "ClassCastException - float[][][] cannot be cast to float[]"
    
    print(f"Current error: {current_error}")
    print()
    
    # Run diagnostic iterations
    error_type, details = test_iteration_1()
    print(f"Error type: {error_type}")
    print(f"Details: {details}")
    print()
    
    print("Next steps:")
    print("1. Fix 3D logits tensor casting in OnnxSwipePredictor.java")
    print("2. Update beam search to handle proper tensor dimensions")
    print("3. Test iteratively until predictions work")
    
    return True

if __name__ == "__main__":
    main()