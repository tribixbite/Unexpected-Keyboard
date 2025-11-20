# Performance Todos v6: Integrating the Quantized ONNX Model

This document provides a step-by-step guide to properly integrate the new statically quantized ONNX models located in `assets/models/bs/` and configure the ONNX Runtime to take full advantage of them for maximum performance.

**Goal:** Replace the existing float32 models with the new int8 quantized models and enable hardware acceleration via the NNAPI execution provider. This should result in significantly lower prediction latency, reduced memory usage, and lower power consumption.

---

## I. Analysis of Current Implementation

The current `OnnxSwipePredictor.java` loads float32 models and uses a very basic configuration for the ONNX Runtime. It enables the NNAPI provider but without the specific flags needed to ensure optimal performance for quantized models. The session creation logic is also spread across several confusing and partially-used methods.

This plan will consolidate and correct that implementation.

---

## II. Integration and Optimization Plan

### Step 1: Update Model Paths

In `OnnxSwipePredictor.java`, inside the `initialize()` method, the `switch` statement for model versions needs to be updated to point to the new models.

**Action:**
*   Change the `v2` (default) case to use the paths in the `assets/models/bs/` directory.

**In `OnnxSwipePredictor.java`:**
```java
// ... inside initialize() method ...
      switch (_currentModelVersion)
      {
        case "v2":
          // NEW: Point to the new, quantized models in the 'bs' directory
          encoderPath = "models/bs/swipe_encoder_android.onnx";
          decoderPath = "models/bs/swipe_decoder_android.onnx";
          _maxSequenceLength = 250; // Assuming this is still correct
          _modelAccuracy = "N/A (quantized)";
          _modelSource = "builtin-quantized";
          Log.d(TAG, "Loading v2 quantized models (bs)");
          break;
        
        // ... other cases ...
      }
// ...
```

---

### Step 2 (Critical): Implement Advanced NNAPI Configuration

To unlock the performance of the quantized model, you must configure the ONNX Runtime session to use the NNAPI provider with specific flags. This tells NNAPI that it can use on-device accelerators (like a DSP or NPU) and that the model is designed for `int8` computation.

**Action:**
*   Replace the existing `createOptimizedSessionOptions()` and `tryEnableHardwareAcceleration()` methods with a single, clear method that correctly configures NNAPI.

**In `OnnxSwipePredictor.java`, add this new method:**

```java
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.NnapiFlags; // You may need to update your ONNX import

// ...

/**
 * Creates an optimized OrtSession.SessionOptions with the NNAPI Execution Provider enabled.
 * This is CRITICAL for leveraging hardware acceleration for quantized models.
 */
private SessionOptions createNnapiSessionOptions(String sessionName) throws OrtException {
    SessionOptions options = new SessionOptions();
    options.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
    
    // These flags are essential for quantized models
    int nnapiFlags = 0;
    // Allows NNAPI to use lower-precision FP16 for even better performance if supported.
    nnapiFlags |= NnapiFlags.NNAPI_FLAG_USE_FP16; 
    
    // --- FOR DEBUGGING ONLY ---
    // This flag FORCES NNAPI. If any operator in your model is not supported 
    // by the device's NNAPI driver, session creation will FAIL.
    // Use this to confirm the model is 100% running on NNAPI.
    // REMOVE THIS FLAG FOR PRODUCTION BUILDS to allow fallback to the CPU.
    nnapiFlags |= NnapiFlags.NNAPI_FLAG_CPU_DISABLED;
    
    try {
        options.addNnapi(nnapiFlags);
        Log.i(TAG, "NNAPI execution provider configured for " + sessionName + " with flags.");
    } catch (Exception e) {
        Log.e(TAG, "NNAPI provider is not available. Falling back to default CPU.", e);
        // Fallback to default CPU provider if NNAPI is not available on the device
    }
    
    return options;
}
```

**Then, update the `initialize()` method to use it:**

```java
// ... inside initialize() ...
// Replace this:
// OrtSession.SessionOptions sessionOptions = createOptimizedSessionOptions("Encoder");
// With this:
OrtSession.SessionOptions sessionOptions = createNnapiSessionOptions("Encoder");
_encoderSession = _ortEnvironment.createSession(encoderModelData, sessionOptions);

// ... and for the decoder ...
// Replace this:
// OrtSession.SessionOptions sessionOptions = createOptimizedSessionOptions("Decoder");
// With this:
OrtSession.SessionOptions sessionOptions = createNnapiSessionOptions("Decoder");
_decoderSession = _ortEnvironment.createSession(decoderModelData, sessionOptions);
// ...
```
**Note:** You may need to ensure your `onnxruntime-android` dependency is recent enough to include `NnapiFlags`.

---

### Step 3: Verify Model Inputs and Outputs

A quantized model may have different input/output data types (e.g., `int8` instead of `float32`). Since we don't have the export script, you must verify this at runtime.

**Action:**
*   Add the following debug code right after you create the sessions in the `initialize()` method to log the exact signature of the new models.

```java
// In initialize(), right after creating _encoderSession
Log.i(TAG, "--- Encoder Model Signature ---");
for (Map.Entry<String, NodeInfo> entry : _encoderSession.getInputInfo().entrySet()) {
    Log.i(TAG, "Input: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
}
for (Map.Entry<String, NodeInfo> entry : _encoderSession.getOutputInfo().entrySet()) {
    Log.i(TAG, "Output: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
}
Log.i(TAG, "---------------------------------");
```
*   Run the app and check the logcat output. Compare the logged data types and tensor shapes with the tensors you create in `createTrajectoryTensor` and other methods. If they don't match (e.g., the model expects `UINT8` but you provide `FLOAT`), you will need to adjust your tensor creation logic. Often, the ONNX runtime handles this for you if the model has the correct Quantize/Dequantize operators, but it's crucial to verify.

---

### Step 4: Implement a Robust NNAPI Fallback (for Production)

The `NNAPI_FLAG_CPU_DISABLED` flag is great for testing but should be removed for production releases. For a robust app, you should also handle the case where session creation with NNAPI fails for other reasons.

**Action:**
*   Modify the session creation logic to fall back to the default CPU provider if the NNAPI-enabled session fails.

**Recommended `initialize()` logic for production:**

```java
// In initialize() for the encoder
try {
    Log.d(TAG, "Attempting to create session with NNAPI provider...");
    OrtSession.SessionOptions nnapiOptions = createNnapiSessionOptions("Encoder"); // Ensure CPU_DISABLED is OFF for production
    _encoderSession = _ortEnvironment.createSession(encoderModelData, nnapiOptions);
    Log.i(TAG, "Successfully created session with NNAPI provider.");
} catch (Exception e) {
    Log.e(TAG, "Failed to create session with NNAPI. Falling back to default CPU provider.", e);
    // Fallback to default options
    OrtSession.SessionOptions defaultOptions = new OrtSession.SessionOptions();
    _encoderSession = _ortEnvironment.createSession(encoderModelData, defaultOptions);
    Log.i(TAG, "Created session with default CPU provider.");
}

// Repeat the same try/catch logic for the decoder session.
```

By following these steps, you will ensure that the new quantized model is not only used but is also run in the most performant way possible on compatible hardware, while remaining robust on devices where hardware acceleration is not available.
