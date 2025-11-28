# AlphaBS ONNX Model: Benchmark and Integration Plan

This document provides a thorough plan for benchmarking, implementing, and maximizing the performance of the new statically quantized ONNX models found in `assets/models/bs/`.

## 1. Executive Summary & Research Findings

The goal is to replace the old float32 models with the new `int8` quantized models to significantly improve swipe prediction latency and reduce power consumption. Your request to investigate hardware acceleration beyond the basic NNAPI implementation is critical for achieving this.

My research confirms the following:

1.  **NNAPI is Not Deprecated:** The Android Neural Networks API (NNAPI) is the standard, actively supported method for hardware acceleration on Android. It acts as a dispatcher, sending computation to the best available hardware (NPU, GPU, or DSP).
2.  **Execution Provider (EP) Options:** For this project, the most relevant EPs are:
    *   **Default CPU:** The baseline.
    *   **XNNPACK:** A highly optimized CPU library for ARM, often faster than the default CPU provider.
    *   **NNAPI:** The standard hardware acceleration API. The best choice if the device has a compatible NPU/GPU/DSP and the model's operators are supported.
    *   **QNN (Qualcomm Neural Network):** A specialized provider for Snapdragon chips, targeting the Hexagon DSP. On supported Qualcomm devices, this is likely the **fastest possible option**. The previous codebase already hinted at its use.
3.  **The Path to Maximum Performance:** For quantized `int8` models, the greatest speedups come from offloading the computation from the CPU to a dedicated NPU or DSP. Therefore, **prioritizing QNN and NNAPI is essential.**

## 2. Proposed Benchmark Methodology

To get definitive data, we will create a standalone Java command-line test harness that can be run directly on an Android device via ADB. This will benchmark each execution provider under identical conditions.

### 2.1. Test Harness: `OnnxBenchmark.java`

This new Java class will be responsible for loading the models and running the benchmark.

**Structure:**
*   A `main` method that accepts a command-line argument (`cpu`, `xnnpack`, `nnapi`, `qnn`) to select the test case.
*   Separate methods to generate the specific `OrtSession.SessionOptions` for each execution provider.
*   A `runInferenceLoop` method to execute the encoder/decoder models 100+ times and calculate the average inference time.
*   Dummy input data (a sample swipe trajectory) to ensure consistent test conditions.

### 2.2. Build & Run Script: `run_benchmark.sh`

This script will automate the test process:
1.  **Compile:** Compile `OnnxBenchmark.java` and package it into a `.dex` or `.jar` file. (This step can be adapted to your build system, e.g., by creating a dedicated `androidTest`).
2.  **Push to Device:** Use `adb push` to move the compiled test, the ONNX models, and any other dependencies to `/data/local/tmp/`.
3.  **Execute Tests:** Use `adb shell` to run the Java benchmark for each execution provider, printing the results to the console.
    ```shell
    # Example execution
    adb shell app_process -cp /data/local/tmp/benchmark.jar com.example.OnnxBenchmark cpu
    adb shell app_process -cp /data/local/tmp/benchmark.jar com.example.OnnxBenchmark xnnpack
    adb shell app_process -cp /data/local/tmp/benchmark.jar com.example.OnnxBenchmark nnapi
    adb shell app_process -cp /data/local/tmp/benchmark.jar com.example.OnnxBenchmark qnn
    ```

## 3. Code Skeletons for `OnnxBenchmark.java`

This skeleton provides the core logic and copy-pasteable code for your coworker to build the test harness.

```java
// OnnxBenchmark.java
import ai.onnxruntime.*;
import java.util.HashMap;
import java.util.Map;

public class OnnxBenchmark {

    private static final String ENCODER_PATH = "/data/local/tmp/swipe_encoder_android.onnx";
    private static final String DECODER_PATH = "/data/local/tmp/swipe_decoder_android.onnx";
    private static final int WARMUP_RUNS = 20;
    private static final int BENCHMARK_RUNS = 100;

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("Usage: OnnxBenchmark <cpu|xnnpack|nnapi|qnn>");
            return;
        }
        String provider = args[0];
        try {
            runBenchmark(provider);
        } catch (Exception e) {
            System.err.println("Benchmark failed for provider '" + provider + "': " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runBenchmark(String provider) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options;

        switch (provider) {
            case "xnnpack":
                options = createXnnpackOptions();
                break;
            case "nnapi":
                options = createNnapiOptions();
                break;
            case "qnn":
                options = createQnnOptions();
                break;
            case "cpu":
            default:
                options = new OrtSession.SessionOptions();
                break;
        }

        System.out.println("---- Benchmarking Provider: " + provider + " ----");

        // Create sessions
        OrtSession encoderSession = env.createSession(ENCODER_PATH, options);
        OrtSession decoderSession = env.createSession(DECODER_PATH, options);

        System.out.println("Encoder Provider: " + encoderSession.getExecutionProviderType());
        System.out.println("Decoder Provider: " + decoderSession.getExecutionProviderType());

        // Create dummy input tensors (replace with real shapes)
        // Map<String, OnnxTensor> dummyInputs = createDummyInputs(env);
        
        // Warmup runs
        // for (int i = 0; i < WARMUP_RUNS; i++) {
        //     encoderSession.run(dummyInputs);
        // }

        // Benchmark runs
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_RUNS; i++) {
            long startTime = System.nanoTime();
            // OrtSession.Result encoderResult = encoderSession.run(dummyInputs);
            // Run decoder here as well...
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }

        double averageTimeMs = (totalTime / (double) BENCHMARK_RUNS) / 1_000_000.0;
        System.out.printf("Average Inference Time: %.4f ms%n", averageTimeMs);
        System.out.println("--------------------------------------\n");

        options.close();
        encoderSession.close();
        decoderSession.close();
        env.close();
    }

    // ---

    private static OrtSession.SessionOptions createXnnpackOptions() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // Enable XNNPACK with recommended threading options
        options.addXNNPACK(0); // Use 0 for default options
        options.addConfigEntry("session.intra_op.allow_spinning", "0");
        return options;
    }

    private static OrtSession.SessionOptions createNnapiOptions() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // Force NNAPI to be used, and fail if not possible.
        // For production, you might remove CPU_DISABLED to allow fallback.
        int nnapiFlags = NnapiFlags.NNAPI_FLAG_USE_FP16 | NnapiFlags.NNAPI_FLAG_CPU_DISABLED;
        options.addNnapi(nnapiFlags);
        return options;
    }

    private static OrtSession.SessionOptions createQnnOptions() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        Map<String, String> qnnOptions = new HashMap<>();
        // Use the Hexagon Tensor Processor (NPU) backend
        qnnOptions.put("backend_path", "libQnnHtp.so");
        qnnOptions.put("htp_performance_mode", "burst");
        
        // This is the modern way to add the QNN provider
        options.addExecutionProvider("QNN", qnnOptions);
        return options;
    }
}
```

## 4. Estimated Performance Gains

Based on research and typical results for quantized models, here are the estimated speedups you can expect compared to the **current float32 CPU implementation**.

*   **XNNPACK (Optimized CPU):**
    *   **Estimated Speedup: 1.5x - 2.5x**
    *   **Analysis:** XNNPACK is highly efficient for `int8` operations on ARM CPUs. It will provide a reliable, significant performance boost on nearly all devices.

*   **NNAPI (via NPU/GPU/DSP):**
    *   **Estimated Speedup: 3x - 10x**
    *   **Analysis:** This is the ideal scenario. If the device has a capable accelerator and the model's operators are fully supported, NNAPI will offload the work from the CPU for a dramatic speed increase and power savings. Performance is device-dependent.

*   **QNN (via Hexagon DSP/NPU):**
    *   **Estimated Speedup: 3x - 12x**
    *   **Analysis:** On supported Qualcomm Snapdragon devices, this is the most direct and likely the fastest path. It uses Qualcomm's own libraries to run the model on the Hexagon DSP, which is designed for this exact type of workload.

This structured benchmark will provide the hard data needed to choose the best execution provider and confidently implement it in the main application.
