package juloo.keyboard2;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import ai.onnxruntime.NnapiFlags;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import android.util.Log;

/**
 * Standalone benchmark for ONNX Execution Providers.
 * This test runs on-device and measures the inference speed of the swipe models
 * using different hardware acceleration backends (CPU, XNNPACK, NNAPI, QNN).
 */
@RunWith(AndroidJUnit4.class)
public class OnnxBenchmarkTest {

    private static final String TAG = "OnnxBenchmarkTest";
    private static final String ENCODER_ASSET_PATH = "models/swipe_encoder_android.onnx";
    private static final String DECODER_ASSET_PATH = "models/swipe_decoder_android.onnx";

    private static final int MAX_SEQUENCE_LENGTH = 250;
    private static final int TRAJECTORY_FEATURES = 6;
    private static final int ACTUAL_LENGTH = 100; // Realistic length for a swipe

    private static final int WARMUP_RUNS = 10;
    private static final int BENCHMARK_RUNS = 50;

    private Context context;
    private OrtEnvironment env;
    private byte[] encoderModelBytes;
    private byte[] decoderModelBytes;
    private Map<String, OnnxTensor> dummyEncoderInputs;

    @Before
    public void setUp() throws Exception {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        env = OrtEnvironment.getEnvironment();
        encoderModelBytes = loadModelFromAssets(ENCODER_ASSET_PATH);
        decoderModelBytes = loadModelFromAssets(DECODER_ASSET_PATH);
        dummyEncoderInputs = createDummyInputs(env);
    }

    private byte[] loadModelFromAssets(String path) throws java.io.IOException {
        java.io.InputStream inputStream = context.getAssets().open(path);
        byte[] modelData = new byte[inputStream.available()];
        inputStream.read(modelData);
        inputStream.close();
        return modelData;
    }

    private Map<String, OnnxTensor> createDummyInputs(OrtEnvironment env) throws OrtException {
        // 1. Create trajectory_features: {1, 250, 6} of type float
        FloatBuffer trajectoryBuffer = FloatBuffer.allocate(1 * MAX_SEQUENCE_LENGTH * TRAJECTORY_FEATURES);
        // Fill with some non-zero data
        for (int i = 0; i < trajectoryBuffer.capacity(); i++) {
            trajectoryBuffer.put(i, (float)Math.random());
        }
        long[] trajectoryShape = {1, MAX_SEQUENCE_LENGTH, TRAJECTORY_FEATURES};
        OnnxTensor trajectoryTensor = OnnxTensor.createTensor(env, trajectoryBuffer, trajectoryShape);

        // 2. Create nearest_keys: {1, 250} of type int32
        IntBuffer keysBuffer = IntBuffer.allocate(1 * MAX_SEQUENCE_LENGTH);
        for (int i = 0; i < keysBuffer.capacity(); i++) {
            keysBuffer.put(i, (int)(Math.random() * 26) + 4); // Random alphabet tokens
        }
        long[] keysShape = {1, MAX_SEQUENCE_LENGTH};
        OnnxTensor keysTensor = OnnxTensor.createTensor(env, keysBuffer, keysShape);

        // 3. Create actual_length: {1} of type int32
        OnnxTensor lengthTensor = OnnxTensor.createTensor(env, new int[]{ACTUAL_LENGTH});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("trajectory_features", trajectoryTensor);
        inputs.put("nearest_keys", keysTensor);
        inputs.put("actual_length", lengthTensor);

        return inputs;
    }

    @Test
    public void benchmarkDefaultCpu() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        runBenchmark("DefaultCPU", options);
    }

    @Test
    public void benchmarkXnnpack() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addXNNPACK(0); // 0 for default options
        options.addConfigEntry("session.intra_op.allow_spinning", "0");
        runBenchmark("XNNPACK", options);
    }

    @Test
    public void benchmarkNnapi() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // Use flags to ensure NNAPI is utilized
        int nnapiFlags = NnapiFlags.NNAPI_FLAG_USE_FP16;
        // The below flag is for strict testing. If session creation fails, it means
        // the model has operators not supported by the device's NNAPI driver.
        // nnapiFlags |= NnapiFlags.NNAPI_FLAG_CPU_DISABLED;
        try {
            options.addNnapi(nnapiFlags);
            runBenchmark("NNAPI", options);
        } catch (Exception e) {
            Log.e(TAG, "NNAPI provider is not available or model is not supported. Skipping benchmark.", e);
        }
    }
    
    @Test
    public void benchmarkQnn() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        Map<String, String> qnnOptions = new HashMap<>();
        // Use the Hexagon Tensor Processor (NPU) backend
        qnnOptions.put("backend_path", "libQnnHtp.so");
        qnnOptions.put("htp_performance_mode", "burst");

        try {
            options.addExecutionProvider("QNN", qnnOptions);
            runBenchmark("QNN-HTP", options);
        } catch (Exception e) {
            Log.e(TAG, "QNN provider is not available or failed to initialize. Skipping benchmark.", e);
        }
    }

    private void runBenchmark(String providerName, OrtSession.SessionOptions options) throws OrtException {
        Log.i(TAG, "---- Starting Benchmark For Provider: " + providerName + " ----");

        OrtSession encoderSession = null;
        try {
            encoderSession = env.createSession(encoderModelBytes, options);
        } catch (Exception e) {
            Log.e(TAG, "Failed to create session for " + providerName + ". Error: " + e.getMessage());
            Log.i(TAG, "---- Benchmark FAILED for " + providerName + " ----");
            return;
        }
        
        Log.i(TAG, "Execution Provider: " + encoderSession.getExecutionProviderType());
        Log.i(TAG, "Warming up...");
        for (int i = 0; i < WARMUP_RUNS; i++) {
            encoderSession.run(dummyEncoderInputs).close();
        }

        Log.i(TAG, "Running benchmark...");
        List<Long> timings = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_RUNS; i++) {
            long startTime = System.nanoTime();
            OrtSession.Result result = encoderSession.run(dummyEncoderInputs);
            long endTime = System.nanoTime();
            timings.add(endTime - startTime);
            result.close(); // Close result to free memory
        }

        // Calculate and print stats
        double totalTime = timings.stream().mapToLong(Long::longValue).sum();
        double averageTimeMs = (totalTime / BENCHMARK_RUNS) / 1_000_000.0;
        
        Log.i(TAG, "---- Results for Provider: " + providerName + " ----");
        Log.i(TAG, String.format("Average Inference Time: %.4f ms", averageTimeMs));
        Log.i(TAG, "------------------------------------------\n");

        encoderSession.close();
        options.close();
    }
}
