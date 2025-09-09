/**
 * Standalone Java CLI test for ONNX neural transformer
 * Tests complete pipeline without Android dependencies
 */

import ai.onnxruntime.*;
import java.io.*;
import java.nio.*;
import java.util.*;

public class TestOnnxDirect {
    
    // Web demo constants
    private static final int MAX_SEQUENCE_LENGTH = 150;
    private static final int DECODER_SEQ_LENGTH = 20;
    private static final int VOCAB_SIZE = 30;
    private static final int PAD_IDX = 0, UNK_IDX = 1, SOS_IDX = 2, EOS_IDX = 3;
    
    public static void main(String[] args) {
        System.out.println("🧪 ONNX Neural Transformer CLI Test");
        System.out.println("===================================");
        
        try {
            // Test complete pipeline
            boolean success = testNeuralPipeline();
            System.out.println(success ? "✅ Neural pipeline test PASSED" : "❌ Neural pipeline test FAILED");
            System.exit(success ? 0 : 1);
        } catch (Exception e) {
            System.err.println("💥 Test failed with exception: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static boolean testNeuralPipeline() throws Exception {
        System.out.println("\n🔄 Loading ONNX models...");
        
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        
        // Load models
        byte[] encoderData = loadModelData("assets/models/swipe_model_character_quant.onnx");
        byte[] decoderData = loadModelData("assets/models/swipe_decoder_character_quant.onnx");
        
        System.out.println("📥 Encoder: " + encoderData.length + " bytes");
        System.out.println("📥 Decoder: " + decoderData.length + " bytes");
        
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession encoderSession = env.createSession(encoderData, options);
        OrtSession decoderSession = env.createSession(decoderData, options);
        
        System.out.println("✅ Sessions created");
        System.out.println("   Encoder inputs: " + encoderSession.getInputNames());
        System.out.println("   Decoder inputs: " + decoderSession.getInputNames());
        
        // Create test inputs (simulating swipe for "hello")
        Map<String, OnnxTensor> encoderInputs = createTestEncoderInputs(env);
        
        System.out.println("\n🚀 Running encoder...");
        OrtSession.Result encoderResult = encoderSession.run(encoderInputs);
        OnnxTensor memory = (OnnxTensor) encoderResult.get(0);
        
        System.out.println("✅ Encoder successful");
        System.out.println("   Memory shape: " + Arrays.toString(memory.getInfo().getShape()));
        
        // Test single decoder step
        System.out.println("\n🔍 Testing decoder step...");
        List<String> predictions = testDecoderStep(env, decoderSession, memory);
        
        System.out.println("🎯 Predictions: " + predictions.size());
        for (int i = 0; i < Math.min(5, predictions.size()); i++) {
            System.out.println("   " + (i+1) + ". " + predictions.get(i));
        }
        
        // Cleanup
        encoderInputs.values().forEach(OnnxTensor::close);
        encoderResult.close();
        encoderSession.close();
        decoderSession.close();
        
        return predictions.size() > 0;
    }
    
    private static Map<String, OnnxTensor> createTestEncoderInputs(OrtEnvironment env) throws OrtException {
        System.out.println("🔧 Creating test encoder inputs...");
        
        // Create trajectory features [1, 150, 6]
        ByteBuffer trajBuffer = ByteBuffer.allocateDirect(MAX_SEQUENCE_LENGTH * 6 * 4);
        trajBuffer.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = trajBuffer.asFloatBuffer();
        
        // Simple test trajectory (horizontal swipe)
        for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
            if (i < 50) { // 50 real points
                floatBuffer.put(i / 50.0f);  // x: 0.0 to 1.0
                floatBuffer.put(0.5f);       // y: middle
                floatBuffer.put(i > 0 ? 0.02f : 0.0f);  // vx
                floatBuffer.put(0.0f);       // vy
                floatBuffer.put(0.0f);       // ax
                floatBuffer.put(0.0f);       // ay
            } else {
                // Padding
                for (int j = 0; j < 6; j++) floatBuffer.put(0.0f);
            }
        }
        floatBuffer.rewind();
        
        // Create nearest keys [1, 150]
        ByteBuffer keysBuffer = ByteBuffer.allocateDirect(MAX_SEQUENCE_LENGTH * 8);
        keysBuffer.order(ByteOrder.nativeOrder());
        LongBuffer longBuffer = keysBuffer.asLongBuffer();
        
        // Test sequence: h-e-l-l-o
        char[] letters = {'h', 'e', 'l', 'l', 'o'};
        for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++) {
            if (i < 50) {
                // Map letters cyclically
                char letter = letters[i % letters.length];
                longBuffer.put(letter - 'a' + 4); // Map to token indices 4-29
            } else {
                longBuffer.put(PAD_IDX); // Padding
            }
        }
        longBuffer.rewind();
        
        // Create source mask [1, 150] - boolean array
        boolean[][] srcMaskData = new boolean[1][MAX_SEQUENCE_LENGTH];
        for (int i = 50; i < MAX_SEQUENCE_LENGTH; i++) {
            srcMaskData[0][i] = true; // Mask padding positions
        }
        
        System.out.println("   Trajectory: [1, 150, 6] - 50 real points + 100 padded");
        System.out.println("   Keys: [1, 150] - h-e-l-l-o pattern");  
        System.out.println("   Mask: [1, 150] - 50 valid + 100 masked");
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("trajectory_features", OnnxTensor.createTensor(env, floatBuffer, new long[]{1, MAX_SEQUENCE_LENGTH, 6}));
        inputs.put("nearest_keys", OnnxTensor.createTensor(env, longBuffer, new long[]{1, MAX_SEQUENCE_LENGTH}));
        inputs.put("src_mask", OnnxTensor.createTensor(env, srcMaskData));
        
        return inputs;
    }
    
    private static List<String> testDecoderStep(OrtEnvironment env, OrtSession decoderSession, OnnxTensor memory) throws OrtException {
        System.out.println("🔍 Testing single decoder step with SOS token...");
        
        // Create decoder inputs for first step
        long[] targetTokens = new long[DECODER_SEQ_LENGTH];
        targetTokens[0] = SOS_IDX; // Start with SOS
        // Rest are padding (0)
        
        ByteBuffer tokensBuffer = ByteBuffer.allocateDirect(DECODER_SEQ_LENGTH * 8);
        tokensBuffer.order(ByteOrder.nativeOrder());
        LongBuffer longBuffer = tokensBuffer.asLongBuffer();
        for (long token : targetTokens) {
            longBuffer.put(token);
        }
        longBuffer.rewind();
        
        // Target mask: mask all positions after SOS
        boolean[][] targetMaskData = new boolean[1][DECODER_SEQ_LENGTH];
        for (int i = 1; i < DECODER_SEQ_LENGTH; i++) {
            targetMaskData[0][i] = true; // Mask positions after SOS
        }
        
        // Source mask: no masking (all encoder positions valid)
        int srcMaskSize = (int) memory.getInfo().getShape()[1];
        boolean[][] srcMaskData = new boolean[1][srcMaskSize];
        // All false (no masking)
        
        System.out.println("   Target tokens: [1, 20] - SOS + padding");
        System.out.println("   Target mask: [1, 20] - 1 valid + 19 masked");
        System.out.println("   Source mask: [1, " + srcMaskSize + "] - all valid");
        
        Map<String, OnnxTensor> decoderInputs = new HashMap<>();
        decoderInputs.put("memory", memory);
        decoderInputs.put("target_tokens", OnnxTensor.createTensor(env, longBuffer, new long[]{1, DECODER_SEQ_LENGTH}));
        decoderInputs.put("target_mask", OnnxTensor.createTensor(env, targetMaskData));
        decoderInputs.put("src_mask", OnnxTensor.createTensor(env, srcMaskData));
        
        // Run decoder
        OrtSession.Result decoderResult = decoderSession.run(decoderInputs);
        OnnxTensor logitsTensor = (OnnxTensor) decoderResult.get(0);
        
        System.out.println("✅ Decoder step successful");
        System.out.println("   Logits shape: " + Arrays.toString(logitsTensor.getInfo().getShape()));
        
        // Extract first token predictions (position 0)
        float[] logitsFlat = (float[]) logitsTensor.getValue();
        System.out.println("   Logits flat length: " + logitsFlat.length);
        
        // Get logits for position 0 (after SOS token)
        int startIdx = 0 * VOCAB_SIZE;
        float[] firstTokenLogits = new float[VOCAB_SIZE];
        System.arraycopy(logitsFlat, startIdx, firstTokenLogits, 0, VOCAB_SIZE);
        
        // Apply softmax
        float[] probs = softmax(firstTokenLogits);
        
        // Get top 5 tokens
        int[] topTokens = getTopKIndices(probs, 5);
        
        List<String> words = new ArrayList<>();
        for (int tokenIdx : topTokens) {
            char letter = (char) ('a' + tokenIdx - 4); // Convert back to letter
            if (tokenIdx >= 4 && tokenIdx <= 29) { // Valid letter range
                words.add(String.valueOf(letter));
            }
        }
        
        System.out.println("   Top tokens: " + Arrays.toString(topTokens));
        System.out.println("   As letters: " + words);
        
        // Cleanup local tensors
        for (OnnxTensor tensor : decoderInputs.values()) {
            if (tensor != memory) { // Don't close memory, it's from encoder
                tensor.close();
            }
        }
        decoderResult.close();
        
        return words;
    }
    
    private static float[] softmax(float[] logits) {
        // Find max manually (Java 8 compatibility)
        float maxLogit = logits[0];
        for (float logit : logits) {
            if (logit > maxLogit) maxLogit = logit;
        }
        
        float[] expScores = new float[logits.length];
        float sum = 0.0f;
        
        for (int i = 0; i < logits.length; i++) {
            expScores[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += expScores[i];
        }
        
        for (int i = 0; i < expScores.length; i++) {
            expScores[i] /= sum;
        }
        
        return expScores;
    }
    
    private static int[] getTopKIndices(float[] array, int k) {
        List<IndexValue> indexed = new ArrayList<>();
        for (int i = 0; i < array.length; i++) {
            indexed.add(new IndexValue(i, array[i]));
        }
        
        indexed.sort((a, b) -> Float.compare(b.value, a.value));
        
        int[] result = new int[Math.min(k, indexed.size())];
        for (int i = 0; i < result.length; i++) {
            result[i] = indexed.get(i).index;
        }
        return result;
    }
    
    private static class IndexValue {
        int index;
        float value;
        IndexValue(int i, float v) { index = i; value = v; }
    }
    
    private static byte[] loadModelData(String path) throws IOException {
        File file = new File(path);
        if (!file.exists()) {
            throw new FileNotFoundException("Model not found: " + path);
        }
        
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int) file.length()];
        fis.read(data);
        fis.close();
        
        return data;
    }
}