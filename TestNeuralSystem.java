import ai.onnxruntime.*;
import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Standalone CLI test for ONNX neural prediction system
 * Tests transformer model loading and tensor creation
 */
public class TestNeuralSystem 
{
    public static void main(String[] args) 
    {
        System.out.println("=== Neural ONNX System CLI Test ===");
        
        try {
            testOnnxModelLoading();
            testBooleanTensorCreation();
            testTransformerPipeline();
            
            System.out.println("✅ All neural system tests passed!");
        }
        catch (Exception e) {
            System.err.println("💥 Neural system test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testOnnxModelLoading() throws Exception
    {
        System.out.println("\n🔧 Testing ONNX model loading...");
        
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        
        // Test encoder model loading
        String encoderPath = "assets/models/swipe_model_character_quant.onnx";
        File encoderFile = new File(encoderPath);
        
        if (!encoderFile.exists()) {
            System.err.println("❌ Encoder model not found at: " + encoderFile.getAbsolutePath());
            return;
        }
        
        byte[] encoderData = loadModelFile(encoderFile);
        System.out.println("📥 Encoder model loaded: " + encoderData.length + " bytes");
        
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession encoderSession = env.createSession(encoderData, sessionOptions);
        
        System.out.println("✅ Encoder session created");
        System.out.println("   Inputs: " + encoderSession.getInputNames());
        System.out.println("   Outputs: " + encoderSession.getOutputNames());
        
        // Test decoder model loading
        String decoderPath = "assets/models/swipe_decoder_character_quant.onnx";
        File decoderFile = new File(decoderPath);
        
        if (!decoderFile.exists()) {
            System.err.println("❌ Decoder model not found at: " + decoderFile.getAbsolutePath());
            return;
        }
        
        byte[] decoderData = loadModelFile(decoderFile);
        System.out.println("📥 Decoder model loaded: " + decoderData.length + " bytes");
        
        OrtSession decoderSession = env.createSession(decoderData, sessionOptions);
        
        System.out.println("✅ Decoder session created");
        System.out.println("   Inputs: " + decoderSession.getInputNames());
        System.out.println("   Outputs: " + decoderSession.getOutputNames());
        
        // Cleanup
        encoderSession.close();
        decoderSession.close();
    }
    
    private static void testBooleanTensorCreation() throws Exception
    {
        System.out.println("\n🎯 Testing boolean tensor creation...");
        
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        
        // Test 2D boolean array creation
        boolean[][] maskData = new boolean[1][100];
        for (int i = 50; i < 100; i++) {
            maskData[0][i] = true; // Mask second half
        }
        
        OnnxTensor boolTensor = OnnxTensor.createTensor(env, maskData);
        
        System.out.println("✅ Boolean tensor created");
        System.out.println("   Shape: " + java.util.Arrays.toString(boolTensor.getInfo().getShape()));
        System.out.println("   Type: " + boolTensor.getInfo().getType());
        
        // Verify shape is [1, 100]
        long[] expectedShape = {1, 100};
        long[] actualShape = boolTensor.getInfo().getShape();
        
        if (java.util.Arrays.equals(expectedShape, actualShape)) {
            System.out.println("✅ Boolean tensor has correct 2D shape");
        } else {
            throw new RuntimeException("Boolean tensor shape mismatch. Expected: " + 
                java.util.Arrays.toString(expectedShape) + ", Got: " + 
                java.util.Arrays.toString(actualShape));
        }
        
        boolTensor.close();
    }
    
    private static void testTransformerPipeline() throws Exception
    {
        System.out.println("\n🚀 Testing transformer pipeline...");
        
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        
        // Load encoder model
        String encoderPath = "assets/models/swipe_model_character_quant.onnx";
        byte[] encoderData = loadModelFile(new File(encoderPath));
        OrtSession encoderSession = env.createSession(encoderData);
        
        // Create sample input tensors
        System.out.println("Creating sample input tensors...");
        
        // Trajectory features: [1, 100, 6] - sample swipe path
        java.nio.ByteBuffer trajBuffer = java.nio.ByteBuffer.allocateDirect(100 * 6 * 4);
        trajBuffer.order(java.nio.ByteOrder.nativeOrder());
        java.nio.FloatBuffer floatBuffer = trajBuffer.asFloatBuffer();
        
        // Create simple linear swipe path
        for (int i = 0; i < 100; i++) {
            floatBuffer.put(i / 100.0f);  // x: 0.0 to 1.0
            floatBuffer.put(0.5f);        // y: constant middle
            floatBuffer.put(i > 0 ? 0.01f : 0.0f);  // vx: small velocity
            floatBuffer.put(0.0f);        // vy: no vertical movement
            floatBuffer.put(0.0f);        // ax: no acceleration
            floatBuffer.put(0.0f);        // ay: no acceleration
        }
        floatBuffer.rewind();
        
        OnnxTensor trajectoryTensor = OnnxTensor.createTensor(env, floatBuffer, new long[]{1, 100, 6});
        System.out.println("   Trajectory tensor: " + java.util.Arrays.toString(trajectoryTensor.getInfo().getShape()));
        
        // Nearest keys: [1, 100] - sample key sequence
        java.nio.ByteBuffer keysBuffer = java.nio.ByteBuffer.allocateDirect(100 * 8);
        keysBuffer.order(java.nio.ByteOrder.nativeOrder());
        java.nio.LongBuffer longBuffer = keysBuffer.asLongBuffer();
        
        // Simple sequence: h-e-l-l-o (using token indices)
        char[] letters = {'h', 'e', 'l', 'l', 'o'};
        for (int i = 0; i < 100; i++) {
            if (i < letters.length) {
                longBuffer.put(letters[i] - 'a' + 4); // Simple char to token mapping
            } else {
                longBuffer.put(0); // Padding
            }
        }
        longBuffer.rewind();
        
        OnnxTensor nearestKeysTensor = OnnxTensor.createTensor(env, longBuffer, new long[]{1, 100});
        System.out.println("   Nearest keys tensor: " + java.util.Arrays.toString(nearestKeysTensor.getInfo().getShape()));
        
        // Source mask: [1, 100] - boolean mask for padding
        boolean[][] srcMaskData = new boolean[1][100];
        for (int i = 20; i < 100; i++) { // Mask positions 20-99
            srcMaskData[0][i] = true;
        }
        
        OnnxTensor srcMaskTensor = OnnxTensor.createTensor(env, srcMaskData);
        System.out.println("   Source mask tensor: " + java.util.Arrays.toString(srcMaskTensor.getInfo().getShape()));
        
        // Run encoder inference
        System.out.println("Running encoder inference...");
        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("trajectory_features", trajectoryTensor);
        encoderInputs.put("nearest_keys", nearestKeysTensor);
        encoderInputs.put("src_mask", srcMaskTensor);
        
        try (OrtSession.Result encoderResult = encoderSession.run(encoderInputs)) {
            System.out.println("✅ Encoder inference successful!");
            
            OnnxTensor memory = (OnnxTensor) encoderResult.get(0);
            System.out.println("   Memory tensor: " + java.util.Arrays.toString(memory.getInfo().getShape()));
            
            // Test simple decoder inference
            testDecoderInference(env, memory, srcMaskTensor);
        }
        
        // Cleanup
        trajectoryTensor.close();
        nearestKeysTensor.close();
        srcMaskTensor.close();
        encoderSession.close();
    }
    
    private static void testDecoderInference(OrtEnvironment env, OnnxTensor memory, OnnxTensor srcMaskTensor) throws Exception
    {
        System.out.println("\nTesting decoder inference...");
        
        // Load decoder model
        String decoderPath = "assets/models/swipe_decoder_character_quant.onnx";
        byte[] decoderData = loadModelFile(new File(decoderPath));
        OrtSession decoderSession = env.createSession(decoderData);
        
        // Create decoder input tensors
        int decoderSeqLength = 20;
        
        // Target tokens: [1, 20] - start with SOS token
        long[] targetTokens = new long[decoderSeqLength];
        targetTokens[0] = 2; // SOS_IDX
        // Rest are padding (0)
        
        java.nio.ByteBuffer tokensBuffer = java.nio.ByteBuffer.allocateDirect(decoderSeqLength * 8);
        tokensBuffer.order(java.nio.ByteOrder.nativeOrder());
        java.nio.LongBuffer longBuffer = tokensBuffer.asLongBuffer();
        for (long token : targetTokens) {
            longBuffer.put(token);
        }
        longBuffer.rewind();
        
        OnnxTensor targetTokensTensor = OnnxTensor.createTensor(env, longBuffer, new long[]{1, decoderSeqLength});
        
        // Target mask: [1, 20] - mask positions after SOS token
        boolean[][] targetMaskData = new boolean[1][decoderSeqLength];
        for (int i = 1; i < decoderSeqLength; i++) {
            targetMaskData[0][i] = true; // Mask all but first position
        }
        
        OnnxTensor targetMaskTensor = OnnxTensor.createTensor(env, targetMaskData);
        
        // Run decoder inference
        Map<String, OnnxTensor> decoderInputs = new HashMap<>();
        decoderInputs.put("memory", memory);
        decoderInputs.put("target_tokens", targetTokensTensor);
        decoderInputs.put("src_mask", srcMaskTensor);
        decoderInputs.put("target_mask", targetMaskTensor);
        
        System.out.println("Decoder inputs:");
        System.out.println("   memory: " + java.util.Arrays.toString(memory.getInfo().getShape()));
        System.out.println("   target_tokens: " + java.util.Arrays.toString(targetTokensTensor.getInfo().getShape()));
        System.out.println("   src_mask: " + java.util.Arrays.toString(srcMaskTensor.getInfo().getShape()));
        System.out.println("   target_mask: " + java.util.Arrays.toString(targetMaskTensor.getInfo().getShape()));
        
        try (OrtSession.Result decoderResult = decoderSession.run(decoderInputs)) {
            System.out.println("✅ Decoder inference successful!");
            
            OnnxTensor logits = (OnnxTensor) decoderResult.get(0);
            System.out.println("   Logits tensor: " + java.util.Arrays.toString(logits.getInfo().getShape()));
            
            float[] logitsData = (float[]) logits.getValue();
            System.out.println("   Logits data length: " + logitsData.length);
            
            // Show first few logits
            System.out.print("   First 10 logits: ");
            for (int i = 0; i < Math.min(10, logitsData.length); i++) {
                System.out.printf("%.3f ", logitsData[i]);
            }
            System.out.println();
        }
        
        // Cleanup
        targetTokensTensor.close();
        targetMaskTensor.close();
        decoderSession.close();
    }
    
    private static byte[] loadModelFile(File file) throws Exception
    {
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int)file.length()];
        fis.read(data);
        fis.close();
        return data;
    }
}