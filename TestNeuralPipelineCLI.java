/**
 * CLI test using actual neural classes with mock swipe data
 */

import juloo.keyboard2.*;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.content.res.AssetManager;
import android.graphics.PointF;
import java.util.*;
import java.io.*;

public class TestNeuralPipelineCLI {
    
    public static void main(String[] args) {
        System.out.println("🧪 Neural Pipeline CLI Test with Mock Swipe Data");
        System.out.println("=================================================");
        
        try {
            testNeuralPipelineComplete();
            System.out.println("✅ Neural pipeline test completed successfully!");
        } catch (Exception e) {
            System.err.println("💥 Neural pipeline test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testNeuralPipelineComplete() throws Exception {
        System.out.println("🔧 Setting up neural engine with mock context...");
        
        // Create mock context that provides access to assets
        MockContext context = new MockContext();
        
        // Create mock config
        Config config = new Config();
        config.neural_prediction_enabled = true;
        config.neural_beam_width = 8;
        config.neural_max_length = 35;
        config.neural_confidence_threshold = 0.1f;
        
        // Initialize neural engine
        NeuralSwipeTypingEngine neuralEngine = new NeuralSwipeTypingEngine(context, config);
        
        // Set up debug logging
        List<String> debugMessages = new ArrayList<>();
        neuralEngine.setDebugLogger(debugMessages::add);
        
        System.out.println("🚀 Initializing neural engine...");
        boolean initialized = neuralEngine.initialize();
        
        System.out.println("Neural engine initialized: " + initialized);
        System.out.println("Neural available: " + neuralEngine.isNeuralAvailable());
        
        // Print all debug messages from initialization
        System.out.println("\n📋 Initialization Debug Messages:");
        for (String msg : debugMessages) {
            System.out.println("   " + msg);
        }
        
        if (!initialized || !neuralEngine.isNeuralAvailable()) {
            System.err.println("❌ Neural engine failed to initialize");
            return;
        }
        
        // Set keyboard dimensions
        neuralEngine.setKeyboardDimensions(1080f, 400f);
        
        // Create QWERTY key positions
        Map<Character, PointF> keyPositions = createQWERTYLayout();
        neuralEngine.setRealKeyPositions(keyPositions);
        
        // Test with mock swipe data for word "hello"
        System.out.println("\n🌀 Creating mock swipe for word 'hello'...");
        SwipeInput testSwipe = createMockSwipeForWord("hello", keyPositions);
        
        System.out.println("Mock swipe created:");
        System.out.println("   Points: " + testSwipe.coordinates.size());
        System.out.println("   Key sequence: " + testSwipe.keySequence);
        
        // Clear debug messages for prediction test
        debugMessages.clear();
        
        // Run neural prediction
        System.out.println("\n🧠 Running neural prediction...");
        long startTime = System.currentTimeMillis();
        
        PredictionResult result = neuralEngine.predict(testSwipe);
        
        long endTime = System.currentTimeMillis();
        
        System.out.println("Neural prediction completed in " + (endTime - startTime) + "ms");
        
        // Print all debug messages from prediction
        System.out.println("\n📋 Prediction Debug Messages:");
        for (String msg : debugMessages) {
            System.out.println("   " + msg);
        }
        
        // Show results
        System.out.println("\n🎯 Neural Prediction Results:");
        System.out.println("   Total predictions: " + result.words.size());
        
        for (int i = 0; i < Math.min(10, result.words.size()); i++) {
            System.out.println("   " + (i+1) + ". " + result.words.get(i) + 
                " (score: " + result.scores.get(i) + ")");
        }
        
        // Check if target word found
        boolean foundTarget = result.words.contains("hello");
        System.out.println("Target word 'hello' found: " + foundTarget);
        
        if (result.words.size() > 0) {
            System.out.println("✅ Neural predictions generated successfully!");
        } else {
            System.out.println("⚠️ No predictions generated - check debug messages above");
        }
    }
    
    private static Map<Character, PointF> createQWERTYLayout() {
        Map<Character, PointF> positions = new HashMap<>();
        
        // QWERTY layout positions (1080x400 screen)
        String[] row1 = {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"};
        String[] row2 = {"a", "s", "d", "f", "g", "h", "j", "k", "l"};
        String[] row3 = {"z", "x", "c", "v", "b", "n", "m"};
        
        float keyWidth = 1080f / 10f;
        float rowHeight = 400f / 4f;
        
        // Layout rows
        for (int i = 0; i < row1.length; i++) {
            positions.put(row1[i].charAt(0), new PointF(i * keyWidth + keyWidth/2, rowHeight/2));
        }
        
        float offset = keyWidth * 0.5f;
        for (int i = 0; i < row2.length; i++) {
            positions.put(row2[i].charAt(0), new PointF(offset + i * keyWidth + keyWidth/2, rowHeight * 1.5f));
        }
        
        offset = keyWidth;
        for (int i = 0; i < row3.length; i++) {
            positions.put(row3[i].charAt(0), new PointF(offset + i * keyWidth + keyWidth/2, rowHeight * 2.5f));
        }
        
        return positions;
    }
    
    private static SwipeInput createMockSwipeForWord(String word, Map<Character, PointF> keyPositions) {
        List<PointF> coordinates = new ArrayList<>();
        List<Long> timestamps = new ArrayList<>();
        long startTime = System.currentTimeMillis();
        
        // Create swipe path through each letter
        for (int i = 0; i < word.length(); i++) {
            char letter = word.charAt(i);
            PointF keyPos = keyPositions.get(letter);
            
            if (keyPos != null) {
                // Add several points around each key
                for (int j = 0; j < 10; j++) {
                    float x = keyPos.x + (float)(Math.random() * 30 - 15);
                    float y = keyPos.y + (float)(Math.random() * 20 - 10);
                    coordinates.add(new PointF(x, y));
                    timestamps.add(startTime + i * 150 + j * 15);
                }
            }
        }
        
        // Create SwipeInput
        return new SwipeInput(coordinates, timestamps, new ArrayList<>());
    }
    
    /**
     * Minimal mock context for neural testing
     */
    private static class MockContext extends Context {
        @Override
        public AssetManager getAssets() {
            // Return a custom asset manager that can load our ONNX models
            return new MockAssetManager();
        }
        
        @Override public String getPackageName() { return "test.neural"; }
        @Override public Resources getResources() { return null; }
        @Override public Object getSystemService(String name) { return null; }
        @Override public String getString(int resId) { return "test"; }
        @Override public int checkCallingOrSelfPermission(String permission) { return 0; }
    }
    
    /**
     * Mock asset manager that loads ONNX models from file system
     */
    private static class MockAssetManager extends AssetManager {
        @Override
        public InputStream open(String fileName) throws IOException {
            // Map assets to actual file paths
            if (fileName.equals("models/swipe_model_character_quant.onnx")) {
                return new FileInputStream("assets/models/swipe_model_character_quant.onnx");
            } else if (fileName.equals("models/swipe_decoder_character_quant.onnx")) {
                return new FileInputStream("assets/models/swipe_decoder_character_quant.onnx");
            } else if (fileName.equals("models/tokenizer.json")) {
                return new FileInputStream("assets/models/tokenizer.json");
            } else {
                throw new FileNotFoundException("Mock asset not found: " + fileName);
            }
        }
    }
}