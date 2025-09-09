package juloo.keyboard2;

import android.content.Context;
import android.graphics.PointF;
import android.content.SharedPreferences;
import android.content.res.Resources;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * CLI test for neural ONNX prediction system
 * Tests complete pipeline without Android device deployment
 */
public class NeuralPredictionTest 
{
    private MockContext _context;
    private Config _config;
    private NeuralSwipeTypingEngine _neuralEngine;
    
    @Before
    public void setUp() throws Exception 
    {
        System.out.println("=== Neural Prediction CLI Test ===");
        
        // Create mock context for testing
        _context = new MockContext();
        
        // Create basic config
        _config = createTestConfig();
        
        // Initialize neural engine
        _neuralEngine = new NeuralSwipeTypingEngine(_context, _config);
        
        // Set up debug logging
        _neuralEngine.setDebugLogger(message -> System.out.println("[TEST] " + message));
    }
    
    @Test
    public void testNeuralEngineInitialization()
    {
        System.out.println("🔧 Testing neural engine initialization...");
        
        try {
            boolean initialized = _neuralEngine.initialize();
            System.out.println("Neural engine initialized: " + initialized);
            System.out.println("Neural available: " + _neuralEngine.isNeuralAvailable());
            System.out.println("Current mode: " + _neuralEngine.getCurrentMode());
            
            assertTrue("Neural engine should initialize", initialized);
            assertTrue("Neural prediction should be available", _neuralEngine.isNeuralAvailable());
            assertEquals("Should be in neural mode", "neural", _neuralEngine.getCurrentMode());
        }
        catch (Exception e) {
            System.err.println("Neural engine initialization failed: " + e.getMessage());
            e.printStackTrace();
            fail("Neural engine initialization should not throw exceptions");
        }
    }
    
    @Test 
    public void testSwipePrediction()
    {
        System.out.println("🧠 Testing swipe prediction pipeline...");
        
        try {
            // Initialize engine
            _neuralEngine.initialize();
            
            // Set keyboard dimensions
            _neuralEngine.setKeyboardDimensions(1080f, 400f);
            
            // Set key positions for QWERTY layout
            Map<Character, PointF> keyPositions = createQWERTYKeyPositions();
            _neuralEngine.setRealKeyPositions(keyPositions);
            
            // Create sample swipe for word "hello"
            SwipeInput testSwipe = createTestSwipe("hello");
            
            System.out.println("Test swipe created: " + testSwipe.coordinates.size() + " points");
            System.out.println("Key sequence: " + testSwipe.keySequence);
            
            // Run neural prediction
            long startTime = System.currentTimeMillis();
            PredictionResult result = _neuralEngine.predict(testSwipe);
            long endTime = System.currentTimeMillis();
            
            System.out.println("Prediction completed in " + (endTime - startTime) + "ms");
            System.out.println("Predictions: " + result.words.size());
            
            for (int i = 0; i < Math.min(5, result.words.size()); i++) {
                System.out.println("  " + (i + 1) + ". " + result.words.get(i) + 
                    " (score: " + result.scores.get(i) + ")");
            }
            
            // Verify results
            assertNotNull("Prediction result should not be null", result);
            assertNotNull("Words list should not be null", result.words);
            assertNotNull("Scores list should not be null", result.scores);
            
            if (result.words.size() > 0) {
                System.out.println("✅ Neural prediction successful!");
                
                // Check if "hello" is in predictions
                boolean foundTarget = result.words.contains("hello");
                System.out.println("Target word 'hello' found: " + foundTarget);
            } else {
                System.out.println("⚠️ No predictions returned");
            }
        }
        catch (Exception e) {
            System.err.println("💥 Neural prediction failed: " + e.getMessage());
            e.printStackTrace();
            fail("Neural prediction should not throw exceptions: " + e.getMessage());
        }
    }
    
    private Config createTestConfig()
    {
        // Create minimal config for testing
        Config config = new Config();
        config.neural_prediction_enabled = true;
        config.neural_beam_width = 8;
        config.neural_max_length = 35;
        config.neural_confidence_threshold = 0.1f;
        return config;
    }
    
    private Map<Character, PointF> createQWERTYKeyPositions()
    {
        Map<Character, PointF> positions = new HashMap<>();
        
        // Simple QWERTY layout positions (normalized to 1080x400 screen)
        String[] row1 = {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"};
        String[] row2 = {"a", "s", "d", "f", "g", "h", "j", "k", "l"};
        String[] row3 = {"z", "x", "c", "v", "b", "n", "m"};
        
        float keyWidth = 1080f / 10f; // 108px per key
        float rowHeight = 400f / 4f;  // 100px per row
        
        // Row 1 (q-p)
        for (int i = 0; i < row1.length; i++) {
            char key = row1[i].charAt(0);
            float x = i * keyWidth + keyWidth/2;
            float y = rowHeight/2;
            positions.put(key, new PointF(x, y));
        }
        
        // Row 2 (a-l) - offset by half key
        float rowOffset = keyWidth * 0.5f;
        for (int i = 0; i < row2.length; i++) {
            char key = row2[i].charAt(0);
            float x = rowOffset + i * keyWidth + keyWidth/2;
            float y = rowHeight + rowHeight/2;
            positions.put(key, new PointF(x, y));
        }
        
        // Row 3 (z-m) - offset by full key
        rowOffset = keyWidth;
        for (int i = 0; i < row3.length; i++) {
            char key = row3[i].charAt(0);
            float x = rowOffset + i * keyWidth + keyWidth/2;
            float y = 2 * rowHeight + rowHeight/2;
            positions.put(key, new PointF(x, y));
        }
        
        return positions;
    }
    
    private SwipeInput createTestSwipe(String word)
    {
        Map<Character, PointF> keyPositions = createQWERTYKeyPositions();
        List<PointF> coordinates = new ArrayList<>();
        List<Long> timestamps = new ArrayList<>();
        long startTime = System.currentTimeMillis();
        
        // Create swipe path through each letter of the word
        for (int i = 0; i < word.length(); i++) {
            char letter = word.charAt(i);
            PointF keyPos = keyPositions.get(letter);
            
            if (keyPos != null) {
                // Add some points around each key to simulate swipe
                for (int j = 0; j < 5; j++) {
                    float x = keyPos.x + (float)(Math.random() * 20 - 10); // Add some randomness
                    float y = keyPos.y + (float)(Math.random() * 20 - 10);
                    coordinates.add(new PointF(x, y));
                    timestamps.add(startTime + i * 100 + j * 20); // 20ms between points
                }
            }
        }
        
        return new SwipeInput(coordinates, timestamps, new ArrayList<>());
    }
    
    /**
     * Mock context for testing ONNX system without Android
     */
    private static class MockContext extends Context
    {
        @Override
        public android.content.res.AssetManager getAssets() {
            // Return real asset manager to load ONNX models
            try {
                // This is a hack to get assets in test environment
                return super.getAssets();
            } catch (Exception e) {
                throw new RuntimeException("Cannot access assets in test environment: " + e.getMessage());
            }
        }
        
        // Minimal overrides for neural engine
        @Override public String getPackageName() { return "test.neural"; }
        @Override public Resources getResources() { return null; }
        @Override public Object getSystemService(String name) { return null; }
        @Override public String getString(int resId) { return "test"; }
        @Override public int checkCallingOrSelfPermission(String permission) { return 0; }
    }
}