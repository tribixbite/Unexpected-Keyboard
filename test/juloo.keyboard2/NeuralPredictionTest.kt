package juloo.keyboard2

import android.content.Context
import android.graphics.PointF
import android.content.res.Resources
import org.junit.Test
import org.junit.Before
import org.junit.Assert.*

/**
 * CLI test for neural ONNX prediction system
 * Tests complete pipeline without Android device deployment
 */
class NeuralPredictionTest {
    private lateinit var _context: MockContext
    private lateinit var _config: Config
    private lateinit var _neuralEngine: NeuralSwipeTypingEngine

    @Before
    fun setUp() {
        println("=== Neural Prediction CLI Test ===")

        // Create mock context for testing
        _context = MockContext()

        // Create basic config
        _config = createTestConfig()

        // Initialize neural engine
        _neuralEngine = NeuralSwipeTypingEngine(_context, _config)

        // Set up debug logging
        _neuralEngine.setDebugLogger { message -> println("[TEST] $message") }
    }

    @Test
    fun testNeuralEngineInitialization() {
        println("🔧 Testing neural engine initialization...")

        try {
            val initialized = _neuralEngine.initialize()
            println("Neural engine initialized: $initialized")
            println("Neural available: ${_neuralEngine.isNeuralAvailable()}")
            println("Current mode: ${_neuralEngine.currentMode}")

            assertTrue("Neural engine should initialize", initialized)
            assertTrue("Neural prediction should be available", _neuralEngine.isNeuralAvailable())
            assertEquals("Should be in neural mode", "neural", _neuralEngine.currentMode)
        } catch (e: Exception) {
            System.err.println("Neural engine initialization failed: ${e.message}")
            e.printStackTrace()
            fail("Neural engine initialization should not throw exceptions")
        }
    }

    @Test
    fun testSwipePrediction() {
        println("🧠 Testing swipe prediction pipeline...")

        try {
            // Initialize engine
            _neuralEngine.initialize()

            // Set keyboard dimensions
            _neuralEngine.setKeyboardDimensions(1080f, 400f)

            // Set key positions for QWERTY layout
            val keyPositions = createQWERTYKeyPositions()
            _neuralEngine.setRealKeyPositions(keyPositions)

            // Create sample swipe for word "hello"
            val testSwipe = createTestSwipe("hello")

            println("Test swipe created: ${testSwipe.coordinates.size} points")
            println("Key sequence: ${testSwipe.keySequence}")

            // Run neural prediction
            val startTime = System.currentTimeMillis()
            val result = _neuralEngine.predict(testSwipe)
            val endTime = System.currentTimeMillis()

            println("Prediction completed in ${endTime - startTime}ms")
            println("Predictions: ${result.words.size}")

            for (i in 0 until minOf(5, result.words.size)) {
                println("  ${i + 1}. ${result.words[i]} (score: ${result.scores[i]})")
            }

            // Verify results
            assertNotNull("Prediction result should not be null", result)
            assertNotNull("Words list should not be null", result.words)
            assertNotNull("Scores list should not be null", result.scores)

            if (result.words.isNotEmpty()) {
                println("✅ Neural prediction successful!")

                // Check if "hello" is in predictions
                val foundTarget = result.words.contains("hello")
                println("Target word 'hello' found: $foundTarget")
            } else {
                println("⚠️ No predictions returned")
            }
        } catch (e: Exception) {
            System.err.println("💥 Neural prediction failed: ${e.message}")
            e.printStackTrace()
            fail("Neural prediction should not throw exceptions: ${e.message}")
        }
    }

    private fun createTestConfig(): Config {
        // Create minimal config for testing
        return Config().apply {
            neural_prediction_enabled = true
            neural_beam_width = 8
            neural_max_length = 35
            neural_confidence_threshold = 0.1f
        }
    }

    private fun createQWERTYKeyPositions(): Map<Char, PointF> {
        val positions = HashMap<Char, PointF>()

        // Simple QWERTY layout positions (normalized to 1080x400 screen)
        val row1 = arrayOf("q", "w", "e", "r", "t", "y", "u", "i", "o", "p")
        val row2 = arrayOf("a", "s", "d", "f", "g", "h", "j", "k", "l")
        val row3 = arrayOf("z", "x", "c", "v", "b", "n", "m")

        val keyWidth = 1080f / 10f // 108px per key
        val rowHeight = 400f / 4f  // 100px per row

        // Row 1 (q-p)
        for (i in row1.indices) {
            val key = row1[i][0]
            val x = i * keyWidth + keyWidth / 2
            val y = rowHeight / 2
            positions[key] = PointF(x, y)
        }

        // Row 2 (a-l) - offset by half key
        var rowOffset = keyWidth * 0.5f
        for (i in row2.indices) {
            val key = row2[i][0]
            val x = rowOffset + i * keyWidth + keyWidth / 2
            val y = rowHeight + rowHeight / 2
            positions[key] = PointF(x, y)
        }

        // Row 3 (z-m) - offset by full key
        rowOffset = keyWidth
        for (i in row3.indices) {
            val key = row3[i][0]
            val x = rowOffset + i * keyWidth + keyWidth / 2
            val y = 2 * rowHeight + rowHeight / 2
            positions[key] = PointF(x, y)
        }

        return positions
    }

    private fun createTestSwipe(word: String): SwipeInput {
        val keyPositions = createQWERTYKeyPositions()
        val coordinates = ArrayList<PointF>()
        val timestamps = ArrayList<Long>()
        val startTime = System.currentTimeMillis()

        // Create swipe path through each letter of the word
        for (i in word.indices) {
            val letter = word[i]
            val keyPos = keyPositions[letter]

            if (keyPos != null) {
                // Add some points around each key to simulate swipe
                for (j in 0 until 5) {
                    val x = keyPos.x + (Math.random() * 20 - 10).toFloat() // Add some randomness
                    val y = keyPos.y + (Math.random() * 20 - 10).toFloat()
                    coordinates.add(PointF(x, y))
                    timestamps.add(startTime + i * 100 + j * 20) // 20ms between points
                }
            }
        }

        return SwipeInput(coordinates, timestamps, ArrayList())
    }

    /**
     * Mock context for testing ONNX system without Android
     */
    private class MockContext : Context() {
        override fun getAssets(): android.content.res.AssetManager {
            // Return real asset manager to load ONNX models
            try {
                // This is a hack to get assets in test environment
                return super.getAssets()
            } catch (e: Exception) {
                throw RuntimeException("Cannot access assets in test environment: ${e.message}")
            }
        }

        // Minimal overrides for neural engine
        override fun getPackageName(): String = "test.neural"
        override fun getResources(): Resources? = null
        override fun getSystemService(name: String): Any? = null
        override fun getString(resId: Int): String = "test"
        override fun checkCallingOrSelfPermission(permission: String): Int = 0
    }
}
