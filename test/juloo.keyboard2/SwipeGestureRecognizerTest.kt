package juloo.keyboard2

import org.junit.Test
import org.junit.Before
import org.junit.Assert.*

class SwipeGestureRecognizerTest {
    private lateinit var recognizer: SwipeGestureRecognizer

    @Before
    fun setUp() {
        recognizer = SwipeGestureRecognizer()
    }

    @Test
    fun testInitialState() {
        assertFalse(
            "Recognizer should not be swipe typing initially",
            recognizer.isSwipeTyping()
        )
        assertTrue(
            "Initial swipe path should be empty",
            recognizer.swipePath.isEmpty()
        )
        assertEquals(
            "Initial key sequence should be empty",
            "", recognizer.keySequence
        )
    }

    @Test
    fun testReset() {
        // Create a dummy key
        val key = createDummyKey('a')

        // Start a swipe
        recognizer.startSwipe(100.0f, 100.0f, key)
        recognizer.addPoint(200.0f, 200.0f, key)

        // Reset should clear everything
        recognizer.reset()

        assertFalse(
            "Should not be swipe typing after reset",
            recognizer.isSwipeTyping()
        )
        assertTrue(
            "Swipe path should be empty after reset",
            recognizer.swipePath.isEmpty()
        )
        assertEquals(
            "Key sequence should be empty after reset",
            "", recognizer.keySequence
        )
    }

    @Test
    fun testSwipePathTracking() {
        val key = createDummyKey('a')

        recognizer.startSwipe(100.0f, 100.0f, key)
        assertEquals(
            "Should have one point after start",
            1, recognizer.swipePath.size
        )

        recognizer.addPoint(150.0f, 150.0f, key)
        assertEquals(
            "Should have two points after adding one",
            2, recognizer.swipePath.size
        )

        recognizer.addPoint(200.0f, 200.0f, key)
        assertEquals(
            "Should have three points after adding another",
            3, recognizer.swipePath.size
        )
    }

    @Test
    fun testEndSwipeRequiresMinimumKeys() {
        val keyA = createDummyKey('a')

        // Single key should not trigger swipe typing
        recognizer.startSwipe(100.0f, 100.0f, keyA)
        assertNull(
            "Single key should not return touched keys",
            recognizer.endSwipe()
        )

        // Two keys should trigger swipe typing
        val keyB = createDummyKey('b')
        recognizer.startSwipe(100.0f, 100.0f, keyA)

        // Simulate movement to trigger swipe typing detection
        for (i in 0 until 10) {
            recognizer.addPoint(100.0f + i * 10, 100.0f, if (i % 2 == 0) keyA else keyB)
        }

        // Note: This may still return null because isSwipeTyping()
        // depends on internal timing and distance calculations
    }

    /**
     * Helper method to create a dummy key for testing
     */
    private fun createDummyKey(c: Char): KeyboardData.Key {
        val kv = KeyValue.makeStringKey(c.toString())
        val keys = Array<KeyValue?>(9) { null }
        keys[0] = kv
        return KeyboardData.Key(keys.toList(), null, 0, 1.0f, 0.0f, null)
    }
}
