package juloo.keyboard2.contextaware

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for BigramEntry data class.
 *
 * Tests:
 * - Probability calculation
 * - Word normalization
 * - Entry matching
 * - String representation
 */
class BigramEntryTest {

    @Test
    fun testProbabilityCalculation() {
        // Test standard probability calculation
        val prob = BigramEntry.calculateProbability(25, 100)
        assertEquals(0.25f, prob, 0.001f)
    }

    @Test
    fun testProbabilityWithZeroTotal() {
        // Division by zero should return 0
        val prob = BigramEntry.calculateProbability(10, 0)
        assertEquals(0f, prob, 0.001f)
    }

    @Test
    fun testProbabilityBoundaries() {
        // Test edge cases
        assertEquals(1.0f, BigramEntry.calculateProbability(100, 100), 0.001f)  // 100%
        assertEquals(0.0f, BigramEntry.calculateProbability(0, 100), 0.001f)    // 0%
        assertEquals(0.5f, BigramEntry.calculateProbability(50, 100), 0.001f)   // 50%
    }

    @Test
    fun testWordNormalization() {
        // Test case normalization
        assertEquals("hello", BigramEntry.normalizeWord("HELLO"))
        assertEquals("hello", BigramEntry.normalizeWord("Hello"))
        assertEquals("hello", BigramEntry.normalizeWord("hello"))
    }

    @Test
    fun testWordNormalizationWithWhitespace() {
        // Test whitespace trimming
        assertEquals("hello", BigramEntry.normalizeWord("  hello  "))
        assertEquals("hello world", BigramEntry.normalizeWord("  hello world  "))
    }

    @Test
    fun testEntryCreation() {
        val entry = BigramEntry(
            word1 = "the",
            word2 = "cat",
            frequency = 42,
            probability = 0.15f
        )

        assertEquals("the", entry.word1)
        assertEquals("cat", entry.word2)
        assertEquals(42, entry.frequency)
        assertEquals(0.15f, entry.probability, 0.001f)
    }

    @Test
    fun testEntryMatching_Exact() {
        val entry = BigramEntry("hello", "world", 10, 0.5f)

        assertTrue(entry.matches("hello", "world"))
    }

    @Test
    fun testEntryMatching_CaseInsensitive() {
        val entry = BigramEntry("hello", "world", 10, 0.5f)

        assertTrue(entry.matches("HELLO", "WORLD"))
        assertTrue(entry.matches("Hello", "World"))
        assertTrue(entry.matches("HeLLo", "WoRLd"))
    }

    @Test
    fun testEntryMatching_WithWhitespace() {
        val entry = BigramEntry("hello", "world", 10, 0.5f)

        assertTrue(entry.matches("  hello  ", "  world  "))
    }

    @Test
    fun testEntryMatching_Negative() {
        val entry = BigramEntry("hello", "world", 10, 0.5f)

        assertFalse(entry.matches("hello", "universe"))
        assertFalse(entry.matches("goodbye", "world"))
        assertFalse(entry.matches("foo", "bar"))
    }

    @Test
    fun testToString() {
        val entry = BigramEntry("I", "am", 50, 0.25f)
        val string = entry.toString()

        assertTrue(string.contains("I"))
        assertTrue(string.contains("am"))
        assertTrue(string.contains("50"))
        assertTrue(string.contains("25"))  // 25% probability
    }

    @Test
    fun testDataClassCopy() {
        val original = BigramEntry("test", "word", 10, 0.1f)
        val modified = original.copy(frequency = 20, probability = 0.2f)

        // Original unchanged
        assertEquals(10, original.frequency)
        assertEquals(0.1f, original.probability, 0.001f)

        // Modified has new values
        assertEquals(20, modified.frequency)
        assertEquals(0.2f, modified.probability, 0.001f)

        // Words unchanged
        assertEquals("test", modified.word1)
        assertEquals("word", modified.word2)
    }

    @Test
    fun testDataClassEquality() {
        val entry1 = BigramEntry("hello", "world", 10, 0.5f)
        val entry2 = BigramEntry("hello", "world", 10, 0.5f)
        val entry3 = BigramEntry("hello", "world", 20, 0.5f)

        assertEquals(entry1, entry2)
        assertNotEquals(entry1, entry3)
    }

    @Test
    fun testRealWorldExample() {
        // "I" → "am" appears 15 out of 60 times after "I"
        val entry = BigramEntry(
            word1 = "I",
            word2 = "am",
            frequency = 15,
            probability = BigramEntry.calculateProbability(15, 60)
        )

        assertEquals("I", entry.word1)
        assertEquals("am", entry.word2)
        assertEquals(15, entry.frequency)
        assertEquals(0.25f, entry.probability, 0.001f)  // 25%

        assertTrue(entry.matches("i", "AM"))  // Case-insensitive
    }

    @Test
    fun testCommonPhraseExample() {
        // "want" → "to" is very common
        val entry = BigramEntry(
            word1 = "want",
            word2 = "to",
            frequency = 89,
            probability = 0.67f
        )

        assertEquals(0.67f, entry.probability, 0.001f)  // 67% probability
        assertTrue(entry.matches("WANT", "TO"))
    }
}
