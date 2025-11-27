package juloo.keyboard2.contextaware

import android.content.Context
import android.content.SharedPreferences
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.mockito.Mockito.*

/**
 * Unit tests for BigramStore.
 *
 * Tests:
 * - Bigram recording and frequency tracking
 * - Prediction retrieval
 * - Probability queries
 * - Persistence (save/load)
 * - Import/export
 * - Statistics
 * - Pruning and limits
 */
class BigramStoreTest {

    private lateinit var mockContext: Context
    private lateinit var mockPrefs: SharedPreferences
    private lateinit var mockEditor: SharedPreferences.Editor
    private lateinit var store: BigramStore

    @Before
    fun setup() {
        // Mock Android Context and SharedPreferences
        mockContext = mock(Context::class.java)
        mockPrefs = mock(SharedPreferences::class.java)
        mockEditor = mock(SharedPreferences.Editor::class.java)

        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockPrefs.getString(anyString(), anyString())).thenReturn(null)

        store = BigramStore(mockContext)
    }

    @Test
    fun testRecordSingleBigram() {
        store.recordBigram("hello", "world")

        val predictions = store.getPredictions("hello")
        assertEquals(1, predictions.size)
        assertEquals("world", predictions[0].word2)
        assertEquals(1, predictions[0].frequency)
    }

    @Test
    fun testRecordMultipleSameBigram() {
        // Record same bigram 5 times
        repeat(5) {
            store.recordBigram("I", "am")
        }

        val predictions = store.getPredictions("I")
        assertEquals(1, predictions.size)
        assertEquals("am", predictions[0].word2)
        assertEquals(5, predictions[0].frequency)
    }

    @Test
    fun testRecordMultipleDifferentBigrams() {
        store.recordBigram("I", "am")
        store.recordBigram("I", "want")
        store.recordBigram("I", "have")

        val predictions = store.getPredictions("I")
        assertEquals(3, predictions.size)

        val words = predictions.map { it.word2 }.toSet()
        assertTrue(words.contains("am"))
        assertTrue(words.contains("want"))
        assertTrue(words.contains("have"))
    }

    @Test
    fun testPredictionsSortedByProbability() {
        // Record with different frequencies
        repeat(10) { store.recordBigram("the", "cat") }  // 10 times
        repeat(5) { store.recordBigram("the", "dog") }   // 5 times
        repeat(2) { store.recordBigram("the", "bird") }  // 2 times

        val predictions = store.getPredictions("the")

        // Should be sorted: cat (10), dog (5), bird (2)
        assertEquals("cat", predictions[0].word2)
        assertEquals("dog", predictions[1].word2)
        assertEquals("bird", predictions[2].word2)

        // Probabilities should be descending
        assertTrue(predictions[0].probability > predictions[1].probability)
        assertTrue(predictions[1].probability > predictions[2].probability)
    }

    @Test
    fun testGetProbability_Exists() {
        repeat(3) { store.recordBigram("want", "to") }  // 3 occurrences
        store.recordBigram("want", "some")              // 1 occurrence
        // Total "want" occurrences: 4

        val prob = store.getProbability("want", "to")
        assertEquals(0.75f, prob, 0.001f)  // 3/4 = 75%
    }

    @Test
    fun testGetProbability_NotExists() {
        store.recordBigram("hello", "world")

        val prob = store.getProbability("hello", "universe")
        assertEquals(0f, prob, 0.001f)
    }

    @Test
    fun testGetProbability_CaseInsensitive() {
        store.recordBigram("Hello", "World")

        val prob1 = store.getProbability("hello", "world")
        val prob2 = store.getProbability("HELLO", "WORLD")
        val prob3 = store.getProbability("HeLLo", "WoRLd")

        assertTrue(prob1 > 0f)
        assertEquals(prob1, prob2, 0.001f)
        assertEquals(prob1, prob3, 0.001f)
    }

    @Test
    fun testIgnoreEmptyWords() {
        store.recordBigram("", "world")
        store.recordBigram("hello", "")
        store.recordBigram("", "")

        assertEquals(0, store.getTotalBigramCount())
    }

    @Test
    fun testIgnoreSelfReferences() {
        store.recordBigram("hello", "hello")

        assertEquals(0, store.getTotalBigramCount())
    }

    @Test
    fun testGetTotalBigramCount() {
        assertEquals(0, store.getTotalBigramCount())

        store.recordBigram("I", "am")
        assertEquals(1, store.getTotalBigramCount())

        store.recordBigram("I", "want")
        assertEquals(2, store.getTotalBigramCount())

        store.recordBigram("you", "are")
        assertEquals(3, store.getTotalBigramCount())
    }

    @Test
    fun testGetContextWordCount() {
        assertEquals(0, store.getContextWordCount())

        store.recordBigram("I", "am")
        assertEquals(1, store.getContextWordCount())  // "I"

        store.recordBigram("I", "want")
        assertEquals(1, store.getContextWordCount())  // Still just "I"

        store.recordBigram("you", "are")
        assertEquals(2, store.getContextWordCount())  // "I" and "you"
    }

    @Test
    fun testClear() {
        store.recordBigram("hello", "world")
        store.recordBigram("foo", "bar")
        assertEquals(2, store.getTotalBigramCount())

        store.clear()
        assertEquals(0, store.getTotalBigramCount())
        assertEquals(0, store.getContextWordCount())
    }

    @Test
    fun testMinimumFrequencyFilter() {
        store.setMinimumFrequency(3)

        store.recordBigram("the", "cat")   // 1 time
        store.recordBigram("the", "dog")   // 1 time
        repeat(5) { store.recordBigram("the", "bird") }  // 5 times

        val predictions = store.getPredictions("the")

        // Only "bird" should appear (frequency 5 >= 3)
        assertEquals(1, predictions.size)
        assertEquals("bird", predictions[0].word2)
    }

    @Test
    fun testMaxBigramsPerWord() {
        // Record 25 different bigrams for "test"
        for (i in 0 until 25) {
            store.recordBigram("test", "word$i")
        }

        val predictions = store.getPredictions("test", maxResults = 100)

        // Should be limited to 20 (MAX_BIGRAMS_PER_WORD)
        assertTrue(predictions.size <= 20)
    }

    @Test
    fun testGetAllBigrams() {
        store.recordBigram("I", "am")
        store.recordBigram("I", "want")
        store.recordBigram("I", "have")

        val allBigrams = store.getAllBigrams("I")
        assertEquals(3, allBigrams.size)
    }

    @Test
    fun testGetStatistics() {
        store.recordBigram("I", "am")
        store.recordBigram("I", "want")
        store.recordBigram("you", "are")

        val stats = store.getStatistics()

        assertEquals(3, stats.totalBigrams)
        assertEquals(2, stats.uniqueContextWords)  // "I" and "you"
        assertEquals(1.5f, stats.averageBigramsPerContext, 0.01f)  // 3/2
        assertTrue(stats.topContextWords.isNotEmpty())
    }

    @Test
    fun testExportToJson() {
        store.recordBigram("hello", "world")
        store.recordBigram("foo", "bar")

        val json = store.exportToJson()

        assertTrue(json.contains("hello"))
        assertTrue(json.contains("world"))
        assertTrue(json.contains("foo"))
        assertTrue(json.contains("bar"))
        assertTrue(json.contains("frequency"))
        assertTrue(json.contains("probability"))
    }

    @Test
    fun testImportFromJson() {
        val json = """
        [
            {"word1":"test","word2":"import","frequency":5,"probability":0.5}
        ]
        """.trimIndent()

        store.importFromJson(json)

        val prob = store.getProbability("test", "import")
        assertTrue(prob > 0f)
    }

    @Test
    fun testRealWorldScenario() {
        // Simulate typing "I want to go"
        val sentences = listOf(
            listOf("I", "want", "to", "go"),
            listOf("I", "want", "to", "stay"),
            listOf("I", "want", "to", "go"),  // Repeat
            listOf("I", "need", "to", "go"),
            listOf("I", "want", "some", "coffee")
        )

        sentences.forEach { words ->
            for (i in 0 until words.size - 1) {
                store.recordBigram(words[i], words[i + 1])
            }
        }

        // "I" → "want" should be most common
        val iPredictions = store.getPredictions("I")
        assertTrue(iPredictions.isNotEmpty())
        assertEquals("want", iPredictions[0].word2)  // Most probable

        // "want" → "to" should be likely
        val wantProb = store.getProbability("want", "to")
        assertTrue(wantProb > 0.5f)  // More than 50%

        // "to" → "go" should exist
        val toProb = store.getProbability("to", "go")
        assertTrue(toProb > 0f)
    }

    @Test
    fun testCommonEnglishPhrases() {
        // Record common phrases
        repeat(10) { store.recordBigram("I", "am") }
        repeat(8) { store.recordBigram("you", "are") }
        repeat(12) { store.recordBigram("want", "to") }
        repeat(15) { store.recordBigram("going", "to") }

        // Test retrieval
        assertEquals("am", store.getPredictions("I")[0].word2)
        assertEquals("are", store.getPredictions("you")[0].word2)
        assertEquals("to", store.getPredictions("want")[0].word2)
        assertEquals("to", store.getPredictions("going")[0].word2)

        // "going" → "to" should have highest probability
        val goingToProb = store.getProbability("going", "to")
        assertEquals(1.0f, goingToProb, 0.001f)  // 100% (only option)
    }
}
