package juloo.keyboard2.contextaware

import android.content.Context
import android.content.SharedPreferences
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.mockito.Mockito.*

/**
 * Unit tests for ContextModel.
 *
 * Tests:
 * - Sequence recording
 * - Context boost calculation
 * - Top predictions
 * - Hybrid scoring with boost formula
 * - Statistics and analytics
 */
class ContextModelTest {

    private lateinit var mockContext: Context
    private lateinit var mockPrefs: SharedPreferences
    private lateinit var mockEditor: SharedPreferences.Editor
    private lateinit var model: ContextModel

    @Before
    fun setup() {
        // Mock Android dependencies
        mockContext = mock(Context::class.java)
        mockPrefs = mock(SharedPreferences::class.java)
        mockEditor = mock(SharedPreferences.Editor::class.java)

        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockPrefs.getString(anyString(), anyString())).thenReturn(null)

        model = ContextModel(mockContext)
    }

    @Test
    fun testRecordSequence_Simple() {
        model.recordSequence(listOf("hello", "world"))

        assertTrue(model.hasContextFor("hello"))
        val boost = model.getContextBoost("world", listOf("hello"))
        assertTrue(boost > 1.0f)  // Should have positive boost
    }

    @Test
    fun testRecordSequence_MultipleWords() {
        // Record "I want to go"
        model.recordSequence(listOf("I", "want", "to", "go"))

        // All bigrams should be recorded
        assertTrue(model.hasContextFor("I"))
        assertTrue(model.hasContextFor("want"))
        assertTrue(model.hasContextFor("to"))
    }

    @Test
    fun testRecordSequence_TooShort() {
        // Single word - should not record anything
        model.recordSequence(listOf("hello"))

        assertFalse(model.hasContextFor("hello"))
    }

    @Test
    fun testRecordSequence_Empty() {
        model.recordSequence(emptyList())

        assertEquals(0, model.getStatistics().totalContextWords)
    }

    @Test
    fun testGetContextBoost_NoContext() {
        // No previous words - should return min boost (1.0)
        val boost = model.getContextBoost("hello", emptyList())
        assertEquals(1.0f, boost, 0.001f)
    }

    @Test
    fun testGetContextBoost_NoMatchingBigram() {
        model.recordSequence(listOf("foo", "bar"))

        // Query non-existent bigram
        val boost = model.getContextBoost("baz", listOf("foo"))
        assertEquals(1.0f, boost, 0.001f)  // No boost for unknown pair
    }

    @Test
    fun testGetContextBoost_WithContext() {
        // Record "hello world" 5 times
        repeat(5) {
            model.recordSequence(listOf("hello", "world"))
        }

        val boost = model.getContextBoost("world", listOf("hello"))
        assertTrue(boost > 1.0f)  // Should have boost
        assertTrue(boost <= 5.0f)  // Max boost is 5.0
    }

    @Test
    fun testGetContextBoost_BoostFormula() {
        // Test boost calculation formula
        // boost = (1 + probability)^2

        // Record bigrams with known probabilities
        repeat(10) { model.recordSequence(listOf("test", "high")) }  // High prob
        repeat(1) { model.recordSequence(listOf("test", "low")) }    // Low prob

        val highBoost = model.getContextBoost("high", listOf("test"))
        val lowBoost = model.getContextBoost("low", listOf("test"))

        // High probability word should have higher boost
        assertTrue(highBoost > lowBoost)

        // Boost should be between 1.0 and 5.0
        assertTrue(highBoost >= 1.0f && highBoost <= 5.0f)
        assertTrue(lowBoost >= 1.0f && lowBoost <= 5.0f)
    }

    @Test
    fun testGetContextBoost_CaseInsensitive() {
        model.recordSequence(listOf("Hello", "World"))

        val boost1 = model.getContextBoost("world", listOf("hello"))
        val boost2 = model.getContextBoost("WORLD", listOf("HELLO"))
        val boost3 = model.getContextBoost("WoRLd", listOf("HeLLo"))

        // All should be the same (case-insensitive)
        assertEquals(boost1, boost2, 0.001f)
        assertEquals(boost1, boost3, 0.001f)
    }

    @Test
    fun testGetTopPredictions_Empty() {
        val predictions = model.getTopPredictions(emptyList())
        assertTrue(predictions.isEmpty())
    }

    @Test
    fun testGetTopPredictions_WithContext() {
        // Record multiple options after "I"
        repeat(10) { model.recordSequence(listOf("I", "am")) }
        repeat(5) { model.recordSequence(listOf("I", "want")) }
        repeat(3) { model.recordSequence(listOf("I", "need")) }

        val predictions = model.getTopPredictions(listOf("I"))

        // Should return predictions sorted by boost
        assertTrue(predictions.isNotEmpty())
        assertEquals(3, predictions.size)

        // First should be "am" (highest frequency)
        assertEquals("am", predictions[0].first)

        // Boosts should be descending
        assertTrue(predictions[0].second >= predictions[1].second)
        assertTrue(predictions[1].second >= predictions[2].second)
    }

    @Test
    fun testGetTopPredictions_LimitResults() {
        // Record many predictions
        for (i in 0 until 20) {
            model.recordSequence(listOf("test", "word$i"))
        }

        val predictions = model.getTopPredictions(listOf("test"), maxResults = 5)

        assertEquals(5, predictions.size)
    }

    @Test
    fun testGetProbability() {
        repeat(3) { model.recordSequence(listOf("want", "to")) }
        model.recordSequence(listOf("want", "some"))

        val prob = model.getProbability("want", "to")
        assertEquals(0.75f, prob, 0.001f)  // 3/4 = 75%
    }

    @Test
    fun testHasContextFor() {
        model.recordSequence(listOf("hello", "world"))

        assertTrue(model.hasContextFor("hello"))
        assertFalse(model.hasContextFor("goodbye"))
    }

    @Test
    fun testClear() {
        model.recordSequence(listOf("hello", "world"))
        assertTrue(model.hasContextFor("hello"))

        model.clear()
        assertFalse(model.hasContextFor("hello"))
    }

    @Test
    fun testGetStatistics() {
        model.recordSequence(listOf("I", "want", "to", "go"))
        model.recordSequence(listOf("you", "need", "to", "stay"))

        val stats = model.getStatistics()

        assertTrue(stats.totalContextWords > 0)
        assertTrue(stats.averageBoostPotential >= 1.0f)
        assertNotNull(stats.bigramStats)
    }

    @Test
    fun testRealWorldScenario_CommonPhrases() {
        // Simulate typing common English phrases
        val sentences = listOf(
            "I want to go home",
            "I want to go there",
            "I want to stay here",
            "I need to go now",
            "I need to leave soon",
            "you are very kind",
            "you are so nice",
            "you are the best"
        )

        sentences.forEach { sentence ->
            val words = sentence.split(" ")
            model.recordSequence(words)
        }

        // Test context boosts for common patterns
        val wantToBoost = model.getContextBoost("to", listOf("want"))
        val needToBoost = model.getContextBoost("to", listOf("need"))
        val youAreBoost = model.getContextBoost("are", listOf("you"))

        // All should have positive boosts
        assertTrue(wantToBoost > 1.0f)
        assertTrue(needToBoost > 1.0f)
        assertTrue(youAreBoost > 1.0f)

        // "want" → "to" should have high boost (appears in 3/3 "want" cases)
        assertTrue(wantToBoost > 2.0f)
    }

    @Test
    fun testRealWorldScenario_TypedSentence() {
        // User types: "I want to"
        // Simulate learning from this
        model.recordSequence(listOf("I", "want", "to"))

        // Now user types "I want" again and we predict next word
        val predictions = model.getTopPredictions(listOf("want"))

        // "to" should be a top prediction
        assertTrue(predictions.isNotEmpty())
        val predictedWords = predictions.map { it.first }
        assertTrue(predictedWords.contains("to"))

        // Get boost for "to" after "want"
        val boost = model.getContextBoost("to", listOf("want"))
        assertTrue(boost > 1.0f)
    }

    @Test
    fun testBoostRanges() {
        // Test that boosts are within expected ranges

        // Very high probability (100%)
        repeat(20) { model.recordSequence(listOf("always", "here")) }
        val maxBoost = model.getContextBoost("here", listOf("always"))
        assertTrue(maxBoost >= 3.0f)  // Should be close to max
        assertTrue(maxBoost <= 5.0f)  // But not exceed max

        // Low probability (~10%)
        repeat(1) { model.recordSequence(listOf("rarely", "seen")) }
        repeat(9) { model.recordSequence(listOf("rarely", "other")) }
        val lowBoost = model.getContextBoost("seen", listOf("rarely"))
        assertTrue(lowBoost > 1.0f)   // Still some boost
        assertTrue(lowBoost < 1.5f)   // But not much
    }

    @Test
    fun testExportImport() {
        model.recordSequence(listOf("test", "export"))
        model.recordSequence(listOf("test", "import"))

        val json = model.exportToJson()
        assertTrue(json.contains("test"))
        assertTrue(json.contains("export"))

        // Create new model and import
        val newModel = ContextModel(mockContext)
        newModel.importFromJson(json)

        assertTrue(newModel.hasContextFor("test"))
        val boost = newModel.getContextBoost("export", listOf("test"))
        assertTrue(boost > 1.0f)
    }

    @Test
    fun testMinimumFrequency() {
        model.setMinimumFrequency(3)

        // Record once (below threshold)
        model.recordSequence(listOf("rare", "word"))

        // Record multiple times (above threshold)
        repeat(5) { model.recordSequence(listOf("common", "word")) }

        // Rare bigram should not provide boost
        val rareBoost = model.getContextBoost("word", listOf("rare"))
        assertEquals(1.0f, rareBoost, 0.001f)  // No boost (below threshold)

        // Common bigram should provide boost
        val commonBoost = model.getContextBoost("word", listOf("common"))
        assertTrue(commonBoost > 1.0f)  // Has boost
    }

    @Test
    fun testMultipleContextWords() {
        // User types several words, but we only use the last one for bigrams
        model.recordSequence(listOf("I", "want", "to", "go"))

        // Test with multiple previous words (should use last one)
        val boost1 = model.getContextBoost("go", listOf("to"))
        val boost2 = model.getContextBoost("go", listOf("want", "to"))
        val boost3 = model.getContextBoost("go", listOf("I", "want", "to"))

        // All should give same boost (using last word "to")
        assertEquals(boost1, boost2, 0.001f)
        assertEquals(boost1, boost3, 0.001f)
    }
}
