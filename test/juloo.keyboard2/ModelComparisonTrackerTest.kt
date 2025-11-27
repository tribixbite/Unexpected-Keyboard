package juloo.keyboard2

import android.content.Context
import android.content.SharedPreferences
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Comprehensive test suite for ModelComparisonTracker.
 *
 * Tests side-by-side model comparison tracking, accuracy comparison,
 * latency tracking, and winner determination logic.
 *
 * @since v1.32.903 - Phase 6.3 test coverage
 */
@RunWith(MockitoJUnitRunner::class)
class ModelComparisonTrackerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var comparisonTracker: ModelComparisonTracker

    @Before
    fun setup() {
        `when`(mockContext.getSharedPreferences("model_comparison_tracker", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putInt(anyString(), anyInt())).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockEditor.putFloat(anyString(), anyFloat())).thenReturn(mockEditor)

        comparisonTracker = ModelComparisonTracker.getInstance(mockContext)
    }

    @After
    fun teardown() {
        reset(mockContext, mockPrefs, mockEditor)
    }

    // === COMPARISON START TESTS ===

    @Test
    fun `startComparison registers new model comparison`() {
        comparisonTracker.startComparison(
            comparisonId = "v1_vs_v2",
            modelAId = "builtin_v1",
            modelBId = "custom_v2"
        )

        verify(mockEditor).putString("v1_vs_v2_modelA", "builtin_v1")
        verify(mockEditor).putString("v1_vs_v2_modelB", "custom_v2")
        verify(mockEditor).putBoolean("v1_vs_v2_active", true)
        verify(mockEditor).putLong(eq("v1_vs_v2_started"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `startComparison sets active comparison ID`() {
        comparisonTracker.startComparison(
            comparisonId = "test_comparison",
            modelAId = "model_a",
            modelBId = "model_b"
        )

        verify(mockEditor).putString("active_comparison_id", "test_comparison")
    }

    // === PREDICTION RECORDING TESTS ===

    @Test
    fun `recordPrediction stores predictions for both models`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")

        val predictionsA = listOf("hello", "help", "held")
        val predictionsB = listOf("hello", "heel", "held")

        comparisonTracker.recordPrediction(
            swipeInput = "h-e-l-l-o",
            predictionsA = predictionsA,
            predictionsB = predictionsB,
            inferenceTimeA = 45L,
            inferenceTimeB = 52L
        )

        verify(mockEditor).putString(eq("comp1_last_swipe"), anyString())
        verify(mockEditor).putString(eq("comp1_modelA_predictions"), anyString())
        verify(mockEditor).putString(eq("comp1_modelB_predictions"), anyString())
        verify(mockEditor).putLong("comp1_modelA_inference", 45L)
        verify(mockEditor).putLong("comp1_modelB_inference", 52L)
        verify(mockEditor).apply()
    }

    @Test
    fun `recordPrediction does nothing when no active comparison`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn(null)

        comparisonTracker.recordPrediction(
            swipeInput = "test",
            predictionsA = listOf("test"),
            predictionsB = listOf("test"),
            inferenceTimeA = 50L,
            inferenceTimeB = 50L
        )

        verify(mockEditor, never()).putString(anyString(), anyString())
    }

    // === SELECTION TRACKING TESTS ===

    @Test
    fun `recordSelection tracks which model won`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"hello\",\"heel\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(10)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_total_predictions", 11)
        verify(mockEditor).apply()
    }

    @Test
    fun `recordSelection increments modelA wins when A has better prediction`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"help\",\"hello\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(5)
        `when`(mockPrefs.getInt("comp1_ties", 0)).thenReturn(2)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_modelA_wins", 6)
        verify(mockEditor).putInt("comp1_total_predictions", 1)
    }

    @Test
    fun `recordSelection increments modelB wins when B has better prediction`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"help\",\"hello\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(3)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_modelB_wins", 4)
        verify(mockEditor).putInt("comp1_total_predictions", 1)
    }

    @Test
    fun `recordSelection increments ties when both have same position`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"hello\",\"heel\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("comp1_ties", 0)).thenReturn(2)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_ties", 3)
        verify(mockEditor).putInt("comp1_total_predictions", 1)
    }

    // === ACCURACY TRACKING TESTS ===

    @Test
    fun `recordSelection tracks top1 accuracy for both models`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"help\",\"hello\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("comp1_modelA_top1", 0)).thenReturn(10)
        `when`(mockPrefs.getInt("comp1_modelB_top1", 0)).thenReturn(8)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_modelA_top1", 11) // First in A's list
        verify(mockEditor).putInt("comp1_modelB_top1", 8)  // Not first in B's list
    }

    @Test
    fun `recordSelection tracks top3 accuracy for both models`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\",\"held\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"help\",\"hello\",\"held\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("comp1_modelA_top3", 0)).thenReturn(15)
        `when`(mockPrefs.getInt("comp1_modelB_top3", 0)).thenReturn(14)

        comparisonTracker.recordSelection("hello")

        verify(mockEditor).putInt("comp1_modelA_top3", 16) // In first 3 of A
        verify(mockEditor).putInt("comp1_modelB_top3", 15) // In first 3 of B
    }

    // === LATENCY TRACKING TESTS ===

    @Test
    fun `recordPrediction accumulates inference times`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getLong("comp1_modelA_total_time", 0)).thenReturn(500L)
        `when`(mockPrefs.getLong("comp1_modelB_total_time", 0)).thenReturn(600L)
        `when`(mockPrefs.getInt("comp1_inference_count", 0)).thenReturn(10)

        comparisonTracker.recordPrediction(
            swipeInput = "test",
            predictionsA = listOf("test"),
            predictionsB = listOf("test"),
            inferenceTimeA = 45L,
            inferenceTimeB = 52L
        )

        verify(mockEditor).putLong("comp1_modelA_total_time", 545L)
        verify(mockEditor).putLong("comp1_modelB_total_time", 652L)
        verify(mockEditor).putInt("comp1_inference_count", 11)
    }

    // === METRICS CALCULATION TESTS ===

    @Test
    fun `getComparisonMetrics calculates win rates correctly`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(60)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(30)
        `when`(mockPrefs.getInt("comp1_ties", 0)).thenReturn(10)

        val metrics = comparisonTracker.getComparisonMetrics("comp1")

        assertNotNull(metrics)
        assertEquals(60.0, metrics.modelAWinRate, 0.01)
        assertEquals(30.0, metrics.modelBWinRate, 0.01)
        assertEquals(10.0, metrics.tieRate, 0.01)
    }

    @Test
    fun `getComparisonMetrics calculates accuracy correctly`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_top1", 0)).thenReturn(70)
        `when`(mockPrefs.getInt("comp1_modelA_top3", 0)).thenReturn(90)
        `when`(mockPrefs.getInt("comp1_modelB_top1", 0)).thenReturn(65)
        `when`(mockPrefs.getInt("comp1_modelB_top3", 0)).thenReturn(85)

        val metrics = comparisonTracker.getComparisonMetrics("comp1")

        assertNotNull(metrics)
        assertEquals(70, metrics.modelATop1Accuracy)
        assertEquals(90, metrics.modelATop3Accuracy)
        assertEquals(65, metrics.modelBTop1Accuracy)
        assertEquals(85, metrics.modelBTop3Accuracy)
    }

    @Test
    fun `getComparisonMetrics calculates average latency correctly`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getLong("comp1_modelA_total_time", 0)).thenReturn(5000L)
        `when`(mockPrefs.getLong("comp1_modelB_total_time", 0)).thenReturn(6000L)
        `when`(mockPrefs.getInt("comp1_inference_count", 0)).thenReturn(100)

        val metrics = comparisonTracker.getComparisonMetrics("comp1")

        assertNotNull(metrics)
        assertEquals(50L, metrics.modelAAvgLatency)
        assertEquals(60L, metrics.modelBAvgLatency)
    }

    @Test
    fun `getComparisonMetrics returns null for non-existent comparison`() {
        `when`(mockPrefs.getString("nonexistent_modelA", null)).thenReturn(null)

        assertNull(comparisonTracker.getComparisonMetrics("nonexistent"))
    }

    @Test
    fun `getComparisonMetrics returns zero rates when no predictions`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)

        val metrics = comparisonTracker.getComparisonMetrics("comp1")

        assertNotNull(metrics)
        assertEquals(0.0, metrics.modelAWinRate, 0.01)
        assertEquals(0.0, metrics.modelBWinRate, 0.01)
        assertEquals(0.0, metrics.tieRate, 0.01)
    }

    // === WINNER DETERMINATION TESTS ===

    @Test
    fun `determineWinner returns model A when A wins more`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(60)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(30)

        val winner = comparisonTracker.determineWinner("comp1")

        assertNotNull(winner)
        assertEquals("model_a", winner.winningModel)
        assertEquals("win_rate", winner.winningMetric)
    }

    @Test
    fun `determineWinner returns model B when B wins more`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(30)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(60)

        val winner = comparisonTracker.determineWinner("comp1")

        assertNotNull(winner)
        assertEquals("model_b", winner.winningModel)
    }

    @Test
    fun `determineWinner uses accuracy as tiebreaker`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(40)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(40)
        `when`(mockPrefs.getInt("comp1_modelA_top1", 0)).thenReturn(75)
        `when`(mockPrefs.getInt("comp1_modelB_top1", 0)).thenReturn(70)

        val winner = comparisonTracker.determineWinner("comp1")

        assertNotNull(winner)
        assertEquals("model_a", winner.winningModel)
        assertEquals("top1_accuracy", winner.winningMetric)
    }

    @Test
    fun `determineWinner uses latency as final tiebreaker`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(40)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(40)
        `when`(mockPrefs.getInt("comp1_modelA_top1", 0)).thenReturn(70)
        `when`(mockPrefs.getInt("comp1_modelB_top1", 0)).thenReturn(70)
        `when`(mockPrefs.getLong("comp1_modelA_total_time", 0)).thenReturn(4500L)
        `when`(mockPrefs.getLong("comp1_modelB_total_time", 0)).thenReturn(5500L)
        `when`(mockPrefs.getInt("comp1_inference_count", 0)).thenReturn(100)

        val winner = comparisonTracker.determineWinner("comp1")

        assertNotNull(winner)
        assertEquals("model_a", winner.winningModel)
        assertEquals("latency", winner.winningMetric)
    }

    @Test
    fun `determineWinner requires minimum sample size`() {
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("model_a")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("model_b")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(10) // Too small
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(8)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(2)

        val winner = comparisonTracker.determineWinner("comp1")

        assertNotNull(winner)
        assertFalse(winner.isSignificant)
    }

    // === COMPARISON STOP TESTS ===

    @Test
    fun `stopComparison deactivates comparison`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")

        comparisonTracker.stopComparison()

        verify(mockEditor).putBoolean("comp1_active", false)
        verify(mockEditor).putLong(eq("comp1_stopped"), anyLong())
        verify(mockEditor).remove("active_comparison_id")
        verify(mockEditor).apply()
    }

    @Test
    fun `stopComparison does nothing when no active comparison`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn(null)

        comparisonTracker.stopComparison()

        verify(mockEditor, never()).putBoolean(anyString(), anyBoolean())
    }

    // === COMPARISON LISTING TESTS ===

    @Test
    fun `listComparisons returns all registered comparisons`() {
        val prefsMap = mutableMapOf<String, Any>(
            "comp1_modelA" to "model_a1",
            "comp1_modelB" to "model_b1",
            "comp1_active" to true,
            "comp2_modelA" to "model_a2",
            "comp2_modelB" to "model_b2",
            "comp2_active" to false
        )
        `when`(mockPrefs.all).thenReturn(prefsMap)

        val comparisons = comparisonTracker.listComparisons()

        assertEquals(2, comparisons.size)
        assertTrue(comparisons.any { it.modelAId == "model_a1" })
        assertTrue(comparisons.any { it.modelAId == "model_a2" })
    }

    // === RESET TESTS ===

    @Test
    fun `resetComparison clears comparison data`() {
        comparisonTracker.resetComparison("comp1")

        verify(mockEditor).remove("comp1_modelA")
        verify(mockEditor).remove("comp1_modelB")
        verify(mockEditor).remove("comp1_active")
        verify(mockEditor).remove("comp1_total_predictions")
        verify(mockEditor).remove("comp1_modelA_wins")
        verify(mockEditor).remove("comp1_modelB_wins")
        verify(mockEditor).remove("comp1_ties")
        verify(mockEditor).apply()
    }

    @Test
    fun `resetAllComparisons clears all data`() {
        comparisonTracker.resetAllComparisons()

        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }

    // === FORMAT TESTS ===

    @Test
    fun `formatComparisonStatus returns no active comparison message`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn(null)

        val status = comparisonTracker.formatComparisonStatus()

        assertTrue(status.contains("No active model comparison"))
    }

    @Test
    fun `formatComparisonStatus includes detailed metrics`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA", null)).thenReturn("builtin_v1")
        `when`(mockPrefs.getString("comp1_modelB", null)).thenReturn("custom_v2")
        `when`(mockPrefs.getBoolean("comp1_active", false)).thenReturn(true)
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("comp1_modelA_wins", 0)).thenReturn(60)
        `when`(mockPrefs.getInt("comp1_modelB_wins", 0)).thenReturn(30)
        `when`(mockPrefs.getInt("comp1_ties", 0)).thenReturn(10)
        `when`(mockPrefs.getInt("comp1_modelA_top1", 0)).thenReturn(75)
        `when`(mockPrefs.getInt("comp1_modelB_top1", 0)).thenReturn(70)
        `when`(mockPrefs.getLong("comp1_modelA_total_time", 0)).thenReturn(5000L)
        `when`(mockPrefs.getLong("comp1_modelB_total_time", 0)).thenReturn(6000L)
        `when`(mockPrefs.getInt("comp1_inference_count", 0)).thenReturn(100)

        val status = comparisonTracker.formatComparisonStatus()

        assertTrue(status.contains("builtin_v1"))
        assertTrue(status.contains("custom_v2"))
        assertTrue(status.contains("60"))
        assertTrue(status.contains("30"))
        assertTrue(status.contains("75"))
        assertTrue(status.contains("70"))
    }

    // === EDGE CASES ===

    @Test
    fun `handles empty predictions list`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)

        comparisonTracker.recordSelection("test")

        verify(mockEditor).putInt("comp1_total_predictions", 1)
        // Should not crash, should handle gracefully
    }

    @Test
    fun `handles selection not in either predictions list`() {
        `when`(mockPrefs.getString("active_comparison_id", null)).thenReturn("comp1")
        `when`(mockPrefs.getString("comp1_modelA_predictions", null))
            .thenReturn("[\"hello\",\"help\"]")
        `when`(mockPrefs.getString("comp1_modelB_predictions", null))
            .thenReturn("[\"hello\",\"help\"]")
        `when`(mockPrefs.getInt("comp1_total_predictions", 0)).thenReturn(0)

        comparisonTracker.recordSelection("goodbye") // Not in either list

        verify(mockEditor).putInt("comp1_total_predictions", 1)
        // Should record but not count as win for either
    }
}
