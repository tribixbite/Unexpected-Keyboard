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
import kotlin.test.assertTrue

/**
 * Test suite for NeuralPerformanceStats.
 *
 * Tests performance metric tracking, accuracy calculations,
 * and privacy integration.
 *
 * @since v1.32.903 - Phase 6.1/6.5 test coverage
 */
@RunWith(MockitoJUnitRunner::class)
class NeuralPerformanceStatsTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    @Mock
    private lateinit var mockPrivacyManager: PrivacyManager

    private lateinit var stats: NeuralPerformanceStats

    @Before
    fun setup() {
        // Setup mock preferences
        `when`(mockContext.getSharedPreferences("neural_performance_stats", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.apply()).then { }

        stats = NeuralPerformanceStats(mockContext)
    }

    @After
    fun teardown() {
        reset(mockContext, mockPrefs, mockEditor)
    }

    // === PREDICTION RECORDING TESTS ===

    @Test
    fun `recordPrediction increments prediction count`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(5L)
        `when`(mockPrefs.getLong("total_inference_time_ms", 0)).thenReturn(250L)

        stats.recordPrediction(50L)

        verify(mockEditor).putLong("total_predictions", 6L)
        verify(mockEditor).putLong("total_inference_time_ms", 300L)
        verify(mockEditor).apply()
    }

    @Test
    fun `recordPrediction sets first stat timestamp on first call`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("total_inference_time_ms", 0)).thenReturn(0L)
        `when`(mockPrefs.contains("first_stat_timestamp")).thenReturn(false)

        stats.recordPrediction(50L)

        verify(mockEditor).putLong(eq("first_stat_timestamp"), anyLong())
    }

    // === SELECTION RECORDING TESTS ===

    @Test
    fun `recordSelection increments total selections`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(10L)

        stats.recordSelection(2)

        verify(mockEditor).putLong("total_selections", 11L)
        verify(mockEditor).apply()
    }

    @Test
    fun `recordSelection increments top1 for index 0`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("top1_selections", 0)).thenReturn(5L)

        stats.recordSelection(0)

        verify(mockEditor).putLong("top1_selections", 6L)
    }

    @Test
    fun `recordSelection increments top3 for indices 0-2`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("top3_selections", 0)).thenReturn(10L)

        stats.recordSelection(1)

        verify(mockEditor).putLong("top3_selections", 11L)
    }

    @Test
    fun `recordSelection does not increment top3 for index 3 or higher`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("top3_selections", 0)).thenReturn(10L)

        stats.recordSelection(3)

        verify(mockEditor, never()).putLong("top3_selections", anyLong())
    }

    // === METRIC CALCULATION TESTS ===

    @Test
    fun `getAverageInferenceTime calculates correctly`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(10L)
        `when`(mockPrefs.getLong("total_inference_time_ms", 0)).thenReturn(500L)

        assertEquals(50L, stats.getAverageInferenceTime())
    }

    @Test
    fun `getAverageInferenceTime returns 0 when no predictions`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("total_inference_time_ms", 0)).thenReturn(0L)

        assertEquals(0L, stats.getAverageInferenceTime())
    }

    @Test
    fun `getTop1Accuracy calculates correct percentage`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(100L)
        `when`(mockPrefs.getLong("top1_selections", 0)).thenReturn(70L)

        assertEquals(70, stats.getTop1Accuracy())
    }

    @Test
    fun `getTop1Accuracy returns 0 when no selections`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(0L)
        `when`(mockPrefs.getLong("top1_selections", 0)).thenReturn(0L)

        assertEquals(0, stats.getTop1Accuracy())
    }

    @Test
    fun `getTop3Accuracy calculates correct percentage`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(100L)
        `when`(mockPrefs.getLong("top3_selections", 0)).thenReturn(90L)

        assertEquals(90, stats.getTop3Accuracy())
    }

    @Test
    fun `getDaysTracked calculates days from first timestamp`() {
        val threeDaysAgo = System.currentTimeMillis() - (3 * 24 * 60 * 60 * 1000)
        `when`(mockPrefs.getLong("first_stat_timestamp", 0)).thenReturn(threeDaysAgo)

        val days = stats.getDaysTracked()

        assertTrue(days >= 3 && days <= 4) // Allow for timing variations
    }

    @Test
    fun `getDaysTracked returns 0 when no data`() {
        `when`(mockPrefs.getLong("first_stat_timestamp", 0)).thenReturn(0L)

        assertEquals(0, stats.getDaysTracked())
    }

    // === MODEL LOAD TIME TESTS ===

    @Test
    fun `recordModelLoadTime stores load time`() {
        stats.recordModelLoadTime(1500L)

        verify(mockEditor).putLong("model_load_time_ms", 1500L)
        verify(mockEditor).apply()
    }

    @Test
    fun `getModelLoadTime retrieves stored value`() {
        `when`(mockPrefs.getLong("model_load_time_ms", 0)).thenReturn(1500L)

        assertEquals(1500L, stats.getModelLoadTime())
    }

    // === RESET FUNCTIONALITY ===

    @Test
    fun `reset clears all statistics`() {
        stats.reset()

        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }

    // === FORMAT TESTS ===

    @Test
    fun `formatSummary returns no data message when empty`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(0L)

        val summary = stats.formatSummary()

        assertTrue(summary.contains("No performance data collected yet"))
    }

    @Test
    fun `formatSummary includes all metrics when data available`() {
        `when`(mockPrefs.getLong("total_predictions", 0)).thenReturn(100L)
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(80L)
        `when`(mockPrefs.getLong("top1_selections", 0)).thenReturn(60L)
        `when`(mockPrefs.getLong("top3_selections", 0)).thenReturn(72L)
        `when`(mockPrefs.getLong("total_inference_time_ms", 0)).thenReturn(5000L)
        `when`(mockPrefs.getLong("model_load_time_ms", 0)).thenReturn(1200L)
        `when`(mockPrefs.getLong("first_stat_timestamp", 0))
            .thenReturn(System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000))

        val summary = stats.formatSummary()

        assertTrue(summary.contains("Usage Statistics"))
        assertTrue(summary.contains("100"))  // predictions
        assertTrue(summary.contains("80"))   // selections
        assertTrue(summary.contains("Performance"))
        assertTrue(summary.contains("50"))   // avg inference (5000/100)
        assertTrue(summary.contains("Accuracy"))
        assertTrue(summary.contains("75%"))  // top1 (60/80)
        assertTrue(summary.contains("90%"))  // top3 (72/80)
    }

    // === EDGE CASES ===

    @Test
    fun `high selection count does not overflow`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(Long.MAX_VALUE - 1)

        stats.recordSelection(0)

        verify(mockEditor).putLong("total_selections", Long.MAX_VALUE)
    }

    @Test
    fun `accuracy calculation handles rounding correctly`() {
        `when`(mockPrefs.getLong("total_selections", 0)).thenReturn(3L)
        `when`(mockPrefs.getLong("top1_selections", 0)).thenReturn(2L)

        // 2/3 = 66.666...% should round to 67
        assertEquals(67, stats.getTop1Accuracy())
    }
}
