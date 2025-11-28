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
        `when`(mockContext.getSharedPreferences("model_comparison_data", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putInt(anyString(), anyInt())).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockEditor.putFloat(anyString(), anyFloat())).thenReturn(mockEditor)
        `when`(mockEditor.putStringSet(anyString(), any())).thenReturn(mockEditor)

        comparisonTracker = ModelComparisonTracker.getInstance(mockContext)
    }

    @Test
    fun `registerModel registers a model`() {
        comparisonTracker.registerModel("modelA", "Model A")
        
        verify(mockEditor).putString("model_modelA_name", "Model A")
        verify(mockEditor).putStringSet(eq("active_models"), any())
        verify(mockEditor).apply()
    }

    @Test
    fun `recordPrediction increments counts`() {
        `when`(mockPrefs.getInt("model_modelA_predictions", 0)).thenReturn(5)
        `when`(mockPrefs.getLong("model_modelA_total_latency", 0)).thenReturn(100L)
        
        comparisonTracker.recordPrediction("modelA", 20L)
        
        verify(mockEditor).putInt("model_modelA_predictions", 6)
        verify(mockEditor).putLong("model_modelA_total_latency", 120L)
    }

    @Test
    fun `recordSelection increments selection counts`() {
        `when`(mockPrefs.getInt("model_modelA_selections", 0)).thenReturn(2)
        `when`(mockPrefs.getInt("model_modelA_top1_hits", 0)).thenReturn(1)
        
        comparisonTracker.recordSelection("modelA", 0) // Top 1
        
        verify(mockEditor).putInt("model_modelA_selections", 3)
        verify(mockEditor).putInt("model_modelA_top1_hits", 2)
    }
    
    @Test
    fun `compareModels calculates stats`() {
        // Mock Model A stats
        `when`(mockPrefs.getString("model_modelA_name", null)).thenReturn("Model A")
        `when`(mockPrefs.getInt("model_modelA_selections", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("model_modelA_top1_hits", 0)).thenReturn(80) // 80%
        
        // Mock Model B stats
        `when`(mockPrefs.getString("model_modelB_name", null)).thenReturn("Model B")
        `when`(mockPrefs.getInt("model_modelB_selections", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("model_modelB_top1_hits", 0)).thenReturn(60) // 60%
        
        val comparison = comparisonTracker.compareModels("modelA", "modelB")
        
        assertNotNull(comparison)
        assertEquals(20.0, comparison.top1AccuracyDiff, 0.01)
        assertEquals(true, comparison.isStatisticallySignificant)
    }
}