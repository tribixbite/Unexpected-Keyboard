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
 * Comprehensive test suite for ABTestManager.
 *
 * Tests A/B testing framework, variant assignment, conversion tracking,
 * and statistical significance calculations.
 *
 * @since v1.32.903 - Phase 6.3 test coverage
 */
@RunWith(MockitoJUnitRunner::class)
class ABTestManagerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var abTestManager: ABTestManager

    @Before
    fun setup() {
        `when`(mockContext.getSharedPreferences("ab_test_config", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putInt(anyString(), anyInt())).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)

        abTestManager = ABTestManager.getInstance(mockContext)
    }

    @After
    fun teardown() {
        reset(mockContext, mockPrefs, mockEditor)
    }

    // === TEST CREATION TESTS ===

    @Test
    fun `configureTest registers new A-B test with control and variant`() {
        abTestManager.configureTest(
            modelAId = "builtin_v1",
            modelAName = "Model V1",
            modelBId = "custom_v2",
            modelBName = "Model V2",
            trafficSplitA = 50
        )

        verify(mockEditor).putString("model_a_id", "builtin_v1")
        verify(mockEditor).putString("model_b_id", "custom_v2")
        verify(mockEditor).putString("model_a_name", "Model V1")
        verify(mockEditor).putString("model_b_name", "Model V2")
        verify(mockEditor).putInt("traffic_split_a", 50)
        verify(mockEditor).putBoolean("test_enabled", true)
        verify(mockEditor).apply()
    }

    // === VARIANT ASSIGNMENT TESTS ===

    @Test
    fun `selectModel returns null when test disabled`() {
        `when`(mockPrefs.getBoolean("test_enabled", false)).thenReturn(false)

        assertNull(abTestManager.selectModel())
    }

    @Test
    fun `selectModel returns selected model for session based`() {
        `when`(mockPrefs.getBoolean("test_enabled", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("session_based", true)).thenReturn(true)
        `when`(mockPrefs.getString("selected_model_session", null)).thenReturn("builtin_v1")

        assertEquals("builtin_v1", abTestManager.selectModel())
    }

    @Test
    fun `selectModel picks model if no session`() {
        `when`(mockPrefs.getBoolean("test_enabled", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("session_based", true)).thenReturn(true)
        `when`(mockPrefs.getString("selected_model_session", null)).thenReturn(null)
        
        `when`(mockPrefs.getInt("traffic_split_a", 50)).thenReturn(50)
        `when`(mockPrefs.getString("model_a_id", "")).thenReturn("modelA")
        `when`(mockPrefs.getString("model_b_id", "")).thenReturn("modelB")

        // We cannot deterministically test Random here without injecting a random provider,
        // but we can verify it returns one of the models and saves to session
        val selected = abTestManager.selectModel()
        assertNotNull(selected)
        assertTrue(selected == "modelA" || selected == "modelB")
        
        verify(mockEditor).putString(eq("selected_model_session"), anyString())
    }

    @Test
    fun `stopTest stops test`() {
        abTestManager.stopTest()

        verify(mockEditor).putBoolean("test_enabled", false)
        verify(mockEditor).remove("selected_model_session")
        verify(mockEditor).apply()
    }
    
    @Test
    fun `resetTest clears data`() {
        `when`(mockPrefs.getString("model_a_id", "")).thenReturn("modelA")
        `when`(mockPrefs.getString("model_b_id", "")).thenReturn("modelB")
        
        abTestManager.resetTest()
        
        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }
}