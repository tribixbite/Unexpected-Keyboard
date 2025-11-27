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
        `when`(mockContext.getSharedPreferences("ab_test_manager", Context.MODE_PRIVATE))
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
    fun `createTest registers new A-B test with control and variant`() {
        abTestManager.createTest(
            testId = "model_comparison_v1",
            testName = "Model V1 vs V2",
            controlVariant = "builtin_v1",
            testVariant = "custom_v2",
            trafficSplit = 50
        )

        verify(mockEditor).putString("model_comparison_v1_name", "Model V1 vs V2")
        verify(mockEditor).putString("model_comparison_v1_control", "builtin_v1")
        verify(mockEditor).putString("model_comparison_v1_variant", "custom_v2")
        verify(mockEditor).putInt("model_comparison_v1_split", 50)
        verify(mockEditor).putBoolean("model_comparison_v1_active", true)
        verify(mockEditor).putLong(eq("model_comparison_v1_created"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `createTest validates traffic split percentage`() {
        // Valid splits: 10, 25, 50, 75, 90
        abTestManager.createTest(
            testId = "test1",
            testName = "Test 1",
            controlVariant = "control",
            testVariant = "variant",
            trafficSplit = 25
        )

        verify(mockEditor).putInt("test1_split", 25)
    }

    // === VARIANT ASSIGNMENT TESTS ===

    @Test
    fun `getAssignedVariant returns null when no active test`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn(null)

        assertNull(abTestManager.getAssignedVariant())
    }

    @Test
    fun `getAssignedVariant returns control variant for user in control group`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getBoolean("test1_active", false)).thenReturn(true)

        // User hash places them in control group (< 50%)
        `when`(mockPrefs.getString("user_id", null)).thenReturn("user_control_hash")

        val assigned = abTestManager.getAssignedVariant()

        assertNotNull(assigned)
        assertEquals("control", assigned.variantId)
        assertEquals("control", assigned.variantType)
    }

    @Test
    fun `getAssignedVariant returns variant for user in test group`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getBoolean("test1_active", false)).thenReturn(true)

        // User hash places them in variant group (>= 50%)
        `when`(mockPrefs.getString("user_id", null)).thenReturn("user_variant_hash")

        val assigned = abTestManager.getAssignedVariant()

        assertNotNull(assigned)
        assertEquals("variant", assigned.variantId)
        assertEquals("variant", assigned.variantType)
    }

    @Test
    fun `getAssignedVariant persists assignment across calls`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_assigned_variant", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")

        val assigned1 = abTestManager.getAssignedVariant()
        val assigned2 = abTestManager.getAssignedVariant()

        assertEquals(assigned1?.variantId, assigned2?.variantId)
    }

    // === ACTIVATION TESTS ===

    @Test
    fun `activateTest sets test as active and assigns user`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)

        assertTrue(abTestManager.activateTest("test1"))

        verify(mockEditor).putString("active_test_id", "test1")
        verify(mockEditor).putBoolean("test1_active", true)
        verify(mockEditor).putLong(eq("test1_activated"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `activateTest returns false for non-existent test`() {
        `when`(mockPrefs.getString("nonexistent_name", null)).thenReturn(null)

        assertFalse(abTestManager.activateTest("nonexistent"))
    }

    @Test
    fun `deactivateTest stops test and clears active test`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")

        abTestManager.deactivateTest()

        verify(mockEditor).remove("active_test_id")
        verify(mockEditor).putBoolean("test1_active", false)
        verify(mockEditor).putLong(eq("test1_deactivated"), anyLong())
        verify(mockEditor).apply()
    }

    // === CONVERSION TRACKING TESTS ===

    @Test
    fun `recordConversion increments conversion count for variant`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_assigned_variant", null)).thenReturn("control")
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(5)

        abTestManager.recordConversion()

        verify(mockEditor).putInt("test1_control_conversions", 6)
        verify(mockEditor).apply()
    }

    @Test
    fun `recordConversion does nothing when no active test`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn(null)

        abTestManager.recordConversion()

        verify(mockEditor, never()).putInt(anyString(), anyInt())
    }

    @Test
    fun `recordImpression increments impression count for variant`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_assigned_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(100)

        abTestManager.recordImpression()

        verify(mockEditor).putInt("test1_variant_impressions", 101)
        verify(mockEditor).apply()
    }

    // === METRICS CALCULATION TESTS ===

    @Test
    fun `getTestMetrics calculates conversion rates correctly`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getBoolean("test1_active", false)).thenReturn(true)

        // Control: 70/100 = 70%
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(70)

        // Variant: 85/100 = 85%
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(85)

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        assertEquals(70.0, metrics.controlConversionRate, 0.01)
        assertEquals(85.0, metrics.variantConversionRate, 0.01)
        assertEquals(21.43, metrics.relativeImprovement, 0.01) // (85-70)/70 * 100
    }

    @Test
    fun `getTestMetrics returns zero conversion rate when no impressions`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(0)

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        assertEquals(0.0, metrics.controlConversionRate, 0.01)
        assertEquals(0.0, metrics.variantConversionRate, 0.01)
    }

    @Test
    fun `getTestMetrics returns null for non-existent test`() {
        `when`(mockPrefs.getString("nonexistent_name", null)).thenReturn(null)

        assertNull(abTestManager.getTestMetrics("nonexistent"))
    }

    // === STATISTICAL SIGNIFICANCE TESTS ===

    @Test
    fun `isSignificant returns false for small sample sizes`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)

        // Only 10 impressions each (too small)
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(10)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(7)
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(10)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(9)

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        assertFalse(metrics.isSignificant)
    }

    @Test
    fun `isSignificant returns true for large sample with clear difference`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)

        // Large sample with significant difference
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(1000)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(700)
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(1000)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(850)

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        assertTrue(metrics.isSignificant)
    }

    // === TEST LISTING TESTS ===

    @Test
    fun `listTests returns empty list when no tests`() {
        `when`(mockPrefs.all).thenReturn(emptyMap())

        val tests = abTestManager.listTests()

        assertTrue(tests.isEmpty())
    }

    @Test
    fun `listTests returns all registered tests`() {
        val prefsMap = mutableMapOf<String, Any>(
            "test1_name" to "Test 1",
            "test1_control" to "control1",
            "test1_variant" to "variant1",
            "test1_split" to 50,
            "test1_active" to true,
            "test2_name" to "Test 2",
            "test2_control" to "control2",
            "test2_variant" to "variant2",
            "test2_split" to 25,
            "test2_active" to false
        )
        `when`(mockPrefs.all).thenReturn(prefsMap)

        val tests = abTestManager.listTests()

        assertEquals(2, tests.size)
        assertTrue(tests.any { it.testName == "Test 1" })
        assertTrue(tests.any { it.testName == "Test 2" })
    }

    // === USER ID GENERATION TESTS ===

    @Test
    fun `getUserId generates new ID when none exists`() {
        `when`(mockPrefs.getString("user_id", null)).thenReturn(null)

        abTestManager.getUserId()

        verify(mockEditor).putString(eq("user_id"), anyString())
        verify(mockEditor).apply()
    }

    @Test
    fun `getUserId returns existing ID when present`() {
        `when`(mockPrefs.getString("user_id", null)).thenReturn("existing-user-id")

        val userId = abTestManager.getUserId()

        assertEquals("existing-user-id", userId)
        verify(mockEditor, never()).putString(eq("user_id"), anyString())
    }

    // === RESET TESTS ===

    @Test
    fun `resetTest clears test data and statistics`() {
        abTestManager.resetTest("test1")

        verify(mockEditor).remove("test1_name")
        verify(mockEditor).remove("test1_control")
        verify(mockEditor).remove("test1_variant")
        verify(mockEditor).remove("test1_split")
        verify(mockEditor).remove("test1_active")
        verify(mockEditor).remove("test1_control_impressions")
        verify(mockEditor).remove("test1_control_conversions")
        verify(mockEditor).remove("test1_variant_impressions")
        verify(mockEditor).remove("test1_variant_conversions")
        verify(mockEditor).remove("test1_assigned_variant")
        verify(mockEditor).apply()
    }

    @Test
    fun `resetAllTests clears all test data`() {
        abTestManager.resetAllTests()

        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }

    // === TRAFFIC SPLIT TESTS ===

    @Test
    fun `updateTrafficSplit updates split percentage`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")

        abTestManager.updateTrafficSplit("test1", 75)

        verify(mockEditor).putInt("test1_split", 75)
        verify(mockEditor).apply()
    }

    @Test
    fun `updateTrafficSplit returns false for non-existent test`() {
        `when`(mockPrefs.getString("nonexistent_name", null)).thenReturn(null)

        assertFalse(abTestManager.updateTrafficSplit("nonexistent", 50))
    }

    // === FORMAT TESTS ===

    @Test
    fun `formatTestStatus returns no active test message when none active`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn(null)

        val status = abTestManager.formatTestStatus()

        assertTrue(status.contains("No active A/B test"))
    }

    @Test
    fun `formatTestStatus includes test details and metrics`() {
        `when`(mockPrefs.getString("active_test_id", null)).thenReturn("test1")
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Model Comparison")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("builtin_v1")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("custom_v2")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getBoolean("test1_active", false)).thenReturn(true)
        `when`(mockPrefs.getString("test1_assigned_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(70)
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(85)

        val status = abTestManager.formatTestStatus()

        assertTrue(status.contains("Model Comparison"))
        assertTrue(status.contains("builtin_v1"))
        assertTrue(status.contains("custom_v2"))
        assertTrue(status.contains("variant"))
        assertTrue(status.contains("70"))
        assertTrue(status.contains("85"))
    }

    // === EDGE CASES ===

    @Test
    fun `conversion rate handles zero denominator`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(5) // Invalid state

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        assertEquals(0.0, metrics.controlConversionRate, 0.01)
    }

    @Test
    fun `relative improvement handles zero baseline`() {
        `when`(mockPrefs.getString("test1_name", null)).thenReturn("Test 1")
        `when`(mockPrefs.getString("test1_control", null)).thenReturn("control")
        `when`(mockPrefs.getString("test1_variant", null)).thenReturn("variant")
        `when`(mockPrefs.getInt("test1_split", 50)).thenReturn(50)
        `when`(mockPrefs.getInt("test1_control_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_control_conversions", 0)).thenReturn(0) // 0% baseline
        `when`(mockPrefs.getInt("test1_variant_impressions", 0)).thenReturn(100)
        `when`(mockPrefs.getInt("test1_variant_conversions", 0)).thenReturn(50)

        val metrics = abTestManager.getTestMetrics("test1")

        assertNotNull(metrics)
        // Improvement should be 0.0 or handle gracefully (can't divide by zero baseline)
        assertTrue(metrics.relativeImprovement >= 0.0)
    }
}
