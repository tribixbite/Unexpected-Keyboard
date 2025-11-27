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
 * Test suite for ModelVersionManager.
 *
 * Tests version tracking, rollback logic, health monitoring,
 * and version history management.
 *
 * @since v1.32.903 - Phase 6.4 test coverage
 */
@RunWith(MockitoJUnitRunner::class)
class ModelVersionManagerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var versionManager: ModelVersionManager

    @Before
    fun setup() {
        `when`(mockContext.getSharedPreferences("model_version_history", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putInt(anyString(), anyInt())).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)

        versionManager = ModelVersionManager(mockContext)
    }

    @After
    fun teardown() {
        reset(mockContext, mockPrefs, mockEditor)
    }

    // === VERSION REGISTRATION TESTS ===

    @Test
    fun `registerVersion stores version information`() {
        versionManager.registerVersion(
            versionId = "v1",
            versionName = "Test Version 1",
            encoderPath = "/path/encoder.onnx",
            decoderPath = "/path/decoder.onnx",
            isBuiltin = true
        )

        verify(mockEditor).putString("current_version", "v1")
        verify(mockEditor).putString("v1_name", "Test Version 1")
        verify(mockEditor).putString("v1_encoder", "/path/encoder.onnx")
        verify(mockEditor).putString("v1_decoder", "/path/decoder.onnx")
        verify(mockEditor).putBoolean("v1_builtin", true)
        verify(mockEditor).putLong(eq("v1_registered"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `registerVersion saves current as previous when changing versions`() {
        `when`(mockPrefs.getString("current_version", null)).thenReturn("v1")

        versionManager.registerVersion(
            versionId = "v2",
            versionName = "Test Version 2",
            encoderPath = "/path/encoder2.onnx",
            decoderPath = "/path/decoder2.onnx"
        )

        verify(mockEditor).putString("previous_version", "v1")
    }

    // === SUCCESS/FAILURE RECORDING TESTS ===

    @Test
    fun `recordSuccess increments success count and resets failures`() {
        `when`(mockPrefs.getInt("v1_successes", 0)).thenReturn(5)
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(2)

        versionManager.recordSuccess("v1")

        verify(mockEditor).putInt("v1_successes", 6)
        verify(mockEditor).putInt("v1_failures", 0)  // Reset to 0
        verify(mockEditor).putLong(eq("v1_last_success"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `recordFailure increments failure count`() {
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(2)

        versionManager.recordFailure("v1", "Test error")

        verify(mockEditor).putInt("v1_failures", 3)
        verify(mockEditor).putLong(eq("v1_last_failure"), anyLong())
        verify(mockEditor).putString("v1_last_error", "Test error")
        verify(mockEditor).apply()
    }

    // === VERSION RETRIEVAL TESTS ===

    @Test
    fun `getCurrentVersion returns null when no version registered`() {
        `when`(mockPrefs.getString("current_version", null)).thenReturn(null)

        assertNull(versionManager.getCurrentVersion())
    }

    @Test
    fun `getCurrentVersion returns version information`() {
        `when`(mockPrefs.getString("current_version", null)).thenReturn("v1")
        `when`(mockPrefs.getString("v1_name", null)).thenReturn("Test Version")
        `when`(mockPrefs.getString("v1_encoder", null)).thenReturn("/encoder.onnx")
        `when`(mockPrefs.getString("v1_decoder", null)).thenReturn("/decoder.onnx")
        `when`(mockPrefs.getInt("v1_successes", 0)).thenReturn(10)
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(2)
        `when`(mockPrefs.getBoolean("v1_builtin", true)).thenReturn(true)

        val version = versionManager.getCurrentVersion()

        assertNotNull(version)
        assertEquals("v1", version.versionId)
        assertEquals("Test Version", version.versionName)
        assertEquals(10, version.successCount)
        assertEquals(2, version.failureCount)
        assertTrue(version.isBuiltin)
    }

    @Test
    fun `getVersion returns null for non-existent version`() {
        `when`(mockPrefs.getString("v99_name", null)).thenReturn(null)

        assertNull(versionManager.getVersion("v99"))
    }

    // === ROLLBACK DECISION TESTS ===

    @Test
    fun `shouldRollback returns false when auto-rollback disabled`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(false)

        val decision = versionManager.shouldRollback()

        assertFalse(decision.shouldRollback)
        assertEquals("Auto-rollback disabled", decision.reason)
    }

    @Test
    fun `shouldRollback returns false when version is pinned`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn("v1")

        val decision = versionManager.shouldRollback()

        assertFalse(decision.shouldRollback)
        assertTrue(decision.reason.contains("Version pinned"))
    }

    @Test
    fun `shouldRollback returns false when in cooldown period`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn(null)
        `when`(mockPrefs.getLong("last_rollback_timestamp", 0))
            .thenReturn(System.currentTimeMillis() - 30000) // 30 seconds ago

        val decision = versionManager.shouldRollback()

        assertFalse(decision.shouldRollback)
        assertTrue(decision.reason.contains("cooldown"))
        assertTrue(decision.isInCooldown)
    }

    @Test
    fun `shouldRollback returns false when no current version`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn(null)
        `when`(mockPrefs.getLong("last_rollback_timestamp", 0)).thenReturn(0L)
        `when`(mockPrefs.getString("current_version", null)).thenReturn(null)

        val decision = versionManager.shouldRollback()

        assertFalse(decision.shouldRollback)
        assertEquals("No current version registered", decision.reason)
    }

    @Test
    fun `shouldRollback returns true when failures exceed threshold`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn(null)
        `when`(mockPrefs.getLong("last_rollback_timestamp", 0)).thenReturn(0L)

        // Current version with 3 failures
        `when`(mockPrefs.getString("current_version", null)).thenReturn("v2")
        `when`(mockPrefs.getString("v2_name", null)).thenReturn("Version 2")
        `when`(mockPrefs.getString("v2_encoder", null)).thenReturn("/enc.onnx")
        `when`(mockPrefs.getString("v2_decoder", null)).thenReturn("/dec.onnx")
        `when`(mockPrefs.getInt("v2_failures", 0)).thenReturn(3)
        `when`(mockPrefs.getInt("v2_successes", 0)).thenReturn(0)

        // Previous healthy version
        `when`(mockPrefs.getString("previous_version", null)).thenReturn("v1")
        `when`(mockPrefs.getString("v1_name", null)).thenReturn("Version 1")
        `when`(mockPrefs.getString("v1_encoder", null)).thenReturn("/enc1.onnx")
        `when`(mockPrefs.getString("v1_decoder", null)).thenReturn("/dec1.onnx")
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("v1_successes", 0)).thenReturn(10)

        val decision = versionManager.shouldRollback()

        assertTrue(decision.shouldRollback)
        assertTrue(decision.reason.contains("Consecutive failures exceeded"))
        assertEquals("v1", decision.targetVersion)
        assertEquals(3, decision.currentFailures)
    }

    @Test
    fun `shouldRollback returns false when previous version is unhealthy`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn(null)
        `when`(mockPrefs.getLong("last_rollback_timestamp", 0)).thenReturn(0L)

        // Current version with 3 failures
        `when`(mockPrefs.getString("current_version", null)).thenReturn("v2")
        `when`(mockPrefs.getString("v2_name", null)).thenReturn("Version 2")
        `when`(mockPrefs.getString("v2_encoder", null)).thenReturn("/enc.onnx")
        `when`(mockPrefs.getString("v2_decoder", null)).thenReturn("/dec.onnx")
        `when`(mockPrefs.getInt("v2_failures", 0)).thenReturn(3)
        `when`(mockPrefs.getInt("v2_successes", 0)).thenReturn(0)

        // Previous unhealthy version (low success rate)
        `when`(mockPrefs.getString("previous_version", null)).thenReturn("v1")
        `when`(mockPrefs.getString("v1_name", null)).thenReturn("Version 1")
        `when`(mockPrefs.getString("v1_encoder", null)).thenReturn("/enc1.onnx")
        `when`(mockPrefs.getString("v1_decoder", null)).thenReturn("/dec1.onnx")
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(6)
        `when`(mockPrefs.getInt("v1_successes", 0)).thenReturn(4)  // 40% success rate

        val decision = versionManager.shouldRollback()

        assertFalse(decision.shouldRollback)
    }

    // === ROLLBACK EXECUTION TESTS ===

    @Test
    fun `rollback returns false when shouldRollback is false`() {
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(false)

        assertFalse(versionManager.rollback())
    }

    @Test
    fun `rollback swaps current and previous versions`() {
        // Setup shouldRollback to return true
        `when`(mockPrefs.getBoolean("auto_rollback_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("pinned_version", null)).thenReturn(null)
        `when`(mockPrefs.getLong("last_rollback_timestamp", 0)).thenReturn(0L)

        `when`(mockPrefs.getString("current_version", null)).thenReturn("v2")
        `when`(mockPrefs.getString("v2_name", null)).thenReturn("Version 2")
        `when`(mockPrefs.getString("v2_encoder", null)).thenReturn("/enc.onnx")
        `when`(mockPrefs.getString("v2_decoder", null)).thenReturn("/dec.onnx")
        `when`(mockPrefs.getInt("v2_failures", 0)).thenReturn(3)
        `when`(mockPrefs.getInt("v2_successes", 0)).thenReturn(0)

        `when`(mockPrefs.getString("previous_version", null)).thenReturn("v1")
        `when`(mockPrefs.getString("v1_name", null)).thenReturn("Version 1")
        `when`(mockPrefs.getString("v1_encoder", null)).thenReturn("/enc1.onnx")
        `when`(mockPrefs.getString("v1_decoder", null)).thenReturn("/dec1.onnx")
        `when`(mockPrefs.getInt("v1_failures", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("v1_successes", 0)).thenReturn(10)

        assertTrue(versionManager.rollback())

        verify(mockEditor).putString("previous_version", "v2")
        verify(mockEditor).putString("current_version", "v1")
        verify(mockEditor).putLong(eq("last_rollback_timestamp"), anyLong())
        verify(mockEditor).apply()
    }

    // === VERSION PINNING TESTS ===

    @Test
    fun `pinCurrentVersion sets pinned version`() {
        `when`(mockPrefs.getString("current_version", null)).thenReturn("v1")
        `when`(mockPrefs.getString("v1_name", null)).thenReturn("Version 1")
        `when`(mockPrefs.getString("v1_encoder", null)).thenReturn("/enc.onnx")
        `when`(mockPrefs.getString("v1_decoder", null)).thenReturn("/dec.onnx")

        versionManager.pinCurrentVersion()

        verify(mockEditor).putString("pinned_version", "v1")
        verify(mockEditor).apply()
    }

    @Test
    fun `unpinVersion removes pinned version`() {
        versionManager.unpinVersion()

        verify(mockEditor).remove("pinned_version")
        verify(mockEditor).apply()
    }

    // === HEALTH CALCULATION TESTS ===

    @Test
    fun `ModelVersion isHealthy returns true for good success rate`() {
        val version = ModelVersionManager.ModelVersion(
            versionId = "v1",
            versionName = "Test",
            encoderPath = "/enc.onnx",
            decoderPath = "/dec.onnx",
            loadTimestamp = 0L,
            successCount = 8,
            failureCount = 2,  // 80% success rate
            isPinned = false,
            isBuiltin = true
        )

        assertTrue(version.isHealthy())
    }

    @Test
    fun `ModelVersion isHealthy returns false for low success rate`() {
        val version = ModelVersionManager.ModelVersion(
            versionId = "v1",
            versionName = "Test",
            encoderPath = "/enc.onnx",
            decoderPath = "/dec.onnx",
            loadTimestamp = 0L,
            successCount = 3,
            failureCount = 7,  // 30% success rate
            isPinned = false,
            isBuiltin = true
        )

        assertFalse(version.isHealthy())
    }

    @Test
    fun `ModelVersion isHealthy returns false for too many failures`() {
        val version = ModelVersionManager.ModelVersion(
            versionId = "v1",
            versionName = "Test",
            encoderPath = "/enc.onnx",
            decoderPath = "/dec.onnx",
            loadTimestamp = 0L,
            successCount = 10,
            failureCount = 3,  // Still has 3 failures
            isPinned = false,
            isBuiltin = true
        )

        assertFalse(version.isHealthy())
    }

    // === RESET TESTS ===

    @Test
    fun `reset clears all version data`() {
        versionManager.reset()

        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }

    // === AUTO-ROLLBACK SETTING ===

    @Test
    fun `setAutoRollbackEnabled updates setting`() {
        versionManager.setAutoRollbackEnabled(false)

        verify(mockEditor).putBoolean("auto_rollback_enabled", false)
        verify(mockEditor).apply()
    }
}
