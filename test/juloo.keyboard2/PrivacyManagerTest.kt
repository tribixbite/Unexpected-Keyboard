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
import kotlin.test.assertTrue

/**
 * Comprehensive test suite for PrivacyManager.
 *
 * Tests privacy controls, consent management, data retention,
 * and audit trail functionality.
 *
 * @since v1.32.903 - Phase 6.5 test coverage
 */
@RunWith(MockitoJUnitRunner::class)
class PrivacyManagerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var privacyManager: PrivacyManager

    @Before
    fun setup() {
        // Setup mock preferences
        `when`(mockContext.getSharedPreferences("privacy_settings", Context.MODE_PRIVATE))
            .thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putInt(anyString(), anyInt())).thenReturn(mockEditor)
        `when`(mockEditor.putLong(anyString(), anyLong())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)

        privacyManager = PrivacyManager(mockContext)
    }

    @After
    fun teardown() {
        // Clear any singleton instances
        reset(mockContext, mockPrefs, mockEditor)
    }

    // === CONSENT MANAGEMENT TESTS ===

    @Test
    fun `hasConsent returns false by default`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(false)
        assertFalse(privacyManager.hasConsent())
    }

    @Test
    fun `grantConsent sets consent flag and records timestamp`() {
        privacyManager.grantConsent()

        verify(mockEditor).putBoolean("consent_given", true)
        verify(mockEditor).putLong(eq("consent_timestamp"), anyLong())
        verify(mockEditor).putInt("consent_version", 1)
        verify(mockEditor).apply()
    }

    @Test
    fun `revokeConsent clears consent flag`() {
        privacyManager.revokeConsent(deleteData = false)

        verify(mockEditor).putBoolean("consent_given", false)
        verify(mockEditor).apply()
    }

    @Test
    fun `getConsentStatus returns correct status`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getLong("consent_timestamp", 0)).thenReturn(123456789L)
        `when`(mockPrefs.getInt("consent_version", 0)).thenReturn(1)

        val status = privacyManager.getConsentStatus()

        assertTrue(status.hasConsent)
        assertEquals(123456789L, status.consentTimestamp)
        assertEquals(1, status.consentVersion)
        assertFalse(status.needsUpdate)
    }

    // === DATA COLLECTION PERMISSION TESTS ===

    @Test
    fun `canCollectSwipeData returns false when no consent`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(false)
        assertFalse(privacyManager.canCollectSwipeData())
    }

    @Test
    fun `canCollectSwipeData returns true when consent given and setting enabled`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_swipe_data", true)).thenReturn(true)

        assertTrue(privacyManager.canCollectSwipeData())
    }

    @Test
    fun `canCollectSwipeData returns false when consent given but setting disabled`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_swipe_data", true)).thenReturn(false)

        assertFalse(privacyManager.canCollectSwipeData())
    }

    @Test
    fun `canCollectPerformanceData respects both consent and setting`() {
        // No consent
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(false)
        assertFalse(privacyManager.canCollectPerformanceData())

        // Consent but disabled
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_performance_data", true)).thenReturn(false)
        assertFalse(privacyManager.canCollectPerformanceData())

        // Consent and enabled
        `when`(mockPrefs.getBoolean("collect_performance_data", true)).thenReturn(true)
        assertTrue(privacyManager.canCollectPerformanceData())
    }

    @Test
    fun `canCollectErrorLogs defaults to false`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_error_logs", false)).thenReturn(false)

        assertFalse(privacyManager.canCollectErrorLogs())
    }

    // === PRIVACY SETTINGS TESTS ===

    @Test
    fun `shouldAnonymizeData defaults to true`() {
        `when`(mockPrefs.getBoolean("anonymize_data", true)).thenReturn(true)
        assertTrue(privacyManager.shouldAnonymizeData())
    }

    @Test
    fun `isLocalOnlyTraining defaults to true`() {
        `when`(mockPrefs.getBoolean("local_only_training", true)).thenReturn(true)
        assertTrue(privacyManager.isLocalOnlyTraining())
    }

    @Test
    fun `isDataExportAllowed defaults to false`() {
        `when`(mockPrefs.getBoolean("allow_data_export", false)).thenReturn(false)
        assertFalse(privacyManager.isDataExportAllowed())
    }

    @Test
    fun `isModelSharingAllowed defaults to false`() {
        `when`(mockPrefs.getBoolean("allow_model_sharing", false)).thenReturn(false)
        assertFalse(privacyManager.isModelSharingAllowed())
    }

    // === DATA RETENTION TESTS ===

    @Test
    fun `getDataRetentionDays returns default 90 days`() {
        `when`(mockPrefs.getInt("data_retention_days", 90)).thenReturn(90)
        assertEquals(90, privacyManager.getDataRetentionDays())
    }

    @Test
    fun `isAutoDeleteEnabled defaults to true`() {
        `when`(mockPrefs.getBoolean("auto_delete_enabled", true)).thenReturn(true)
        assertTrue(privacyManager.isAutoDeleteEnabled())
    }

    @Test
    fun `getDataRetentionCutoff calculates correct timestamp`() {
        `when`(mockPrefs.getInt("data_retention_days", 90)).thenReturn(90)

        val cutoff = privacyManager.getDataRetentionCutoff()
        val now = System.currentTimeMillis()
        val expectedCutoff = now - (90 * 24L * 60 * 60 * 1000)

        // Allow 1 second tolerance for execution time
        assertTrue(Math.abs(cutoff - expectedCutoff) < 1000)
    }

    @Test
    fun `shouldPerformCleanup returns false when auto-delete disabled`() {
        `when`(mockPrefs.getBoolean("auto_delete_enabled", true)).thenReturn(false)
        assertFalse(privacyManager.shouldPerformCleanup())
    }

    @Test
    fun `shouldPerformCleanup returns true when more than 24 hours elapsed`() {
        `when`(mockPrefs.getBoolean("auto_delete_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getLong("last_cleanup_timestamp", 0))
            .thenReturn(System.currentTimeMillis() - (25 * 60 * 60 * 1000)) // 25 hours ago

        assertTrue(privacyManager.shouldPerformCleanup())
    }

    @Test
    fun `shouldPerformCleanup returns false when less than 24 hours elapsed`() {
        `when`(mockPrefs.getBoolean("auto_delete_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getLong("last_cleanup_timestamp", 0))
            .thenReturn(System.currentTimeMillis() - (23 * 60 * 60 * 1000)) // 23 hours ago

        assertFalse(privacyManager.shouldPerformCleanup())
    }

    // === SETTINGS UPDATE TESTS ===

    @Test
    fun `updateSettings applies all settings correctly`() {
        val settings = PrivacyManager.PrivacySettings(
            collectSwipeData = true,
            collectPerformanceData = false,
            collectErrorLogs = true,
            anonymizeData = false,
            localOnlyTraining = false,
            allowDataExport = true,
            allowModelSharing = true,
            dataRetentionDays = 30,
            autoDeleteEnabled = false
        )

        privacyManager.updateSettings(settings)

        verify(mockEditor).putBoolean("collect_swipe_data", true)
        verify(mockEditor).putBoolean("collect_performance_data", false)
        verify(mockEditor).putBoolean("collect_error_logs", true)
        verify(mockEditor).putBoolean("anonymize_data", false)
        verify(mockEditor).putBoolean("local_only_training", false)
        verify(mockEditor).putBoolean("allow_data_export", true)
        verify(mockEditor).putBoolean("allow_model_sharing", true)
        verify(mockEditor).putInt("data_retention_days", 30)
        verify(mockEditor).putBoolean("auto_delete_enabled", false)
        verify(mockEditor).apply()
    }

    @Test
    fun `individual setters update correct preferences`() {
        privacyManager.setCollectSwipeData(false)
        verify(mockEditor).putBoolean("collect_swipe_data", false)

        privacyManager.setAnonymizeData(true)
        verify(mockEditor).putBoolean("anonymize_data", true)

        privacyManager.setDataRetentionDays(180)
        verify(mockEditor).putInt("data_retention_days", 180)
    }

    // === AUDIT TRAIL TESTS ===

    @Test
    fun `recordCleanupPerformed updates timestamp`() {
        privacyManager.recordCleanupPerformed()
        verify(mockEditor).putLong(eq("last_cleanup_timestamp"), anyLong())
        verify(mockEditor).apply()
    }

    @Test
    fun `getAuditTrail returns empty list when no audit data`() {
        `when`(mockPrefs.getString("audit_trail", "[]")).thenReturn("[]")

        val trail = privacyManager.getAuditTrail()

        assertTrue(trail.isEmpty())
    }

    @Test
    fun `formatStatus includes consent and settings information`() {
        `when`(mockPrefs.getBoolean("consent_given", false)).thenReturn(true)
        `when`(mockPrefs.getLong("consent_timestamp", 0)).thenReturn(123456789L)
        `when`(mockPrefs.getInt("consent_version", 0)).thenReturn(1)
        `when`(mockPrefs.getBoolean("collect_swipe_data", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("anonymize_data", true)).thenReturn(true)
        `when`(mockPrefs.getInt("data_retention_days", 90)).thenReturn(90)

        val status = privacyManager.formatStatus()

        assertTrue(status.contains("Privacy & Data Control"))
        assertTrue(status.contains("Consent Status"))
        assertTrue(status.contains("Granted"))
    }

    // === EDGE CASES AND ERROR HANDLING ===

    @Test
    fun `resetAll clears all preferences`() {
        privacyManager.resetAll()
        verify(mockEditor).clear()
        verify(mockEditor).apply()
    }

    @Test
    fun `getSettings returns correct defaults`() {
        `when`(mockPrefs.getBoolean("collect_swipe_data", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_performance_data", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("collect_error_logs", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("anonymize_data", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("local_only_training", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("allow_data_export", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("allow_model_sharing", false)).thenReturn(false)
        `when`(mockPrefs.getInt("data_retention_days", 90)).thenReturn(90)
        `when`(mockPrefs.getBoolean("auto_delete_enabled", true)).thenReturn(true)

        val settings = privacyManager.getSettings()

        assertTrue(settings.collectSwipeData)
        assertTrue(settings.collectPerformanceData)
        assertFalse(settings.collectErrorLogs)
        assertTrue(settings.anonymizeData)
        assertTrue(settings.localOnlyTraining)
        assertFalse(settings.allowDataExport)
        assertFalse(settings.allowModelSharing)
        assertEquals(90, settings.dataRetentionDays)
        assertTrue(settings.autoDeleteEnabled)
    }
}
