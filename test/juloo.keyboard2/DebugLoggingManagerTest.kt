package juloo.keyboard2

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentCaptor
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for DebugLoggingManager.
 *
 * Tests cover:
 * - Log writer initialization
 * - Debug mode broadcast receiver registration/unregistration
 * - Debug mode change listener management
 * - Debug mode state management
 * - Debug log message broadcasting
 * - Log file writing
 * - Resource cleanup
 * - Edge cases and error handling
 */
@RunWith(MockitoJUnitRunner::class)
class DebugLoggingManagerTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var debugLoggingManager: DebugLoggingManager
    private val testPackageName = "juloo.keyboard2.test"

    @Before
    fun setUp() {
        debugLoggingManager = DebugLoggingManager(mockContext, testPackageName)
    }

    // ========== Log Writer Initialization Tests ==========

    @Test
    fun testInitializeLogWriter_doesNotCrash() {
        // Act & Assert - should not throw
        // Note: Will fail to create file in test environment, but should handle gracefully
        val result = debugLoggingManager.initializeLogWriter()

        // Assert - should return false since file path doesn't exist in test environment
        // (or true if running in proper Android environment)
        assertNotNull("Result should not be null", result)
    }

    @Test
    fun testGetLogFilePath_returnsCorrectPath() {
        // Act
        val logFilePath = debugLoggingManager.getLogFilePath()

        // Assert
        assertEquals("Should return correct log file path",
            "/data/data/com.termux/files/home/swipe_log.txt", logFilePath)
    }

    // ========== Debug Mode Receiver Registration Tests ==========

    @Test
    fun testRegisterDebugModeReceiver_registersReceiver() {
        // Act
        debugLoggingManager.registerDebugModeReceiver(mockContext)

        // Assert
        verify(mockContext).registerReceiver(
            any(BroadcastReceiver::class.java),
            any(IntentFilter::class.java),
            eq(Context.RECEIVER_NOT_EXPORTED)
        )
    }

    @Test
    fun testRegisterDebugModeReceiver_registersWithCorrectAction() {
        // Arrange
        val filterCaptor = ArgumentCaptor.forClass(IntentFilter::class.java)

        // Act
        debugLoggingManager.registerDebugModeReceiver(mockContext)

        // Assert
        verify(mockContext).registerReceiver(
            any(BroadcastReceiver::class.java),
            filterCaptor.capture(),
            eq(Context.RECEIVER_NOT_EXPORTED)
        )

        val filter = filterCaptor.value
        assertTrue("Filter should match SET_DEBUG_MODE action",
            filter.hasAction("juloo.keyboard2.SET_DEBUG_MODE"))
    }

    @Test
    fun testRegisterDebugModeReceiver_calledTwice_doesNotRegisterTwice() {
        // Act
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        debugLoggingManager.registerDebugModeReceiver(mockContext)

        // Assert - should only register once
        verify(mockContext, times(1)).registerReceiver(
            any(BroadcastReceiver::class.java),
            any(IntentFilter::class.java),
            eq(Context.RECEIVER_NOT_EXPORTED)
        )
    }

    // ========== Debug Mode Receiver Unregistration Tests ==========

    @Test
    fun testUnregisterDebugModeReceiver_afterRegistration_unregisters() {
        // Arrange
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Act
        debugLoggingManager.unregisterDebugModeReceiver(mockContext)

        // Assert
        verify(mockContext).unregisterReceiver(receiverCaptor.value)
    }

    @Test
    fun testUnregisterDebugModeReceiver_withoutRegistration_doesNotCrash() {
        // Act & Assert - should not throw
        debugLoggingManager.unregisterDebugModeReceiver(mockContext)

        // Verify no calls to unregisterReceiver
        verify(mockContext, never()).unregisterReceiver(any())
    }

    @Test
    fun testUnregisterDebugModeReceiver_whenExceptionThrown_handlesGracefully() {
        // Arrange
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        doThrow(IllegalArgumentException("Receiver not registered"))
            .`when`(mockContext).unregisterReceiver(any())

        // Act & Assert - should not throw
        debugLoggingManager.unregisterDebugModeReceiver(mockContext)
    }

    // ========== Debug Mode Listener Tests ==========

    @Test
    fun testRegisterDebugModeListener_addsListener() {
        // Arrange
        val listener = mock(DebugLoggingManager.DebugModeListener::class.java)

        // Act
        debugLoggingManager.registerDebugModeListener(listener)

        // Assert - trigger debug mode change to verify listener is called
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        val intent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        intent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, intent)

        verify(listener).onDebugModeChanged(true)
    }

    @Test
    fun testUnregisterDebugModeListener_removesListener() {
        // Arrange
        val listener = mock(DebugLoggingManager.DebugModeListener::class.java)
        debugLoggingManager.registerDebugModeListener(listener)

        // Act
        debugLoggingManager.unregisterDebugModeListener(listener)

        // Assert - trigger debug mode change, listener should NOT be called
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        val intent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        intent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, intent)

        verify(listener, never()).onDebugModeChanged(anyBoolean())
    }

    @Test
    fun testRegisterDebugModeListener_sameListerTwice_onlyAddsOnce() {
        // Arrange
        val listener = mock(DebugLoggingManager.DebugModeListener::class.java)

        // Act
        debugLoggingManager.registerDebugModeListener(listener)
        debugLoggingManager.registerDebugModeListener(listener)

        // Assert - trigger debug mode change, listener should be called only once
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        val intent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        intent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, intent)

        verify(listener, times(1)).onDebugModeChanged(true)
    }

    // ========== Debug Mode State Tests ==========

    @Test
    fun testIsDebugMode_initiallyFalse() {
        // Assert
        assertFalse("Debug mode should initially be false",
            debugLoggingManager.isDebugMode())
    }

    @Test
    fun testDebugModeReceiver_enablesDebugMode() {
        // Arrange
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Act
        val intent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        intent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, intent)

        // Assert
        assertTrue("Debug mode should be enabled",
            debugLoggingManager.isDebugMode())
    }

    @Test
    fun testDebugModeReceiver_disablesDebugMode() {
        // Arrange
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Enable first
        val enableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        enableIntent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, enableIntent)

        // Act - disable
        val disableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        disableIntent.putExtra("debug_enabled", false)
        receiverCaptor.value.onReceive(mockContext, disableIntent)

        // Assert
        assertFalse("Debug mode should be disabled",
            debugLoggingManager.isDebugMode())
    }

    @Test
    fun testDebugModeReceiver_missingExtra_defaultsToFalse() {
        // Arrange
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Act
        val intent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        // No extra included
        receiverCaptor.value.onReceive(mockContext, intent)

        // Assert
        assertFalse("Debug mode should default to false",
            debugLoggingManager.isDebugMode())
    }

    // ========== Debug Log Sending Tests ==========

    @Test
    fun testSendDebugLog_whenDebugModeDisabled_doesNotSendBroadcast() {
        // Act
        debugLoggingManager.sendDebugLog("Test message")

        // Assert
        verify(mockContext, never()).sendBroadcast(any())
    }

    @Test
    fun testSendDebugLog_whenDebugModeEnabled_sendsBroadcast() {
        // Arrange
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        val enableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        enableIntent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, enableIntent)

        // Act
        debugLoggingManager.sendDebugLog("Test debug message")

        // Assert
        val intentCaptor = ArgumentCaptor.forClass(Intent::class.java)
        verify(mockContext, atLeastOnce()).sendBroadcast(intentCaptor.capture())

        val capturedIntents = intentCaptor.allValues
        val debugLogIntent = capturedIntents.find {
            it.action == SwipeDebugActivity.ACTION_DEBUG_LOG
        }

        assertNotNull("Should send debug log broadcast", debugLogIntent)
        assertEquals("Should include log message",
            "Test debug message",
            debugLogIntent!!.getStringExtra(SwipeDebugActivity.EXTRA_LOG_MESSAGE))
        assertEquals("Should set explicit package",
            testPackageName,
            debugLogIntent.getPackage())
    }

    @Test
    fun testSendDebugLog_whenDebugModeEnabled_sendsDebugModeEnabledMessage() {
        // Arrange
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Act
        val enableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        enableIntent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, enableIntent)

        // Assert
        val intentCaptor = ArgumentCaptor.forClass(Intent::class.java)
        verify(mockContext, atLeastOnce()).sendBroadcast(intentCaptor.capture())

        val capturedIntents = intentCaptor.allValues
        val debugLogIntent = capturedIntents.find {
            it.action == SwipeDebugActivity.ACTION_DEBUG_LOG &&
            it.getStringExtra(SwipeDebugActivity.EXTRA_LOG_MESSAGE)?.contains("Debug mode enabled") == true
        }

        assertNotNull("Should send 'Debug mode enabled' message", debugLogIntent)
    }

    // ========== Log File Writing Tests ==========

    @Test
    fun testWriteToLogFile_doesNotCrash() {
        // Act & Assert - should not throw even if log writer is not initialized
        debugLoggingManager.writeToLogFile("Test log entry")
    }

    // ========== Cleanup Tests ==========

    @Test
    fun testClose_doesNotCrash() {
        // Act & Assert - should not throw
        debugLoggingManager.close()
    }

    @Test
    fun testClose_afterInitialization_doesNotCrash() {
        // Arrange
        debugLoggingManager.initializeLogWriter()

        // Act & Assert - should not throw
        debugLoggingManager.close()
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_registerEnableDisableUnregister() {
        // Arrange
        val listener = mock(DebugLoggingManager.DebugModeListener::class.java)
        debugLoggingManager.registerDebugModeListener(listener)

        // Register receiver
        debugLoggingManager.registerDebugModeReceiver(mockContext)
        val receiverCaptor = ArgumentCaptor.forClass(BroadcastReceiver::class.java)
        verify(mockContext).registerReceiver(receiverCaptor.capture(), any(), any())

        // Enable debug mode
        val enableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        enableIntent.putExtra("debug_enabled", true)
        receiverCaptor.value.onReceive(mockContext, enableIntent)

        assertTrue("Debug mode should be enabled", debugLoggingManager.isDebugMode())
        verify(listener).onDebugModeChanged(true)

        // Send debug log
        debugLoggingManager.sendDebugLog("Test message")
        verify(mockContext, atLeastOnce()).sendBroadcast(any())

        // Disable debug mode
        val disableIntent = Intent("juloo.keyboard2.SET_DEBUG_MODE")
        disableIntent.putExtra("debug_enabled", false)
        receiverCaptor.value.onReceive(mockContext, disableIntent)

        assertFalse("Debug mode should be disabled", debugLoggingManager.isDebugMode())
        verify(listener).onDebugModeChanged(false)

        // Unregister receiver
        debugLoggingManager.unregisterDebugModeReceiver(mockContext)
        verify(mockContext).unregisterReceiver(receiverCaptor.value)

        // Close
        debugLoggingManager.close()
    }
}
