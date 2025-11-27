package juloo.keyboard2

import android.content.Context
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.InOrder
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for CleanupHandler.
 *
 * Tests cover:
 * - Full cleanup with all managers
 * - Null manager handling (individual and all null)
 * - Cleanup order verification
 * - Factory method
 * - Multiple cleanup calls
 * - Integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class CleanupHandlerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockConfigManager: ConfigurationManager

    @Mock
    private lateinit var mockFoldStateTracker: FoldStateTracker

    @Mock
    private lateinit var mockClipboardManager: ClipboardManager

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockDebugLoggingManager: DebugLoggingManager

    private lateinit var handler: CleanupHandler

    @Before
    fun setUp() {
        // Mock fold state tracker from config manager
        `when`(mockConfigManager.getFoldStateTracker()).thenReturn(mockFoldStateTracker)

        handler = CleanupHandler(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )
    }

    // ========== Full Cleanup Tests ==========

    @Test
    fun testCleanup_allManagers_cleansUpAll() {
        // Act
        handler.cleanup()

        // Assert - all cleanup methods called
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockFoldStateTracker).close()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
        verify(mockDebugLoggingManager).close()
    }

    @Test
    fun testCleanup_verifyCleanupOrder() {
        // Arrange
        val inOrder: InOrder = inOrder(
            mockConfigManager,
            mockFoldStateTracker,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Act
        handler.cleanup()

        // Assert - verify order: fold tracker → clipboard → prediction → debug
        inOrder.verify(mockConfigManager).getFoldStateTracker()
        inOrder.verify(mockFoldStateTracker).close()
        inOrder.verify(mockClipboardManager).cleanup()
        inOrder.verify(mockPredictionCoordinator).shutdown()
        inOrder.verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
        inOrder.verify(mockDebugLoggingManager).close()
    }

    // ========== Null Manager Tests ==========

    @Test
    fun testCleanup_nullConfigManager_doesNotCrash() {
        // Arrange - create handler with null config manager
        val handler = CleanupHandler(
            mockContext,
            null, // null ConfigManager
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        handler.cleanup()

        // Other managers should still be cleaned up
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
    }

    @Test
    fun testCleanup_nullClipboardManager_doesNotCrash() {
        // Arrange
        val handler = CleanupHandler(
            mockContext,
            mockConfigManager,
            null, // null ClipboardManager
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        handler.cleanup()

        // Other managers should still be cleaned up
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockPredictionCoordinator).shutdown()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
    }

    @Test
    fun testCleanup_nullPredictionCoordinator_doesNotCrash() {
        // Arrange
        val handler = CleanupHandler(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            null, // null PredictionCoordinator
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        handler.cleanup()

        // Other managers should still be cleaned up
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockClipboardManager).cleanup()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
    }

    @Test
    fun testCleanup_nullDebugLoggingManager_doesNotCrash() {
        // Arrange
        val handler = CleanupHandler(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            null // null DebugLoggingManager
        )

        // Act & Assert - should not throw
        handler.cleanup()

        // Other managers should still be cleaned up
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
    }

    @Test
    fun testCleanup_allManagersNull_doesNotCrash() {
        // Arrange - all managers null
        val handler = CleanupHandler(
            mockContext,
            null, // null ConfigManager
            null, // null ClipboardManager
            null, // null PredictionCoordinator
            null  // null DebugLoggingManager
        )

        // Act & Assert - should not throw
        handler.cleanup()

        // No managers to verify, just ensure no crash
    }

    @Test
    fun testCleanup_nullFoldStateTracker_doesNotCrash() {
        // Arrange - config manager returns null fold state tracker
        `when`(mockConfigManager.getFoldStateTracker()).thenReturn(null)

        // Act & Assert - should not throw
        handler.cleanup()

        // Other cleanup should still happen
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesHandler() {
        // Act
        val handler = CleanupHandler.create(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Assert
        assertNotNull("Factory method should create handler", handler)
    }

    @Test
    fun testCreate_factoryMethodHandlerWorks() {
        // Arrange
        val handler = CleanupHandler.create(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Act
        handler.cleanup()

        // Assert
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
    }

    @Test
    fun testCreate_withNullManagers() {
        // Act
        val handler = CleanupHandler.create(
            mockContext,
            null, // null ConfigManager
            null, // null ClipboardManager
            null, // null PredictionCoordinator
            null  // null DebugLoggingManager
        )

        // Assert
        assertNotNull("Factory should create handler with null managers", handler)

        // Should not crash when cleaning up
        handler.cleanup()
    }

    // ========== Multiple Cleanup Tests ==========

    @Test
    fun testCleanup_calledTwice_cleansUpTwice() {
        // Act
        handler.cleanup()
        handler.cleanup()

        // Assert - all cleanup methods called twice
        verify(mockConfigManager, times(2)).getFoldStateTracker()
        verify(mockFoldStateTracker, times(2)).close()
        verify(mockClipboardManager, times(2)).cleanup()
        verify(mockPredictionCoordinator, times(2)).shutdown()
        verify(mockDebugLoggingManager, times(2)).unregisterDebugModeReceiver(mockContext)
        verify(mockDebugLoggingManager, times(2)).close()
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_createAndCleanup() {
        // Act - simulate typical usage
        val handler = CleanupHandler.create(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        handler.cleanup()

        // Assert - all cleanup performed
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockFoldStateTracker).close()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
        verify(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)
        verify(mockDebugLoggingManager).close()
    }

    @Test
    fun testIntegration_multipleHandlersIndependent() {
        // Arrange - create two handlers
        val handler1 = CleanupHandler.create(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        val handler2 = CleanupHandler.create(
            mockContext,
            mockConfigManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockDebugLoggingManager
        )

        // Act - cleanup both
        handler1.cleanup()
        handler2.cleanup()

        // Assert - cleanup called twice total
        verify(mockConfigManager, times(2)).getFoldStateTracker()
        verify(mockClipboardManager, times(2)).cleanup()
        verify(mockPredictionCoordinator, times(2)).shutdown()
        verify(mockDebugLoggingManager, times(2)).unregisterDebugModeReceiver(mockContext)
    }

    @Test
    fun testIntegration_partialManagerSet() {
        // Arrange - some managers null, some not
        val handler = CleanupHandler.create(
            mockContext,
            mockConfigManager, // present
            null, // null ClipboardManager
            mockPredictionCoordinator, // present
            null  // null DebugLoggingManager
        )

        // Act
        handler.cleanup()

        // Assert - only present managers cleaned up
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockPredictionCoordinator).shutdown()

        // Null managers not called
        verifyNoInteractions(mockClipboardManager)
        verifyNoInteractions(mockDebugLoggingManager)
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_debugLoggingManagerThrowsException() {
        // Arrange - debug logging manager throws exception
        doThrow(RuntimeException("Test exception"))
            .`when`(mockDebugLoggingManager).unregisterDebugModeReceiver(mockContext)

        // Act & Assert - exception propagates (handler doesn't catch)
        try {
            handler.cleanup()
            fail("Should throw exception from debug logging manager")
        } catch (e: RuntimeException) {
            assertEquals("Test exception", e.message)
        }

        // Verify other cleanup still attempted before exception
        verify(mockConfigManager).getFoldStateTracker()
        verify(mockClipboardManager).cleanup()
        verify(mockPredictionCoordinator).shutdown()
    }
}
