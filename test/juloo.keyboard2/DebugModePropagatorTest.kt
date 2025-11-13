package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentCaptor
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for DebugModePropagator.
 *
 * Tests cover:
 * - Debug mode propagation to SuggestionHandler
 * - Debug mode propagation to NeuralLayoutHelper
 * - Logger adapter creation for NeuralLayoutHelper
 * - Null manager handling
 * - Enable and disable scenarios
 * - Factory method
 * - Integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class DebugModePropagatorTest {

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockDebugLogger: SuggestionHandler.DebugLogger

    @Mock
    private lateinit var mockDebugLoggingManager: DebugLoggingManager

    private lateinit var propagator: DebugModePropagator

    @Before
    fun setUp() {
        propagator = DebugModePropagator(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLogger,
            mockDebugLoggingManager
        )
    }

    // ========== Debug Mode Enabled Tests ==========

    @Test
    fun testOnDebugModeChanged_enabled_propagatesToSuggestionHandler() {
        // Act
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
    }

    @Test
    fun testOnDebugModeChanged_enabled_propagatesToNeuralLayoutHelper() {
        // Act
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    @Test
    fun testOnDebugModeChanged_enabled_propagatesToBothManagers() {
        // Act
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    // ========== Debug Mode Disabled Tests ==========

    @Test
    fun testOnDebugModeChanged_disabled_propagatesToSuggestionHandler() {
        // Act
        propagator.onDebugModeChanged(false)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
    }

    @Test
    fun testOnDebugModeChanged_disabled_propagatesToNeuralLayoutHelper() {
        // Act
        propagator.onDebugModeChanged(false)

        // Assert
        verify(mockNeuralLayoutHelper).setDebugMode(eq(false), any())
    }

    @Test
    fun testOnDebugModeChanged_disabled_propagatesToBothManagers() {
        // Act
        propagator.onDebugModeChanged(false)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(false), any())
    }

    // ========== Logger Adapter Tests ==========

    @Test
    fun testOnDebugModeChanged_neuralLayoutHelperReceivesLoggerAdapter() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)

        // Act
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())
        assertNotNull("Logger adapter should be created", loggerCaptor.value)
    }

    @Test
    fun testLoggerAdapter_sendsDebugLogToManager() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)
        propagator.onDebugModeChanged(true)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())

        // Act - use the captured logger adapter
        val adapter = loggerCaptor.value
        adapter.sendDebugLog("test message")

        // Assert
        verify(mockDebugLoggingManager).sendDebugLog("test message")
    }

    @Test
    fun testLoggerAdapter_forwardsMultipleMessages() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)
        propagator.onDebugModeChanged(true)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())
        val adapter = loggerCaptor.value

        // Act
        adapter.sendDebugLog("message 1")
        adapter.sendDebugLog("message 2")
        adapter.sendDebugLog("message 3")

        // Assert
        verify(mockDebugLoggingManager).sendDebugLog("message 1")
        verify(mockDebugLoggingManager).sendDebugLog("message 2")
        verify(mockDebugLoggingManager).sendDebugLog("message 3")
    }

    // ========== Null Manager Tests ==========

    @Test
    fun testOnDebugModeChanged_withNullSuggestionHandler_doesNotCrash() {
        // Arrange
        val propagator = DebugModePropagator(
            null, // null SuggestionHandler
            mockNeuralLayoutHelper,
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        propagator.onDebugModeChanged(true)

        // Only NeuralLayoutHelper should be called
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    @Test
    fun testOnDebugModeChanged_withNullNeuralLayoutHelper_doesNotCrash() {
        // Arrange
        val propagator = DebugModePropagator(
            mockSuggestionHandler,
            null, // null NeuralLayoutHelper
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        propagator.onDebugModeChanged(true)

        // Only SuggestionHandler should be called
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
    }

    @Test
    fun testOnDebugModeChanged_withBothManagersNull_doesNotCrash() {
        // Arrange
        val propagator = DebugModePropagator(
            null, // null SuggestionHandler
            null, // null NeuralLayoutHelper
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Act & Assert - should not throw
        propagator.onDebugModeChanged(true)
        propagator.onDebugModeChanged(false)
    }

    // ========== Multiple Propagation Tests ==========

    @Test
    fun testOnDebugModeChanged_calledMultipleTimes_propagatesEachTime() {
        // Act
        propagator.onDebugModeChanged(true)
        propagator.onDebugModeChanged(false)
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
        verify(mockSuggestionHandler, times(2)).setDebugMode(true, mockDebugLogger)

        verify(mockNeuralLayoutHelper, times(3)).setDebugMode(anyBoolean(), any())
    }

    @Test
    fun testOnDebugModeChanged_toggleDebugMode_propagatesCorrectly() {
        // Act
        propagator.onDebugModeChanged(true)
        propagator.onDebugModeChanged(false)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesPropagator() {
        // Act
        val propagator = DebugModePropagator.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Assert
        assertNotNull("Factory method should create propagator", propagator)
    }

    @Test
    fun testCreate_factoryMethodPropagatorWorks() {
        // Arrange
        val propagator = DebugModePropagator.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Act
        propagator.onDebugModeChanged(true)

        // Assert
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    @Test
    fun testCreate_withNullManagers() {
        // Act
        val propagator = DebugModePropagator.create(
            null,
            null,
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Assert
        assertNotNull("Factory should create propagator with null managers", propagator)

        // Should not crash
        propagator.onDebugModeChanged(true)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_enableDisableDebugMode() {
        // Act - simulate full debug mode lifecycle
        propagator.onDebugModeChanged(true)
        propagator.onDebugModeChanged(false)

        // Assert - verify correct propagation
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
        verify(mockNeuralLayoutHelper).setDebugMode(eq(false), any())
    }

    @Test
    fun testIntegration_loggerAdapterWorksWithRealMessages() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)
        propagator.onDebugModeChanged(true)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())
        val adapter = loggerCaptor.value

        // Act - simulate real log messages
        adapter.sendDebugLog("CGR: Processing swipe path")
        adapter.sendDebugLog("CGR: Prediction confidence: 0.95")
        adapter.sendDebugLog("CGR: Word selected: hello")

        // Assert
        verify(mockDebugLoggingManager).sendDebugLog("CGR: Processing swipe path")
        verify(mockDebugLoggingManager).sendDebugLog("CGR: Prediction confidence: 0.95")
        verify(mockDebugLoggingManager).sendDebugLog("CGR: Word selected: hello")
    }

    // ========== Edge Case Tests ==========

    @Test
    fun testLoggerAdapter_withEmptyMessage() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)
        propagator.onDebugModeChanged(true)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())
        val adapter = loggerCaptor.value

        // Act
        adapter.sendDebugLog("")

        // Assert
        verify(mockDebugLoggingManager).sendDebugLog("")
    }

    @Test
    fun testLoggerAdapter_withNullMessage() {
        // Arrange
        val loggerCaptor = ArgumentCaptor.forClass(NeuralLayoutHelper.DebugLogger::class.java)
        propagator.onDebugModeChanged(true)
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), loggerCaptor.capture())
        val adapter = loggerCaptor.value

        // Act
        adapter.sendDebugLog(null)

        // Assert
        verify(mockDebugLoggingManager).sendDebugLog(null)
    }

    @Test
    fun testOnDebugModeChanged_multiplePropagators_independent() {
        // Arrange
        val propagator2 = DebugModePropagator.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLogger,
            mockDebugLoggingManager
        )

        // Act
        propagator.onDebugModeChanged(true)
        propagator2.onDebugModeChanged(false)

        // Assert - verify both propagators work independently
        verify(mockSuggestionHandler).setDebugMode(true, mockDebugLogger)
        verify(mockSuggestionHandler).setDebugMode(false, mockDebugLogger)
    }
}
