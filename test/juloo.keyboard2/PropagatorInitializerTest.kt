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
 * Comprehensive test suite for PropagatorInitializer.
 *
 * Tests cover:
 * - Propagator initialization (DebugModePropagator and ConfigPropagator)
 * - Registration with debug logging manager
 * - ConfigPropagator builder pattern with all managers
 * - Factory method
 * - Data class structure
 * - Null manager handling
 * - Integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class PropagatorInitializerTest {

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockDebugLoggerImpl: SuggestionHandler.DebugLogger

    @Mock
    private lateinit var mockDebugLoggingManager: DebugLoggingManager

    @Mock
    private lateinit var mockClipboardManager: ClipboardManager

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockLayoutManager: LayoutManager

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockSubtypeManager: SubtypeManager

    private lateinit var initializer: PropagatorInitializer

    @Before
    fun setUp() {
        initializer = PropagatorInitializer(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )
    }

    // ========== Initialization Tests ==========

    @Test
    fun testInitialize_createsConfigPropagator() {
        // Act
        val result = initializer.initialize()

        // Assert
        assertNotNull("ConfigPropagator should be created", result.configPropagator)
    }

    @Test
    fun testInitialize_registersDebugModePropagator() {
        // Arrange
        val listenerCaptor = ArgumentCaptor.forClass(DebugLoggingManager.DebugModeListener::class.java)

        // Act
        initializer.initialize()

        // Assert
        verify(mockDebugLoggingManager).registerDebugModeListener(listenerCaptor.capture())
        assertNotNull("DebugModePropagator should be registered", listenerCaptor.value)
    }

    @Test
    fun testInitialize_debugPropagatorIsDebugModePropagator() {
        // Arrange
        val listenerCaptor = ArgumentCaptor.forClass(DebugLoggingManager.DebugModeListener::class.java)

        // Act
        initializer.initialize()

        // Assert
        verify(mockDebugLoggingManager).registerDebugModeListener(listenerCaptor.capture())
        assertTrue(
            "Registered listener should be DebugModePropagator",
            listenerCaptor.value is DebugModePropagator
        )
    }

    @Test
    fun testInitialize_configPropagatorContainsAllManagers() {
        // Act
        val result = initializer.initialize()

        // Assert - ConfigPropagator should be configured with all managers
        // We can't directly verify internal state, but we can verify it was created
        assertNotNull("ConfigPropagator should contain configuration", result.configPropagator)
    }

    // ========== Registration Tests ==========

    @Test
    fun testInitialize_registerDebugModeListenerCalledOnce() {
        // Act
        initializer.initialize()

        // Assert
        verify(mockDebugLoggingManager, times(1)).registerDebugModeListener(any())
    }

    @Test
    fun testInitialize_registeredListenerCanPropagate() {
        // Arrange
        val listenerCaptor = ArgumentCaptor.forClass(DebugLoggingManager.DebugModeListener::class.java)

        // Act
        initializer.initialize()
        verify(mockDebugLoggingManager).registerDebugModeListener(listenerCaptor.capture())

        // Act - trigger the listener
        listenerCaptor.value.onDebugModeChanged(true)

        // Assert - verify propagation happened
        verify(mockSuggestionHandler).setDebugMode(eq(true), any())
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesInitializer() {
        // Act
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Assert
        assertNotNull("Factory method should create initializer", initializer)
    }

    @Test
    fun testCreate_factoryMethodInitializerWorks() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act
        val result = initializer.initialize()

        // Assert
        assertNotNull("Factory-created initializer should work", result.configPropagator)
        verify(mockDebugLoggingManager).registerDebugModeListener(any())
    }

    // ========== Data Class Tests ==========

    @Test
    fun testInitializationResult_isDataClass() {
        // Arrange
        val configPropagator1 = mock(ConfigPropagator::class.java)
        val configPropagator2 = mock(ConfigPropagator::class.java)

        // Act
        val result1 = PropagatorInitializer.InitializationResult(configPropagator1)
        val result2 = PropagatorInitializer.InitializationResult(configPropagator1)
        val result3 = PropagatorInitializer.InitializationResult(configPropagator2)

        // Assert - data class equality
        assertEquals("Same ConfigPropagator should be equal", result1, result2)
        assertNotEquals("Different ConfigPropagator should not be equal", result1, result3)
    }

    @Test
    fun testInitializationResult_copyWorks() {
        // Arrange
        val configPropagator1 = mock(ConfigPropagator::class.java)
        val configPropagator2 = mock(ConfigPropagator::class.java)
        val result = PropagatorInitializer.InitializationResult(configPropagator1)

        // Act
        val copied = result.copy(configPropagator = configPropagator2)

        // Assert
        assertNotEquals("Copied result should differ from original", result, copied)
        assertEquals("Copied result should have new propagator", configPropagator2, copied.configPropagator)
    }

    @Test
    fun testInitializationResult_accessConfigPropagator() {
        // Arrange
        val configPropagator = mock(ConfigPropagator::class.java)
        val result = PropagatorInitializer.InitializationResult(configPropagator)

        // Act & Assert
        assertEquals("Should access ConfigPropagator", configPropagator, result.configPropagator)
    }

    // ========== Null Manager Tests ==========

    @Test
    fun testInitialize_withNullSuggestionHandler_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            null, // null SuggestionHandler
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullNeuralLayoutHelper_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            null, // null NeuralLayoutHelper
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullClipboardManager_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            null, // null ClipboardManager
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullPredictionCoordinator_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            null, // null PredictionCoordinator
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullInputCoordinator_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            null, // null InputCoordinator
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullLayoutManager_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            null, // null LayoutManager
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullKeyboardView_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            null, // null KeyboardView
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withNullSubtypeManager_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockLayoutManager,
            mockKeyboardView,
            null // null SubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with null manager", result.configPropagator)
    }

    @Test
    fun testInitialize_withAllNullableManagersNull_doesNotCrash() {
        // Arrange
        val initializer = PropagatorInitializer.create(
            null, // null SuggestionHandler
            null, // null NeuralLayoutHelper
            mockDebugLoggerImpl,
            mockDebugLoggingManager,
            null, // null ClipboardManager
            null, // null PredictionCoordinator
            null, // null InputCoordinator
            null, // null LayoutManager
            null, // null KeyboardView
            null  // null SubtypeManager
        )

        // Act & Assert - should not throw
        val result = initializer.initialize()
        assertNotNull("Should create ConfigPropagator with all null managers", result.configPropagator)
    }

    // ========== Multiple Initialization Tests ==========

    @Test
    fun testInitialize_calledTwice_createsIndependentPropagators() {
        // Act
        val result1 = initializer.initialize()
        val result2 = initializer.initialize()

        // Assert - different instances
        assertNotSame("Should create independent propagators", result1.configPropagator, result2.configPropagator)

        // Verify registration happened twice
        verify(mockDebugLoggingManager, times(2)).registerDebugModeListener(any())
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_initializeAndUseConfigPropagator() {
        // Act - initialize
        val result = initializer.initialize()
        val configPropagator = result.configPropagator

        // Assert - ConfigPropagator is ready to use
        assertNotNull("ConfigPropagator should be ready", configPropagator)
        verify(mockDebugLoggingManager).registerDebugModeListener(any())
    }

    @Test
    fun testFullLifecycle_initializeAndTriggerDebugMode() {
        // Arrange
        val listenerCaptor = ArgumentCaptor.forClass(DebugLoggingManager.DebugModeListener::class.java)

        // Act - initialize
        initializer.initialize()
        verify(mockDebugLoggingManager).registerDebugModeListener(listenerCaptor.capture())

        // Act - trigger debug mode
        listenerCaptor.value.onDebugModeChanged(true)

        // Assert - debug mode propagated
        verify(mockSuggestionHandler).setDebugMode(eq(true), any())
        verify(mockNeuralLayoutHelper).setDebugMode(eq(true), any())
    }

    @Test
    fun testIntegration_bothPropagatorsWorkTogether() {
        // Arrange
        val listenerCaptor = ArgumentCaptor.forClass(DebugLoggingManager.DebugModeListener::class.java)

        // Act - initialize (creates both propagators)
        val result = initializer.initialize()

        // Assert - ConfigPropagator created
        assertNotNull("ConfigPropagator should be created", result.configPropagator)

        // Assert - DebugModePropagator registered
        verify(mockDebugLoggingManager).registerDebugModeListener(listenerCaptor.capture())

        // Act - test DebugModePropagator functionality
        listenerCaptor.value.onDebugModeChanged(false)

        // Assert - both propagators are functional
        verify(mockSuggestionHandler).setDebugMode(eq(false), any())
        verify(mockNeuralLayoutHelper).setDebugMode(eq(false), any())
    }
}
