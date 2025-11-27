package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for PredictionInitializer.
 *
 * Tests cover:
 * - Predictions disabled (no initialization)
 * - Word prediction enabled scenarios
 * - Swipe typing enabled scenarios (with/without available engine)
 * - Factory method
 * - Multiple initialization calls
 * - Integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class PredictionInitializerTest {

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockWordPredictor: Any

    private lateinit var initializer: PredictionInitializer

    @Before
    fun setUp() {
        // Default config: predictions disabled
        mockConfig.word_prediction_enabled = false
        mockConfig.swipe_typing_enabled = false

        initializer = PredictionInitializer(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )
    }

    // ========== Predictions Disabled Tests ==========

    @Test
    fun testInitializeIfEnabled_predictionsDisabled_doesNotInitialize() {
        // Arrange - predictions disabled (default config)

        // Act
        initializer.initializeIfEnabled()

        // Assert - prediction coordinator should not be called
        verify(mockPredictionCoordinator, never()).initialize()
        verify(mockPredictionCoordinator, never()).isSwipeTypingAvailable()
    }

    // ========== Word Prediction Enabled Tests ==========

    @Test
    fun testInitializeIfEnabled_wordPredictionEnabled_initializesCoordinator() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
    }

    @Test
    fun testInitializeIfEnabled_wordPredictionEnabledSwipeDisabled_doesNotSetSwipeComponents() {
        // Arrange
        mockConfig.word_prediction_enabled = true
        mockConfig.swipe_typing_enabled = false

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockPredictionCoordinator, never()).isSwipeTypingAvailable()
        verify(mockKeyboardView, never()).setSwipeTypingComponents(any(), any())
    }

    // ========== Swipe Typing Enabled Tests ==========

    @Test
    fun testInitializeIfEnabled_swipeTypingEnabled_initializesCoordinator() {
        // Arrange
        mockConfig.swipe_typing_enabled = true

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
    }

    @Test
    fun testInitializeIfEnabled_swipeTypingEnabled_checksAvailability() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(false)

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockPredictionCoordinator).isSwipeTypingAvailable()
    }

    @Test
    fun testInitializeIfEnabled_swipeTypingAvailable_setsComponents() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(true)
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockKeyboardView).setSwipeTypingComponents(mockWordPredictor, mockKeyboard2)
    }

    @Test
    fun testInitializeIfEnabled_swipeTypingNotAvailable_doesNotSetComponents() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(false)

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockPredictionCoordinator).isSwipeTypingAvailable()
        verify(mockKeyboardView, never()).setSwipeTypingComponents(any(), any())
    }

    // ========== Both Enabled Tests ==========

    @Test
    fun testInitializeIfEnabled_bothEnabled_initializesAndSetsComponents() {
        // Arrange
        mockConfig.word_prediction_enabled = true
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(true)
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockKeyboardView).setSwipeTypingComponents(mockWordPredictor, mockKeyboard2)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesInitializer() {
        // Act
        val initializer = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        // Assert
        assertNotNull("Factory method should create initializer", initializer)
    }

    @Test
    fun testCreate_factoryMethodInitializerWorks() {
        // Arrange
        mockConfig.word_prediction_enabled = true
        val initializer = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        // Act
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
    }

    // ========== Multiple Initialization Tests ==========

    @Test
    fun testInitializeIfEnabled_calledTwice_initializesTwice() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        initializer.initializeIfEnabled()
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator, times(2)).initialize()
    }

    @Test
    fun testInitializeIfEnabled_calledTwiceWithSwipe_setsTwice() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(true)
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)

        // Act
        initializer.initializeIfEnabled()
        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator, times(2)).initialize()
        verify(mockKeyboardView, times(2)).setSwipeTypingComponents(mockWordPredictor, mockKeyboard2)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_predictionsDisabled() {
        // Act - create and initialize with predictions disabled
        val initializer = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        initializer.initializeIfEnabled()

        // Assert - no initialization occurred
        verify(mockPredictionCoordinator, never()).initialize()
    }

    @Test
    fun testFullLifecycle_wordPredictionEnabled() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        val initializer = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockKeyboardView, never()).setSwipeTypingComponents(any(), any())
    }

    @Test
    fun testFullLifecycle_swipeTypingEnabledAndAvailable() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(true)
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)

        // Act
        val initializer = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        initializer.initializeIfEnabled()

        // Assert
        verify(mockPredictionCoordinator).initialize()
        verify(mockPredictionCoordinator).isSwipeTypingAvailable()
        verify(mockPredictionCoordinator).getWordPredictor()
        verify(mockKeyboardView).setSwipeTypingComponents(mockWordPredictor, mockKeyboard2)
    }

    @Test
    fun testIntegration_multipleInitializersIndependent() {
        // Arrange - create two initializers
        mockConfig.word_prediction_enabled = true

        val initializer1 = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        val initializer2 = PredictionInitializer.create(
            mockConfig,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockKeyboard2
        )

        // Act
        initializer1.initializeIfEnabled()
        initializer2.initializeIfEnabled()

        // Assert - both initialized (called on same mock twice)
        verify(mockPredictionCoordinator, times(2)).initialize()
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_togglePredictions() {
        // Act & Assert - predictions disabled
        initializer.initializeIfEnabled()
        verify(mockPredictionCoordinator, never()).initialize()

        // Act & Assert - enable predictions
        mockConfig.word_prediction_enabled = true
        initializer.initializeIfEnabled()
        verify(mockPredictionCoordinator).initialize()

        // Act & Assert - disable again (mock already called once)
        mockConfig.word_prediction_enabled = false
        initializer.initializeIfEnabled()
        verify(mockPredictionCoordinator, times(1)).initialize() // Still just once total
    }

    @Test
    fun testEdgeCase_swipeAvailableBecomesFalse() {
        // Arrange
        mockConfig.swipe_typing_enabled = true
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(true)
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)

        // Act - first call with available
        initializer.initializeIfEnabled()
        verify(mockKeyboardView).setSwipeTypingComponents(mockWordPredictor, mockKeyboard2)

        // Arrange - becomes unavailable
        `when`(mockPredictionCoordinator.isSwipeTypingAvailable()).thenReturn(false)

        // Act - second call with unavailable
        initializer.initializeIfEnabled()

        // Assert - components only set once (when available)
        verify(mockKeyboardView, times(1)).setSwipeTypingComponents(any(), any())
    }
}
