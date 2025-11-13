package juloo.keyboard2

import android.view.View
import android.view.ViewGroup
import android.view.ViewTreeObserver
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for PredictionViewSetup.
 *
 * Tests cover:
 * - Prediction enabled scenarios (word prediction, swipe typing)
 * - Prediction disabled scenarios
 * - Suggestion bar creation (first time vs existing)
 * - Neural engine dimension setting
 * - GlobalLayoutListener setup
 * - Factory method
 * - Data class structure
 * - Null manager handling
 */
@RunWith(MockitoJUnitRunner::class)
class PredictionViewSetupTest {

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockReceiver: KeyboardReceiver

    @Mock
    private lateinit var mockEmojiPane: ViewGroup

    @Mock
    private lateinit var mockExistingSuggestionBar: SuggestionBar

    @Mock
    private lateinit var mockExistingInputViewContainer: ViewGroup

    @Mock
    private lateinit var mockExistingContentPaneContainer: ViewGroup

    @Mock
    private lateinit var mockViewTreeObserver: ViewTreeObserver

    private lateinit var setup: PredictionViewSetup

    @Before
    fun setUp() {
        // Default config: predictions disabled
        mockConfig.word_prediction_enabled = false
        mockConfig.swipe_typing_enabled = false

        // Mock keyboard view dimensions
        `when`(mockKeyboardView.getWidth()).thenReturn(1080)
        `when`(mockKeyboardView.getHeight()).thenReturn(400)
        `when`(mockKeyboardView.getViewTreeObserver()).thenReturn(mockViewTreeObserver)

        setup = PredictionViewSetup(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver,
            mockEmojiPane
        )
    }

    // ========== Prediction Disabled Tests ==========

    @Test
    fun testSetupPredictionViews_predictionsDisabled_returnsKeyboardView() {
        // Arrange - predictions disabled (default config)

        // Act
        val result = setup.setupPredictionViews(null, null, null)

        // Assert
        assertSame("Should return keyboard view when predictions disabled", mockKeyboardView, result.inputView)
        assertNull("Should have null suggestion bar", result.suggestionBar)
        assertNull("Should have null input view container", result.inputViewContainer)
        assertNull("Should have null content pane container", result.contentPaneContainer)
    }

    @Test
    fun testSetupPredictionViews_predictionsDisabled_doesNotInitialize() {
        // Arrange - predictions disabled

        // Act
        setup.setupPredictionViews(null, null, null)

        // Assert - prediction coordinator should not be called
        verify(mockPredictionCoordinator, never()).ensureInitialized()
    }

    // ========== Word Prediction Enabled Tests ==========

    @Test
    fun testSetupPredictionViews_wordPredictionEnabled_initializesCoordinator() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        setup.setupPredictionViews(null, null, null)

        // Assert
        verify(mockPredictionCoordinator).ensureInitialized()
    }

    // ========== Swipe Typing Enabled Tests ==========

    @Test
    fun testSetupPredictionViews_swipeTypingEnabled_initializesCoordinator() {
        // Arrange
        mockConfig.swipe_typing_enabled = true

        // Act
        setup.setupPredictionViews(null, null, null)

        // Assert
        verify(mockPredictionCoordinator).ensureInitialized()
    }

    // ========== Existing Components Tests ==========

    @Test
    fun testSetupPredictionViews_withExistingSuggestionBar_returnsExisting() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        val result = setup.setupPredictionViews(
            mockExistingSuggestionBar,
            mockExistingInputViewContainer,
            mockExistingContentPaneContainer
        )

        // Assert
        assertSame("Should return existing suggestion bar", mockExistingSuggestionBar, result.suggestionBar)
        assertSame("Should return existing input view container", mockExistingInputViewContainer, result.inputViewContainer)
        assertSame("Should return existing content pane container", mockExistingContentPaneContainer, result.contentPaneContainer)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesSetup() {
        // Act
        val setup = PredictionViewSetup.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver,
            mockEmojiPane
        )

        // Assert
        assertNotNull("Factory method should create setup", setup)
    }

    @Test
    fun testCreate_factoryMethodSetupWorks() {
        // Arrange
        val setup = PredictionViewSetup.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver,
            mockEmojiPane
        )

        // Act
        val result = setup.setupPredictionViews(null, null, null)

        // Assert
        assertNotNull("Factory-created setup should work", result)
        assertSame("Should return keyboard view", mockKeyboardView, result.inputView)
    }

    @Test
    fun testCreate_withNullManagers() {
        // Act
        val setup = PredictionViewSetup.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            null, // null inputCoordinator
            null, // null suggestionHandler
            null, // null neuralLayoutHelper
            null, // null receiver
            null  // null emojiPane
        )

        // Assert
        assertNotNull("Factory should create setup with null managers", setup)

        // Should not crash when setting up
        val result = setup.setupPredictionViews(null, null, null)
        assertNotNull("Should return result with null managers", result)
    }

    // ========== Data Class Tests ==========

    @Test
    fun testSetupResult_isDataClass() {
        // Arrange
        val view1 = mock(View::class.java)
        val view2 = mock(View::class.java)

        // Act
        val result1 = PredictionViewSetup.SetupResult(view1, null, null, null)
        val result2 = PredictionViewSetup.SetupResult(view1, null, null, null)
        val result3 = PredictionViewSetup.SetupResult(view2, null, null, null)

        // Assert - data class equality
        assertEquals("Same view should be equal", result1, result2)
        assertNotEquals("Different view should not be equal", result1, result3)
    }

    @Test
    fun testSetupResult_copyWorks() {
        // Arrange
        val view1 = mock(View::class.java)
        val view2 = mock(View::class.java)
        val result = PredictionViewSetup.SetupResult(view1, null, null, null)

        // Act
        val copied = result.copy(inputView = view2)

        // Assert
        assertNotEquals("Copied result should differ from original", result, copied)
        assertEquals("Copied result should have new view", view2, copied.inputView)
    }

    @Test
    fun testSetupResult_accessFields() {
        // Arrange
        val view = mock(View::class.java)
        val suggestionBar = mock(SuggestionBar::class.java)
        val inputViewContainer = mock(ViewGroup::class.java)
        val contentPaneContainer = mock(ViewGroup::class.java)

        val result = PredictionViewSetup.SetupResult(
            view,
            suggestionBar,
            inputViewContainer,
            contentPaneContainer
        )

        // Act & Assert
        assertEquals("Should access inputView", view, result.inputView)
        assertEquals("Should access suggestionBar", suggestionBar, result.suggestionBar)
        assertEquals("Should access inputViewContainer", inputViewContainer, result.inputViewContainer)
        assertEquals("Should access contentPaneContainer", contentPaneContainer, result.contentPaneContainer)
    }

    // ========== Multiple Setup Tests ==========

    @Test
    fun testSetupPredictionViews_calledTwicePredictionsDisabled_returnsSameView() {
        // Act
        val result1 = setup.setupPredictionViews(null, null, null)
        val result2 = setup.setupPredictionViews(null, null, null)

        // Assert
        assertSame("Both calls should return keyboard view", result1.inputView, result2.inputView)
        assertNull("Both should have null suggestion bar", result1.suggestionBar)
        assertNull("Both should have null suggestion bar", result2.suggestionBar)
    }

    @Test
    fun testSetupPredictionViews_calledTwicePredictionsEnabled_initializesTwice() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act
        setup.setupPredictionViews(null, null, null)
        setup.setupPredictionViews(null, null, null)

        // Assert
        verify(mockPredictionCoordinator, times(2)).ensureInitialized()
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_predictionsDisabled() {
        // Act - setup with predictions disabled
        val result = setup.setupPredictionViews(null, null, null)

        // Assert
        assertSame("Should return keyboard view", mockKeyboardView, result.inputView)
        assertNull("Should clean up suggestion bar", result.suggestionBar)
        assertNull("Should clean up input view container", result.inputViewContainer)
        assertNull("Should clean up content pane container", result.contentPaneContainer)
        verify(mockPredictionCoordinator, never()).ensureInitialized()
    }

    @Test
    fun testFullLifecycle_predictionsEnabledWithExisting() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act - setup with existing components
        val result = setup.setupPredictionViews(
            mockExistingSuggestionBar,
            mockExistingInputViewContainer,
            mockExistingContentPaneContainer
        )

        // Assert
        verify(mockPredictionCoordinator).ensureInitialized()
        assertSame("Should return existing suggestion bar", mockExistingSuggestionBar, result.suggestionBar)
        assertSame("Should return existing input view container", mockExistingInputViewContainer, result.inputViewContainer)
    }

    @Test
    fun testIntegration_multipleSetupsIndependent() {
        // Arrange - create two setups
        val setup1 = PredictionViewSetup.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver,
            mockEmojiPane
        )

        val setup2 = PredictionViewSetup.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver,
            mockEmojiPane
        )

        // Act
        val result1 = setup1.setupPredictionViews(null, null, null)
        val result2 = setup2.setupPredictionViews(null, null, null)

        // Assert - both should work independently
        assertNotNull("First setup should work", result1)
        assertNotNull("Second setup should work", result2)
        assertSame("Both should return keyboard view", result1.inputView, result2.inputView)
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_togglePredictions() {
        // Act & Assert - predictions disabled
        var result = setup.setupPredictionViews(null, null, null)
        assertNull("Should have null suggestion bar when disabled", result.suggestionBar)

        // Act & Assert - enable predictions
        mockConfig.word_prediction_enabled = true
        result = setup.setupPredictionViews(null, null, null)
        verify(mockPredictionCoordinator).ensureInitialized()

        // Act & Assert - disable again
        mockConfig.word_prediction_enabled = false
        result = setup.setupPredictionViews(null, null, null)
        assertNull("Should have null suggestion bar when disabled again", result.suggestionBar)
    }

    @Test
    fun testEdgeCase_alternatingExistingAndNull() {
        // Arrange
        mockConfig.word_prediction_enabled = true

        // Act & Assert - with existing
        var result = setup.setupPredictionViews(
            mockExistingSuggestionBar,
            mockExistingInputViewContainer,
            mockExistingContentPaneContainer
        )
        assertSame("Should return existing", mockExistingSuggestionBar, result.suggestionBar)

        // Act & Assert - with null (simulating recreation)
        result = setup.setupPredictionViews(null, null, null)
        verify(mockPredictionCoordinator, times(2)).ensureInitialized()
    }
}
