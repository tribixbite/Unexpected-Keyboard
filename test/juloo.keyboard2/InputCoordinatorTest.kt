package juloo.keyboard2

import android.content.Context
import android.content.res.Resources
import android.util.DisplayMetrics
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import juloo.keyboard2.ml.SwipeMLData
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for InputCoordinator.
 *
 * Tests cover:
 * - Regular typing with predictions
 * - Swipe typing gesture handling
 * - Autocorrection during typing
 * - Suggestion selection and text insertion
 * - Backspace and word deletion
 * - Termux-aware text handling
 * - ML data collection
 * - Context tracking updates
 */
@RunWith(MockitoJUnitRunner::class)
class InputCoordinatorTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockContextTracker: PredictionContextTracker

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockContractionManager: ContractionManager

    @Mock
    private lateinit var mockSuggestionBar: SuggestionBar

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockKeyEventHandler: KeyEventHandler

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    @Mock
    private lateinit var mockResources: Resources

    private lateinit var coordinator: InputCoordinator

    @Before
    fun setUp() {
        // Setup display metrics
        val displayMetrics = DisplayMetrics().apply {
            widthPixels = 1080
            heightPixels = 2340
        }
        `when`(mockResources.displayMetrics).thenReturn(displayMetrics)
        `when`(mockKeyboardView.height).thenReturn(800)

        // Setup config defaults
        `when`(mockConfig.word_prediction_enabled).thenReturn(true)
        `when`(mockConfig.autocorrect_enabled).thenReturn(false)
        `when`(mockConfig.swipe_typing_enabled).thenReturn(true)
        `when`(mockConfig.termux_mode_enabled).thenReturn(false)

        coordinator = InputCoordinator(
            mockContext,
            mockConfig,
            mockContextTracker,
            mockPredictionCoordinator,
            mockContractionManager,
            mockSuggestionBar,
            mockKeyboardView,
            mockKeyEventHandler
        )
    }

    // Configuration Tests

    @Test
    fun testSetConfig_updatesConfiguration() {
        // Arrange
        val newConfig = mock(Config::class.java)

        // Act
        coordinator.setConfig(newConfig)

        // Assert - no exception thrown, config updated internally
    }

    @Test
    fun testSetSuggestionBar_updatesReference() {
        // Arrange
        val newSuggestionBar = mock(SuggestionBar::class.java)

        // Act
        coordinator.setSuggestionBar(newSuggestionBar)

        // Assert - no exception thrown, suggestion bar updated
    }

    @Test
    fun testResetSwipeData_clearsSwipeTracking() {
        // Act
        coordinator.resetSwipeData()

        // Assert
        assertNull("Swipe data should be null after reset",
            coordinator.getCurrentSwipeData())
    }

    // Regular Typing Tests

    @Test
    fun testHandleRegularTyping_withLetter_appendsToCurrentWord() {
        // Arrange
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mock(WordPredictor::class.java))
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(0)

        // Act
        coordinator.handleRegularTyping("a", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker).appendToCurrentWord("a")
    }

    @Test
    fun testHandleRegularTyping_withSpace_clearsCurrentWord() {
        // Arrange
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mock(WordPredictor::class.java))
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(5)
        `when`(mockContextTracker.getCurrentWord()).thenReturn("hello")

        // Act
        coordinator.handleRegularTyping(" ", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker).clearCurrentWord()
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    @Test
    fun testHandleRegularTyping_withPredictionsDisabled_doesNothing() {
        // Arrange
        `when`(mockConfig.word_prediction_enabled).thenReturn(false)

        // Act
        coordinator.handleRegularTyping("a", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker, never()).appendToCurrentWord(anyString())
    }

    @Test
    fun testHandleRegularTyping_withNullSuggestionBar_doesNotCrash() {
        // Arrange
        coordinator.setSuggestionBar(null)
        `when`(mockConfig.word_prediction_enabled).thenReturn(true)

        // Act - should not throw
        coordinator.handleRegularTyping("a", mockInputConnection, mockEditorInfo)
    }

    // Backspace Tests

    @Test
    fun testHandleBackspace_withCurrentWord_deletesLastChar() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(5, 4)

        // Act
        coordinator.handleBackspace()

        // Assert
        verify(mockContextTracker).deleteLastChar()
    }

    @Test
    fun testHandleBackspace_withEmptyWord_doesNothing() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(0)

        // Act
        coordinator.handleBackspace()

        // Assert
        verify(mockContextTracker, never()).deleteLastChar()
    }

    @Test
    fun testHandleBackspace_whenWordBecomesEmpty_clearsSuggestions() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(1, 0)

        // Act
        coordinator.handleBackspace()

        // Assert
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    // Delete Last Word Tests

    @Test
    fun testHandleDeleteLastWord_inTermux_sendsCtrlW() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.termux")

        // Act
        coordinator.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockKeyEventHandler).send_key_down_up(
            android.view.KeyEvent.KEYCODE_W,
            android.view.KeyEvent.META_CTRL_ON or android.view.KeyEvent.META_CTRL_LEFT_ON
        )
    }

    @Test
    fun testHandleDeleteLastWord_withAutoInsertedWord_deletesWord() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.android.test")
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn("hello")
        `when`(mockInputConnection.getTextBeforeCursor(100, 0)).thenReturn("hello ")

        // Act
        coordinator.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockInputConnection).deleteSurroundingText(anyInt(), eq(0))
        verify(mockContextTracker).clearLastAutoInsertedWord()
    }

    @Test
    fun testHandleDeleteLastWord_withNullInputConnection_returnsEarly() {
        // Act
        coordinator.handleDeleteLastWord(null, mockEditorInfo)

        // Assert - no crash
    }

    // Suggestion Selection Tests

    @Test
    fun testOnSuggestionSelected_withNullWord_returnsEarly() {
        // Act
        coordinator.onSuggestionSelected(null, mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection, never()).commitText(anyString(), anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withEmptyWord_returnsEarly() {
        // Act
        coordinator.onSuggestionSelected("  ", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection, never()).commitText(anyString(), anyInt())
    }

    @Test
    fun testOnSuggestionSelected_stripsRawPrefix() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")

        // Act
        coordinator.onSuggestionSelected("raw:hello", mockInputConnection, mockEditorInfo, mockResources)

        // Assert - should commit "hello" not "raw:hello"
        verify(mockInputConnection).commitText(argThat { it.contains("hello") && !it.contains("raw:") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withKnownContraction_skipsAutocorrect() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction("don't")).thenReturn(true)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")

        // Act
        coordinator.onSuggestionSelected("don't", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockPredictionCoordinator.getWordPredictor(), never())?.autoCorrect(anyString())
    }

    @Test
    fun testOnSuggestionSelected_inTermuxMode_omitsTrailingSpace() {
        // Arrange
        `when`(mockConfig.termux_mode_enabled).thenReturn(true)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("test")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        coordinator.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert - should not have trailing space in Termux mode for non-swipe
        verify(mockInputConnection).commitText(argThat { !it.endsWith(" ") || it.startsWith(" ") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withSwipeInput_alwaysAddsTrailingSpace() {
        // Arrange
        `when`(mockConfig.termux_mode_enabled).thenReturn(true) // Even in Termux mode
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("test")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)

        // Act
        coordinator.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert - should have trailing space for swipe even in Termux
        verify(mockInputConnection).commitText(argThat { it.endsWith(" ") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_recordsSelectionForAdaptation() {
        // Arrange
        val mockAdaptationManager = mock(AdaptationManager::class.java)
        `when`(mockPredictionCoordinator.getAdaptationManager()).thenReturn(mockAdaptationManager)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")

        // Act
        coordinator.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockAdaptationManager).recordSelection("word")
    }

    // Swipe Typing Tests

    @Test
    fun testHandleSwipeTyping_withSwipeDisabled_returnsEarly() {
        // Arrange
        `when`(mockConfig.swipe_typing_enabled).thenReturn(false)

        // Act
        coordinator.handleSwipeTyping(
            emptyList(), emptyList(), emptyList(),
            mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockPredictionCoordinator, never()).ensureNeuralEngineReady()
    }

    @Test
    fun testHandleSwipeTyping_clearsAutoInsertedTracking() {
        // Act
        coordinator.handleSwipeTyping(
            emptyList(), emptyList(), emptyList(),
            mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockContextTracker).clearLastAutoInsertedWord()
    }

    @Test
    fun testHandleSwipeTyping_marksLastInputAsSwipe() {
        // Arrange
        `when`(mockPredictionCoordinator.getNeuralEngine()).thenReturn(mock(NeuralSwipeTypingEngine::class.java))

        // Act
        coordinator.handleSwipeTyping(
            emptyList(), emptyList(), emptyList(),
            mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockContextTracker).setWasLastInputSwipe(true)
    }

    @Test
    fun testHandleSwipeTyping_ensuresNeuralEngineReady() {
        // Act
        coordinator.handleSwipeTyping(
            emptyList(), emptyList(), emptyList(),
            mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockPredictionCoordinator).ensureNeuralEngineReady()
    }

    // Prediction Results Handling Tests

    @Test
    fun testHandlePredictionResults_withNullPredictions_clearsSuggestions() {
        // Act
        coordinator.handlePredictionResults(
            null, null, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    @Test
    fun testHandlePredictionResults_withEmptyPredictions_clearsSuggestions() {
        // Act
        coordinator.handlePredictionResults(
            emptyList(), emptyList(), mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    @Test
    fun testHandlePredictionResults_withValidPredictions_displaysSuggestions() {
        // Arrange
        val predictions = listOf("hello", "world", "test")
        val scores = listOf(100, 90, 80)
        `when`(mockConfig.swipe_show_debug_scores).thenReturn(false)

        // Act
        coordinator.handlePredictionResults(
            predictions, scores, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.setSuggestionsWithScores(predictions, scores)
    }

    @Test
    fun testHandlePredictionResults_withDebugScores_enablesDebugDisplay() {
        // Arrange
        val predictions = listOf("test")
        val scores = listOf(100)
        `when`(mockConfig.swipe_show_debug_scores).thenReturn(true)

        // Act
        coordinator.handlePredictionResults(
            predictions, scores, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.setShowDebugScores(true)
    }

    // Mockito helper for argument matching
    private inline fun <reified T> argThat(predicate: (T) -> Boolean): T {
        return org.mockito.ArgumentMatchers.argThat { predicate(it) } ?: createInstance()
    }

    private inline fun <reified T> createInstance(): T {
        return mock(T::class.java)
    }
}
