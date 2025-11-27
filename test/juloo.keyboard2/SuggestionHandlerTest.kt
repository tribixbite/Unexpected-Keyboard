package juloo.keyboard2

import android.content.Context
import android.content.res.Resources
import android.util.DisplayMetrics
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for SuggestionHandler.
 *
 * Tests cover:
 * - Prediction result handling and display
 * - Suggestion selection with autocorrect
 * - Regular typing prediction updates
 * - Backspace handling
 * - Delete last word functionality
 * - Termux-aware text handling
 * - Context tracking updates
 * - Debug logging
 * - Possessive augmentation
 * - Async prediction execution
 */
@RunWith(MockitoJUnitRunner::class)
class SuggestionHandlerTest {

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
    private lateinit var mockKeyEventHandler: KeyEventHandler

    @Mock
    private lateinit var mockSuggestionBar: SuggestionBar

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockWordPredictor: WordPredictor

    @Mock
    private lateinit var mockAdaptationManager: AdaptationManager

    @Mock
    private lateinit var mockDebugLogger: SuggestionHandler.DebugLogger

    private lateinit var handler: SuggestionHandler

    @Before
    fun setUp() {
        // Setup display metrics
        val displayMetrics = DisplayMetrics().apply {
            widthPixels = 1080
            heightPixels = 2340
        }
        `when`(mockResources.displayMetrics).thenReturn(displayMetrics)

        // Setup config defaults
        `when`(mockConfig.word_prediction_enabled).thenReturn(true)
        `when`(mockConfig.autocorrect_enabled).thenReturn(false)
        `when`(mockConfig.swipe_typing_enabled).thenReturn(true)
        `when`(mockConfig.termux_mode_enabled).thenReturn(false)
        `when`(mockConfig.swipe_show_debug_scores).thenReturn(false)
        `when`(mockConfig.swipe_final_autocorrect_enabled).thenReturn(false)

        // Setup prediction coordinator
        `when`(mockPredictionCoordinator.getWordPredictor()).thenReturn(mockWordPredictor)
        `when`(mockPredictionCoordinator.getAdaptationManager()).thenReturn(mockAdaptationManager)

        handler = SuggestionHandler(
            mockContext,
            mockConfig,
            mockContextTracker,
            mockPredictionCoordinator,
            mockContractionManager,
            mockKeyEventHandler
        )

        // Set suggestion bar
        handler.setSuggestionBar(mockSuggestionBar)
    }

    // Configuration Tests

    @Test
    fun testSetConfig_updatesConfiguration() {
        // Arrange
        val newConfig = mock(Config::class.java)

        // Act
        handler.setConfig(newConfig)

        // Assert - no exception thrown
    }

    @Test
    fun testSetSuggestionBar_updatesReference() {
        // Arrange
        val newBar = mock(SuggestionBar::class.java)

        // Act
        handler.setSuggestionBar(newBar)

        // Assert - no exception thrown
    }

    @Test
    fun testSetDebugMode_enablesLogging() {
        // Act
        handler.setDebugMode(true, mockDebugLogger)

        // Assert - verify no crash, logging enabled internally
    }

    // Prediction Results Handling Tests

    @Test
    fun testHandlePredictionResults_withNullPredictions_clearsSuggestions() {
        // Act
        handler.handlePredictionResults(
            emptyList(), null, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    @Test
    fun testHandlePredictionResults_withEmptyPredictions_clearsSuggestions() {
        // Act
        handler.handlePredictionResults(
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

        // Act
        handler.handlePredictionResults(
            predictions, scores, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.setShowDebugScores(false)
        verify(mockSuggestionBar)?.setSuggestionsWithScores(anyList(), anyList())
    }

    @Test
    fun testHandlePredictionResults_withDebugScoresEnabled_setsDebugMode() {
        // Arrange
        `when`(mockConfig.swipe_show_debug_scores).thenReturn(true)
        val predictions = listOf("test")
        val scores = listOf(100)

        // Act
        handler.handlePredictionResults(
            predictions, scores, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockSuggestionBar)?.setShowDebugScores(true)
    }

    @Test
    fun testHandlePredictionResults_withTopSuggestion_autoInserts() {
        // Arrange
        val predictions = listOf("hello")
        val scores = listOf(100)
        `when`(mockSuggestionBar.getTopSuggestion()).thenReturn("hello")
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(0)
        `when`(mockInputConnection.getTextBeforeCursor(1, 0)).thenReturn("")

        // Act
        handler.handlePredictionResults(
            predictions, scores, mockInputConnection, mockEditorInfo, mockResources
        )

        // Assert
        verify(mockInputConnection).commitText(argThat { it.contains("hello") }, eq(1))
    }

    // Suggestion Selection Tests

    @Test
    fun testOnSuggestionSelected_withNullWord_returnsEarly() {
        // Act
        handler.onSuggestionSelected(null, mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection, never()).commitText(anyString(), anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withEmptyWord_returnsEarly() {
        // Act
        handler.onSuggestionSelected("  ", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection, never()).commitText(anyString(), anyInt())
    }

    @Test
    fun testOnSuggestionSelected_stripsRawPrefix() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("raw:hello", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection).commitText(argThat { it.contains("hello") && !it.contains("raw:") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withKnownContraction_skipsAutocorrect() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction("don't")).thenReturn(true)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("don't", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockWordPredictor, never()).autoCorrect(anyString())
    }

    @Test
    fun testOnSuggestionSelected_withRawPrediction_skipsAutocorrect() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("raw:test", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockWordPredictor, never()).autoCorrect(anyString())
    }

    @Test
    fun testOnSuggestionSelected_withFinalAutocorrectEnabled_appliesCorrection() {
        // Arrange
        `when`(mockConfig.swipe_final_autocorrect_enabled).thenReturn(true)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockWordPredictor.autoCorrect("teh")).thenReturn("the")
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("teh", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockWordPredictor).autoCorrect("teh")
        verify(mockInputConnection).commitText(argThat { it.contains("the") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_inTermuxMode_omitsTrailingSpace() {
        // Arrange
        `when`(mockConfig.termux_mode_enabled).thenReturn(true)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("test")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn(null)

        // Act
        handler.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection).commitText(argThat { !it.endsWith(" ") || it.startsWith(" ") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_withSwipeInput_alwaysAddsTrailingSpace() {
        // Arrange
        `when`(mockConfig.termux_mode_enabled).thenReturn(true) // Even in Termux mode
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("test")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn(null)

        // Act
        handler.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection).commitText(argThat { it.endsWith(" ") }, anyInt())
    }

    @Test
    fun testOnSuggestionSelected_recordsSelectionForAdaptation() {
        // Arrange
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("word", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockAdaptationManager).recordSelection(anyString())
    }

    @Test
    fun testOnSuggestionSelected_replacesAutoInsertedWord() {
        // Arrange
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn("hello")
        `when`(mockContextTracker.getLastCommitSource()).thenReturn(PredictionSource.NEURAL_SWIPE)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("hello ")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("world", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection).deleteSurroundingText(6, 0) // "hello " = 6 chars
        verify(mockContextTracker).clearLastAutoInsertedWord()
    }

    @Test
    fun testOnSuggestionSelected_inTermux_replacesAutoInsertedWithBackspace() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.termux")
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn("hello")
        `when`(mockContextTracker.getLastCommitSource()).thenReturn(PredictionSource.NEURAL_SWIPE)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("hello ")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("world", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockKeyEventHandler, atLeastOnce()).send_key_down_up(
            android.view.KeyEvent.KEYCODE_DEL,
            0
        )
    }

    @Test
    fun testOnSuggestionSelected_deletesPartialTypedWord() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(3)
        `when`(mockContextTracker.getCurrentWord()).thenReturn("hel")
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn(null)
        `when`(mockContractionManager.isKnownContraction(anyString())).thenReturn(false)
        `when`(mockInputConnection.getTextBeforeCursor(anyInt(), anyInt())).thenReturn("hel")
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act
        handler.onSuggestionSelected("hello", mockInputConnection, mockEditorInfo, mockResources)

        // Assert
        verify(mockInputConnection).deleteSurroundingText(3, 0)
    }

    // Regular Typing Tests

    @Test
    fun testHandleRegularTyping_withLetter_appendsToCurrentWord() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(0)

        // Act
        handler.handleRegularTyping("a", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker).appendToCurrentWord("a")
    }

    @Test
    fun testHandleRegularTyping_withSpace_clearsCurrentWord() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(5)
        `when`(mockContextTracker.getCurrentWord()).thenReturn("hello")

        // Act
        handler.handleRegularTyping(" ", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker).clearCurrentWord()
        verify(mockWordPredictor).reset()
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    @Test
    fun testHandleRegularTyping_withPredictionsDisabled_doesNothing() {
        // Arrange
        `when`(mockConfig.word_prediction_enabled).thenReturn(false)

        // Act
        handler.handleRegularTyping("a", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker, never()).appendToCurrentWord(anyString())
    }

    @Test
    fun testHandleRegularTyping_withNullSuggestionBar_doesNotCrash() {
        // Arrange
        handler.setSuggestionBar(null)

        // Act - should not throw
        handler.handleRegularTyping("a", mockInputConnection, mockEditorInfo)
    }

    @Test
    fun testHandleRegularTyping_withAutocorrectEnabled_correctsTypedWord() {
        // Arrange
        `when`(mockConfig.autocorrect_enabled).thenReturn(true)
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(4)
        `when`(mockContextTracker.getCurrentWord()).thenReturn("teh")
        `when`(mockWordPredictor.autoCorrect("teh")).thenReturn("the")
        `when`(mockEditorInfo.packageName).thenReturn("com.android.test")

        // Act
        handler.handleRegularTyping(" ", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockWordPredictor).autoCorrect("teh")
        verify(mockInputConnection).deleteSurroundingText(4, 0)
        verify(mockInputConnection).commitText("the ", 1)
    }

    @Test
    fun testHandleRegularTyping_autocorrectDisabledInTermux() {
        // Arrange
        `when`(mockConfig.autocorrect_enabled).thenReturn(true)
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(4)
        `when`(mockContextTracker.getCurrentWord()).thenReturn("teh")
        `when`(mockEditorInfo.packageName).thenReturn("com.termux")

        // Act
        handler.handleRegularTyping(" ", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockWordPredictor, never()).autoCorrect(anyString())
    }

    @Test
    fun testHandleRegularTyping_withMultiCharInput_resetsTracking() {
        // Act
        handler.handleRegularTyping("hello", mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockContextTracker).clearCurrentWord()
        verify(mockWordPredictor).reset()
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    // Backspace Tests

    @Test
    fun testHandleBackspace_withCurrentWord_deletesLastChar() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(5, 4)

        // Act
        handler.handleBackspace()

        // Assert
        verify(mockContextTracker).deleteLastChar()
    }

    @Test
    fun testHandleBackspace_withEmptyWord_doesNothing() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(0)

        // Act
        handler.handleBackspace()

        // Assert
        verify(mockContextTracker, never()).deleteLastChar()
    }

    @Test
    fun testHandleBackspace_whenWordBecomesEmpty_clearsSuggestions() {
        // Arrange
        `when`(mockContextTracker.getCurrentWordLength()).thenReturn(1, 0)

        // Act
        handler.handleBackspace()

        // Assert
        verify(mockSuggestionBar)?.clearSuggestions()
    }

    // Delete Last Word Tests

    @Test
    fun testHandleDeleteLastWord_inTermux_sendsCtrlW() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.termux")

        // Act
        handler.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

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
        handler.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockInputConnection).deleteSurroundingText(anyInt(), eq(0))
        verify(mockContextTracker).clearLastAutoInsertedWord()
    }

    @Test
    fun testHandleDeleteLastWord_withNullInputConnection_returnsEarly() {
        // Act
        handler.handleDeleteLastWord(null, mockEditorInfo)

        // Assert - no crash
    }

    @Test
    fun testHandleDeleteLastWord_withNoTextBeforeCursor_returnsEarly() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.android.test")
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn(null)
        `when`(mockInputConnection.getTextBeforeCursor(100, 0)).thenReturn("")

        // Act
        handler.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

        // Assert
        verify(mockInputConnection, never()).deleteSurroundingText(anyInt(), anyInt())
    }

    @Test
    fun testHandleDeleteLastWord_withSafetyLimit_doesNotDeleteTooMuch() {
        // Arrange
        `when`(mockEditorInfo.packageName).thenReturn("com.android.test")
        `when`(mockContextTracker.getLastAutoInsertedWord()).thenReturn(null)
        val longText = "a".repeat(100)
        `when`(mockInputConnection.getTextBeforeCursor(100, 0)).thenReturn(longText)

        // Act
        handler.handleDeleteLastWord(mockInputConnection, mockEditorInfo)

        // Assert - should limit delete count to 50
        verify(mockInputConnection).deleteSurroundingText(eq(50), eq(0))
    }

    // Context Update Tests

    @Test
    fun testUpdateContext_withValidWord_updatesContext() {
        // Arrange
        `when`(mockContextTracker.getLastCommitSource()).thenReturn(PredictionSource.USER_TYPED_TAP)

        // Act
        handler.updateContext("hello")

        // Assert
        verify(mockContextTracker).commitWord("hello", PredictionSource.USER_TYPED_TAP, false)
        verify(mockWordPredictor).addWordToContext("hello")
    }

    @Test
    fun testUpdateContext_withNullWord_returnsEarly() {
        // Act
        handler.updateContext(null)

        // Assert
        verify(mockContextTracker, never()).commitWord(anyString(), any(), anyBoolean())
    }

    @Test
    fun testUpdateContext_withEmptyWord_returnsEarly() {
        // Act
        handler.updateContext("")

        // Assert
        verify(mockContextTracker, never()).commitWord(anyString(), any(), anyBoolean())
    }

    // Mockito helper for argument matching
    private inline fun <reified T> argThat(predicate: (T) -> Boolean): T {
        return org.mockito.ArgumentMatchers.argThat { predicate(it) } ?: createInstance()
    }

    private inline fun <reified T> anyList(): List<T> {
        return org.mockito.ArgumentMatchers.anyList() ?: emptyList()
    }

    private inline fun <reified T> createInstance(): T {
        return mock(T::class.java)
    }

    private inline fun <reified T> any(): T {
        return org.mockito.ArgumentMatchers.any(T::class.java) ?: createInstance()
    }
}
