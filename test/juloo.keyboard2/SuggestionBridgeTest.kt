package juloo.keyboard2

import android.content.res.Resources
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import juloo.keyboard2.ml.SwipeMLData
import juloo.keyboard2.ml.SwipeMLDataStore
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentCaptor
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for SuggestionBridge.
 *
 * Tests cover:
 * - Prediction result handling
 * - Regular typing handling
 * - Backspace handling
 * - Delete last word handling
 * - Suggestion selection (with/without ML data collection)
 * - Null handler scenarios
 * - Factory method
 * - Multiple calls and integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class SuggestionBridgeTest {

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockMLDataCollector: MLDataCollector

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockContextTracker: PredictionContextTracker

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockSwipeMLData: SwipeMLData

    @Mock
    private lateinit var mockMLDataStore: SwipeMLDataStore

    private lateinit var bridge: SuggestionBridge

    @Before
    fun setUp() {
        // Mock keyboard2 to return input context
        `when`(mockKeyboard2.currentInputConnection).thenReturn(mockInputConnection)
        `when`(mockKeyboard2.currentInputEditorInfo).thenReturn(mockEditorInfo)
        `when`(mockKeyboard2.resources).thenReturn(mockResources)

        // Default: no swipe data
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(null)

        bridge = SuggestionBridge(
            mockKeyboard2,
            mockSuggestionHandler,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )
    }

    // ========== Prediction Results Tests ==========

    @Test
    fun testHandlePredictionResults_withHandler_delegatesWithContext() {
        // Arrange
        val predictions = listOf("hello", "world", "test")
        val scores = listOf(100, 90, 80)

        // Act
        bridge.handlePredictionResults(predictions, scores)

        // Assert - handler called with gathered context
        verify(mockSuggestionHandler).handlePredictionResults(
            predictions,
            scores,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testHandlePredictionResults_nullHandler_doesNotCrash() {
        // Arrange - bridge with null handler
        val bridgeNullHandler = SuggestionBridge(
            mockKeyboard2,
            null, // null handler
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        val predictions = listOf("hello", "world")
        val scores = listOf(100, 90)

        // Act & Assert - should not throw
        bridgeNullHandler.handlePredictionResults(predictions, scores)

        // No handler to verify
    }

    @Test
    fun testHandlePredictionResults_emptyLists_passedThrough() {
        // Arrange
        val predictions = emptyList<String>()
        val scores = emptyList<Int>()

        // Act
        bridge.handlePredictionResults(predictions, scores)

        // Assert
        verify(mockSuggestionHandler).handlePredictionResults(
            predictions,
            scores,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    // ========== Regular Typing Tests ==========

    @Test
    fun testHandleRegularTyping_withHandler_delegatesWithContext() {
        // Arrange
        val text = "hello"

        // Act
        bridge.handleRegularTyping(text)

        // Assert - handler called with gathered context (no resources for regular typing)
        verify(mockSuggestionHandler).handleRegularTyping(
            text,
            mockInputConnection,
            mockEditorInfo
        )
    }

    @Test
    fun testHandleRegularTyping_nullHandler_doesNotCrash() {
        // Arrange
        val bridgeNullHandler = SuggestionBridge(
            mockKeyboard2,
            null,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act & Assert - should not throw
        bridgeNullHandler.handleRegularTyping("test")
    }

    @Test
    fun testHandleRegularTyping_emptyText_passedThrough() {
        // Act
        bridge.handleRegularTyping("")

        // Assert
        verify(mockSuggestionHandler).handleRegularTyping(
            "",
            mockInputConnection,
            mockEditorInfo
        )
    }

    // ========== Backspace Tests ==========

    @Test
    fun testHandleBackspace_withHandler_delegates() {
        // Act
        bridge.handleBackspace()

        // Assert
        verify(mockSuggestionHandler).handleBackspace()
    }

    @Test
    fun testHandleBackspace_nullHandler_doesNotCrash() {
        // Arrange
        val bridgeNullHandler = SuggestionBridge(
            mockKeyboard2,
            null,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act & Assert - should not throw
        bridgeNullHandler.handleBackspace()
    }

    @Test
    fun testHandleBackspace_calledMultipleTimes_delegatesEachTime() {
        // Act
        bridge.handleBackspace()
        bridge.handleBackspace()
        bridge.handleBackspace()

        // Assert
        verify(mockSuggestionHandler, times(3)).handleBackspace()
    }

    // ========== Delete Last Word Tests ==========

    @Test
    fun testHandleDeleteLastWord_withHandler_delegatesWithContext() {
        // Act
        bridge.handleDeleteLastWord()

        // Assert
        verify(mockSuggestionHandler).handleDeleteLastWord(
            mockInputConnection,
            mockEditorInfo
        )
    }

    @Test
    fun testHandleDeleteLastWord_nullHandler_doesNotCrash() {
        // Arrange
        val bridgeNullHandler = SuggestionBridge(
            mockKeyboard2,
            null,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act & Assert - should not throw
        bridgeNullHandler.handleDeleteLastWord()
    }

    // ========== Suggestion Selection Tests (Simple Cases) ==========

    @Test
    fun testOnSuggestionSelected_regularTyping_delegatesWithContext() {
        // Arrange - not a swipe
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)
        val word = "hello"

        // Act
        bridge.onSuggestionSelected(word)

        // Assert - no ML data collection, just delegation
        verify(mockMLDataCollector, never()).collectAndStoreSwipeData(
            any(), any(), anyInt(), any()
        )
        verify(mockInputCoordinator).resetSwipeData()
        verify(mockSuggestionHandler).onSuggestionSelected(
            word,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testOnSuggestionSelected_nullHandler_doesNotCrash() {
        // Arrange
        val bridgeNullHandler = SuggestionBridge(
            mockKeyboard2,
            null,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act & Assert - should not throw
        bridgeNullHandler.onSuggestionSelected("test")

        // Swipe data should still be reset
        verify(mockInputCoordinator).resetSwipeData()
    }

    // ========== Suggestion Selection Tests (Swipe with ML) ==========

    @Test
    fun testOnSuggestionSelected_swipeWithMLStore_collectsData() {
        // Arrange - swipe with ML data
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(mockSwipeMLData)
        `when`(mockPredictionCoordinator.getMlDataStore()).thenReturn(mockMLDataStore)
        `when`(mockKeyboardView.height).thenReturn(800)

        val word = "hello"

        // Act
        bridge.onSuggestionSelected(word)

        // Assert - ML data collected
        verify(mockMLDataCollector).collectAndStoreSwipeData(
            word,
            mockSwipeMLData,
            800,
            mockMLDataStore
        )

        // Swipe data reset
        verify(mockInputCoordinator).resetSwipeData()

        // Delegated to handler
        verify(mockSuggestionHandler).onSuggestionSelected(
            word,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testOnSuggestionSelected_swipeWithoutMLStore_noCollection() {
        // Arrange - swipe but no ML data store
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(mockSwipeMLData)
        `when`(mockPredictionCoordinator.getMlDataStore()).thenReturn(null) // No ML store

        val word = "hello"

        // Act
        bridge.onSuggestionSelected(word)

        // Assert - no ML data collection
        verify(mockMLDataCollector, never()).collectAndStoreSwipeData(
            any(), any(), anyInt(), any()
        )

        // Swipe data still reset
        verify(mockInputCoordinator).resetSwipeData()

        // Still delegated
        verify(mockSuggestionHandler).onSuggestionSelected(
            word,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testOnSuggestionSelected_swipeWithoutSwipeData_noCollection() {
        // Arrange - was swipe but no current swipe data
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(null) // No swipe data
        `when`(mockPredictionCoordinator.getMlDataStore()).thenReturn(mockMLDataStore)

        val word = "hello"

        // Act
        bridge.onSuggestionSelected(word)

        // Assert - no ML data collection (no data to collect)
        verify(mockMLDataCollector, never()).collectAndStoreSwipeData(
            any(), any(), anyInt(), any()
        )

        // Swipe data still reset
        verify(mockInputCoordinator).resetSwipeData()

        // Still delegated
        verify(mockSuggestionHandler).onSuggestionSelected(
            word,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testOnSuggestionSelected_notSwipe_noCollection() {
        // Arrange - not a swipe
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(mockSwipeMLData)
        `when`(mockPredictionCoordinator.getMlDataStore()).thenReturn(mockMLDataStore)

        val word = "hello"

        // Act
        bridge.onSuggestionSelected(word)

        // Assert - no ML data collection (not a swipe)
        verify(mockMLDataCollector, never()).collectAndStoreSwipeData(
            any(), any(), anyInt(), any()
        )

        // Swipe data still reset
        verify(mockInputCoordinator).resetSwipeData()

        // Still delegated
        verify(mockSuggestionHandler).onSuggestionSelected(
            word,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesBridge() {
        // Act
        val bridge = SuggestionBridge.create(
            mockKeyboard2,
            mockSuggestionHandler,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Assert
        assertNotNull("Factory method should create bridge", bridge)
    }

    @Test
    fun testCreate_factoryMethodBridgeWorks() {
        // Arrange
        val bridge = SuggestionBridge.create(
            mockKeyboard2,
            mockSuggestionHandler,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act
        bridge.handleBackspace()

        // Assert
        verify(mockSuggestionHandler).handleBackspace()
    }

    @Test
    fun testCreate_withNullHandler() {
        // Act
        val bridge = SuggestionBridge.create(
            mockKeyboard2,
            null, // null handler
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Assert - should not crash
        assertNotNull("Factory should create bridge with null handler", bridge)

        // Should not crash when calling methods
        bridge.handleBackspace()
        bridge.handleRegularTyping("test")
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_typingAndBackspace() {
        // Act - simulate typing workflow
        bridge.handleRegularTyping("h")
        bridge.handleRegularTyping("he")
        bridge.handleRegularTyping("hel")
        bridge.handleBackspace()
        bridge.handleRegularTyping("hel")
        bridge.handleRegularTyping("hello")

        // Assert - all calls made
        verify(mockSuggestionHandler, times(5)).handleRegularTyping(
            anyString(), eq(mockInputConnection), eq(mockEditorInfo)
        )
        verify(mockSuggestionHandler).handleBackspace()
    }

    @Test
    fun testFullLifecycle_swipePredictionAndSelection() {
        // Arrange - swipe prediction workflow
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(true)
        `when`(mockInputCoordinator.getCurrentSwipeData()).thenReturn(mockSwipeMLData)
        `when`(mockPredictionCoordinator.getMlDataStore()).thenReturn(mockMLDataStore)
        `when`(mockKeyboardView.height).thenReturn(800)

        val predictions = listOf("hello", "world", "test")
        val scores = listOf(100, 90, 80)

        // Act - simulate swipe workflow
        bridge.handlePredictionResults(predictions, scores)
        bridge.onSuggestionSelected("hello")

        // Assert - prediction results handled
        verify(mockSuggestionHandler).handlePredictionResults(
            predictions,
            scores,
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )

        // ML data collected
        verify(mockMLDataCollector).collectAndStoreSwipeData(
            "hello",
            mockSwipeMLData,
            800,
            mockMLDataStore
        )

        // Swipe data reset
        verify(mockInputCoordinator).resetSwipeData()

        // Suggestion selected
        verify(mockSuggestionHandler).onSuggestionSelected(
            "hello",
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }

    @Test
    fun testIntegration_multipleBridgesIndependent() {
        // Arrange - create second bridge with different mocks
        val mockKeyboard2_2 = mock(Keyboard2::class.java)
        val mockSuggestionHandler_2 = mock(SuggestionHandler::class.java)

        `when`(mockKeyboard2_2.currentInputConnection).thenReturn(mockInputConnection)
        `when`(mockKeyboard2_2.currentInputEditorInfo).thenReturn(mockEditorInfo)

        val bridge2 = SuggestionBridge.create(
            mockKeyboard2_2,
            mockSuggestionHandler_2,
            mockMLDataCollector,
            mockInputCoordinator,
            mockContextTracker,
            mockPredictionCoordinator,
            mockKeyboardView
        )

        // Act - call both bridges
        bridge.handleBackspace()
        bridge2.handleBackspace()

        // Assert - each bridge calls its own handler
        verify(mockSuggestionHandler).handleBackspace()
        verify(mockSuggestionHandler_2).handleBackspace()
    }

    @Test
    fun testIntegration_deleteLastWord() {
        // Act - simulate delete last word workflow
        bridge.handleRegularTyping("hello world")
        bridge.handleDeleteLastWord()

        // Assert
        verify(mockSuggestionHandler).handleRegularTyping(
            "hello world",
            mockInputConnection,
            mockEditorInfo
        )
        verify(mockSuggestionHandler).handleDeleteLastWord(
            mockInputConnection,
            mockEditorInfo
        )
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_multipleSelections() {
        // Arrange - regular typing
        `when`(mockContextTracker.wasLastInputSwipe()).thenReturn(false)

        // Act - select multiple suggestions
        bridge.onSuggestionSelected("hello")
        bridge.onSuggestionSelected("world")
        bridge.onSuggestionSelected("test")

        // Assert - all selections delegated
        verify(mockSuggestionHandler, times(3)).onSuggestionSelected(
            anyString(),
            eq(mockInputConnection),
            eq(mockEditorInfo),
            eq(mockResources)
        )

        // Swipe data reset each time
        verify(mockInputCoordinator, times(3)).resetSwipeData()
    }

    @Test
    fun testEdgeCase_emptyWordSelection() {
        // Act
        bridge.onSuggestionSelected("")

        // Assert - empty word passed through
        verify(mockSuggestionHandler).onSuggestionSelected(
            "",
            mockInputConnection,
            mockEditorInfo,
            mockResources
        )
    }
}
