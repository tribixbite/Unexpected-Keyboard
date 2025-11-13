package juloo.keyboard2

import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for KeyboardReceiver.
 *
 * Tests cover:
 * - Event key handling for all event types
 * - Layout switching (text, numeric, emoji, clipboard)
 * - State management (shift, compose, selection)
 * - Input method switching
 * - View management
 * - Clipboard and emoji pane handling
 */
@RunWith(MockitoJUnitRunner::class)
class KeyboardReceiverTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockLayoutManager: LayoutManager

    @Mock
    private lateinit var mockClipboardManager: ClipboardManager

    @Mock
    private lateinit var mockContextTracker: PredictionContextTracker

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockSubtypeManager: SubtypeManager

    @Mock
    private lateinit var mockHandler: Handler

    @Mock
    private lateinit var mockInputMethodManager: InputMethodManager

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockLayout: KeyboardData

    private lateinit var receiver: KeyboardReceiver

    @Before
    fun setUp() {
        `when`(mockKeyboard2.getCurrentInputConnection()).thenReturn(mockInputConnection)
        `when`(mockKeyboard2.getConfig()).thenReturn(mockConfig)
        `when`(mockSubtypeManager.inputMethodManager).thenReturn(mockInputMethodManager)

        receiver = KeyboardReceiver(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler
        )
    }

    @Test
    fun testHandleEventKey_CONFIG_startsSettingsActivity() {
        // Act
        receiver.handle_event_key(KeyValue.Event.CONFIG)

        // Assert
        verify(mockContext).startActivity(any(Intent::class.java))
    }

    @Test
    fun testHandleEventKey_SWITCH_TEXT_clearsSpecialLayout() {
        // Arrange
        `when`(mockLayoutManager.clearSpecialLayout()).thenReturn(mockLayout)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_TEXT)

        // Assert
        verify(mockLayoutManager).clearSpecialLayout()
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandleEventKey_SWITCH_NUMERIC_loadsNumericLayout() {
        // Arrange
        `when`(mockLayoutManager.loadNumpad(R.xml.numeric)).thenReturn(mockLayout)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_NUMERIC)

        // Assert
        verify(mockLayoutManager).loadNumpad(R.xml.numeric)
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandleEventKey_SWITCH_GREEKMATH_loadsGreekmathLayout() {
        // Arrange
        `when`(mockLayoutManager.loadNumpad(R.xml.greekmath)).thenReturn(mockLayout)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_GREEKMATH)

        // Assert
        verify(mockLayoutManager).loadNumpad(R.xml.greekmath)
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandleEventKey_SWITCH_FORWARD_incrementsLayout() {
        // Arrange
        `when`(mockLayoutManager.incrTextLayout(1)).thenReturn(mockLayout)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_FORWARD)

        // Assert
        verify(mockLayoutManager).incrTextLayout(1)
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandleEventKey_SWITCH_BACKWARD_decrementsLayout() {
        // Arrange
        `when`(mockLayoutManager.incrTextLayout(-1)).thenReturn(mockLayout)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_BACKWARD)

        // Assert
        verify(mockLayoutManager).incrTextLayout(-1)
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandleEventKey_CHANGE_METHOD_PICKER_showsInputMethodPicker() {
        // Act
        receiver.handle_event_key(KeyValue.Event.CHANGE_METHOD_PICKER)

        // Assert
        verify(mockInputMethodManager).showInputMethodPicker()
    }

    @Test
    fun testHandleEventKey_ACTION_performsEditorAction() {
        // Arrange
        val actionId = 123
        `when`(mockKeyboard2.actionId).thenReturn(actionId)

        // Act
        receiver.handle_event_key(KeyValue.Event.ACTION)

        // Assert
        verify(mockInputConnection).performEditorAction(actionId)
    }

    @Test
    fun testHandleEventKey_ACTION_withNullConnection_doesNotCrash() {
        // Arrange
        `when`(mockKeyboard2.getCurrentInputConnection()).thenReturn(null)

        // Act - should not throw exception
        receiver.handle_event_key(KeyValue.Event.ACTION)

        // Assert
        verify(mockInputConnection, never()).performEditorAction(anyInt())
    }

    @Test
    fun testHandleEventKey_CAPS_LOCK_setsShiftState() {
        // Act
        receiver.handle_event_key(KeyValue.Event.CAPS_LOCK)

        // Assert
        verify(mockKeyboardView).set_shift_state(true, true)
    }

    @Test
    fun testSetShiftState_delegatesToView() {
        // Act
        receiver.set_shift_state(true, false)

        // Assert
        verify(mockKeyboardView).set_shift_state(true, false)
    }

    @Test
    fun testSetComposePending_delegatesToView() {
        // Act
        receiver.set_compose_pending(true)

        // Assert
        verify(mockKeyboardView).set_compose_pending(true)
    }

    @Test
    fun testSelectionStateChanged_delegatesToView() {
        // Act
        receiver.selection_state_changed(true)

        // Assert
        verify(mockKeyboardView).set_selection_state(true)
    }

    @Test
    fun testGetCurrentInputConnection_returnsKeyboard2Connection() {
        // Act
        val result = receiver.getCurrentInputConnection()

        // Assert
        assertSame("Should return Keyboard2's input connection",
            mockInputConnection, result)
    }

    @Test
    fun testGetHandler_returnsProvidedHandler() {
        // Act
        val result = receiver.getHandler()

        // Assert
        assertSame("Should return provided handler", mockHandler, result)
    }

    @Test
    fun testHandleTextTyped_resetsSwipeTracking() {
        // Act
        receiver.handle_text_typed("test")

        // Assert
        verify(mockContextTracker).setWasLastInputSwipe(false)
        verify(mockInputCoordinator).resetSwipeData()
        verify(mockKeyboard2).handleRegularTyping("test")
    }

    @Test
    fun testHandleBackspace_delegatesToKeyboard2() {
        // Act
        receiver.handle_backspace()

        // Assert
        verify(mockKeyboard2).handleBackspace()
    }

    @Test
    fun testHandleDeleteLastWord_delegatesToKeyboard2() {
        // Act
        receiver.handle_delete_last_word()

        // Assert
        verify(mockKeyboard2).handleDeleteLastWord()
    }

    @Test
    fun testIsClipboardSearchMode_delegatesToClipboardManager() {
        // Arrange
        `when`(mockClipboardManager.isInSearchMode()).thenReturn(true)

        // Act
        val result = receiver.isClipboardSearchMode()

        // Assert
        assertTrue("Should return clipboard manager's search mode", result)
        verify(mockClipboardManager).isInSearchMode()
    }

    @Test
    fun testAppendToClipboardSearch_delegatesToClipboardManager() {
        // Act
        receiver.appendToClipboardSearch("test")

        // Assert
        verify(mockClipboardManager).appendToSearch("test")
    }

    @Test
    fun testBackspaceClipboardSearch_delegatesToClipboardManager() {
        // Act
        receiver.backspaceClipboardSearch()

        // Assert
        verify(mockClipboardManager).deleteFromSearch()
    }

    @Test
    fun testExitClipboardSearchMode_delegatesToClipboardManager() {
        // Act
        receiver.exitClipboardSearchMode()

        // Assert
        verify(mockClipboardManager).clearSearch()
    }

    @Test
    fun testHandleEventKey_SWITCH_BACK_EMOJI_resetsClipboardSearch() {
        // Arrange
        val mockContentPane = mock(ViewGroup::class.java)
        receiver.setViewReferences(null, mockContentPane)
        `when`(mockContentPane.visibility).thenReturn(View.VISIBLE)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_BACK_EMOJI)

        // Assert
        verify(mockClipboardManager).resetSearchOnHide()
        verify(mockContentPane).visibility = View.GONE
    }

    @Test
    fun testHandleEventKey_SWITCH_BACK_CLIPBOARD_resetsClipboardSearch() {
        // Arrange
        val mockContentPane = mock(ViewGroup::class.java)
        receiver.setViewReferences(null, mockContentPane)
        `when`(mockContentPane.visibility).thenReturn(View.VISIBLE)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_BACK_CLIPBOARD)

        // Assert
        verify(mockClipboardManager).resetSearchOnHide()
        verify(mockContentPane).visibility = View.GONE
    }

    @Test
    fun testSetViewReferences_storesViewsForLaterUse() {
        // Arrange
        val mockEmojiPane = mock(ViewGroup::class.java)
        val mockContentPane = mock(ViewGroup::class.java)

        // Act
        receiver.setViewReferences(mockEmojiPane, mockContentPane)

        // Assert - verify views are used in subsequent operations
        `when`(mockContentPane.visibility).thenReturn(View.VISIBLE)
        receiver.handle_event_key(KeyValue.Event.SWITCH_BACK_CLIPBOARD)
        verify(mockContentPane).visibility = View.GONE
    }

    @Test
    fun testHandleEventKey_SWITCH_CLIPBOARD_resetsSearchOnShow() {
        // Arrange
        val mockClipboardPane = mock(ViewGroup::class.java)
        `when`(mockClipboardManager.getClipboardPane(any())).thenReturn(mockClipboardPane)

        val mockContentPane = mock(ViewGroup::class.java)
        receiver.setViewReferences(null, mockContentPane)

        // Act
        receiver.handle_event_key(KeyValue.Event.SWITCH_CLIPBOARD)

        // Assert
        verify(mockClipboardManager).resetSearchOnShow()
    }
}
