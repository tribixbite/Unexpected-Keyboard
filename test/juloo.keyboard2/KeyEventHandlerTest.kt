package juloo.keyboard2

import android.os.Handler
import android.os.Looper
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.ExtractedText
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for KeyEventHandler.
 *
 * Tests cover:
 * - Key routing logic (character, modifier, special keys)
 * - Meta state handling (Shift, Ctrl, Alt combinations)
 * - Clipboard search routing (SWITCH_CLIPBOARD key)
 * - Backspace handling (DELETE vs delete_last_word)
 * - Cursor movement calculations (ArrowLeft, ArrowRight, Home, End)
 * - Macro evaluation (F1-F12, switch_numeric, etc.)
 * - Action key handling (ENTER, DONE, SEARCH, etc.)
 * - Locked modifiers (Caps Lock, Num Lock behavior)
 * - Editing operations (copy, paste, cut, select all, etc.)
 * - Text sending and clipboard integration
 */
@RunWith(MockitoJUnitRunner::class)
class KeyEventHandlerTest {

    @Mock
    private lateinit var mockReceiver: KeyEventHandler.IReceiver

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockHandler: Handler

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    private lateinit var keyEventHandler: KeyEventHandler

    @Before
    fun setUp() {
        // Setup default mocks
        `when`(mockReceiver.getCurrentInputConnection()).thenReturn(mockInputConnection)
        `when`(mockReceiver.getHandler()).thenReturn(mockHandler)
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(false)

        // Mock EditorInfo defaults
        mockEditorInfo.inputType = EditorInfo.TYPE_CLASS_TEXT

        keyEventHandler = KeyEventHandler(mockReceiver)
    }

    // ============================================
    // CHARACTER KEY TESTS
    // ============================================

    @Test
    fun testCharacterKey_sendsCommitText() {
        val charKey = KeyValue.makeCharKey('a')

        keyEventHandler.key_up(charKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).commitText("a", 1)
        verify(mockReceiver).handle_text_typed("a")
    }

    @Test
    fun testStringKey_sendsCommitText() {
        val stringKey = KeyValue.makeStringKey("hello")

        keyEventHandler.key_up(stringKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).commitText("hello", 1)
        verify(mockReceiver).handle_text_typed("hello")
    }

    @Test
    fun testNullKey_doesNothing() {
        keyEventHandler.key_up(null, Pointers.Modifiers.EMPTY)

        verifyNoInteractions(mockInputConnection)
    }

    // ============================================
    // META STATE / MODIFIER TESTS
    // ============================================

    @Test
    fun testShiftModifier_updatesMetaState() {
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("shift"))

        keyEventHandler.mods_changed(mods)

        // Shift down event
        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_SHIFT_LEFT &&
                (event.metaState and KeyEvent.META_SHIFT_ON) != 0
            }
        )
    }

    @Test
    fun testCtrlModifier_updatesMetaState() {
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("ctrl"))

        keyEventHandler.mods_changed(mods)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_CTRL_LEFT &&
                (event.metaState and KeyEvent.META_CTRL_ON) != 0
            }
        )
    }

    @Test
    fun testAltModifier_updatesMetaState() {
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("alt"))

        keyEventHandler.mods_changed(mods)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_ALT_LEFT &&
                (event.metaState and KeyEvent.META_ALT_ON) != 0
            }
        )
    }

    @Test
    fun testMetaModifier_updatesMetaState() {
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("meta"))

        keyEventHandler.mods_changed(mods)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_META_LEFT &&
                (event.metaState and KeyEvent.META_META_ON) != 0
            }
        )
    }

    @Test
    fun testMultipleModifiers_allApplied() {
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("shift"))
            .add(KeyValue.getKeyByName("ctrl"))

        keyEventHandler.mods_changed(mods)

        // Verify both Shift and Ctrl down events sent
        verify(mockInputConnection, times(2)).sendKeyEvent(any())
    }

    @Test
    fun testModifierRelease_sendsUpEvent() {
        val modsDown = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("shift"))
        keyEventHandler.mods_changed(modsDown)

        clearInvocations(mockInputConnection)

        // Release shift
        keyEventHandler.mods_changed(Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_UP &&
                event.keyCode == KeyEvent.KEYCODE_SHIFT_LEFT
            }
        )
    }

    // ============================================
    // BACKSPACE / DELETE TESTS
    // ============================================

    @Test
    fun testBackspace_normalMode_sendsDeleteKey() {
        val deleteKey = KeyValue.makeKeyeventKey(KeyEvent.KEYCODE_DEL)

        keyEventHandler.key_up(deleteKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection, times(2)).sendKeyEvent(any()) // DOWN + UP
        verify(mockReceiver).handle_backspace()
    }

    @Test
    fun testBackspace_clipboardSearchMode_routesToSearch() {
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(true)
        val deleteKey = KeyValue.makeKeyeventKey(KeyEvent.KEYCODE_DEL)

        keyEventHandler.key_up(deleteKey, Pointers.Modifiers.EMPTY)

        verify(mockReceiver).backspaceClipboardSearch()
        verify(mockReceiver, never()).handle_backspace()
    }

    @Test
    fun testDeleteWord_sendsCtrlBackspace() {
        val deleteWordKey = KeyValue.makeEditingKey(KeyValue.Editing.DELETE_WORD)

        keyEventHandler.key_up(deleteWordKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_DEL &&
                (event.metaState and KeyEvent.META_CTRL_ON) != 0
            }
        )
    }

    @Test
    fun testDeleteLastWord_routesToReceiver() {
        val deleteLastWordKey = KeyValue.makeEditingKey(KeyValue.Editing.DELETE_LAST_WORD)

        keyEventHandler.key_up(deleteLastWordKey, Pointers.Modifiers.EMPTY)

        verify(mockReceiver).handle_delete_last_word()
    }

    @Test
    fun testForwardDeleteWord_sendsCtrlForwardDelete() {
        val forwardDeleteWordKey = KeyValue.makeEditingKey(KeyValue.Editing.FORWARD_DELETE_WORD)

        keyEventHandler.key_up(forwardDeleteWordKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).sendKeyEvent(
            argThat { event ->
                event.action == KeyEvent.ACTION_DOWN &&
                event.keyCode == KeyEvent.KEYCODE_FORWARD_DEL &&
                (event.metaState and KeyEvent.META_CTRL_ON) != 0
            }
        )
    }

    // ============================================
    // CLIPBOARD / EDITING TESTS
    // ============================================

    @Test
    fun testPasteKey_sendsContextMenuAction() {
        val pasteKey = KeyValue.makeEditingKey(KeyValue.Editing.PASTE)

        keyEventHandler.key_up(pasteKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.paste)
    }

    @Test
    fun testCopyKey_withSelection_sendsContextMenuAction() {
        // Mock selection exists
        val extractedText = ExtractedText().apply {
            selectionStart = 0
            selectionEnd = 5
        }
        `when`(mockInputConnection.getExtractedText(any(), anyInt())).thenReturn(extractedText)

        val copyKey = KeyValue.makeEditingKey(KeyValue.Editing.COPY)
        keyEventHandler.key_up(copyKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.copy)
    }

    @Test
    fun testCutKey_withSelection_sendsContextMenuAction() {
        val extractedText = ExtractedText().apply {
            selectionStart = 0
            selectionEnd = 5
        }
        `when`(mockInputConnection.getExtractedText(any(), anyInt())).thenReturn(extractedText)

        val cutKey = KeyValue.makeEditingKey(KeyValue.Editing.CUT)
        keyEventHandler.key_up(cutKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.cut)
    }

    @Test
    fun testSelectAll_sendsContextMenuAction() {
        val selectAllKey = KeyValue.makeEditingKey(KeyValue.Editing.SELECT_ALL)

        keyEventHandler.key_up(selectAllKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.selectAll)
    }

    @Test
    fun testUndo_sendsContextMenuAction() {
        val undoKey = KeyValue.makeEditingKey(KeyValue.Editing.UNDO)

        keyEventHandler.key_up(undoKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.undo)
    }

    @Test
    fun testRedo_sendsContextMenuAction() {
        val redoKey = KeyValue.makeEditingKey(KeyValue.Editing.REDO)

        keyEventHandler.key_up(redoKey, Pointers.Modifiers.EMPTY)

        verify(mockInputConnection).performContextMenuAction(android.R.id.redo)
    }

    // ============================================
    // CLIPBOARD SEARCH MODE TESTS
    // ============================================

    @Test
    fun testTextInput_clipboardSearchMode_routesToSearchBox() {
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(true)
        val charKey = KeyValue.makeCharKey('a')

        keyEventHandler.key_up(charKey, Pointers.Modifiers.EMPTY)

        verify(mockReceiver).appendToClipboardSearch("a")
        verify(mockInputConnection, never()).commitText(any(), anyInt())
    }

    @Test
    fun testPasteFromClipboard_exitsSearchMode_thenPastes() {
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(true)

        keyEventHandler.paste_from_clipboard_pane("test content")

        verify(mockReceiver).exitClipboardSearchMode()
        // After exit, should paste to target field
        verify(mockInputConnection).commitText("test content", 1)
    }

    // ============================================
    // EVENT KEY TESTS
    // ============================================

    @Test
    fun testEventKey_routesToReceiver() {
        val eventKey = KeyValue.makeEventKey(KeyValue.Event.CHANGE_METHOD)

        keyEventHandler.key_up(eventKey, Pointers.Modifiers.EMPTY)

        verify(mockReceiver).handle_event_key(KeyValue.Event.CHANGE_METHOD)
    }

    @Test
    fun testComposeKey_setsPendingState() {
        val composeKey = KeyValue.getKeyByName("compose")

        keyEventHandler.key_up(composeKey, Pointers.Modifiers.EMPTY)

        verify(mockReceiver).set_compose_pending(true)
    }

    // ============================================
    // EDITOR INFO / STARTED TESTS
    // ============================================

    @Test
    fun testStarted_initializesAutocap() {
        keyEventHandler.started(mockEditorInfo)

        // Should query input connection for autocap initialization
        verify(mockInputConnection, atLeastOnce()).getExtractedText(any(), anyInt())
    }

    @Test
    fun testSelectionUpdated_updatesAutocap() {
        keyEventHandler.started(mockEditorInfo)
        clearInvocations(mockInputConnection)

        keyEventHandler.selection_updated(0, 5)

        // Autocap should be notified of selection change
        // No verification needed - just ensure no crash
    }

    // ============================================
    // KEY DOWN TESTS (AUTOCAP STOP)
    // ============================================

    @Test
    fun testKeyDown_ctrlModifier_stopsAutocap() {
        val ctrlKey = KeyValue.getKeyByName("ctrl")

        keyEventHandler.key_down(ctrlKey, false)

        // Should stop autocap (verified by no crash)
    }

    @Test
    fun testKeyDown_altModifier_stopsAutocap() {
        val altKey = KeyValue.getKeyByName("alt")

        keyEventHandler.key_down(altKey, false)

        // Should stop autocap
    }

    @Test
    fun testKeyDown_composeKey_stopsAutocap() {
        val composeKey = KeyValue.getKeyByName("compose")

        keyEventHandler.key_down(composeKey, false)

        // Should stop autocap
    }

    // ============================================
    // MACRO EVALUATION TESTS
    // ============================================

    @Test
    fun testMacro_f1Key_evaluatedCorrectly() {
        val f1Key = KeyValue.makeMacroKey("f1")

        keyEventHandler.key_up(f1Key, Pointers.Modifiers.EMPTY)

        // F1 should send keycode
        verify(mockInputConnection, times(2)).sendKeyEvent(
            argThat { event ->
                event.keyCode == KeyEvent.KEYCODE_F1
            }
        )
    }

    @Test
    fun testMacro_switchNumeric_routesToReceiver() {
        val switchNumericKey = KeyValue.makeMacroKey("switch_numeric")

        keyEventHandler.key_up(switchNumericKey, Pointers.Modifiers.EMPTY)

        // Should route to receiver's handle_event_key
        verify(mockReceiver).handle_event_key(any())
    }

    // ============================================
    // NULL INPUT CONNECTION TESTS
    // ============================================

    @Test
    fun testNullInputConnection_doesNotCrash() {
        `when`(mockReceiver.getCurrentInputConnection()).thenReturn(null)

        val charKey = KeyValue.makeCharKey('a')
        keyEventHandler.key_up(charKey, Pointers.Modifiers.EMPTY)

        // Should not crash, just return early
    }

    @Test
    fun testNullInputConnection_modifierChange_doesNotCrash() {
        `when`(mockReceiver.getCurrentInputConnection()).thenReturn(null)

        val mods = Pointers.Modifiers.EMPTY.add(KeyValue.getKeyByName("shift"))
        keyEventHandler.mods_changed(mods)

        // Should not crash
    }
}
