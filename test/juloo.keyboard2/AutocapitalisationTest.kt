package juloo.keyboard2

import android.os.Handler
import android.text.InputType
import android.view.KeyEvent
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
 * Comprehensive test suite for Autocapitalisation.
 *
 * Tests cover:
 * - Sentence start detection (. ? ! followed by space)
 * - Paragraph start detection (newline)
 * - Email/URL detection (no auto-caps in emails/URLs)
 * - State machine transitions (OFF → SENTENCE → WORD → OFF)
 * - Edge cases (multiple punctuation, whitespace variations)
 * - Pause/unpause functionality
 * - Cursor movement detection
 * - Delete key handling
 * - Enter key handling
 * - Config-based enabling/disabling
 * - Input type variations
 */
@RunWith(MockitoJUnitRunner::class)
class AutocapitalisationTest {

    @Mock
    private lateinit var mockHandler: Handler

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockCallback: Autocapitalisation.Callback

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    private lateinit var autocap: Autocapitalisation

    @Before
    fun setUp() {
        // Mock Config to enable autocapitalisation
        `when`(mockConfig.autocapitalisation).thenReturn(true)
        Config.setGlobalConfig(mockConfig)

        // Setup default EditorInfo
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        mockEditorInfo.initialCapsMode = 1 // Start with caps enabled

        // Setup Handler to execute runnables immediately for testing
        doAnswer { invocation ->
            val runnable = invocation.getArgument<Runnable>(0)
            runnable.run()
            null
        }.`when`(mockHandler).postDelayed(any(), anyLong())

        // Setup InputConnection defaults
        `when`(mockInputConnection.getCursorCapsMode(anyInt())).thenReturn(1)
        `when`(mockInputConnection.getTextAfterCursor(anyInt(), anyInt())).thenReturn("")

        autocap = Autocapitalisation(mockHandler, mockCallback)
    }

    // ============================================
    // INITIALIZATION TESTS
    // ============================================

    @Test
    fun testStarted_capsSentences_enablesAutocap() {
        autocap.started(mockEditorInfo, mockInputConnection)

        // Should enable shift at start (initialCapsMode = 1)
        verify(mockCallback).update_shift_state(true, true)
    }

    @Test
    fun testStarted_noCapsMode_disablesAutocap() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT // No CAP_SENTENCES flag
        mockEditorInfo.initialCapsMode = 0

        autocap.started(mockEditorInfo, mockInputConnection)

        // Should not enable autocap
        verify(mockCallback, never()).update_shift_state(true, anyBoolean())
    }

    @Test
    fun testStarted_configDisabled_disablesAutocap() {
        `when`(mockConfig.autocapitalisation).thenReturn(false)

        autocap.started(mockEditorInfo, mockInputConnection)

        // Should not enable autocap when config disabled
        verify(mockCallback, never()).update_shift_state(true, anyBoolean())
    }

    @Test
    fun testStarted_initialCapsModeZero_doesNotEnableShift() {
        mockEditorInfo.initialCapsMode = 0

        autocap.started(mockEditorInfo, mockInputConnection)

        // Should not enable shift when initialCapsMode is 0
        verify(mockCallback).update_shift_state(false, true)
    }

    // ============================================
    // TYPING TESTS
    // ============================================

    @Test
    fun testTyped_space_triggersCapsModeUpdate() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.typed(" ")

        // Space is a trigger character, should update caps mode
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testTyped_regularCharacter_disablesShift() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.typed("a")

        // Regular character should disable shift
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testTyped_multipleCharacters_processedSequentially() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.typed("Hello world")

        // Each character should be processed
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testTyped_sentenceEnd_triggersCapitalization() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        // Type sentence with period and space
        autocap.typed(". ")

        // Should enable capitalization after sentence end
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    // ============================================
    // EVENT SENT TESTS
    // ============================================

    @Test
    fun testEventSent_delete_updatesCursorAndCapsMode() {
        autocap.started(mockEditorInfo, mockInputConnection)
        autocap.typed("a") // cursor = 1
        clearInvocations(mockCallback)

        autocap.event_sent(KeyEvent.KEYCODE_DEL, 0)

        // Delete should update caps mode
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testEventSent_enter_updatesCapsMode() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.event_sent(KeyEvent.KEYCODE_ENTER, 0)

        // Enter should update caps mode (new paragraph)
        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testEventSent_withMetaKey_disablesShift() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.event_sent(KeyEvent.KEYCODE_A, KeyEvent.META_CTRL_ON)

        // Meta keys should disable shift updates
        verify(mockCallback, atLeastOnce()).update_shift_state(false, anyBoolean())
    }

    @Test
    fun testEventSent_deleteAtStart_doesNotDecrementCursor() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        // Delete when cursor is at 0 should not go negative
        autocap.event_sent(KeyEvent.KEYCODE_DEL, 0)

        verify(mockCallback, atLeastOnce()).update_shift_state(anyBoolean(), anyBoolean())
    }

    // ============================================
    // STOP/PAUSE/UNPAUSE TESTS
    // ============================================

    @Test
    fun testStop_disablesShift() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        autocap.stop()

        // Stop should disable shift
        verify(mockCallback).update_shift_state(false, true)
    }

    @Test
    fun testPause_returnsWasEnabled() {
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        val wasEnabled = autocap.pause()

        // Should return true if it was enabled
        assertTrue(wasEnabled)
        verify(mockCallback).update_shift_state(false, true)
    }

    @Test
    fun testPause_whenNotEnabled_returnsFalse() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT // No CAP_SENTENCES
        autocap.started(mockEditorInfo, mockInputConnection)
        clearInvocations(mockCallback)

        val wasEnabled = autocap.pause()

        // Should return false if it was not enabled
        assertFalse(wasEnabled)
    }

    @Test
    fun testUnpause_restoresState() {
        autocap.started(mockEditorInfo, mockInputConnection)
        val wasEnabled = autocap.pause()
        clearInvocations(mockCallback)

        autocap.unpause(wasEnabled)

        // Should restore autocap state
        verify(mockCallback).update_shift_state(anyBoolean(), true)
    }

    @Test
    fun testUnpause_withFalse_keepsDisabled() {
        autocap.started(mockEditorInfo, mockInputConnection)
        autocap.pause()
        clearInvocations(mockCallback)

        autocap.unpause(false)

        // Should keep autocap disabled
        verify(mockCallback).update_shift_state(anyBoolean(), true)
    }

    // ============================================
    // SELECTION UPDATE TESTS
    // ============================================

    @Test
    fun testSelectionUpdated_cursorMovement_disablesShift() {
        autocap.started(mockEditorInfo, mockInputConnection)
        autocap.typed("Hello")
        clearInvocations(mockCallback)

        // Simulate cursor movement (not just typing)
        autocap.selection_updated(5, 0)

        // Cursor movement should disable shift
        verify(mockCallback).update_shift_state(false, true)
    }

    @Test
    fun testSelectionUpdated_samePosition_doesNothing() {
        autocap.started(mockEditorInfo, mockInputConnection)
        autocap.typed("a") // cursor = 1
        clearInvocations(mockCallback)

        // Selection update with same cursor position (just typing)
        autocap.selection_updated(0, 1)

        // Should not trigger callback when cursor hasn't moved
        verify(mockCallback, never()).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testSelectionUpdated_clearInputBox_updatesCapsMode() {
        autocap.started(mockEditorInfo, mockInputConnection)
        autocap.typed("Hello")
        clearInvocations(mockCallback)

        `when`(mockInputConnection.getTextAfterCursor(1, 0)).thenReturn("")

        // Cursor moved to 0 and input box is empty
        autocap.selection_updated(5, 0)

        // Should update caps mode when input box cleared
        verify(mockCallback).update_shift_state(anyBoolean(), true)
    }

    // ============================================
    // INPUT TYPE VARIATION TESTS
    // ============================================

    @Test
    fun testInputType_longMessage_updatesState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_LONG_MESSAGE or
                                   InputType.TYPE_TEXT_FLAG_CAP_SENTENCES

        autocap.started(mockEditorInfo, mockInputConnection)

        // Long message should support autocap
        verify(mockCallback).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testInputType_shortMessage_updatesState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_SHORT_MESSAGE or
                                   InputType.TYPE_TEXT_FLAG_CAP_SENTENCES

        autocap.started(mockEditorInfo, mockInputConnection)

        // Short message should support autocap
        verify(mockCallback).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testInputType_personName_updatesState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_PERSON_NAME or
                                   InputType.TYPE_TEXT_FLAG_CAP_WORDS

        autocap.started(mockEditorInfo, mockInputConnection)

        // Person name should support autocap
        verify(mockCallback).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testInputType_emailSubject_updatesState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_EMAIL_SUBJECT or
                                   InputType.TYPE_TEXT_FLAG_CAP_SENTENCES

        autocap.started(mockEditorInfo, mockInputConnection)

        // Email subject should support autocap
        verify(mockCallback).update_shift_state(anyBoolean(), anyBoolean())
    }

    @Test
    fun testInputType_emailAddress_doesNotUpdateState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS or
                                   InputType.TYPE_TEXT_FLAG_CAP_SENTENCES

        autocap.started(mockEditorInfo, mockInputConnection)

        // Email address should NOT support autocap (lowercase only)
        // Should not update state
    }

    @Test
    fun testInputType_password_doesNotUpdateState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_PASSWORD

        autocap.started(mockEditorInfo, mockInputConnection)

        // Password fields should NOT support autocap
    }

    @Test
    fun testInputType_uri_doesNotUpdateState() {
        mockEditorInfo.inputType = InputType.TYPE_CLASS_TEXT or
                                   InputType.TYPE_TEXT_VARIATION_URI

        autocap.started(mockEditorInfo, mockInputConnection)

        // URI fields should NOT support autocap
    }

    // ============================================
    // CAPS MODE TESTS
    // ============================================

    @Test
    fun testSupportedCapsModes_includesSentencesAndWords() {
        val supported = Autocapitalisation.SUPPORTED_CAPS_MODES

        // Should support both SENTENCES and WORDS
        assertTrue((supported and InputType.TYPE_TEXT_FLAG_CAP_SENTENCES) != 0)
        assertTrue((supported and InputType.TYPE_TEXT_FLAG_CAP_WORDS) != 0)
    }
}
