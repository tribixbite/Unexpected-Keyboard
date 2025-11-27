package juloo.keyboard2

import android.content.res.Resources
import android.view.inputmethod.EditorInfo
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for EditorInfoHelper.
 *
 * Tests cover:
 * - Action info extraction (custom labels and IME actions)
 * - Action label mapping for all IME action types
 * - Resource ID mapping
 * - Enter/Action key swap behavior
 * - Edge cases and null handling
 */
@RunWith(MockitoJUnitRunner::class)
class EditorInfoHelperTest {

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockEditorInfo: EditorInfo

    @Before
    fun setUp() {
        // Setup mock resources with standard action strings
        `when`(mockResources.getString(R.string.key_action_next)).thenReturn("Next")
        `when`(mockResources.getString(R.string.key_action_done)).thenReturn("Done")
        `when`(mockResources.getString(R.string.key_action_go)).thenReturn("Go")
        `when`(mockResources.getString(R.string.key_action_prev)).thenReturn("Previous")
        `when`(mockResources.getString(R.string.key_action_search)).thenReturn("Search")
        `when`(mockResources.getString(R.string.key_action_send)).thenReturn("Send")
    }

    // ========== extractActionInfo Tests (Custom Action Label) ==========

    @Test
    fun testExtractActionInfo_withCustomActionLabel_usesCustomLabel() {
        // Arrange
        mockEditorInfo.actionLabel = "Submit"
        mockEditorInfo.actionId = 12345
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_DONE

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should use custom action label", "Submit", result.actionLabel)
        assertEquals("Should use custom action ID", 12345, result.actionId)
        assertFalse("Should not swap Enter/Action when using custom label",
            result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withCustomActionLabel_ignoresImeOptions() {
        // Arrange
        mockEditorInfo.actionLabel = "Custom"
        mockEditorInfo.actionId = 99
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_SEARCH or EditorInfo.IME_FLAG_NO_ENTER_ACTION

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert - should use custom label, not IME action
        assertEquals("Should use custom label, not 'Search'", "Custom", result.actionLabel)
        assertEquals("Should use custom ID, not IME_ACTION_SEARCH", 99, result.actionId)
        assertFalse("Should always be false with custom label", result.swapEnterActionKey)
    }

    // ========== extractActionInfo Tests (IME Actions) ==========

    @Test
    fun testExtractActionInfo_withImeActionNext_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_NEXT

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Next' label", "Next", result.actionLabel)
        assertEquals("Should extract IME_ACTION_NEXT", EditorInfo.IME_ACTION_NEXT, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionDone_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_DONE

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Done' label", "Done", result.actionLabel)
        assertEquals("Should extract IME_ACTION_DONE", EditorInfo.IME_ACTION_DONE, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionGo_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_GO

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Go' label", "Go", result.actionLabel)
        assertEquals("Should extract IME_ACTION_GO", EditorInfo.IME_ACTION_GO, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionSearch_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_SEARCH

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Search' label", "Search", result.actionLabel)
        assertEquals("Should extract IME_ACTION_SEARCH", EditorInfo.IME_ACTION_SEARCH, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionSend_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_SEND

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Send' label", "Send", result.actionLabel)
        assertEquals("Should extract IME_ACTION_SEND", EditorInfo.IME_ACTION_SEND, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionPrevious_extractsCorrectly() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_PREVIOUS

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertEquals("Should extract 'Previous' label", "Previous", result.actionLabel)
        assertEquals("Should extract IME_ACTION_PREVIOUS", EditorInfo.IME_ACTION_PREVIOUS, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionNone_extractsNull() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_NONE

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertNull("Should extract null label for IME_ACTION_NONE", result.actionLabel)
        assertEquals("Should extract IME_ACTION_NONE", EditorInfo.IME_ACTION_NONE, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withImeActionUnspecified_extractsNull() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_UNSPECIFIED

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertNull("Should extract null label for IME_ACTION_UNSPECIFIED", result.actionLabel)
        assertEquals("Should extract IME_ACTION_UNSPECIFIED",
            EditorInfo.IME_ACTION_UNSPECIFIED, result.actionId)
        assertTrue("Should swap by default", result.swapEnterActionKey)
    }

    // ========== Enter/Action Key Swap Tests ==========

    @Test
    fun testExtractActionInfo_withNoEnterActionFlag_disablesSwap() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_DONE or EditorInfo.IME_FLAG_NO_ENTER_ACTION

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertFalse("Should not swap when IME_FLAG_NO_ENTER_ACTION is set",
            result.swapEnterActionKey)
    }

    @Test
    fun testExtractActionInfo_withoutNoEnterActionFlag_enablesSwap() {
        // Arrange
        mockEditorInfo.actionLabel = null
        mockEditorInfo.imeOptions = EditorInfo.IME_ACTION_DONE
        // No IME_FLAG_NO_ENTER_ACTION flag

        // Act
        val result = EditorInfoHelper.extractActionInfo(mockEditorInfo, mockResources)

        // Assert
        assertTrue("Should swap when IME_FLAG_NO_ENTER_ACTION is not set",
            result.swapEnterActionKey)
    }

    // ========== actionLabelFor Tests ==========

    @Test
    fun testActionLabelFor_allValidActions_returnsCorrectStrings() {
        // Test all valid IME actions
        assertEquals("Next", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_NEXT, mockResources))
        assertEquals("Done", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_DONE, mockResources))
        assertEquals("Go", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_GO, mockResources))
        assertEquals("Previous", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_PREVIOUS, mockResources))
        assertEquals("Search", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_SEARCH, mockResources))
        assertEquals("Send", EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_SEND, mockResources))
    }

    @Test
    fun testActionLabelFor_noneAndUnspecified_returnsNull() {
        assertNull(EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_NONE, mockResources))
        assertNull(EditorInfoHelper.actionLabelFor(EditorInfo.IME_ACTION_UNSPECIFIED, mockResources))
    }

    @Test
    fun testActionLabelFor_unknownAction_returnsNull() {
        assertNull("Should return null for unknown action",
            EditorInfoHelper.actionLabelFor(999, mockResources))
    }

    // ========== actionResourceIdFor Tests ==========

    @Test
    fun testActionResourceIdFor_allValidActions_returnsCorrectResourceIds() {
        assertEquals(R.string.key_action_next,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_NEXT))
        assertEquals(R.string.key_action_done,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_DONE))
        assertEquals(R.string.key_action_go,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_GO))
        assertEquals(R.string.key_action_prev,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_PREVIOUS))
        assertEquals(R.string.key_action_search,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_SEARCH))
        assertEquals(R.string.key_action_send,
            EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_SEND))
    }

    @Test
    fun testActionResourceIdFor_noneAndUnspecified_returnsNull() {
        assertNull(EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_NONE))
        assertNull(EditorInfoHelper.actionResourceIdFor(EditorInfo.IME_ACTION_UNSPECIFIED))
    }

    @Test
    fun testActionResourceIdFor_unknownAction_returnsNull() {
        assertNull("Should return null for unknown action",
            EditorInfoHelper.actionResourceIdFor(999))
    }

    // ========== Data Class Tests ==========

    @Test
    fun testEditorActionInfo_dataClassEquality() {
        val info1 = EditorInfoHelper.EditorActionInfo("Done", 123, true)
        val info2 = EditorInfoHelper.EditorActionInfo("Done", 123, true)
        val info3 = EditorInfoHelper.EditorActionInfo("Next", 123, true)

        assertEquals("Same data should be equal", info1, info2)
        assertNotEquals("Different data should not be equal", info1, info3)
    }

    @Test
    fun testEditorActionInfo_nullActionLabel_handlesCorrectly() {
        val info = EditorInfoHelper.EditorActionInfo(null, 0, false)

        assertNull("Action label can be null", info.actionLabel)
        assertEquals("Action ID should be 0", 0, info.actionId)
        assertFalse("Swap flag should be false", info.swapEnterActionKey)
    }
}
