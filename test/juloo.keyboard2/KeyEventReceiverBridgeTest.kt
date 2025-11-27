package juloo.keyboard2

import android.os.Handler
import android.view.inputmethod.InputConnection
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for KeyEventReceiverBridge.
 *
 * Tests cover:
 * - Delegation to KeyboardReceiver
 * - Lazy initialization pattern (receiver set after creation)
 * - All IReceiver methods delegated correctly
 * - Direct methods (getCurrentInputConnection, getHandler)
 * - Null safety when receiver not set
 * - Factory method
 */
@RunWith(MockitoJUnitRunner::class)
class KeyEventReceiverBridgeTest {

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockHandler: Handler

    @Mock
    private lateinit var mockReceiver: KeyboardReceiver

    @Mock
    private lateinit var mockInputConnection: InputConnection

    @Mock
    private lateinit var mockKeyValueEvent: KeyValue.Event

    private lateinit var bridge: KeyEventReceiverBridge

    @Before
    fun setUp() {
        bridge = KeyEventReceiverBridge(mockKeyboard2, mockHandler)

        // Setup Keyboard2 mocks
        `when`(mockKeyboard2.getCurrentInputConnection()).thenReturn(mockInputConnection)
    }

    // ========== Basic Delegation Tests ==========

    @Test
    fun testHandleEventKey_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_event_key(mockKeyValueEvent)

        // Assert
        verify(mockReceiver).handle_event_key(mockKeyValueEvent)
    }

    @Test
    fun testSetShiftState_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.set_shift_state(true, false)

        // Assert
        verify(mockReceiver).set_shift_state(true, false)
    }

    @Test
    fun testSetComposePending_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.set_compose_pending(true)

        // Assert
        verify(mockReceiver).set_compose_pending(true)
    }

    @Test
    fun testSelectionStateChanged_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.selection_state_changed(true)

        // Assert
        verify(mockReceiver).selection_state_changed(true)
    }

    @Test
    fun testHandleTextTyped_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_text_typed("test")

        // Assert
        verify(mockReceiver).handle_text_typed("test")
    }

    @Test
    fun testHandleBackspace_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_backspace()

        // Assert
        verify(mockReceiver).handle_backspace()
    }

    @Test
    fun testHandleDeleteLastWord_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_delete_last_word()

        // Assert
        verify(mockReceiver).handle_delete_last_word()
    }

    @Test
    fun testIsClipboardSearchMode_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(true)

        // Act
        val result = bridge.isClipboardSearchMode()

        // Assert
        assertTrue("Should return true from receiver", result)
        verify(mockReceiver).isClipboardSearchMode()
    }

    @Test
    fun testAppendToClipboardSearch_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.appendToClipboardSearch("test")

        // Assert
        verify(mockReceiver).appendToClipboardSearch("test")
    }

    @Test
    fun testBackspaceClipboardSearch_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.backspaceClipboardSearch()

        // Assert
        verify(mockReceiver).backspaceClipboardSearch()
    }

    @Test
    fun testExitClipboardSearchMode_delegatesToReceiver() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.exitClipboardSearchMode()

        // Assert
        verify(mockReceiver).exitClipboardSearchMode()
    }

    // ========== Direct Method Tests (Not Delegated to Receiver) ==========

    @Test
    fun testGetCurrentInputConnection_returnsFromKeyboard2() {
        // Act
        val result = bridge.getCurrentInputConnection()

        // Assert
        assertSame("Should return InputConnection from Keyboard2",
            mockInputConnection, result)
        verify(mockKeyboard2).getCurrentInputConnection()
    }

    @Test
    fun testGetHandler_returnsHandler() {
        // Act
        val result = bridge.getHandler()

        // Assert
        assertSame("Should return the handler", mockHandler, result)
    }

    // ========== Null Safety Tests (Receiver Not Set) ==========

    @Test
    fun testHandleEventKey_withoutReceiver_doesNotCrash() {
        // Act & Assert - should not throw
        bridge.handle_event_key(mockKeyValueEvent)
    }

    @Test
    fun testSetShiftState_withoutReceiver_doesNotCrash() {
        // Act & Assert - should not throw
        bridge.set_shift_state(true, false)
    }

    @Test
    fun testHandleTextTyped_withoutReceiver_doesNotCrash() {
        // Act & Assert - should not throw
        bridge.handle_text_typed("test")
    }

    @Test
    fun testIsClipboardSearchMode_withoutReceiver_returnsFalse() {
        // Act
        val result = bridge.isClipboardSearchMode()

        // Assert
        assertFalse("Should return false when receiver not set", result)
    }

    @Test
    fun testGetCurrentInputConnection_withoutReceiver_stillWorks() {
        // Act
        val result = bridge.getCurrentInputConnection()

        // Assert
        assertSame("Should still return InputConnection even without receiver",
            mockInputConnection, result)
    }

    @Test
    fun testGetHandler_withoutReceiver_stillWorks() {
        // Act
        val result = bridge.getHandler()

        // Assert
        assertSame("Should still return handler even without receiver",
            mockHandler, result)
    }

    // ========== Receiver Lifecycle Tests ==========

    @Test
    fun testSetReceiver_setsReceiverCorrectly() {
        // Act
        bridge.setReceiver(mockReceiver)
        bridge.handle_event_key(mockKeyValueEvent)

        // Assert
        verify(mockReceiver).handle_event_key(mockKeyValueEvent)
    }

    @Test
    fun testSetReceiver_canBeCalledMultipleTimes() {
        // Arrange
        val mockReceiver2 = mock(KeyboardReceiver::class.java)

        // Act
        bridge.setReceiver(mockReceiver)
        bridge.setReceiver(mockReceiver2) // Replace with new receiver
        bridge.handle_event_key(mockKeyValueEvent)

        // Assert - only new receiver should be called
        verify(mockReceiver, never()).handle_event_key(any())
        verify(mockReceiver2).handle_event_key(mockKeyValueEvent)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesBridge() {
        // Act
        val bridge = KeyEventReceiverBridge.create(mockKeyboard2, mockHandler)

        // Assert
        assertNotNull("Factory method should create bridge", bridge)
    }

    @Test
    fun testCreate_factoryMethodBridgeWorks() {
        // Arrange
        val bridge = KeyEventReceiverBridge.create(mockKeyboard2, mockHandler)
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_event_key(mockKeyValueEvent)

        // Assert
        verify(mockReceiver).handle_event_key(mockKeyValueEvent)
    }

    @Test
    fun testCreate_factoryMethodReturnsCorrectHandler() {
        // Arrange
        val bridge = KeyEventReceiverBridge.create(mockKeyboard2, mockHandler)

        // Act
        val result = bridge.getHandler()

        // Assert
        assertSame("Factory-created bridge should return correct handler",
            mockHandler, result)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_createSetReceiver_allMethodsWork() {
        // Arrange
        bridge.setReceiver(mockReceiver)
        `when`(mockReceiver.isClipboardSearchMode()).thenReturn(true)

        // Act & Assert - test multiple methods
        bridge.handle_event_key(mockKeyValueEvent)
        verify(mockReceiver).handle_event_key(mockKeyValueEvent)

        bridge.handle_text_typed("hello")
        verify(mockReceiver).handle_text_typed("hello")

        assertTrue(bridge.isClipboardSearchMode())
        verify(mockReceiver).isClipboardSearchMode()

        assertSame(mockInputConnection, bridge.getCurrentInputConnection())
        assertSame(mockHandler, bridge.getHandler())
    }

    // ========== Edge Case Tests ==========

    @Test
    fun testHandleTextTyped_withEmptyString() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_text_typed("")

        // Assert
        verify(mockReceiver).handle_text_typed("")
    }

    @Test
    fun testHandleTextTyped_withNullString() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.handle_text_typed(null)

        // Assert
        verify(mockReceiver).handle_text_typed(null)
    }

    @Test
    fun testAppendToClipboardSearch_withEmptyString() {
        // Arrange
        bridge.setReceiver(mockReceiver)

        // Act
        bridge.appendToClipboardSearch("")

        // Assert
        verify(mockReceiver).appendToClipboardSearch("")
    }
}
