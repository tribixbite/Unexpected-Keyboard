package juloo.keyboard2

import android.os.Handler
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for ReceiverInitializer.
 *
 * Tests cover:
 * - Lazy initialization pattern (return existing vs create new)
 * - KeyboardReceiver creation with all dependencies
 * - Bridge registration (receiver set on bridge)
 * - Null bridge handling
 * - Factory method
 * - Multiple initialization scenarios
 * - Integration tests
 */
@RunWith(MockitoJUnitRunner::class)
class ReceiverInitializerTest {

    @Mock
    private lateinit var mockContext: Keyboard2

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
    private lateinit var mockReceiverBridge: KeyEventReceiverBridge

    @Mock
    private lateinit var mockExistingReceiver: KeyboardReceiver

    private lateinit var initializer: ReceiverInitializer

    @Before
    fun setUp() {
        initializer = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )
    }

    // ========== Lazy Initialization Tests ==========

    @Test
    fun testInitializeIfNeeded_withExistingReceiver_returnsExisting() {
        // Act
        val result = initializer.initializeIfNeeded(mockExistingReceiver)

        // Assert
        assertSame("Should return existing receiver", mockExistingReceiver, result)
    }

    @Test
    fun testInitializeIfNeeded_withExistingReceiver_doesNotCreateNew() {
        // Act
        initializer.initializeIfNeeded(mockExistingReceiver)

        // Assert - bridge should not be called if receiver already exists
        verifyNoInteractions(mockReceiverBridge)
    }

    @Test
    fun testInitializeIfNeeded_withNullReceiver_createsNew() {
        // Act
        val result = initializer.initializeIfNeeded(null)

        // Assert
        assertNotNull("Should create new receiver", result)
        assertNotSame("Should not return mock", mockExistingReceiver, result)
    }

    @Test
    fun testInitializeIfNeeded_withNullReceiver_setsReceiverOnBridge() {
        // Act
        val result = initializer.initializeIfNeeded(null)

        // Assert
        verify(mockReceiverBridge).setReceiver(result)
    }

    // ========== Bridge Registration Tests ==========

    @Test
    fun testInitializeIfNeeded_registersReceiverWithBridge() {
        // Act
        val receiver = initializer.initializeIfNeeded(null)

        // Assert - verify bridge receives the new receiver
        verify(mockReceiverBridge, times(1)).setReceiver(receiver)
    }

    @Test
    fun testInitializeIfNeeded_withNullBridge_doesNotCrash() {
        // Arrange - create initializer with null bridge
        val initializerWithNullBridge = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            null // null bridge
        )

        // Act & Assert - should not throw
        val receiver = initializerWithNullBridge.initializeIfNeeded(null)
        assertNotNull("Should create receiver even with null bridge", receiver)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesInitializer() {
        // Act
        val initializer = ReceiverInitializer.create(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Assert
        assertNotNull("Factory method should create initializer", initializer)
    }

    @Test
    fun testCreate_factoryMethodInitializerWorks() {
        // Arrange
        val initializer = ReceiverInitializer.create(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act
        val receiver = initializer.initializeIfNeeded(null)

        // Assert
        assertNotNull("Factory-created initializer should work", receiver)
        verify(mockReceiverBridge).setReceiver(receiver)
    }

    @Test
    fun testCreate_withNullBridge() {
        // Act
        val initializer = ReceiverInitializer.create(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            null // null bridge
        )

        // Assert
        assertNotNull("Factory should create initializer with null bridge", initializer)

        // Should not crash when initializing
        val receiver = initializer.initializeIfNeeded(null)
        assertNotNull("Should create receiver with null bridge", receiver)
    }

    // ========== Multiple Initialization Tests ==========

    @Test
    fun testInitializeIfNeeded_calledTwiceWithNull_createsDifferentReceivers() {
        // Act
        val receiver1 = initializer.initializeIfNeeded(null)
        val receiver2 = initializer.initializeIfNeeded(null)

        // Assert - different instances
        assertNotSame("Should create independent receivers", receiver1, receiver2)

        // Both should be registered with bridge
        verify(mockReceiverBridge).setReceiver(receiver1)
        verify(mockReceiverBridge).setReceiver(receiver2)
    }

    @Test
    fun testInitializeIfNeeded_calledTwiceWithSameExisting_returnsSameReceiver() {
        // Act
        val receiver1 = initializer.initializeIfNeeded(mockExistingReceiver)
        val receiver2 = initializer.initializeIfNeeded(mockExistingReceiver)

        // Assert - same instance
        assertSame("Should return same existing receiver", receiver1, receiver2)
        assertSame("Both should be the existing receiver", mockExistingReceiver, receiver1)

        // Bridge should never be called when receiver exists
        verifyNoInteractions(mockReceiverBridge)
    }

    @Test
    fun testInitializeIfNeeded_existingThenNull_returnsExistingThenCreatesNew() {
        // Act - first call with existing receiver
        val receiver1 = initializer.initializeIfNeeded(mockExistingReceiver)

        // Assert
        assertSame("First call should return existing", mockExistingReceiver, receiver1)
        verifyNoInteractions(mockReceiverBridge)

        // Act - second call with null (simulating first call on next onStartInputView)
        val receiver2 = initializer.initializeIfNeeded(null)

        // Assert
        assertNotNull("Second call should create new", receiver2)
        assertNotSame("Second call should not return existing", mockExistingReceiver, receiver2)
        verify(mockReceiverBridge).setReceiver(receiver2)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_firstCall_createsAndRegisters() {
        // Act - simulate first onStartInputView call
        val receiver = initializer.initializeIfNeeded(null)

        // Assert - receiver created and registered
        assertNotNull("Should create receiver on first call", receiver)
        verify(mockReceiverBridge).setReceiver(receiver)
    }

    @Test
    fun testFullLifecycle_subsequentCalls_returnsExisting() {
        // Arrange - first call creates receiver
        val firstReceiver = initializer.initializeIfNeeded(null)
        verify(mockReceiverBridge).setReceiver(firstReceiver)

        // Act - subsequent calls with existing receiver
        val secondReceiver = initializer.initializeIfNeeded(firstReceiver)
        val thirdReceiver = initializer.initializeIfNeeded(firstReceiver)

        // Assert - all return the same receiver
        assertSame("Second call should return first receiver", firstReceiver, secondReceiver)
        assertSame("Third call should return first receiver", firstReceiver, thirdReceiver)

        // Bridge should only be called once (on first creation)
        verify(mockReceiverBridge, times(1)).setReceiver(any())
    }

    @Test
    fun testIntegration_multipleInitializersIndependent() {
        // Arrange - create two initializers
        val initializer1 = ReceiverInitializer.create(
            mockKeyboard2,
            mockKeyboard2ViewModel,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        val initializer2 = ReceiverInitializer.create(
            mockKeyboard2,
            mockKeyboard2ViewModel,
            mockKeyboardView,
            mockLayoutManager,
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act - initialize with both
        val receiver1 = initializer1.initializeIfNeeded(null)
        val receiver2 = initializer2.initializeIfNeeded(null)

        // Assert - different receivers created
        assertNotSame("Different initializers should create different receivers", receiver1, receiver2)

        // Both should register with bridge
        verify(mockReceiverBridge).setReceiver(receiver1)
        verify(mockReceiverBridge).setReceiver(receiver2)
    }

    @Test
    fun testIntegration_lazyInitializationPattern() {
        // Arrange - simulate typical usage pattern
        var receiver: KeyboardReceiver? = null

        // Act - first call (onCreate/first onStartInputView)
        receiver = initializer.initializeIfNeeded(receiver)
        assertNotNull("Should create on first call", receiver)
        verify(mockReceiverBridge).setReceiver(receiver)

        // Act - subsequent calls (later onStartInputView calls)
        receiver = initializer.initializeIfNeeded(receiver)
        receiver = initializer.initializeIfNeeded(receiver)
        receiver = initializer.initializeIfNeeded(receiver)

        // Assert - bridge only called once
        verify(mockReceiverBridge, times(1)).setReceiver(any())
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_alternatingNullAndExisting() {
        // Act & Assert - alternate between null and existing
        val receiver1 = initializer.initializeIfNeeded(null)
        assertNotNull("First: should create", receiver1)

        val receiver2 = initializer.initializeIfNeeded(mockExistingReceiver)
        assertSame("Second: should return existing", mockExistingReceiver, receiver2)

        val receiver3 = initializer.initializeIfNeeded(null)
        assertNotNull("Third: should create new", receiver3)
        assertNotSame("Third: should not return first", receiver1, receiver3)

        // Bridge should be called twice (for null cases)
        verify(mockReceiverBridge, times(2)).setReceiver(any())
    }

    // ========== Null LayoutManager Tests (v1.32.413: initialization order fix) ==========

    @Test
    fun testNullLayoutManager_initializeIfNeeded_returnsNull() {
        // Arrange - create initializer with null layoutManager
        val initializerWithNullLayout = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            null,  // null layoutManager
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act
        val result = initializerWithNullLayout.initializeIfNeeded(null)

        // Assert
        assertNull("Should return null when layoutManager is null", result)
    }

    @Test
    fun testNullLayoutManager_doesNotCreateReceiver() {
        // Arrange - create initializer with null layoutManager
        val initializerWithNullLayout = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            null,  // null layoutManager
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act
        initializerWithNullLayout.initializeIfNeeded(null)

        // Assert - bridge should not be called since receiver wasn't created
        verifyNoInteractions(mockReceiverBridge)
    }

    @Test
    fun testNullLayoutManager_withExistingReceiver_returnsExisting() {
        // Arrange - create initializer with null layoutManager
        val initializerWithNullLayout = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            null,  // null layoutManager
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act - even with null layoutManager, should return existing receiver
        val result = initializerWithNullLayout.initializeIfNeeded(mockExistingReceiver)

        // Assert
        assertSame("Should return existing receiver even with null layoutManager",
                   mockExistingReceiver, result)
    }

    @Test
    fun testFactoryMethod_withNullLayoutManager_createsInitializer() {
        // Act
        val initializer = ReceiverInitializer.create(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            null,  // null layoutManager
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Assert
        assertNotNull("Factory should create initializer with null layoutManager", initializer)

        // Verify behavior - should return null when trying to initialize
        val result = initializer.initializeIfNeeded(null)
        assertNull("Should defer creation when layoutManager is null", result)
    }

    @Test
    fun testNullLayoutManager_multipleCallsWithExisting_returnsExisting() {
        // Arrange
        val initializerWithNullLayout = ReceiverInitializer(
            mockContext,
            mockKeyboard2,
            mockKeyboardView,
            null,  // null layoutManager
            mockClipboardManager,
            mockContextTracker,
            mockInputCoordinator,
            mockSubtypeManager,
            mockHandler,
            mockReceiverBridge
        )

        // Act - multiple calls with existing receiver
        val result1 = initializerWithNullLayout.initializeIfNeeded(mockExistingReceiver)
        val result2 = initializerWithNullLayout.initializeIfNeeded(mockExistingReceiver)
        val result3 = initializerWithNullLayout.initializeIfNeeded(mockExistingReceiver)

        // Assert
        assertSame("All calls should return same existing receiver", mockExistingReceiver, result1)
        assertSame("All calls should return same existing receiver", mockExistingReceiver, result2)
        assertSame("All calls should return same existing receiver", mockExistingReceiver, result3)
    }
}
