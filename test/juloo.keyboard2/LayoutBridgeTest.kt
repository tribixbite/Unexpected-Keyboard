package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for LayoutBridge.
 *
 * Tests cover:
 * - Current layout retrieval (unmodified and modified)
 * - Text layout operations (set by index, increment/decrement)
 * - Special layout setting
 * - Layout loading (standard, numpad, pinentry)
 * - Keyboard view updates after layout changes
 * - Factory method
 * - Multiple calls and integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class LayoutBridgeTest {

    @Mock
    private lateinit var mockLayoutManager: LayoutManager

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockKeyboardData: KeyboardData

    @Mock
    private lateinit var mockKeyboardData2: KeyboardData

    @Mock
    private lateinit var mockSpecialLayout: KeyboardData

    private lateinit var bridge: LayoutBridge

    @Before
    fun setUp() {
        bridge = LayoutBridge(mockLayoutManager, mockKeyboardView)
    }

    // ========== Current Layout Tests ==========

    @Test
    fun testGetCurrentLayoutUnmodified_delegatesToManager() {
        // Arrange
        `when`(mockLayoutManager.current_layout_unmodified()).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.getCurrentLayoutUnmodified()

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).current_layout_unmodified()
    }

    @Test
    fun testGetCurrentLayoutUnmodified_multipleCallsDelegatesToManager() {
        // Arrange
        `when`(mockLayoutManager.current_layout_unmodified())
            .thenReturn(mockKeyboardData, mockKeyboardData2)

        // Act
        val result1 = bridge.getCurrentLayoutUnmodified()
        val result2 = bridge.getCurrentLayoutUnmodified()

        // Assert
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager, times(2)).current_layout_unmodified()
    }

    @Test
    fun testGetCurrentLayout_delegatesToManager() {
        // Arrange
        `when`(mockLayoutManager.current_layout()).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.getCurrentLayout()

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).current_layout()
    }

    @Test
    fun testGetCurrentLayout_multipleCallsDelegatesToManager() {
        // Arrange
        `when`(mockLayoutManager.current_layout())
            .thenReturn(mockKeyboardData, mockKeyboardData2)

        // Act
        val result1 = bridge.getCurrentLayout()
        val result2 = bridge.getCurrentLayout()

        // Assert
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager, times(2)).current_layout()
    }

    // ========== Set Text Layout Tests ==========

    @Test
    fun testSetTextLayout_delegatesAndUpdatesView() {
        // Arrange
        val layoutIndex = 0
        `when`(mockLayoutManager.setTextLayout(layoutIndex)).thenReturn(mockKeyboardData)

        // Act
        bridge.setTextLayout(layoutIndex)

        // Assert
        verify(mockLayoutManager).setTextLayout(layoutIndex)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testSetTextLayout_differentIndices_delegatesEach() {
        // Arrange
        `when`(mockLayoutManager.setTextLayout(0)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.setTextLayout(1)).thenReturn(mockKeyboardData2)

        // Act
        bridge.setTextLayout(0)
        bridge.setTextLayout(1)

        // Assert
        verify(mockLayoutManager).setTextLayout(0)
        verify(mockLayoutManager).setTextLayout(1)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData2)
    }

    @Test
    fun testSetTextLayout_negativeIndex_passedThrough() {
        // Arrange
        val layoutIndex = -1
        `when`(mockLayoutManager.setTextLayout(layoutIndex)).thenReturn(mockKeyboardData)

        // Act
        bridge.setTextLayout(layoutIndex)

        // Assert
        verify(mockLayoutManager).setTextLayout(layoutIndex)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    // ========== Increment Text Layout Tests ==========

    @Test
    fun testIncrTextLayout_positiveDelga_delegatesAndUpdatesView() {
        // Arrange
        val delta = 1
        `when`(mockLayoutManager.incrTextLayout(delta)).thenReturn(mockKeyboardData)

        // Act
        bridge.incrTextLayout(delta)

        // Assert
        verify(mockLayoutManager).incrTextLayout(delta)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testIncrTextLayout_negativeDelta_delegatesAndUpdatesView() {
        // Arrange
        val delta = -1
        `when`(mockLayoutManager.incrTextLayout(delta)).thenReturn(mockKeyboardData)

        // Act
        bridge.incrTextLayout(delta)

        // Assert
        verify(mockLayoutManager).incrTextLayout(delta)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testIncrTextLayout_zeroDelta_delegatesAndUpdatesView() {
        // Arrange
        val delta = 0
        `when`(mockLayoutManager.incrTextLayout(delta)).thenReturn(mockKeyboardData)

        // Act
        bridge.incrTextLayout(delta)

        // Assert
        verify(mockLayoutManager).incrTextLayout(delta)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testIncrTextLayout_multipleCalls_delegatesEach() {
        // Arrange
        `when`(mockLayoutManager.incrTextLayout(1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.incrTextLayout(-1)).thenReturn(mockKeyboardData2)

        // Act
        bridge.incrTextLayout(1)
        bridge.incrTextLayout(-1)

        // Assert
        verify(mockLayoutManager).incrTextLayout(1)
        verify(mockLayoutManager).incrTextLayout(-1)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData2)
    }

    // ========== Set Special Layout Tests ==========

    @Test
    fun testSetSpecialLayout_delegatesAndUpdatesView() {
        // Arrange
        `when`(mockLayoutManager.setSpecialLayout(mockSpecialLayout))
            .thenReturn(mockKeyboardData)

        // Act
        bridge.setSpecialLayout(mockSpecialLayout)

        // Assert
        verify(mockLayoutManager).setSpecialLayout(mockSpecialLayout)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testSetSpecialLayout_differentLayouts_delegatesEach() {
        // Arrange
        val specialLayout1 = mock(KeyboardData::class.java)
        val specialLayout2 = mock(KeyboardData::class.java)
        `when`(mockLayoutManager.setSpecialLayout(specialLayout1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.setSpecialLayout(specialLayout2)).thenReturn(mockKeyboardData2)

        // Act
        bridge.setSpecialLayout(specialLayout1)
        bridge.setSpecialLayout(specialLayout2)

        // Assert
        verify(mockLayoutManager).setSpecialLayout(specialLayout1)
        verify(mockLayoutManager).setSpecialLayout(specialLayout2)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData2)
    }

    // ========== Load Layout Tests ==========

    @Test
    fun testLoadLayout_delegatesToManager() {
        // Arrange
        val layoutId = 123
        `when`(mockLayoutManager.loadLayout(layoutId)).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.loadLayout(layoutId)

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).loadLayout(layoutId)
        verifyNoInteractions(mockKeyboardView) // No view update for load
    }

    @Test
    fun testLoadLayout_multipleIds_delegatesEach() {
        // Arrange
        val layoutId1 = 123
        val layoutId2 = 456
        `when`(mockLayoutManager.loadLayout(layoutId1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.loadLayout(layoutId2)).thenReturn(mockKeyboardData2)

        // Act
        val result1 = bridge.loadLayout(layoutId1)
        val result2 = bridge.loadLayout(layoutId2)

        // Assert
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager).loadLayout(layoutId1)
        verify(mockLayoutManager).loadLayout(layoutId2)
    }

    @Test
    fun testLoadNumpad_delegatesToManager() {
        // Arrange
        val layoutId = 789
        `when`(mockLayoutManager.loadNumpad(layoutId)).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.loadNumpad(layoutId)

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).loadNumpad(layoutId)
        verifyNoInteractions(mockKeyboardView) // No view update for load
    }

    @Test
    fun testLoadNumpad_multipleIds_delegatesEach() {
        // Arrange
        val layoutId1 = 789
        val layoutId2 = 101112
        `when`(mockLayoutManager.loadNumpad(layoutId1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.loadNumpad(layoutId2)).thenReturn(mockKeyboardData2)

        // Act
        val result1 = bridge.loadNumpad(layoutId1)
        val result2 = bridge.loadNumpad(layoutId2)

        // Assert
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager).loadNumpad(layoutId1)
        verify(mockLayoutManager).loadNumpad(layoutId2)
    }

    @Test
    fun testLoadPinentry_delegatesToManager() {
        // Arrange
        val layoutId = 131415
        `when`(mockLayoutManager.loadPinentry(layoutId)).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.loadPinentry(layoutId)

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).loadPinentry(layoutId)
        verifyNoInteractions(mockKeyboardView) // No view update for load
    }

    @Test
    fun testLoadPinentry_multipleIds_delegatesEach() {
        // Arrange
        val layoutId1 = 131415
        val layoutId2 = 161718
        `when`(mockLayoutManager.loadPinentry(layoutId1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.loadPinentry(layoutId2)).thenReturn(mockKeyboardData2)

        // Act
        val result1 = bridge.loadPinentry(layoutId1)
        val result2 = bridge.loadPinentry(layoutId2)

        // Assert
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager).loadPinentry(layoutId1)
        verify(mockLayoutManager).loadPinentry(layoutId2)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesBridge() {
        // Act
        val bridge = LayoutBridge.create(mockLayoutManager, mockKeyboardView)

        // Assert
        assertNotNull("Factory method should create bridge", bridge)
    }

    @Test
    fun testCreate_factoryMethodBridgeWorks() {
        // Arrange
        val bridge = LayoutBridge.create(mockLayoutManager, mockKeyboardView)
        `when`(mockLayoutManager.current_layout()).thenReturn(mockKeyboardData)

        // Act
        val result = bridge.getCurrentLayout()

        // Assert
        assertEquals(mockKeyboardData, result)
        verify(mockLayoutManager).current_layout()
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_layoutSwitchingWorkflow() {
        // Arrange
        `when`(mockLayoutManager.setTextLayout(0)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.incrTextLayout(1)).thenReturn(mockKeyboardData2)
        `when`(mockLayoutManager.current_layout()).thenReturn(mockKeyboardData2)

        // Act - simulate typical layout switching
        bridge.setTextLayout(0) // Set initial layout
        bridge.incrTextLayout(1) // Cycle to next
        val current = bridge.getCurrentLayout() // Get current

        // Assert
        verify(mockLayoutManager).setTextLayout(0)
        verify(mockLayoutManager).incrTextLayout(1)
        verify(mockLayoutManager).current_layout()
        verify(mockKeyboardView, times(2)).setKeyboard(any())
        assertEquals(mockKeyboardData2, current)
    }

    @Test
    fun testFullLifecycle_loadAndApplyWorkflow() {
        // Arrange
        val layoutId = 123
        `when`(mockLayoutManager.loadLayout(layoutId)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.setSpecialLayout(mockKeyboardData)).thenReturn(mockKeyboardData)

        // Act - simulate load and apply
        val loaded = bridge.loadLayout(layoutId)
        bridge.setSpecialLayout(loaded)

        // Assert
        verify(mockLayoutManager).loadLayout(layoutId)
        verify(mockLayoutManager).setSpecialLayout(mockKeyboardData)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
        assertEquals(mockKeyboardData, loaded)
    }

    @Test
    fun testFullLifecycle_numpadWorkflow() {
        // Arrange
        val numpadId = 789
        `when`(mockLayoutManager.loadNumpad(numpadId)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.setSpecialLayout(mockKeyboardData)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.current_layout()).thenReturn(mockKeyboardData)

        // Act - simulate numpad activation
        val numpad = bridge.loadNumpad(numpadId)
        bridge.setSpecialLayout(numpad)
        val current = bridge.getCurrentLayout()

        // Assert
        verify(mockLayoutManager).loadNumpad(numpadId)
        verify(mockLayoutManager).setSpecialLayout(mockKeyboardData)
        verify(mockLayoutManager).current_layout()
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
        assertEquals(mockKeyboardData, current)
    }

    @Test
    fun testFullLifecycle_pinentryWorkflow() {
        // Arrange
        val pinentryId = 131415
        `when`(mockLayoutManager.loadPinentry(pinentryId)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.setSpecialLayout(mockKeyboardData)).thenReturn(mockKeyboardData)

        // Act - simulate pinentry activation
        val pinentry = bridge.loadPinentry(pinentryId)
        bridge.setSpecialLayout(pinentry)

        // Assert
        verify(mockLayoutManager).loadPinentry(pinentryId)
        verify(mockLayoutManager).setSpecialLayout(mockKeyboardData)
        verify(mockKeyboardView).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testIntegration_multipleBridgesIndependent() {
        // Arrange - create second bridge with different mocks
        val mockLayoutManager2 = mock(LayoutManager::class.java)
        val mockKeyboardView2 = mock(Keyboard2View::class.java)
        val bridge2 = LayoutBridge.create(mockLayoutManager2, mockKeyboardView2)

        `when`(mockLayoutManager.current_layout()).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager2.current_layout()).thenReturn(mockKeyboardData2)

        // Act - call both bridges
        val result1 = bridge.getCurrentLayout()
        val result2 = bridge2.getCurrentLayout()

        // Assert - each bridge uses its own manager
        assertEquals(mockKeyboardData, result1)
        assertEquals(mockKeyboardData2, result2)
        verify(mockLayoutManager).current_layout()
        verify(mockLayoutManager2).current_layout()
    }

    @Test
    fun testIntegration_allLoadMethods() {
        // Arrange
        val layoutId = 123
        val numpadId = 456
        val pinentryId = 789
        val layout1 = mock(KeyboardData::class.java)
        val layout2 = mock(KeyboardData::class.java)
        val layout3 = mock(KeyboardData::class.java)

        `when`(mockLayoutManager.loadLayout(layoutId)).thenReturn(layout1)
        `when`(mockLayoutManager.loadNumpad(numpadId)).thenReturn(layout2)
        `when`(mockLayoutManager.loadPinentry(pinentryId)).thenReturn(layout3)

        // Act - load all types
        val loadedLayout = bridge.loadLayout(layoutId)
        val loadedNumpad = bridge.loadNumpad(numpadId)
        val loadedPinentry = bridge.loadPinentry(pinentryId)

        // Assert - all loaded correctly
        assertEquals(layout1, loadedLayout)
        assertEquals(layout2, loadedNumpad)
        assertEquals(layout3, loadedPinentry)
        verify(mockLayoutManager).loadLayout(layoutId)
        verify(mockLayoutManager).loadNumpad(numpadId)
        verify(mockLayoutManager).loadPinentry(pinentryId)
        verifyNoInteractions(mockKeyboardView) // No view updates for loads
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_setTextLayoutSameIndexMultipleTimes() {
        // Arrange
        val layoutIndex = 0
        `when`(mockLayoutManager.setTextLayout(layoutIndex)).thenReturn(mockKeyboardData)

        // Act - set same layout multiple times
        bridge.setTextLayout(layoutIndex)
        bridge.setTextLayout(layoutIndex)
        bridge.setTextLayout(layoutIndex)

        // Assert - all delegated
        verify(mockLayoutManager, times(3)).setTextLayout(layoutIndex)
        verify(mockKeyboardView, times(3)).setKeyboard(mockKeyboardData)
    }

    @Test
    fun testEdgeCase_incrementDecrementCycle() {
        // Arrange
        `when`(mockLayoutManager.incrTextLayout(1)).thenReturn(mockKeyboardData)
        `when`(mockLayoutManager.incrTextLayout(-1)).thenReturn(mockKeyboardData2)

        // Act - cycle forward and back
        bridge.incrTextLayout(1)
        bridge.incrTextLayout(-1)
        bridge.incrTextLayout(1)
        bridge.incrTextLayout(-1)

        // Assert
        verify(mockLayoutManager, times(2)).incrTextLayout(1)
        verify(mockLayoutManager, times(2)).incrTextLayout(-1)
        verify(mockKeyboardView, times(4)).setKeyboard(any())
    }

    @Test
    fun testEdgeCase_loadWithoutApplying() {
        // Arrange
        val layoutId = 123
        `when`(mockLayoutManager.loadLayout(layoutId)).thenReturn(mockKeyboardData)

        // Act - load but never apply to view
        val loaded = bridge.loadLayout(layoutId)

        // Assert - loaded but view not updated
        assertEquals(mockKeyboardData, loaded)
        verify(mockLayoutManager).loadLayout(layoutId)
        verifyNoInteractions(mockKeyboardView)
    }
}
