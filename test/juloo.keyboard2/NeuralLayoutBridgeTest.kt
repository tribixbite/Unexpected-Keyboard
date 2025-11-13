package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for NeuralLayoutBridge.
 *
 * Tests cover:
 * - Dynamic keyboard height calculation (with/without helper/view)
 * - User keyboard height percentage (with/without helper)
 * - CGR prediction updates (with/without helper)
 * - Swipe prediction management (with/without helper)
 * - Neural keyboard layout configuration
 * - Factory method
 * - Multiple calls and integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class NeuralLayoutBridgeTest {

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    private lateinit var bridge: NeuralLayoutBridge

    @Before
    fun setUp() {
        bridge = NeuralLayoutBridge(mockNeuralLayoutHelper, mockKeyboardView)
    }

    // ========== Dynamic Keyboard Height Tests ==========

    @Test
    fun testCalculateDynamicKeyboardHeight_withHelper_delegatesToHelper() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight()).thenReturn(800f)

        // Act
        val result = bridge.calculateDynamicKeyboardHeight()

        // Assert
        assertEquals(800f, result, 0.01f)
        verify(mockNeuralLayoutHelper).calculateDynamicKeyboardHeight()
    }

    @Test
    fun testCalculateDynamicKeyboardHeight_nullHelper_fallsBackToView() {
        // Arrange - bridge with null helper
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)
        `when`(mockKeyboardView.height).thenReturn(1000)

        // Act
        val result = bridgeNullHelper.calculateDynamicKeyboardHeight()

        // Assert
        assertEquals(1000f, result, 0.01f)
        verify(mockKeyboardView).height
    }

    @Test
    fun testCalculateDynamicKeyboardHeight_nullHelperAndView_returnsZero() {
        // Arrange - bridge with null helper and null view
        val bridgeNullBoth = NeuralLayoutBridge(null, null)

        // Act
        val result = bridgeNullBoth.calculateDynamicKeyboardHeight()

        // Assert
        assertEquals(0f, result, 0.01f)
    }

    @Test
    fun testCalculateDynamicKeyboardHeight_multipleCallsWithHelper_delegatesEachTime() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight())
            .thenReturn(800f, 850f, 900f)

        // Act
        val result1 = bridge.calculateDynamicKeyboardHeight()
        val result2 = bridge.calculateDynamicKeyboardHeight()
        val result3 = bridge.calculateDynamicKeyboardHeight()

        // Assert - each call returns different value
        assertEquals(800f, result1, 0.01f)
        assertEquals(850f, result2, 0.01f)
        assertEquals(900f, result3, 0.01f)
        verify(mockNeuralLayoutHelper, times(3)).calculateDynamicKeyboardHeight()
    }

    @Test
    fun testCalculateDynamicKeyboardHeight_zeroHeight_returnsZero() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight()).thenReturn(0f)

        // Act
        val result = bridge.calculateDynamicKeyboardHeight()

        // Assert
        assertEquals(0f, result, 0.01f)
    }

    // ========== User Keyboard Height Percent Tests ==========

    @Test
    fun testGetUserKeyboardHeightPercent_withHelper_delegatesToHelper() {
        // Arrange
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(45)

        // Act
        val result = bridge.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(45, result)
        verify(mockNeuralLayoutHelper).getUserKeyboardHeightPercent()
    }

    @Test
    fun testGetUserKeyboardHeightPercent_nullHelper_returnsDefault35() {
        // Arrange - bridge with null helper
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)

        // Act
        val result = bridgeNullHelper.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(35, result)
    }

    @Test
    fun testGetUserKeyboardHeightPercent_multipleCallsWithHelper_delegatesEachTime() {
        // Arrange
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent())
            .thenReturn(40, 45, 50)

        // Act
        val result1 = bridge.getUserKeyboardHeightPercent()
        val result2 = bridge.getUserKeyboardHeightPercent()
        val result3 = bridge.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(40, result1)
        assertEquals(45, result2)
        assertEquals(50, result3)
        verify(mockNeuralLayoutHelper, times(3)).getUserKeyboardHeightPercent()
    }

    @Test
    fun testGetUserKeyboardHeightPercent_edgeValues_handlesCorrectly() {
        // Arrange
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent())
            .thenReturn(0, 100)

        // Act
        val result1 = bridge.getUserKeyboardHeightPercent()
        val result2 = bridge.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(0, result1)
        assertEquals(100, result2)
    }

    // ========== CGR Prediction Tests ==========

    @Test
    fun testUpdateCGRPredictions_withHelper_delegatesToHelper() {
        // Act
        bridge.updateCGRPredictions()

        // Assert
        verify(mockNeuralLayoutHelper).updateCGRPredictions()
    }

    @Test
    fun testUpdateCGRPredictions_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)

        // Act & Assert - should not throw
        bridgeNullHelper.updateCGRPredictions()
    }

    @Test
    fun testUpdateCGRPredictions_multipleCalls_delegatesEachTime() {
        // Act
        bridge.updateCGRPredictions()
        bridge.updateCGRPredictions()
        bridge.updateCGRPredictions()

        // Assert
        verify(mockNeuralLayoutHelper, times(3)).updateCGRPredictions()
    }

    @Test
    fun testCheckCGRPredictions_withHelper_delegatesToHelper() {
        // Act
        bridge.checkCGRPredictions()

        // Assert
        verify(mockNeuralLayoutHelper).checkCGRPredictions()
    }

    @Test
    fun testCheckCGRPredictions_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)

        // Act & Assert - should not throw
        bridgeNullHelper.checkCGRPredictions()
    }

    @Test
    fun testCheckCGRPredictions_multipleCalls_delegatesEachTime() {
        // Act
        bridge.checkCGRPredictions()
        bridge.checkCGRPredictions()

        // Assert
        verify(mockNeuralLayoutHelper, times(2)).checkCGRPredictions()
    }

    // ========== Swipe Prediction Tests ==========

    @Test
    fun testUpdateSwipePredictions_withHelper_delegatesToHelper() {
        // Arrange
        val predictions = listOf("hello", "world", "test")

        // Act
        bridge.updateSwipePredictions(predictions)

        // Assert
        verify(mockNeuralLayoutHelper).updateSwipePredictions(predictions)
    }

    @Test
    fun testUpdateSwipePredictions_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)
        val predictions = listOf("hello", "world")

        // Act & Assert - should not throw
        bridgeNullHelper.updateSwipePredictions(predictions)
    }

    @Test
    fun testUpdateSwipePredictions_emptyList_passedThrough() {
        // Arrange
        val predictions = emptyList<String>()

        // Act
        bridge.updateSwipePredictions(predictions)

        // Assert
        verify(mockNeuralLayoutHelper).updateSwipePredictions(predictions)
    }

    @Test
    fun testCompleteSwipePredictions_withHelper_delegatesToHelper() {
        // Arrange
        val finalPredictions = listOf("hello", "world")

        // Act
        bridge.completeSwipePredictions(finalPredictions)

        // Assert
        verify(mockNeuralLayoutHelper).completeSwipePredictions(finalPredictions)
    }

    @Test
    fun testCompleteSwipePredictions_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)
        val finalPredictions = listOf("hello")

        // Act & Assert - should not throw
        bridgeNullHelper.completeSwipePredictions(finalPredictions)
    }

    @Test
    fun testCompleteSwipePredictions_emptyList_passedThrough() {
        // Arrange
        val finalPredictions = emptyList<String>()

        // Act
        bridge.completeSwipePredictions(finalPredictions)

        // Assert
        verify(mockNeuralLayoutHelper).completeSwipePredictions(finalPredictions)
    }

    @Test
    fun testClearSwipePredictions_withHelper_delegatesToHelper() {
        // Act
        bridge.clearSwipePredictions()

        // Assert
        verify(mockNeuralLayoutHelper).clearSwipePredictions()
    }

    @Test
    fun testClearSwipePredictions_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)

        // Act & Assert - should not throw
        bridgeNullHelper.clearSwipePredictions()
    }

    @Test
    fun testClearSwipePredictions_multipleCalls_delegatesEachTime() {
        // Act
        bridge.clearSwipePredictions()
        bridge.clearSwipePredictions()

        // Assert
        verify(mockNeuralLayoutHelper, times(2)).clearSwipePredictions()
    }

    // ========== Neural Keyboard Layout Tests ==========

    @Test
    fun testSetNeuralKeyboardLayout_withHelper_delegatesToHelper() {
        // Act
        bridge.setNeuralKeyboardLayout()

        // Assert
        verify(mockNeuralLayoutHelper).setNeuralKeyboardLayout()
    }

    @Test
    fun testSetNeuralKeyboardLayout_nullHelper_doesNotCrash() {
        // Arrange
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)

        // Act & Assert - should not throw
        bridgeNullHelper.setNeuralKeyboardLayout()
    }

    @Test
    fun testSetNeuralKeyboardLayout_multipleCalls_delegatesEachTime() {
        // Act
        bridge.setNeuralKeyboardLayout()
        bridge.setNeuralKeyboardLayout()

        // Assert
        verify(mockNeuralLayoutHelper, times(2)).setNeuralKeyboardLayout()
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesBridge() {
        // Act
        val bridge = NeuralLayoutBridge.create(mockNeuralLayoutHelper, mockKeyboardView)

        // Assert
        assertNotNull("Factory method should create bridge", bridge)
    }

    @Test
    fun testCreate_factoryMethodBridgeWorks() {
        // Arrange
        val bridge = NeuralLayoutBridge.create(mockNeuralLayoutHelper, mockKeyboardView)
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(50)

        // Act
        val result = bridge.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(50, result)
        verify(mockNeuralLayoutHelper).getUserKeyboardHeightPercent()
    }

    @Test
    fun testCreate_withNullHelper() {
        // Act
        val bridge = NeuralLayoutBridge.create(null, mockKeyboardView)

        // Assert - should not crash
        assertNotNull("Factory should create bridge with null helper", bridge)

        // Should not crash when calling methods
        bridge.updateCGRPredictions()
        val result = bridge.getUserKeyboardHeightPercent()
        assertEquals(35, result) // Default value
    }

    @Test
    fun testCreate_withNullView() {
        // Act
        val bridge = NeuralLayoutBridge.create(mockNeuralLayoutHelper, null)

        // Assert
        assertNotNull("Factory should create bridge with null view", bridge)

        // Should work normally with helper
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(40)
        val result = bridge.getUserKeyboardHeightPercent()
        assertEquals(40, result)
    }

    @Test
    fun testCreate_withBothNull() {
        // Act
        val bridge = NeuralLayoutBridge.create(null, null)

        // Assert - should not crash
        assertNotNull("Factory should create bridge with null helper and view", bridge)

        // Should return defaults without crashing
        val height = bridge.calculateDynamicKeyboardHeight()
        assertEquals(0f, height, 0.01f)

        val percent = bridge.getUserKeyboardHeightPercent()
        assertEquals(35, percent)

        bridge.updateCGRPredictions() // No-op
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_swipePredictionWorkflow() {
        // Arrange
        val predictions1 = listOf("hel", "hel", "help")
        val predictions2 = listOf("hell", "hello", "help")
        val finalPredictions = listOf("hello", "hello", "help")

        // Act - simulate swipe workflow
        bridge.updateSwipePredictions(predictions1)
        bridge.updateSwipePredictions(predictions2)
        bridge.completeSwipePredictions(finalPredictions)
        bridge.clearSwipePredictions()

        // Assert - all methods called
        verify(mockNeuralLayoutHelper).updateSwipePredictions(predictions1)
        verify(mockNeuralLayoutHelper).updateSwipePredictions(predictions2)
        verify(mockNeuralLayoutHelper).completeSwipePredictions(finalPredictions)
        verify(mockNeuralLayoutHelper).clearSwipePredictions()
    }

    @Test
    fun testFullLifecycle_cgrPredictionWorkflow() {
        // Act - simulate CGR workflow
        bridge.setNeuralKeyboardLayout() // Set layout first
        bridge.updateCGRPredictions()
        bridge.checkCGRPredictions()
        bridge.updateCGRPredictions()
        bridge.checkCGRPredictions()

        // Assert
        verify(mockNeuralLayoutHelper).setNeuralKeyboardLayout()
        verify(mockNeuralLayoutHelper, times(2)).updateCGRPredictions()
        verify(mockNeuralLayoutHelper, times(2)).checkCGRPredictions()
    }

    @Test
    fun testFullLifecycle_heightCalculationWorkflow() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight()).thenReturn(800f)
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(45)

        // Act - simulate height calculation workflow
        val height = bridge.calculateDynamicKeyboardHeight()
        val percent = bridge.getUserKeyboardHeightPercent()

        // Assert
        assertEquals(800f, height, 0.01f)
        assertEquals(45, percent)
        verify(mockNeuralLayoutHelper).calculateDynamicKeyboardHeight()
        verify(mockNeuralLayoutHelper).getUserKeyboardHeightPercent()
    }

    @Test
    fun testIntegration_multipleBridgesIndependent() {
        // Arrange - create second bridge with different mocks
        val mockHelper2 = mock(NeuralLayoutHelper::class.java)
        val bridge2 = NeuralLayoutBridge.create(mockHelper2, mockKeyboardView)

        // Act - call both bridges
        bridge.updateCGRPredictions()
        bridge2.updateCGRPredictions()

        // Assert - each bridge calls its own helper
        verify(mockNeuralLayoutHelper).updateCGRPredictions()
        verify(mockHelper2).updateCGRPredictions()
    }

    @Test
    fun testIntegration_nullHelperAllMethods() {
        // Arrange - bridge with null helper
        val bridgeNullHelper = NeuralLayoutBridge(null, mockKeyboardView)
        `when`(mockKeyboardView.height).thenReturn(1000)

        // Act - call all methods (should not crash)
        val height = bridgeNullHelper.calculateDynamicKeyboardHeight()
        val percent = bridgeNullHelper.getUserKeyboardHeightPercent()
        bridgeNullHelper.updateCGRPredictions()
        bridgeNullHelper.checkCGRPredictions()
        bridgeNullHelper.updateSwipePredictions(listOf("test"))
        bridgeNullHelper.completeSwipePredictions(listOf("test"))
        bridgeNullHelper.clearSwipePredictions()
        bridgeNullHelper.setNeuralKeyboardLayout()

        // Assert - defaults returned, no crashes
        assertEquals(1000f, height, 0.01f) // Fallback to view
        assertEquals(35, percent) // Default
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_largeHeightValues() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight()).thenReturn(10000f)
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(200)

        // Act
        val height = bridge.calculateDynamicKeyboardHeight()
        val percent = bridge.getUserKeyboardHeightPercent()

        // Assert - large values passed through
        assertEquals(10000f, height, 0.01f)
        assertEquals(200, percent)
    }

    @Test
    fun testEdgeCase_negativeHeightValues() {
        // Arrange
        `when`(mockNeuralLayoutHelper.calculateDynamicKeyboardHeight()).thenReturn(-100f)
        `when`(mockNeuralLayoutHelper.getUserKeyboardHeightPercent()).thenReturn(-10)

        // Act
        val height = bridge.calculateDynamicKeyboardHeight()
        val percent = bridge.getUserKeyboardHeightPercent()

        // Assert - negative values passed through (validation in helper)
        assertEquals(-100f, height, 0.01f)
        assertEquals(-10, percent)
    }

    @Test
    fun testEdgeCase_singlePredictionInList() {
        // Arrange
        val predictions = listOf("hello")

        // Act
        bridge.updateSwipePredictions(predictions)

        // Assert
        verify(mockNeuralLayoutHelper).updateSwipePredictions(predictions)
    }
}
