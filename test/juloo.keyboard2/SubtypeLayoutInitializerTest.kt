package juloo.keyboard2

import android.content.res.Resources
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for SubtypeLayoutInitializer.
 *
 * Tests cover:
 * - First initialization (creates SubtypeManager, LayoutManager, LayoutBridge)
 * - Subsequent refresh (updates layout, no new bridge)
 * - Null default layout handling (fallback to QWERTY)
 * - Factory method and data class
 * - Multiple refresh cycles
 * - Edge cases (null resources, repeated initialization)
 */
@RunWith(MockitoJUnitRunner::class)
class SubtypeLayoutInitializerTest {

    @Mock
    private lateinit var mockKeyboard2: Keyboard2

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockSubtypeManager: SubtypeManager

    @Mock
    private lateinit var mockLayoutManager: LayoutManager

    @Mock
    private lateinit var mockDefaultLayout: KeyboardData

    @Mock
    private lateinit var mockQwertyLayout: KeyboardData

    private lateinit var initializer: SubtypeLayoutInitializer

    @Before
    fun setUp() {
        initializer = SubtypeLayoutInitializer(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView
        )

        // Mock SubtypeManager to return a default layout
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources))
            .thenReturn(mockDefaultLayout)
    }

    // ========== First Initialization Tests ==========

    @Test
    fun testRefreshSubtypeAndLayout_firstCall_createsSubtypeManager() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert
        assertNotNull("Should create SubtypeManager", result.subtypeManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_firstCall_createsLayoutManager() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert
        assertNotNull("Should create LayoutManager", result.layoutManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_firstCall_createsLayoutBridge() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert
        assertNotNull("Should create LayoutBridge on first call", result.layoutBridge)
    }

    // ========== Subsequent Refresh Tests ==========

    @Test
    fun testRefreshSubtypeAndLayout_subsequentCall_returnsExistingSubtypeManager() {
        // Act
        val result = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert
        assertSame("Should return existing SubtypeManager", mockSubtypeManager, result.subtypeManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_subsequentCall_returnsExistingLayoutManager() {
        // Act
        val result = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert
        assertSame("Should return existing LayoutManager", mockLayoutManager, result.layoutManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_subsequentCall_doesNotCreateBridge() {
        // Act
        val result = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert
        assertNull("Should not create LayoutBridge on subsequent call", result.layoutBridge)
    }

    @Test
    fun testRefreshSubtypeAndLayout_subsequentCall_updatesLayoutManager() {
        // Act
        initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert
        verify(mockLayoutManager).setLocaleTextLayout(mockDefaultLayout)
    }

    @Test
    fun testRefreshSubtypeAndLayout_subsequentCall_refreshesSubtype() {
        // Act
        initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert
        verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)
    }

    // ========== Null Default Layout Tests ==========

    @Test
    fun testRefreshSubtypeAndLayout_nullDefaultLayout_createsLayoutManager() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources)).thenReturn(null)

        // Act - first call with null default layout
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert - should still create LayoutManager (using fallback QWERTY)
        assertNotNull("Should create LayoutManager even with null default layout", result.layoutManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_nullDefaultLayout_createsLayoutBridge() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources)).thenReturn(null)

        // Act
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert
        assertNotNull("Should create LayoutBridge even with null default layout", result.layoutBridge)
    }

    @Test
    fun testRefreshSubtypeAndLayout_nullDefaultLayout_updatesExistingLayoutManager() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources)).thenReturn(null)

        // Act - subsequent call with null default layout
        val result = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert - should update with fallback layout
        verify(mockLayoutManager).setLocaleTextLayout(any())
        assertSame("Should return existing LayoutManager", mockLayoutManager, result.layoutManager)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesInitializer() {
        // Act
        val initializer = SubtypeLayoutInitializer.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView
        )

        // Assert
        assertNotNull("Factory method should create initializer", initializer)
    }

    @Test
    fun testCreate_factoryMethodInitializerWorks() {
        // Arrange
        val initializer = SubtypeLayoutInitializer.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView
        )
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act
        val result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert
        assertNotNull("Factory-created initializer should work", result)
        assertNotNull("Should create LayoutManager", result.layoutManager)
    }

    // ========== Data Class Tests ==========

    @Test
    fun testInitializationResult_isDataClass() {
        // Arrange
        val result1 = SubtypeLayoutInitializer.InitializationResult(
            mockSubtypeManager,
            mockLayoutManager,
            null
        )
        val result2 = SubtypeLayoutInitializer.InitializationResult(
            mockSubtypeManager,
            mockLayoutManager,
            null
        )

        // Act & Assert - data class equality
        assertEquals("Same managers should be equal", result1, result2)
    }

    @Test
    fun testInitializationResult_copyWorks() {
        // Arrange
        val bridge = mock(LayoutBridge::class.java)
        val result = SubtypeLayoutInitializer.InitializationResult(
            mockSubtypeManager,
            mockLayoutManager,
            null
        )

        // Act
        val copied = result.copy(layoutBridge = bridge)

        // Assert
        assertNotEquals("Copied result should differ from original", result, copied)
        assertEquals("Copied result should have new bridge", bridge, copied.layoutBridge)
    }

    @Test
    fun testInitializationResult_accessFields() {
        // Arrange
        val bridge = mock(LayoutBridge::class.java)
        val result = SubtypeLayoutInitializer.InitializationResult(
            mockSubtypeManager,
            mockLayoutManager,
            bridge
        )

        // Act & Assert
        assertEquals("Should access subtypeManager", mockSubtypeManager, result.subtypeManager)
        assertEquals("Should access layoutManager", mockLayoutManager, result.layoutManager)
        assertEquals("Should access layoutBridge", bridge, result.layoutBridge)
    }

    // ========== Multiple Refresh Cycles ==========

    @Test
    fun testRefreshSubtypeAndLayout_multipleFirstCalls_createMultipleManagers() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - two first calls (simulating multiple initializations)
        val result1 = initializer.refreshSubtypeAndLayout(null, null, mockResources)
        val result2 = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert - both should create managers and bridges
        assertNotNull("First call should create managers", result1.layoutManager)
        assertNotNull("First call should create bridge", result1.layoutBridge)
        assertNotNull("Second call should create managers", result2.layoutManager)
        assertNotNull("Second call should create bridge", result2.layoutBridge)
        assertNotSame("Should create different managers", result1.layoutManager, result2.layoutManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_alternatingFirstAndSubsequent() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - first call
        val result1 = initializer.refreshSubtypeAndLayout(null, null, mockResources)
        assertNotNull("First call should create bridge", result1.layoutBridge)

        // Act - subsequent call with managers from first call
        val result2 = initializer.refreshSubtypeAndLayout(
            result1.subtypeManager,
            result1.layoutManager,
            mockResources
        )

        // Assert
        assertNull("Subsequent call should not create bridge", result2.layoutBridge)
        assertSame("Should return same SubtypeManager", result1.subtypeManager, result2.subtypeManager)
        assertSame("Should return same LayoutManager", result1.layoutManager, result2.layoutManager)
    }

    @Test
    fun testRefreshSubtypeAndLayout_multipleSubsequentCalls_updatesLayoutEachTime() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources))
            .thenReturn(mockDefaultLayout)
            .thenReturn(mockQwertyLayout)
            .thenReturn(mockDefaultLayout)

        // Act - three subsequent calls
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)

        // Assert - should update layout three times
        verify(mockLayoutManager, times(3)).setLocaleTextLayout(any())
        verify(mockSubtypeManager, times(3)).refreshSubtype(mockConfig, mockResources)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_initializeThenRefresh() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - first call (initialize)
        val initResult = initializer.refreshSubtypeAndLayout(null, null, mockResources)
        assertNotNull("Should create bridge on init", initResult.layoutBridge)

        // Act - second call (refresh with created managers)
        val refreshResult = initializer.refreshSubtypeAndLayout(
            initResult.subtypeManager,
            initResult.layoutManager,
            mockResources
        )

        // Assert
        assertNull("Should not create bridge on refresh", refreshResult.layoutBridge)
        assertSame("Should use same managers", initResult.subtypeManager, refreshResult.subtypeManager)
        verify(mockLayoutManager).setLocaleTextLayout(mockDefaultLayout)
    }

    @Test
    fun testFullLifecycle_multipleRefreshes() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - initialize
        var result = initializer.refreshSubtypeAndLayout(null, null, mockResources)

        // Act - refresh 5 times
        for (i in 1..5) {
            result = initializer.refreshSubtypeAndLayout(
                result.subtypeManager,
                result.layoutManager,
                mockResources
            )
            assertNull("Refresh $i should not create bridge", result.layoutBridge)
        }

        // Assert - layout should be updated 5 times
        verify(mockLayoutManager, times(5)).setLocaleTextLayout(mockDefaultLayout)
    }

    @Test
    fun testIntegration_multipleInitializersIndependent() {
        // Arrange - create two initializers
        val initializer1 = SubtypeLayoutInitializer.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView
        )

        val initializer2 = SubtypeLayoutInitializer.create(
            mockKeyboard2,
            mockConfig,
            mockKeyboardView
        )

        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act
        val result1 = initializer1.refreshSubtypeAndLayout(null, null, mockResources)
        val result2 = initializer2.refreshSubtypeAndLayout(null, null, mockResources)

        // Assert - both should work independently
        assertNotNull("First initializer should work", result1)
        assertNotNull("Second initializer should work", result2)
        assertNotNull("First should create bridge", result1.layoutBridge)
        assertNotNull("Second should create bridge", result2.layoutBridge)
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_onlySubtypeManagerProvided() {
        // Arrange
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - provide SubtypeManager but null LayoutManager
        val result = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            null,
            mockResources
        )

        // Assert - should create LayoutManager and bridge
        assertSame("Should use provided SubtypeManager", mockSubtypeManager, result.subtypeManager)
        assertNotNull("Should create new LayoutManager", result.layoutManager)
        assertNotNull("Should create LayoutBridge", result.layoutBridge)
    }

    @Test
    fun testEdgeCase_onlyLayoutManagerProvided() {
        // Arrange - this shouldn't happen in practice, but test robustness
        // If SubtypeManager is null but LayoutManager is provided, create SubtypeManager
        // but update the LayoutManager (treat as subsequent call)
        `when`(mockSubtypeManager.refreshSubtype(any(), any())).thenReturn(mockDefaultLayout)

        // Act - provide LayoutManager but null SubtypeManager
        val result = initializer.refreshSubtypeAndLayout(
            null,
            mockLayoutManager,
            mockResources
        )

        // Assert - should create SubtypeManager
        assertNotNull("Should create SubtypeManager", result.subtypeManager)
        assertSame("Should use provided LayoutManager", mockLayoutManager, result.layoutManager)
        assertNull("Should not create bridge (LayoutManager already exists)", result.layoutBridge)
    }

    @Test
    fun testEdgeCase_sameManagersRepeated() {
        // Act - call with same managers repeatedly
        val result1 = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )
        val result2 = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )
        val result3 = initializer.refreshSubtypeAndLayout(
            mockSubtypeManager,
            mockLayoutManager,
            mockResources
        )

        // Assert - all should return same managers, no bridges
        assertSame("Should return same SubtypeManager", mockSubtypeManager, result1.subtypeManager)
        assertSame("Should return same LayoutManager", mockLayoutManager, result1.layoutManager)
        assertNull("Should not create bridge", result1.layoutBridge)
        assertNull("Should not create bridge", result2.layoutBridge)
        assertNull("Should not create bridge", result3.layoutBridge)
    }

    @Test
    fun testEdgeCase_changeLayoutBetweenCalls() {
        // Arrange - first call returns one layout, second call returns different layout
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources))
            .thenReturn(mockDefaultLayout)
            .thenReturn(mockQwertyLayout)

        // Act - first call
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)
        verify(mockLayoutManager).setLocaleTextLayout(mockDefaultLayout)

        // Act - second call (layout changes)
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)

        // Assert - should update with new layout
        verify(mockLayoutManager).setLocaleTextLayout(mockQwertyLayout)
    }

    @Test
    fun testEdgeCase_nullThenNonNullLayout() {
        // Arrange - first call returns null, second call returns layout
        `when`(mockSubtypeManager.refreshSubtype(mockConfig, mockResources))
            .thenReturn(null)
            .thenReturn(mockDefaultLayout)

        // Act - first call with null layout
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)
        verify(mockLayoutManager).setLocaleTextLayout(any())  // Should use fallback

        // Act - second call with valid layout
        initializer.refreshSubtypeAndLayout(mockSubtypeManager, mockLayoutManager, mockResources)

        // Assert - should update with new layout
        verify(mockLayoutManager).setLocaleTextLayout(mockDefaultLayout)
        verify(mockLayoutManager, times(2)).setLocaleTextLayout(any())
    }
}
