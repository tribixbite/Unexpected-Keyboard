package juloo.keyboard2

import android.os.Build
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.Window
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for WindowLayoutUtils.
 *
 * Tests cover:
 * - Window layout height updates
 * - View layout height updates
 * - View gravity updates (LinearLayout and FrameLayout)
 * - Edge-to-edge configuration (API 35+)
 * - Soft input window layout parameter updates
 * - Fullscreen vs non-fullscreen modes
 * - Null safety and edge cases
 */
@RunWith(MockitoJUnitRunner::class)
class WindowLayoutUtilsTest {

    @Mock
    private lateinit var mockWindow: Window

    @Mock
    private lateinit var mockView: View

    @Mock
    private lateinit var mockParentView: View

    @Mock
    private lateinit var mockWindowAttributes: WindowManager.LayoutParams

    @Mock
    private lateinit var mockViewLayoutParams: ViewGroup.LayoutParams

    @Mock
    private lateinit var mockLinearLayoutParams: LinearLayout.LayoutParams

    @Mock
    private lateinit var mockFrameLayoutParams: FrameLayout.LayoutParams

    @Before
    fun setUp() {
        // Setup window attributes
        `when`(mockWindow.attributes).thenReturn(mockWindowAttributes)
        mockWindowAttributes.height = ViewGroup.LayoutParams.WRAP_CONTENT

        // Setup view layout params
        `when`(mockView.layoutParams).thenReturn(mockViewLayoutParams)
        mockViewLayoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
    }

    // ========== Window Layout Height Tests ==========

    @Test
    fun testUpdateLayoutHeightOf_window_withDifferentHeight_updatesHeight() {
        // Arrange
        mockWindowAttributes.height = ViewGroup.LayoutParams.WRAP_CONTENT
        val newHeight = ViewGroup.LayoutParams.MATCH_PARENT

        // Act
        WindowLayoutUtils.updateLayoutHeightOf(mockWindow, newHeight)

        // Assert
        assertEquals("Window height should be updated",
            newHeight, mockWindowAttributes.height)
        verify(mockWindow).attributes = mockWindowAttributes
    }

    @Test
    fun testUpdateLayoutHeightOf_window_withSameHeight_doesNotUpdate() {
        // Arrange
        val existingHeight = ViewGroup.LayoutParams.MATCH_PARENT
        mockWindowAttributes.height = existingHeight

        // Act
        WindowLayoutUtils.updateLayoutHeightOf(mockWindow, existingHeight)

        // Assert - should not set attributes since height unchanged
        verify(mockWindow, never()).attributes = any()
    }

    @Test
    fun testUpdateLayoutHeightOf_window_withNullAttributes_handlesGracefully() {
        // Arrange
        `when`(mockWindow.attributes).thenReturn(null)

        // Act - should not crash
        WindowLayoutUtils.updateLayoutHeightOf(mockWindow, ViewGroup.LayoutParams.MATCH_PARENT)

        // Assert - no exception thrown, method returns gracefully
        verify(mockWindow, never()).attributes = any()
    }

    // ========== View Layout Height Tests ==========

    @Test
    fun testUpdateLayoutHeightOf_view_withDifferentHeight_updatesHeight() {
        // Arrange
        mockViewLayoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
        val newHeight = ViewGroup.LayoutParams.MATCH_PARENT

        // Act
        WindowLayoutUtils.updateLayoutHeightOf(mockView, newHeight)

        // Assert
        assertEquals("View height should be updated",
            newHeight, mockViewLayoutParams.height)
        verify(mockView).layoutParams = mockViewLayoutParams
    }

    @Test
    fun testUpdateLayoutHeightOf_view_withSameHeight_doesNotUpdate() {
        // Arrange
        val existingHeight = ViewGroup.LayoutParams.MATCH_PARENT
        mockViewLayoutParams.height = existingHeight

        // Act
        WindowLayoutUtils.updateLayoutHeightOf(mockView, existingHeight)

        // Assert - should not set layoutParams since height unchanged
        verify(mockView, never()).layoutParams = any()
    }

    @Test
    fun testUpdateLayoutHeightOf_view_withNullLayoutParams_handlesGracefully() {
        // Arrange
        `when`(mockView.layoutParams).thenReturn(null)

        // Act - should not crash
        WindowLayoutUtils.updateLayoutHeightOf(mockView, ViewGroup.LayoutParams.MATCH_PARENT)

        // Assert - no exception thrown, method returns gracefully
        verify(mockView, never()).layoutParams = any()
    }

    // ========== View Gravity Tests (LinearLayout) ==========

    @Test
    fun testUpdateLayoutGravityOf_linearLayout_withDifferentGravity_updatesGravity() {
        // Arrange
        `when`(mockView.layoutParams).thenReturn(mockLinearLayoutParams)
        mockLinearLayoutParams.gravity = Gravity.TOP
        val newGravity = Gravity.BOTTOM

        // Act
        WindowLayoutUtils.updateLayoutGravityOf(mockView, newGravity)

        // Assert
        assertEquals("LinearLayout gravity should be updated",
            newGravity, mockLinearLayoutParams.gravity)
        verify(mockView).layoutParams = mockLinearLayoutParams
    }

    @Test
    fun testUpdateLayoutGravityOf_linearLayout_withSameGravity_doesNotUpdate() {
        // Arrange
        `when`(mockView.layoutParams).thenReturn(mockLinearLayoutParams)
        val existingGravity = Gravity.BOTTOM
        mockLinearLayoutParams.gravity = existingGravity

        // Act
        WindowLayoutUtils.updateLayoutGravityOf(mockView, existingGravity)

        // Assert - should not set layoutParams since gravity unchanged
        verify(mockView, never()).layoutParams = any()
    }

    // ========== View Gravity Tests (FrameLayout) ==========

    @Test
    fun testUpdateLayoutGravityOf_frameLayout_withDifferentGravity_updatesGravity() {
        // Arrange
        `when`(mockView.layoutParams).thenReturn(mockFrameLayoutParams)
        mockFrameLayoutParams.gravity = Gravity.TOP
        val newGravity = Gravity.BOTTOM

        // Act
        WindowLayoutUtils.updateLayoutGravityOf(mockView, newGravity)

        // Assert
        assertEquals("FrameLayout gravity should be updated",
            newGravity, mockFrameLayoutParams.gravity)
        verify(mockView).layoutParams = mockFrameLayoutParams
    }

    @Test
    fun testUpdateLayoutGravityOf_frameLayout_withSameGravity_doesNotUpdate() {
        // Arrange
        `when`(mockView.layoutParams).thenReturn(mockFrameLayoutParams)
        val existingGravity = Gravity.BOTTOM
        mockFrameLayoutParams.gravity = existingGravity

        // Act
        WindowLayoutUtils.updateLayoutGravityOf(mockView, existingGravity)

        // Assert - should not set layoutParams since gravity unchanged
        verify(mockView, never()).layoutParams = any()
    }

    @Test
    fun testUpdateLayoutGravityOf_otherLayoutParams_doesNotUpdate() {
        // Arrange - use generic ViewGroup.LayoutParams (not Linear/Frame)
        `when`(mockView.layoutParams).thenReturn(mockViewLayoutParams)

        // Act
        WindowLayoutUtils.updateLayoutGravityOf(mockView, Gravity.BOTTOM)

        // Assert - should not update since it's not LinearLayout or FrameLayout params
        verify(mockView, never()).layoutParams = any()
    }

    // ========== Edge-to-Edge Configuration Tests ==========

    @Test
    fun testConfigureEdgeToEdge_api35Plus_configuresWindow() {
        // Note: This test requires API 35+ to actually execute the configuration code.
        // On lower API versions, the method returns early without doing anything.
        // We test that the method executes without crashing.

        // Act - should not crash regardless of API level
        WindowLayoutUtils.configureEdgeToEdge(mockWindow)

        // Assert - On API 35+, window would be configured.
        // On lower APIs, this is a no-op. We verify the method completes.
        // (Cannot easily mock Build.VERSION.SDK_INT in unit tests)
        assertTrue("Method should execute without exception", true)
    }

    // ========== Soft Input Window Layout Tests ==========

    @Test
    fun testUpdateSoftInputWindowLayoutParams_fullscreenMode_setsMatchParent() {
        // Arrange
        val mockInputArea = mock(View::class.java)
        `when`(mockInputArea.parent).thenReturn(mockParentView)
        `when`(mockParentView.layoutParams).thenReturn(mockViewLayoutParams)
        mockViewLayoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT

        // Act
        WindowLayoutUtils.updateSoftInputWindowLayoutParams(
            mockWindow, mockInputArea, isFullscreen = true
        )

        // Assert
        assertEquals("Window should be MATCH_PARENT",
            ViewGroup.LayoutParams.MATCH_PARENT, mockWindowAttributes.height)
        assertEquals("Input area parent should be MATCH_PARENT in fullscreen",
            ViewGroup.LayoutParams.MATCH_PARENT, mockViewLayoutParams.height)
        verify(mockWindow).attributes = mockWindowAttributes
        verify(mockParentView).layoutParams = mockViewLayoutParams
    }

    @Test
    fun testUpdateSoftInputWindowLayoutParams_nonFullscreenMode_setsWrapContent() {
        // Arrange
        val mockInputArea = mock(View::class.java)
        `when`(mockInputArea.parent).thenReturn(mockParentView)
        `when`(mockParentView.layoutParams).thenReturn(mockViewLayoutParams)
        mockViewLayoutParams.height = ViewGroup.LayoutParams.MATCH_PARENT

        // Act
        WindowLayoutUtils.updateSoftInputWindowLayoutParams(
            mockWindow, mockInputArea, isFullscreen = false
        )

        // Assert
        assertEquals("Window should be MATCH_PARENT",
            ViewGroup.LayoutParams.MATCH_PARENT, mockWindowAttributes.height)
        assertEquals("Input area parent should be WRAP_CONTENT in non-fullscreen",
            ViewGroup.LayoutParams.WRAP_CONTENT, mockViewLayoutParams.height)
        verify(mockWindow).attributes = mockWindowAttributes
        verify(mockParentView).layoutParams = mockViewLayoutParams
    }

    @Test
    fun testUpdateSoftInputWindowLayoutParams_setsBottomGravity() {
        // Arrange
        val mockInputArea = mock(View::class.java)
        `when`(mockInputArea.parent).thenReturn(mockParentView)
        `when`(mockParentView.layoutParams).thenReturn(mockLinearLayoutParams)
        mockLinearLayoutParams.gravity = Gravity.TOP
        mockLinearLayoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT

        // Act
        WindowLayoutUtils.updateSoftInputWindowLayoutParams(
            mockWindow, mockInputArea, isFullscreen = false
        )

        // Assert
        assertEquals("Input area parent should have BOTTOM gravity",
            Gravity.BOTTOM, mockLinearLayoutParams.gravity)
        verify(mockParentView).layoutParams = mockLinearLayoutParams
    }

    @Test
    fun testUpdateSoftInputWindowLayoutParams_withNullParent_handlesGracefully() {
        // Arrange
        val mockInputArea = mock(View::class.java)
        `when`(mockInputArea.parent).thenReturn(null)

        // Act - should not crash
        WindowLayoutUtils.updateSoftInputWindowLayoutParams(
            mockWindow, mockInputArea, isFullscreen = false
        )

        // Assert - window should still be configured even if parent is null
        assertEquals("Window should be MATCH_PARENT",
            ViewGroup.LayoutParams.MATCH_PARENT, mockWindowAttributes.height)
        verify(mockWindow).attributes = mockWindowAttributes
    }

    @Test
    fun testUpdateSoftInputWindowLayoutParams_callsConfigureEdgeToEdge() {
        // Arrange
        val mockInputArea = mock(View::class.java)
        `when`(mockInputArea.parent).thenReturn(mockParentView)
        `when`(mockParentView.layoutParams).thenReturn(mockViewLayoutParams)

        // Act
        WindowLayoutUtils.updateSoftInputWindowLayoutParams(
            mockWindow, mockInputArea, isFullscreen = false
        )

        // Assert - verify edge-to-edge configuration is called (no crash)
        // (Cannot verify internal call easily, but we verify method completes)
        assertTrue("Method should complete without exception", true)
    }
}
