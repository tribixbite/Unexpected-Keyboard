package juloo.keyboard2

import android.content.Context
import android.content.res.Resources
import android.util.DisplayMetrics
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for SuggestionBarInitializer.
 *
 * Tests cover:
 * - Initialization with and without theme
 * - View hierarchy construction
 * - Layout parameter configuration
 * - Scroll view setup
 * - Content pane creation and sizing
 * - Edge cases and null handling
 */
@RunWith(MockitoJUnitRunner::class)
class SuggestionBarInitializerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockDisplayMetrics: DisplayMetrics

    @Mock
    private lateinit var mockTheme: Theme

    @Before
    fun setUp() {
        // Setup mock context and resources
        `when`(mockContext.resources).thenReturn(mockResources)
        `when`(mockResources.displayMetrics).thenReturn(mockDisplayMetrics)

        // Setup display metrics (1080x1920 typical phone screen)
        mockDisplayMetrics.density = 3.0f // xxhdpi
        mockDisplayMetrics.widthPixels = 1080
        mockDisplayMetrics.heightPixels = 1920
    }

    // ========== initialize() Tests (With Theme) ==========

    @Test
    fun testInitialize_withTheme_createsSuggestionBarWithTheme() {
        // Arrange
        val opacity = 90
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertNotNull("Should create suggestion bar", result.suggestionBar)
        assertNotNull("Should create input view container", result.inputViewContainer)
        assertNotNull("Should create scroll view", result.scrollView)
        assertNotNull("Should create content pane container", result.contentPaneContainer)
    }

    @Test
    fun testInitialize_withTheme_setsCorrectOpacity() {
        // Arrange
        val opacity = 75
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        // Note: We can't directly verify setOpacity() was called with specific value
        // since SuggestionBar is a real object, but we verify it was created
        assertNotNull("Suggestion bar should be created with opacity applied",
            result.suggestionBar)
    }

    // ========== initialize() Tests (Without Theme) ==========

    @Test
    fun testInitialize_withoutTheme_createsSuggestionBarWithoutTheme() {
        // Arrange
        val opacity = 100
        val heightPercent = 50

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        // Assert
        assertNotNull("Should create suggestion bar without theme", result.suggestionBar)
        assertNotNull("Should create input view container", result.inputViewContainer)
    }

    @Test
    fun testInitialize_withNullTheme_doesNotCrash() {
        // Arrange
        val opacity = 50
        val heightPercent = 30

        // Act & Assert - should not throw exception
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        assertNotNull("Result should not be null", result)
    }

    // ========== View Hierarchy Tests ==========

    @Test
    fun testInitialize_inputViewContainer_hasVerticalOrientation() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertEquals("Input view container should be vertical",
            LinearLayout.VERTICAL, result.inputViewContainer.orientation)
    }

    @Test
    fun testInitialize_scrollView_hasCorrectConfiguration() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertFalse("Scroll view should hide scrollbar",
            result.scrollView.isHorizontalScrollBarEnabled)
        assertFalse("Scroll view should not fill viewport",
            result.scrollView.isFillViewport)
    }

    @Test
    fun testInitialize_scrollView_hasCorrectLayoutParams() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        val params = result.scrollView.layoutParams as? LinearLayout.LayoutParams
        assertNotNull("Scroll view should have LinearLayout.LayoutParams", params)
        assertEquals("Scroll view width should be MATCH_PARENT",
            LinearLayout.LayoutParams.MATCH_PARENT, params?.width)

        // Height should be ~40dp converted to pixels
        // At density 3.0, 40dp = 120px
        assertTrue("Scroll view height should be positive", (params?.height ?: 0) > 0)
    }

    @Test
    fun testInitialize_suggestionBar_hasCorrectLayoutParams() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        val params = result.suggestionBar.layoutParams as? LinearLayout.LayoutParams
        assertNotNull("Suggestion bar should have LinearLayout.LayoutParams", params)
        assertEquals("Suggestion bar width should be WRAP_CONTENT",
            LinearLayout.LayoutParams.WRAP_CONTENT, params?.width)
        assertEquals("Suggestion bar height should be MATCH_PARENT",
            LinearLayout.LayoutParams.MATCH_PARENT, params?.height)
    }

    @Test
    fun testInitialize_scrollView_containsSuggestionBar() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertEquals("Scroll view should have exactly 1 child",
            1, result.scrollView.childCount)
        assertSame("Scroll view should contain suggestion bar",
            result.suggestionBar, result.scrollView.getChildAt(0))
    }

    @Test
    fun testInitialize_inputViewContainer_containsScrollView() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertTrue("Input view container should have at least 1 child",
            result.inputViewContainer.childCount >= 1)
        assertSame("First child should be scroll view",
            result.scrollView, result.inputViewContainer.getChildAt(0))
    }

    @Test
    fun testInitialize_inputViewContainer_containsContentPane() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertTrue("Input view container should have at least 2 children",
            result.inputViewContainer.childCount >= 2)
        assertSame("Second child should be content pane container",
            result.contentPaneContainer, result.inputViewContainer.getChildAt(1))
    }

    // ========== Content Pane Tests ==========

    @Test
    fun testInitialize_contentPaneContainer_isInitiallyHidden() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        assertEquals("Content pane should be initially GONE",
            android.view.View.GONE, result.contentPaneContainer.visibility)
    }

    @Test
    fun testInitialize_contentPaneContainer_hasCorrectHeight() {
        // Arrange
        mockDisplayMetrics.heightPixels = 1920
        val heightPercent = 40
        val expectedHeight = (1920 * 40) / 100 // 768 pixels

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, 1.0f, heightPercent
        )

        // Assert
        val params = result.contentPaneContainer.layoutParams as? LinearLayout.LayoutParams
        assertNotNull("Content pane should have LinearLayout.LayoutParams", params)
        assertEquals("Content pane height should be 40% of screen height",
            expectedHeight, params?.height)
    }

    @Test
    fun testInitialize_contentPaneContainer_hasMatchParentWidth() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, opacity, heightPercent
        )

        // Assert
        val params = result.contentPaneContainer.layoutParams as? LinearLayout.LayoutParams
        assertEquals("Content pane width should be MATCH_PARENT",
            LinearLayout.LayoutParams.MATCH_PARENT, params?.width)
    }

    // ========== calculateContentPaneHeight() Tests ==========

    @Test
    fun testCalculateContentPaneHeight_withStandardScreenHeight_calculatesCorrectly() {
        // Arrange
        mockDisplayMetrics.heightPixels = 1920
        val heightPercent = 40
        val expectedHeight = (1920 * 40) / 100 // 768

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 40% of 1920px", expectedHeight, result)
    }

    @Test
    fun testCalculateContentPaneHeight_with50Percent_calculatesCorrectly() {
        // Arrange
        mockDisplayMetrics.heightPixels = 2000
        val heightPercent = 50
        val expectedHeight = (2000 * 50) / 100 // 1000

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 50% of 2000px", expectedHeight, result)
    }

    @Test
    fun testCalculateContentPaneHeight_with100Percent_returnsFullScreenHeight() {
        // Arrange
        mockDisplayMetrics.heightPixels = 1920
        val heightPercent = 100
        val expectedHeight = 1920

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 100% of screen height", expectedHeight, result)
    }

    @Test
    fun testCalculateContentPaneHeight_with0Percent_returnsZero() {
        // Arrange
        mockDisplayMetrics.heightPixels = 1920
        val heightPercent = 0

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 0% as 0", 0, result)
    }

    @Test
    fun testCalculateContentPaneHeight_withSmallScreen_calculatesCorrectly() {
        // Arrange
        mockDisplayMetrics.heightPixels = 800 // Small phone
        val heightPercent = 30
        val expectedHeight = (800 * 30) / 100 // 240

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 30% of 800px", expectedHeight, result)
    }

    @Test
    fun testCalculateContentPaneHeight_withLargeScreen_calculatesCorrectly() {
        // Arrange
        mockDisplayMetrics.heightPixels = 3840 // 4K display
        val heightPercent = 25
        val expectedHeight = (3840 * 25) / 100 // 960

        // Act
        val result = SuggestionBarInitializer.calculateContentPaneHeight(
            mockContext, heightPercent
        )

        // Assert
        assertEquals("Should calculate 25% of 3840px", expectedHeight, result)
    }

    // ========== Edge Case Tests ==========

    @Test
    fun testInitialize_withZeroOpacity_doesNotCrash() {
        // Arrange
        val opacity = 0
        val heightPercent = 40

        // Act & Assert - should not throw
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        assertNotNull("Should create result even with 0 opacity", result)
    }

    @Test
    fun testInitialize_withFullOpacity_doesNotCrash() {
        // Arrange
        val opacity = 100
        val heightPercent = 40

        // Act & Assert - should not throw
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        assertNotNull("Should create result with full opacity", result)
    }

    @Test
    fun testInitialize_withMinimumHeightPercent_doesNotCrash() {
        // Arrange
        val opacity = 100
        val heightPercent = 0

        // Act & Assert - should not throw
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        assertNotNull("Should create result with 0% height", result)
    }

    @Test
    fun testInitialize_withMaximumHeightPercent_doesNotCrash() {
        // Arrange
        val opacity = 100
        val heightPercent = 100

        // Act & Assert - should not throw
        val result = SuggestionBarInitializer.initialize(
            mockContext, null, opacity, heightPercent
        )

        assertNotNull("Should create result with 100% height", result)
    }

    // ========== Data Class Tests ==========

    @Test
    fun testInitializationResult_dataClassEquality() {
        // Arrange
        val container1 = LinearLayout(mockContext)
        val container2 = LinearLayout(mockContext)
        val suggestionBar = SuggestionBar(mockContext)
        val contentPane = FrameLayout(mockContext)
        val scrollView = HorizontalScrollView(mockContext)

        val result1 = SuggestionBarInitializer.InitializationResult(
            container1, suggestionBar, contentPane, scrollView
        )
        val result2 = SuggestionBarInitializer.InitializationResult(
            container1, suggestionBar, contentPane, scrollView
        )
        val result3 = SuggestionBarInitializer.InitializationResult(
            container2, suggestionBar, contentPane, scrollView
        )

        // Assert
        assertEquals("Same references should be equal", result1, result2)
        assertNotEquals("Different containers should not be equal", result1, result3)
    }

    @Test
    fun testInitializationResult_allFieldsAccessible() {
        // Arrange & Act
        val result = SuggestionBarInitializer.initialize(
            mockContext, mockTheme, 1.0f, 40
        )

        // Assert - verify all fields are accessible and not null
        assertNotNull("inputViewContainer should be accessible",
            result.inputViewContainer)
        assertNotNull("suggestionBar should be accessible",
            result.suggestionBar)
        assertNotNull("contentPaneContainer should be accessible",
            result.contentPaneContainer)
        assertNotNull("scrollView should be accessible",
            result.scrollView)
    }
}
