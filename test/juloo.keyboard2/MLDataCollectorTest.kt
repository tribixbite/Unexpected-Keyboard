package juloo.keyboard2

import android.content.Context
import android.content.res.Resources
import android.util.DisplayMetrics
import juloo.keyboard2.ml.SwipeMLData
import juloo.keyboard2.ml.SwipeMLDataStore
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for MLDataCollector.
 *
 * Tests cover:
 * - Data collection with valid inputs
 * - Null handling and edge cases
 * - Coordinate normalization/denormalization
 * - Registered key copying
 * - Error handling
 * - Store interaction
 */
@RunWith(MockitoJUnitRunner::class)
class MLDataCollectorTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockDataStore: SwipeMLDataStore

    private lateinit var collector: MLDataCollector
    private lateinit var displayMetrics: DisplayMetrics

    @Before
    fun setUp() {
        // Setup display metrics
        displayMetrics = DisplayMetrics().apply {
            widthPixels = 1080
            heightPixels = 2340
        }

        `when`(mockContext.resources).thenReturn(mockResources)
        `when`(mockResources.displayMetrics).thenReturn(displayMetrics)
        
        // Mock SharedPreferences for PrivacyManager
        val mockPrefs = mock(android.content.SharedPreferences::class.java)
        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockContext.applicationContext).thenReturn(mockContext)

        collector = MLDataCollector(mockContext)
    }

    @Test
    fun testCollectAndStoreSwipeData_withValidData_returnsTrue() {
        // Arrange
        val word = "hello"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertTrue("Should successfully collect valid data", result)
        verify(mockDataStore, times(1)).storeSwipeData(any(SwipeMLData::class.java))
    }

    @Test
    fun testCollectAndStoreSwipeData_withNullSwipeData_returnsFalse() {
        // Arrange
        val word = "hello"
        val keyboardHeight = 800

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, null, keyboardHeight, mockDataStore
        )

        // Assert
        assertFalse("Should return false for null swipe data", result)
        verify(mockDataStore, never()).storeSwipeData(any(SwipeMLData::class.java))
    }

    @Test
    fun testCollectAndStoreSwipeData_withNullDataStore_returnsFalse() {
        // Arrange
        val word = "hello"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, null
        )

        // Assert
        assertFalse("Should return false for null data store", result)
    }

    @Test
    fun testCollectAndStoreSwipeData_stripsRawPrefix() {
        // Arrange
        val wordWithPrefix = "raw:hello"
        val expectedCleanWord = "hello"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Capture the stored ML data
        val captor = argumentCaptor<SwipeMLData>()

        // Act
        collector.collectAndStoreSwipeData(
            wordWithPrefix, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        verify(mockDataStore).storeSwipeData(captor.capture())
        val storedData = captor.value
        assertEquals("Should strip 'raw:' prefix", expectedCleanWord, storedData.targetWord)
    }

    @Test
    fun testCollectAndStoreSwipeData_copiesTracePoints() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeDataWithTracePoints(3)

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertTrue("Should successfully collect data", result)

        // Verify trace points were copied
        val captor = argumentCaptor<SwipeMLData>()
        verify(mockDataStore).storeSwipeData(captor.capture())
        val storedData = captor.value
        assertEquals("Should have 3 trace points", 3, storedData.getTracePoints().size)
    }

    @Test
    fun testCollectAndStoreSwipeData_copiesRegisteredKeys() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val registeredKeys = listOf("t", "e", "s", "t")
        val swipeData = createMockSwipeDataWithKeys(registeredKeys)

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertTrue("Should successfully collect data", result)

        // Verify registered keys were copied
        val captor = argumentCaptor<SwipeMLData>()
        verify(mockDataStore).storeSwipeData(captor.capture())
        val storedData = captor.value
        assertEquals("Should have 4 registered keys", 4, storedData.getRegisteredKeys().size)
        assertTrue("Should contain key 't'", storedData.getRegisteredKeys().contains("t"))
        assertTrue("Should contain key 'e'", storedData.getRegisteredKeys().contains("e"))
        assertTrue("Should contain key 's'", storedData.getRegisteredKeys().contains("s"))
    }

    @Test
    fun testCollectAndStoreSwipeData_handlesEmptyTracePoints() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeDataWithTracePoints(0)

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertTrue("Should handle empty trace points", result)
        verify(mockDataStore, times(1)).storeSwipeData(any(SwipeMLData::class.java))
    }

    @Test
    fun testCollectAndStoreSwipeData_handlesEmptyRegisteredKeys() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeDataWithKeys(emptyList())

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertTrue("Should handle empty registered keys", result)
        verify(mockDataStore, times(1)).storeSwipeData(any(SwipeMLData::class.java))
    }

    @Test
    fun testCollectAndStoreSwipeData_handlesExceptionGracefully() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Mock store to throw exception
        `when`(mockDataStore.storeSwipeData(any(SwipeMLData::class.java)))
            .thenThrow(RuntimeException("Storage error"))

        // Act
        val result = collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        assertFalse("Should return false on exception", result)
    }

    @Test
    fun testCollectAndStoreSwipeData_usesCorrectDimensions() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Act
        collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        val captor = argumentCaptor<SwipeMLData>()
        verify(mockDataStore).storeSwipeData(captor.capture())
        val storedData = captor.value

        // Verify dimensions match display metrics
        assertEquals("Width should match display metrics",
            displayMetrics.widthPixels, storedData.screenWidthPx)
        assertEquals("Height should match display metrics",
            displayMetrics.heightPixels, storedData.screenHeightPx)
        assertEquals("Keyboard height should match parameter",
            keyboardHeight, storedData.keyboardHeightPx)
    }

    @Test
    fun testCollectAndStoreSwipeData_setsUserSelectionSource() {
        // Arrange
        val word = "test"
        val keyboardHeight = 800
        val swipeData = createMockSwipeData()

        // Act
        collector.collectAndStoreSwipeData(
            word, swipeData, keyboardHeight, mockDataStore
        )

        // Assert
        val captor = argumentCaptor<SwipeMLData>()
        verify(mockDataStore).storeSwipeData(captor.capture())
        val storedData = captor.value
        assertEquals("Source should be user_selection",
            "user_selection", storedData.collectionSource)
    }

    // Helper methods for creating mock data

    private fun createMockSwipeData(): SwipeMLData {
        return mock(SwipeMLData::class.java).apply {
            `when`(getTracePoints()).thenReturn(emptyList())
            `when`(getRegisteredKeys()).thenReturn(emptyList())
        }
    }

    private fun createMockSwipeDataWithTracePoints(count: Int): SwipeMLData {
        val points = mutableListOf<SwipeMLData.TracePoint>()
        for (i in 0 until count) {
            val point = SwipeMLData.TracePoint(
                x = 0.5f + i * 0.1f,
                y = 0.5f + i * 0.1f,
                tDeltaMs = i * 10L
            )
            points.add(point)
        }

        return mock(SwipeMLData::class.java).apply {
            `when`(getTracePoints()).thenReturn(points)
            `when`(getRegisteredKeys()).thenReturn(emptyList())
        }
    }

    private fun createMockSwipeDataWithKeys(keys: List<String>): SwipeMLData {
        return mock(SwipeMLData::class.java).apply {
            `when`(getTracePoints()).thenReturn(emptyList())
            `when`(getRegisteredKeys()).thenReturn(keys)
        }
    }

    // Mockito helper for argument capture
    private inline fun <reified T> argumentCaptor(): org.mockito.ArgumentCaptor<T> {
        return org.mockito.ArgumentCaptor.forClass(T::class.java)
    }
}
