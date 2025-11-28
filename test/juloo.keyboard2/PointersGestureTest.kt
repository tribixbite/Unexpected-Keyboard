package juloo.keyboard2

import android.content.Context
import android.content.res.Resources
import android.util.DisplayMetrics
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import org.mockito.ArgumentMatchers.any
import org.mockito.stubbing.Answer

@RunWith(MockitoJUnitRunner::class)
class PointersGestureTest {

    @Mock private lateinit var mockHandler: Pointers.IPointerEventHandler
    @Mock private lateinit var mockConfig: Config
    @Mock private lateinit var mockContext: Context
    @Mock private lateinit var mockResources: Resources
    @Mock private lateinit var mockDisplayMetrics: DisplayMetrics
    @Mock private lateinit var mockKey: KeyboardData.Key
    @Mock private lateinit var mockKeyValue: KeyValue

    private lateinit var pointers: Pointers

    @Before
    fun setUp() {
        `when`(mockContext.resources).thenReturn(mockResources)
        `when`(mockResources.displayMetrics).thenReturn(mockDisplayMetrics)
        mockDisplayMetrics.density = 1.0f // simplified density

        // Setup default config values
        mockConfig.short_gestures_enabled = true
        mockConfig.short_gesture_min_distance = 20
        mockConfig.swipe_dist_px = 50.0f // Absolute threshold
        mockConfig.swipe_typing_enabled = true // Enable to test interaction

        pointers = Pointers(mockHandler, mockConfig, mockContext)
    }

    @Test
    fun testShortGesture_onNonCharKey_usesLegacyLogic() {
        // Arrange
        val pointerId = 0
        val downX = 100f
        val downY = 100f
        val moveX = 160f // moved 60px right
        val moveY = 100f

        // Mock a non-char key (e.g. Fn)
        `when`(mockKeyValue.getKind()).thenReturn(KeyValue.Kind.Modifier)
        
        // Mock modifyKey to return center key first, then directional key
        // We need to handle the arguments correctly
        `when`(mockHandler.modifyKey(any(), any())).thenAnswer { invocation ->
            val key = invocation.arguments[0] as? KeyValue
            if (key == mockKeyValue) mockKeyValue else mock(KeyValue::class.java)
        }
        
        `when`(mockHandler.getKeyHypotenuse(mockKey)).thenReturn(100f)
        
        // Use a list for keys
        val keysList = MutableList<KeyValue?>(9) { null }
        keysList[0] = mockKeyValue
        // Fill others with mocks
        for (i in 1..8) keysList[i] = mock(KeyValue::class.java)
        
        `when`(mockKey.keys).thenReturn(keysList)

        // Act
        pointers.onTouchDown(downX, downY, pointerId, mockKey)
        pointers.onTouchMove(moveX, moveY, pointerId)
        pointers.onTouchUp(pointerId)

        // Assert
        // onPointerDown should be called twice:
        // 1. For the initial press (center key)
        // 2. For the short gesture result (directional key)
        verify(mockHandler, times(2)).onPointerDown(any(), anyBoolean())
    }
}
