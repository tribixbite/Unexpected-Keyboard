package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for PreferenceUIUpdateHandler.
 *
 * Tests cover:
 * - Keyboard layout updates
 * - Suggestion bar opacity updates
 * - Neural engine config updates for model settings
 * - Null handling for all dependencies
 * - Factory method
 * - Multiple update cycles
 * - Edge cases (null keys, unrelated keys)
 */
@RunWith(MockitoJUnitRunner::class)
class PreferenceUIUpdateHandlerTest {

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockLayoutBridge: LayoutBridge

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockSuggestionBar: SuggestionBar

    @Mock
    private lateinit var mockNeuralEngine: NeuralEngine

    @Mock
    private lateinit var mockLayout: KeyboardData

    private lateinit var handler: PreferenceUIUpdateHandler

    @Before
    fun setUp() {
        // Setup config defaults
        mockConfig.suggestion_bar_opacity = 0.8f

        // Setup layout bridge to return a layout
        `when`(mockLayoutBridge.getCurrentLayout()).thenReturn(mockLayout)

        // Setup prediction coordinator to return neural engine
        `when`(mockPredictionCoordinator.getNeuralEngine()).thenReturn(mockNeuralEngine)

        handler = PreferenceUIUpdateHandler(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )
    }

    // ========== Keyboard Layout Update Tests ==========

    @Test
    fun testHandlePreferenceChange_updatesKeyboardLayout() {
        // Act
        handler.handlePreferenceChange(null)

        // Assert
        verify(mockLayoutBridge).getCurrentLayout()
        verify(mockKeyboardView).setKeyboard(mockLayout)
    }

    @Test
    fun testHandlePreferenceChange_nullLayoutBridge_doesNotCrash() {
        // Arrange
        val handler = PreferenceUIUpdateHandler(
            mockConfig,
            null,  // null layout bridge
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )

        // Act & Assert - should not crash
        handler.handlePreferenceChange(null)
        verify(mockKeyboardView, never()).setKeyboard(any())
    }

    @Test
    fun testHandlePreferenceChange_nullKeyboardView_doesNotCrash() {
        // Arrange
        val handler = PreferenceUIUpdateHandler(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            null,  // null keyboard view
            mockSuggestionBar
        )

        // Act & Assert - should not crash
        handler.handlePreferenceChange(null)
        verify(mockKeyboardView, never()).setKeyboard(any())
    }

    // ========== Suggestion Bar Opacity Update Tests ==========

    @Test
    fun testHandlePreferenceChange_updatesSuggestionBarOpacity() {
        // Arrange
        mockConfig.suggestion_bar_opacity = 0.5f

        // Act
        handler.handlePreferenceChange(null)

        // Assert
        verify(mockSuggestionBar).setOpacity(0.5f)
    }

    @Test
    fun testHandlePreferenceChange_differentOpacityValues() {
        // Test various opacity values
        val opacityValues = listOf(0.0f, 0.25f, 0.5f, 0.75f, 1.0f)

        for (opacity in opacityValues) {
            // Arrange
            mockConfig.suggestion_bar_opacity = opacity
            reset(mockSuggestionBar)

            // Act
            handler.handlePreferenceChange(null)

            // Assert
            verify(mockSuggestionBar).setOpacity(opacity)
        }
    }

    @Test
    fun testHandlePreferenceChange_nullSuggestionBar_doesNotCrash() {
        // Arrange
        val handler = PreferenceUIUpdateHandler(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            null  // null suggestion bar
        )

        // Act & Assert - should not crash
        handler.handlePreferenceChange(null)
    }

    // ========== Neural Engine Config Update Tests ==========

    @Test
    fun testHandlePreferenceChange_neuralCustomEncoderUri_updatesEngine() {
        // Act
        handler.handlePreferenceChange("neural_custom_encoder_uri")

        // Assert
        verify(mockNeuralEngine).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_neuralCustomDecoderUri_updatesEngine() {
        // Act
        handler.handlePreferenceChange("neural_custom_decoder_uri")

        // Assert
        verify(mockNeuralEngine).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_neuralModelVersion_updatesEngine() {
        // Act
        handler.handlePreferenceChange("neural_model_version")

        // Assert
        verify(mockNeuralEngine).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_neuralUserMaxSeqLength_updatesEngine() {
        // Act
        handler.handlePreferenceChange("neural_user_max_seq_length")

        // Assert
        verify(mockNeuralEngine).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_unrelatedKey_doesNotUpdateEngine() {
        // Act
        handler.handlePreferenceChange("some_other_setting")

        // Assert
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testHandlePreferenceChange_nullKey_doesNotUpdateEngine() {
        // Act
        handler.handlePreferenceChange(null)

        // Assert
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testHandlePreferenceChange_nullNeuralEngine_doesNotCrash() {
        // Arrange
        `when`(mockPredictionCoordinator.getNeuralEngine()).thenReturn(null)

        // Act & Assert - should not crash
        handler.handlePreferenceChange("neural_custom_encoder_uri")
    }

    @Test
    fun testHandlePreferenceChange_nullPredictionCoordinator_doesNotCrash() {
        // Arrange
        val handler = PreferenceUIUpdateHandler(
            mockConfig,
            mockLayoutBridge,
            null,  // null prediction coordinator
            mockKeyboardView,
            mockSuggestionBar
        )

        // Act & Assert - should not crash
        handler.handlePreferenceChange("neural_custom_encoder_uri")
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesHandler() {
        // Act
        val handler = PreferenceUIUpdateHandler.create(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )

        // Assert
        assertNotNull("Factory method should create handler", handler)
    }

    @Test
    fun testCreate_factoryMethodHandlerWorks() {
        // Arrange
        val handler = PreferenceUIUpdateHandler.create(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )

        // Act
        handler.handlePreferenceChange(null)

        // Assert
        verify(mockKeyboardView).setKeyboard(any())
        verify(mockSuggestionBar).setOpacity(anyFloat())
    }

    @Test
    fun testCreate_withAllNullDependencies_doesNotCrash() {
        // Act
        val handler = PreferenceUIUpdateHandler.create(
            mockConfig,
            null,  // null layout bridge
            null,  // null prediction coordinator
            null,  // null keyboard view
            null   // null suggestion bar
        )

        // Assert
        assertNotNull("Should create handler with null dependencies", handler)

        // Should not crash when handling changes
        handler.handlePreferenceChange("any_key")
    }

    // ========== Multiple Update Cycles ==========

    @Test
    fun testHandlePreferenceChange_multipleCallsWithSameKey() {
        // Act
        handler.handlePreferenceChange("neural_custom_encoder_uri")
        handler.handlePreferenceChange("neural_custom_encoder_uri")
        handler.handlePreferenceChange("neural_custom_encoder_uri")

        // Assert
        verify(mockNeuralEngine, times(3)).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_multipleCallsWithDifferentKeys() {
        // Act
        handler.handlePreferenceChange("neural_custom_encoder_uri")
        handler.handlePreferenceChange("neural_custom_decoder_uri")
        handler.handlePreferenceChange("unrelated_key")

        // Assert
        verify(mockNeuralEngine, times(2)).setConfig(mockConfig)
    }

    @Test
    fun testHandlePreferenceChange_alternatingModelAndNonModelKeys() {
        // Act
        handler.handlePreferenceChange("neural_model_version")
        handler.handlePreferenceChange("some_other_setting")
        handler.handlePreferenceChange("neural_user_max_seq_length")
        handler.handlePreferenceChange("another_setting")

        // Assert
        verify(mockNeuralEngine, times(2)).setConfig(mockConfig)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_allUpdatesTriggered() {
        // Arrange
        mockConfig.suggestion_bar_opacity = 0.7f

        // Act
        handler.handlePreferenceChange("neural_custom_encoder_uri")

        // Assert - all three update types should be triggered
        verify(mockLayoutBridge).getCurrentLayout()
        verify(mockKeyboardView).setKeyboard(mockLayout)
        verify(mockSuggestionBar).setOpacity(0.7f)
        verify(mockNeuralEngine).setConfig(mockConfig)
    }

    @Test
    fun testFullLifecycle_nonModelKey_noEngineUpdate() {
        // Act
        handler.handlePreferenceChange("unrelated_setting")

        // Assert - layout and opacity updated, but not engine
        verify(mockLayoutBridge).getCurrentLayout()
        verify(mockKeyboardView).setKeyboard(mockLayout)
        verify(mockSuggestionBar).setOpacity(anyFloat())
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testIntegration_multipleHandlersIndependent() {
        // Arrange - create two handlers
        val handler1 = PreferenceUIUpdateHandler.create(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )

        val handler2 = PreferenceUIUpdateHandler.create(
            mockConfig,
            mockLayoutBridge,
            mockPredictionCoordinator,
            mockKeyboardView,
            mockSuggestionBar
        )

        // Act
        handler1.handlePreferenceChange("neural_model_version")
        handler2.handlePreferenceChange("neural_custom_encoder_uri")

        // Assert - both should work independently
        verify(mockNeuralEngine, times(2)).setConfig(mockConfig)
    }

    // ========== Edge Cases ==========

    @Test
    fun testEdgeCase_emptyKey_doesNotUpdateEngine() {
        // Act
        handler.handlePreferenceChange("")

        // Assert
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testEdgeCase_caseSlightlyDifferent_doesNotMatch() {
        // Act - keys are case-sensitive
        handler.handlePreferenceChange("Neural_Custom_Encoder_Uri")

        // Assert - should not match (case-sensitive)
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testEdgeCase_partialKeyMatch_doesNotMatch() {
        // Act
        handler.handlePreferenceChange("neural_custom")

        // Assert
        verify(mockNeuralEngine, never()).setConfig(any())
    }

    @Test
    fun testEdgeCase_allModelKeysSequentially() {
        // Act - test all four model keys
        handler.handlePreferenceChange("neural_custom_encoder_uri")
        handler.handlePreferenceChange("neural_custom_decoder_uri")
        handler.handlePreferenceChange("neural_model_version")
        handler.handlePreferenceChange("neural_user_max_seq_length")

        // Assert
        verify(mockNeuralEngine, times(4)).setConfig(mockConfig)
    }

    @Test
    fun testEdgeCase_configUpdatedBetweenCalls() {
        // Arrange - first call
        mockConfig.suggestion_bar_opacity = 0.3f
        handler.handlePreferenceChange(null)
        verify(mockSuggestionBar).setOpacity(0.3f)

        // Act - config changes, second call
        mockConfig.suggestion_bar_opacity = 0.9f
        reset(mockSuggestionBar)
        handler.handlePreferenceChange(null)

        // Assert - should use new config value
        verify(mockSuggestionBar).setOpacity(0.9f)
    }

    @Test
    fun testEdgeCase_layoutChangedBetweenCalls() {
        // Arrange - first call
        val layout1 = mock(KeyboardData::class.java)
        `when`(mockLayoutBridge.getCurrentLayout()).thenReturn(layout1)
        handler.handlePreferenceChange(null)
        verify(mockKeyboardView).setKeyboard(layout1)

        // Act - layout changes, second call
        val layout2 = mock(KeyboardData::class.java)
        `when`(mockLayoutBridge.getCurrentLayout()).thenReturn(layout2)
        reset(mockKeyboardView)
        handler.handlePreferenceChange(null)

        // Assert - should use new layout
        verify(mockKeyboardView).setKeyboard(layout2)
    }
}
