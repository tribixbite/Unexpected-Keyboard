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
 * Comprehensive test suite for ConfigPropagator.
 *
 * Tests cover:
 * - Config propagation to all managers
 * - Null manager handling
 * - Manager update order
 * - Keyboard view reset
 * - Builder pattern functionality
 * - Edge cases and error handling
 */
@RunWith(MockitoJUnitRunner::class)
class ConfigPropagatorTest {

    @Mock
    private lateinit var mockClipboardManager: ClipboardManager

    @Mock
    private lateinit var mockPredictionCoordinator: PredictionCoordinator

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockLayoutManager: LayoutManager

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockSubtypeManager: SubtypeManager

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockResources: Resources

    private lateinit var configPropagator: ConfigPropagator

    @Before
    fun setUp() {
        configPropagator = ConfigPropagator(
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )
    }

    // ========== Config Propagation Tests ==========

    @Test
    fun testPropagateConfig_propagatesToAllManagers() {
        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert - verify all managers receive config
        verify(mockClipboardManager).setConfig(mockConfig)
        verify(mockPredictionCoordinator).setConfig(mockConfig)
        verify(mockInputCoordinator).setConfig(mockConfig)
        verify(mockSuggestionHandler).setConfig(mockConfig)
        verify(mockNeuralLayoutHelper).setConfig(mockConfig)
        verify(mockLayoutManager).setConfig(mockConfig)
    }

    @Test
    fun testPropagateConfig_refreshesSubtype() {
        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert
        verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)
    }

    @Test
    fun testPropagateConfig_resetsKeyboardView() {
        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert
        verify(mockKeyboardView).reset()
    }

    @Test
    fun testPropagateConfig_callsSubtypeRefreshBeforeManagerUpdates() {
        // Arrange - use InOrder to verify call order
        val inOrder = inOrder(mockSubtypeManager, mockLayoutManager)

        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert - subtype refresh should happen before layout manager update
        inOrder.verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)
        inOrder.verify(mockLayoutManager).setConfig(mockConfig)
    }

    @Test
    fun testPropagateConfig_resetsViewAfterManagerUpdates() {
        // Arrange - use InOrder to verify call order
        val inOrder = inOrder(mockLayoutManager, mockKeyboardView)

        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert - keyboard view reset should happen after manager updates
        inOrder.verify(mockLayoutManager).setConfig(mockConfig)
        inOrder.verify(mockKeyboardView).reset()
    }

    // ========== Null Manager Handling Tests ==========

    @Test
    fun testPropagateConfig_withNullClipboardManager_doesNotCrash() {
        // Arrange
        val propagator = ConfigPropagator(
            null, // null clipboard manager
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockLayoutManager,
            mockKeyboardView,
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        propagator.propagateConfig(mockConfig, mockResources)

        // Verify other managers still receive config
        verify(mockPredictionCoordinator).setConfig(mockConfig)
    }

    @Test
    fun testPropagateConfig_withAllNullManagers_doesNotCrash() {
        // Arrange
        val propagator = ConfigPropagator(
            null, null, null, null, null, null, null, null
        )

        // Act & Assert - should not throw
        propagator.propagateConfig(mockConfig, mockResources)
    }

    @Test
    fun testPropagateConfig_withNullSubtypeManager_doesNotCrash() {
        // Arrange
        val propagator = ConfigPropagator(
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockLayoutManager,
            mockKeyboardView,
            null // null subtype manager
        )

        // Act & Assert - should not throw
        propagator.propagateConfig(mockConfig, mockResources)

        // Verify other managers still receive config
        verify(mockClipboardManager).setConfig(mockConfig)
    }

    @Test
    fun testPropagateConfig_withNullKeyboardView_doesNotCrash() {
        // Arrange
        val propagator = ConfigPropagator(
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockLayoutManager,
            null, // null keyboard view
            mockSubtypeManager
        )

        // Act & Assert - should not throw
        propagator.propagateConfig(mockConfig, mockResources)

        // Verify managers still receive config
        verify(mockClipboardManager).setConfig(mockConfig)
    }

    // ========== Reset Keyboard View Tests ==========

    @Test
    fun testResetKeyboardView_resetsView() {
        // Act
        configPropagator.resetKeyboardView()

        // Assert
        verify(mockKeyboardView).reset()
    }

    @Test
    fun testResetKeyboardView_withNullView_doesNotCrash() {
        // Arrange
        val propagator = ConfigPropagator(
            null, null, null, null, null, null, null, null
        )

        // Act & Assert - should not throw
        propagator.resetKeyboardView()
    }

    // ========== Builder Tests ==========

    @Test
    fun testBuilder_buildsWithAllManagers() {
        // Act
        val propagator = ConfigPropagator.builder()
            .setClipboardManager(mockClipboardManager)
            .setPredictionCoordinator(mockPredictionCoordinator)
            .setInputCoordinator(mockInputCoordinator)
            .setSuggestionHandler(mockSuggestionHandler)
            .setNeuralLayoutHelper(mockNeuralLayoutHelper)
            .setLayoutManager(mockLayoutManager)
            .setKeyboardView(mockKeyboardView)
            .setSubtypeManager(mockSubtypeManager)
            .build()

        // Assert - verify propagator works correctly
        propagator.propagateConfig(mockConfig, mockResources)

        verify(mockClipboardManager).setConfig(mockConfig)
        verify(mockPredictionCoordinator).setConfig(mockConfig)
        verify(mockInputCoordinator).setConfig(mockConfig)
        verify(mockSuggestionHandler).setConfig(mockConfig)
        verify(mockNeuralLayoutHelper).setConfig(mockConfig)
        verify(mockLayoutManager).setConfig(mockConfig)
        verify(mockKeyboardView).reset()
        verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)
    }

    @Test
    fun testBuilder_buildsWithSomeNullManagers() {
        // Act
        val propagator = ConfigPropagator.builder()
            .setClipboardManager(mockClipboardManager)
            .setPredictionCoordinator(null)
            .setInputCoordinator(mockInputCoordinator)
            .setSuggestionHandler(null)
            .setNeuralLayoutHelper(mockNeuralLayoutHelper)
            .setLayoutManager(null)
            .setKeyboardView(mockKeyboardView)
            .setSubtypeManager(mockSubtypeManager)
            .build()

        // Assert - verify propagator works correctly with nulls
        propagator.propagateConfig(mockConfig, mockResources)

        verify(mockClipboardManager).setConfig(mockConfig)
        verify(mockInputCoordinator).setConfig(mockConfig)
        verify(mockNeuralLayoutHelper).setConfig(mockConfig)
        verify(mockKeyboardView).reset()
        verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)

        // Null managers should not be called
        verifyNoInteractions(mockPredictionCoordinator)
        verifyNoInteractions(mockSuggestionHandler)
        verifyNoInteractions(mockLayoutManager)
    }

    @Test
    fun testBuilder_buildsWithNoManagers() {
        // Act
        val propagator = ConfigPropagator.builder().build()

        // Assert - should not crash
        propagator.propagateConfig(mockConfig, mockResources)
        propagator.resetKeyboardView()
    }

    @Test
    fun testBuilder_fluentAPI_returnsBuilderForChaining() {
        // Act & Assert - verify fluent API works
        val builder = ConfigPropagator.builder()
        assertSame("setClipboardManager should return builder", builder,
            builder.setClipboardManager(mockClipboardManager))
        assertSame("setPredictionCoordinator should return builder", builder,
            builder.setPredictionCoordinator(mockPredictionCoordinator))
        assertSame("setInputCoordinator should return builder", builder,
            builder.setInputCoordinator(mockInputCoordinator))
        assertSame("setSuggestionHandler should return builder", builder,
            builder.setSuggestionHandler(mockSuggestionHandler))
        assertSame("setNeuralLayoutHelper should return builder", builder,
            builder.setNeuralLayoutHelper(mockNeuralLayoutHelper))
        assertSame("setLayoutManager should return builder", builder,
            builder.setLayoutManager(mockLayoutManager))
        assertSame("setKeyboardView should return builder", builder,
            builder.setKeyboardView(mockKeyboardView))
        assertSame("setSubtypeManager should return builder", builder,
            builder.setSubtypeManager(mockSubtypeManager))
    }

    // ========== Multiple Propagation Tests ==========

    @Test
    fun testPropagateConfig_calledMultipleTimes_propagatesEachTime() {
        // Arrange
        val config1 = mock(Config::class.java)
        val config2 = mock(Config::class.java)

        // Act
        configPropagator.propagateConfig(config1)
        configPropagator.propagateConfig(config2)

        // Assert
        verify(mockClipboardManager).setConfig(config1)
        verify(mockClipboardManager).setConfig(config2)
        verify(mockKeyboardView, times(2)).reset()
    }

    // ========== Integration Tests ==========

    @Test
    fun testPropagateConfig_fullIntegration_allManagersUpdatedInOrder() {
        // Arrange - use InOrder to verify complete call sequence
        val inOrder = inOrder(
            mockSubtypeManager,
            mockClipboardManager,
            mockPredictionCoordinator,
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockLayoutManager,
            mockKeyboardView
        )

        // Act
        configPropagator.propagateConfig(mockConfig, mockResources)

        // Assert - verify complete call order
        inOrder.verify(mockSubtypeManager).refreshSubtype(mockConfig, mockResources)
        inOrder.verify(mockClipboardManager).setConfig(mockConfig)
        inOrder.verify(mockPredictionCoordinator).setConfig(mockConfig)
        inOrder.verify(mockInputCoordinator).setConfig(mockConfig)
        inOrder.verify(mockSuggestionHandler).setConfig(mockConfig)
        inOrder.verify(mockNeuralLayoutHelper).setConfig(mockConfig)
        inOrder.verify(mockLayoutManager).setConfig(mockConfig)
        inOrder.verify(mockKeyboardView).reset()
    }
}
