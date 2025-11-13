package juloo.keyboard2

import android.view.ViewGroup
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for SuggestionBarPropagator.
 *
 * Tests cover:
 * - SuggestionBar propagation to managers
 * - View reference propagation to receiver
 * - Combined propagation
 * - Null manager handling
 * - Factory method
 * - Integration scenarios
 */
@RunWith(MockitoJUnitRunner::class)
class SuggestionBarPropagatorTest {

    @Mock
    private lateinit var mockInputCoordinator: InputCoordinator

    @Mock
    private lateinit var mockSuggestionHandler: SuggestionHandler

    @Mock
    private lateinit var mockNeuralLayoutHelper: NeuralLayoutHelper

    @Mock
    private lateinit var mockReceiver: KeyboardReceiver

    @Mock
    private lateinit var mockSuggestionBar: SuggestionBar

    @Mock
    private lateinit var mockEmojiPane: ViewGroup

    @Mock
    private lateinit var mockContentPaneContainer: ViewGroup

    private lateinit var propagator: SuggestionBarPropagator

    @Before
    fun setUp() {
        propagator = SuggestionBarPropagator(
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver
        )
    }

    // ========== SuggestionBar Propagation Tests ==========

    @Test
    fun testPropagateSuggestionBar_propagatesToInputCoordinator() {
        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_propagatesToSuggestionHandler() {
        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_propagatesToNeuralLayoutHelper() {
        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_propagatesToAllManagers() {
        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert - verify all managers receive the suggestion bar
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_doesNotPropagateToReceiver() {
        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert - receiver should not be called for suggestion bar
        verifyNoInteractions(mockReceiver)
    }

    // ========== View Reference Propagation Tests ==========

    @Test
    fun testPropagateViewReferences_propagatesToReceiver() {
        // Act
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)

        // Assert
        verify(mockReceiver).setViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    @Test
    fun testPropagateViewReferences_withNullViews() {
        // Act
        propagator.propagateViewReferences(null, null)

        // Assert
        verify(mockReceiver).setViewReferences(null, null)
    }

    @Test
    fun testPropagateViewReferences_withOnlyEmojiPane() {
        // Act
        propagator.propagateViewReferences(mockEmojiPane, null)

        // Assert
        verify(mockReceiver).setViewReferences(mockEmojiPane, null)
    }

    @Test
    fun testPropagateViewReferences_withOnlyContentPane() {
        // Act
        propagator.propagateViewReferences(null, mockContentPaneContainer)

        // Assert
        verify(mockReceiver).setViewReferences(null, mockContentPaneContainer)
    }

    @Test
    fun testPropagateViewReferences_doesNotPropagateToManagers() {
        // Act
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)

        // Assert - managers should not be called for view references
        verifyNoInteractions(mockInputCoordinator)
        verifyNoInteractions(mockSuggestionHandler)
        verifyNoInteractions(mockNeuralLayoutHelper)
    }

    // ========== Combined Propagation Tests ==========

    @Test
    fun testPropagateAll_propagatesSuggestionBarToManagers() {
        // Act
        propagator.propagateAll(mockSuggestionBar, mockEmojiPane, mockContentPaneContainer)

        // Assert - verify suggestion bar propagated to all managers
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateAll_propagatesViewReferencesToReceiver() {
        // Act
        propagator.propagateAll(mockSuggestionBar, mockEmojiPane, mockContentPaneContainer)

        // Assert - verify view references propagated to receiver
        verify(mockReceiver).setViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    @Test
    fun testPropagateAll_propagatesEverything() {
        // Act
        propagator.propagateAll(mockSuggestionBar, mockEmojiPane, mockContentPaneContainer)

        // Assert - verify all propagations happened
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
        verify(mockReceiver).setViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    @Test
    fun testPropagateAll_withNullViews() {
        // Act
        propagator.propagateAll(mockSuggestionBar, null, null)

        // Assert
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
        verify(mockReceiver).setViewReferences(null, null)
    }

    // ========== Null Manager Tests ==========

    @Test
    fun testPropagateSuggestionBar_withNullInputCoordinator_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(
            null, // null InputCoordinator
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver
        )

        // Act & Assert - should not throw
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Only non-null managers should be called
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_withNullSuggestionHandler_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(
            mockInputCoordinator,
            null, // null SuggestionHandler
            mockNeuralLayoutHelper,
            mockReceiver
        )

        // Act & Assert - should not throw
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Only non-null managers should be called
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_withNullNeuralLayoutHelper_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(
            mockInputCoordinator,
            mockSuggestionHandler,
            null, // null NeuralLayoutHelper
            mockReceiver
        )

        // Act & Assert - should not throw
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Only non-null managers should be called
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateSuggestionBar_withAllManagersNull_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(
            null, // null InputCoordinator
            null, // null SuggestionHandler
            null, // null NeuralLayoutHelper
            mockReceiver
        )

        // Act & Assert - should not throw
        propagator.propagateSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testPropagateViewReferences_withNullReceiver_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            null // null receiver
        )

        // Act & Assert - should not throw
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    @Test
    fun testPropagateAll_withAllNull_doesNotCrash() {
        // Arrange
        val propagator = SuggestionBarPropagator(null, null, null, null)

        // Act & Assert - should not throw
        propagator.propagateAll(mockSuggestionBar, mockEmojiPane, mockContentPaneContainer)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesPropagator() {
        // Act
        val propagator = SuggestionBarPropagator.create(
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver
        )

        // Assert
        assertNotNull("Factory method should create propagator", propagator)
    }

    @Test
    fun testCreate_factoryMethodPropagatorWorks() {
        // Arrange
        val propagator = SuggestionBarPropagator.create(
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver
        )

        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)

        // Assert
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
    }

    @Test
    fun testCreate_withNullManagers() {
        // Act
        val propagator = SuggestionBarPropagator.create(null, null, null, null)

        // Assert
        assertNotNull("Factory should create propagator with null managers", propagator)

        // Should not crash
        propagator.propagateSuggestionBar(mockSuggestionBar)
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    // ========== Multiple Propagation Tests ==========

    @Test
    fun testPropagateSuggestionBar_calledMultipleTimes_propagatesEachTime() {
        // Arrange
        val mockSuggestionBar2 = mock(SuggestionBar::class.java)

        // Act
        propagator.propagateSuggestionBar(mockSuggestionBar)
        propagator.propagateSuggestionBar(mockSuggestionBar2)

        // Assert
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar2)
    }

    @Test
    fun testPropagateViewReferences_calledMultipleTimes_propagatesEachTime() {
        // Arrange
        val mockEmojiPane2 = mock(ViewGroup::class.java)
        val mockContentPane2 = mock(ViewGroup::class.java)

        // Act
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)
        propagator.propagateViewReferences(mockEmojiPane2, mockContentPane2)

        // Assert
        verify(mockReceiver).setViewReferences(mockEmojiPane, mockContentPaneContainer)
        verify(mockReceiver).setViewReferences(mockEmojiPane2, mockContentPane2)
    }

    // ========== Integration Tests ==========

    @Test
    fun testFullLifecycle_createPropagateSuggestionBarThenViews() {
        // Act - simulate full lifecycle
        propagator.propagateSuggestionBar(mockSuggestionBar)
        propagator.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)

        // Assert - verify correct propagation
        verify(mockInputCoordinator).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper).setSuggestionBar(mockSuggestionBar)
        verify(mockReceiver).setViewReferences(mockEmojiPane, mockContentPaneContainer)
    }

    @Test
    fun testIntegration_propagateAllEquivalentToSeparateCalls() {
        // Arrange
        val propagator2 = SuggestionBarPropagator.create(
            mockInputCoordinator,
            mockSuggestionHandler,
            mockNeuralLayoutHelper,
            mockReceiver
        )

        // Act - test both approaches
        propagator.propagateAll(mockSuggestionBar, mockEmojiPane, mockContentPaneContainer)

        propagator2.propagateSuggestionBar(mockSuggestionBar)
        propagator2.propagateViewReferences(mockEmojiPane, mockContentPaneContainer)

        // Assert - both approaches should have same effect
        verify(mockInputCoordinator, times(2)).setSuggestionBar(mockSuggestionBar)
        verify(mockSuggestionHandler, times(2)).setSuggestionBar(mockSuggestionBar)
        verify(mockNeuralLayoutHelper, times(2)).setSuggestionBar(mockSuggestionBar)
        verify(mockReceiver, times(2)).setViewReferences(mockEmojiPane, mockContentPaneContainer)
    }
}
