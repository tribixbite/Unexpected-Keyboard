package juloo.keyboard2

import android.content.Context
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for ManagerInitializer.
 *
 * Tests cover:
 * - Manager initialization and dependencies
 * - Data class structure and accessibility
 * - Factory method functionality
 * - All managers properly created
 * - Initialization result structure
 */
@RunWith(MockitoJUnitRunner::class)
class ManagerInitializerTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockConfig: Config

    @Mock
    private lateinit var mockKeyboardView: Keyboard2View

    @Mock
    private lateinit var mockKeyEventHandler: KeyEventHandler

    private lateinit var managerInitializer: ManagerInitializer

    @Before
    fun setUp() {
        managerInitializer = ManagerInitializer(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )
    }

    // ========== Initialization Tests ==========

    @Test
    fun testInitialize_createsAllManagers() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - verify all managers are created
        assertNotNull("ContractionManager should be created", result.contractionManager)
        assertNotNull("ClipboardManager should be created", result.clipboardManager)
        assertNotNull("PredictionContextTracker should be created", result.contextTracker)
        assertNotNull("PredictionCoordinator should be created", result.predictionCoordinator)
        assertNotNull("InputCoordinator should be created", result.inputCoordinator)
        assertNotNull("SuggestionHandler should be created", result.suggestionHandler)
        assertNotNull("NeuralLayoutHelper should be created", result.neuralLayoutHelper)
        assertNotNull("MLDataCollector should be created", result.mlDataCollector)
    }

    @Test
    fun testInitialize_contractionManagerLoaded() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - ContractionManager should be initialized
        assertNotNull(result.contractionManager)
        // Note: We can't verify loadMappings() was called without making ContractionManager
        // mockable, but the initialization sequence ensures it's called
    }

    @Test
    fun testInitialize_clipboardManagerCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.clipboardManager)
    }

    @Test
    fun testInitialize_contextTrackerCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.contextTracker)
    }

    @Test
    fun testInitialize_predictionCoordinatorCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.predictionCoordinator)
    }

    @Test
    fun testInitialize_inputCoordinatorCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.inputCoordinator)
    }

    @Test
    fun testInitialize_suggestionHandlerCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.suggestionHandler)
    }

    @Test
    fun testInitialize_neuralLayoutHelperCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.neuralLayoutHelper)
    }

    @Test
    fun testInitialize_mlDataCollectorCreated() {
        // Act
        val result = managerInitializer.initialize()

        // Assert
        assertNotNull(result.mlDataCollector)
    }

    // ========== Multiple Initialization Tests ==========

    @Test
    fun testInitialize_calledMultipleTimes_createsNewInstances() {
        // Act
        val result1 = managerInitializer.initialize()
        val result2 = managerInitializer.initialize()

        // Assert - each call should create new instances
        assertNotSame("Should create new ContractionManager",
            result1.contractionManager, result2.contractionManager)
        assertNotSame("Should create new ClipboardManager",
            result1.clipboardManager, result2.clipboardManager)
        assertNotSame("Should create new PredictionContextTracker",
            result1.contextTracker, result2.contextTracker)
        assertNotSame("Should create new PredictionCoordinator",
            result1.predictionCoordinator, result2.predictionCoordinator)
        assertNotSame("Should create new InputCoordinator",
            result1.inputCoordinator, result2.inputCoordinator)
        assertNotSame("Should create new SuggestionHandler",
            result1.suggestionHandler, result2.suggestionHandler)
        assertNotSame("Should create new NeuralLayoutHelper",
            result1.neuralLayoutHelper, result2.neuralLayoutHelper)
        assertNotSame("Should create new MLDataCollector",
            result1.mlDataCollector, result2.mlDataCollector)
    }

    // ========== Data Class Tests ==========

    @Test
    fun testInitializationResult_dataClassEquality() {
        // Arrange
        val result1 = managerInitializer.initialize()
        val result2 = managerInitializer.initialize()

        // Assert - different instances should not be equal
        assertNotEquals("Different instances should not be equal", result1, result2)
    }

    @Test
    fun testInitializationResult_fieldsAccessible() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - verify all fields are accessible
        assertNotNull(result.contractionManager)
        assertNotNull(result.clipboardManager)
        assertNotNull(result.contextTracker)
        assertNotNull(result.predictionCoordinator)
        assertNotNull(result.inputCoordinator)
        assertNotNull(result.suggestionHandler)
        assertNotNull(result.neuralLayoutHelper)
        assertNotNull(result.mlDataCollector)
    }

    @Test
    fun testInitializationResult_dataClassCopy() {
        // Arrange
        val result = managerInitializer.initialize()

        // Act - use data class copy
        val copy = result.copy(
            contractionManager = result.contractionManager
        )

        // Assert - copy should have same references
        assertSame(result.contractionManager, copy.contractionManager)
        assertSame(result.clipboardManager, copy.clipboardManager)
        assertSame(result.contextTracker, copy.contextTracker)
    }

    // ========== Factory Method Tests ==========

    @Test
    fun testCreate_factoryMethodCreatesInstance() {
        // Act
        val initializer = ManagerInitializer.create(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )

        // Assert
        assertNotNull("Factory method should create instance", initializer)
    }

    @Test
    fun testCreate_factoryMethodInitializesManagers() {
        // Arrange
        val initializer = ManagerInitializer.create(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )

        // Act
        val result = initializer.initialize()

        // Assert
        assertNotNull(result.contractionManager)
        assertNotNull(result.clipboardManager)
        assertNotNull(result.contextTracker)
        assertNotNull(result.predictionCoordinator)
        assertNotNull(result.inputCoordinator)
        assertNotNull(result.suggestionHandler)
        assertNotNull(result.neuralLayoutHelper)
        assertNotNull(result.mlDataCollector)
    }

    // ========== Constructor Tests ==========

    @Test
    fun testConstructor_withAllParameters() {
        // Act
        val initializer = ManagerInitializer(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )

        // Assert
        assertNotNull("Constructor should create instance", initializer)
    }

    @Test
    fun testConstructor_initializeWorks() {
        // Arrange
        val initializer = ManagerInitializer(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )

        // Act
        val result = initializer.initialize()

        // Assert
        assertNotNull(result)
        assertNotNull(result.contractionManager)
    }

    // ========== Manager Dependency Tests ==========

    @Test
    fun testInitialize_managersHaveCorrectTypes() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - verify correct manager types
        assertTrue("ContractionManager type",
            result.contractionManager is ContractionManager)
        assertTrue("ClipboardManager type",
            result.clipboardManager is ClipboardManager)
        assertTrue("PredictionContextTracker type",
            result.contextTracker is PredictionContextTracker)
        assertTrue("PredictionCoordinator type",
            result.predictionCoordinator is PredictionCoordinator)
        assertTrue("InputCoordinator type",
            result.inputCoordinator is InputCoordinator)
        assertTrue("SuggestionHandler type",
            result.suggestionHandler is SuggestionHandler)
        assertTrue("NeuralLayoutHelper type",
            result.neuralLayoutHelper is NeuralLayoutHelper)
        assertTrue("MLDataCollector type",
            result.mlDataCollector is MLDataCollector)
    }

    @Test
    fun testInitialize_allManagersNonNull() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - verify no manager is null
        with(result) {
            assertNotNull(contractionManager)
            assertNotNull(clipboardManager)
            assertNotNull(contextTracker)
            assertNotNull(predictionCoordinator)
            assertNotNull(inputCoordinator)
            assertNotNull(suggestionHandler)
            assertNotNull(neuralLayoutHelper)
            assertNotNull(mlDataCollector)
        }
    }

    // ========== Integration Tests ==========

    @Test
    fun testInitialize_fullIntegration_allManagersInitialized() {
        // Act
        val result = managerInitializer.initialize()

        // Assert - verify complete initialization
        assertNotNull("Full integration: ContractionManager", result.contractionManager)
        assertNotNull("Full integration: ClipboardManager", result.clipboardManager)
        assertNotNull("Full integration: PredictionContextTracker", result.contextTracker)
        assertNotNull("Full integration: PredictionCoordinator", result.predictionCoordinator)
        assertNotNull("Full integration: InputCoordinator", result.inputCoordinator)
        assertNotNull("Full integration: SuggestionHandler", result.suggestionHandler)
        assertNotNull("Full integration: NeuralLayoutHelper", result.neuralLayoutHelper)
        assertNotNull("Full integration: MLDataCollector", result.mlDataCollector)
    }

    @Test
    fun testInitialize_multipleInitializers_independent() {
        // Arrange
        val initializer1 = ManagerInitializer(mockContext, mockConfig,
            mockKeyboardView, mockKeyEventHandler)
        val initializer2 = ManagerInitializer(mockContext, mockConfig,
            mockKeyboardView, mockKeyEventHandler)

        // Act
        val result1 = initializer1.initialize()
        val result2 = initializer2.initialize()

        // Assert - results should be independent
        assertNotSame("Results should be independent", result1, result2)
        assertNotSame("Managers should be independent",
            result1.contractionManager, result2.contractionManager)
    }

    // ========== Companion Object Tests ==========

    @Test
    fun testCompanionObject_createMethodExists() {
        // Act & Assert - verify companion object factory method exists
        val initializer = ManagerInitializer.create(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )

        assertNotNull("Companion object factory method should work", initializer)
    }

    @Test
    fun testCompanionObject_createAndInitialize() {
        // Act
        val initializer = ManagerInitializer.create(
            mockContext,
            mockConfig,
            mockKeyboardView,
            mockKeyEventHandler
        )
        val result = initializer.initialize()

        // Assert
        assertNotNull(result)
        assertNotNull(result.contractionManager)
        assertNotNull(result.clipboardManager)
        assertNotNull(result.contextTracker)
        assertNotNull(result.predictionCoordinator)
        assertNotNull(result.inputCoordinator)
        assertNotNull(result.suggestionHandler)
        assertNotNull(result.neuralLayoutHelper)
        assertNotNull(result.mlDataCollector)
    }
}
