package juloo.keyboard2

/**
 * Initializes prediction components during onCreate().
 *
 * This class handles initialization of prediction engines when word prediction
 * or swipe typing is enabled:
 * - Initializes PredictionCoordinator
 * - Sets swipe typing components on keyboard view if available
 *
 * The initializer pattern simplifies onCreate() by consolidating prediction
 * initialization into a single, testable operation.
 *
 * This utility is extracted from Keyboard2.java as part of Phase 4 refactoring
 * to reduce the main class size (v1.32.405).
 *
 * @since v1.32.405
 */
class PredictionInitializer(
    private val config: Config,
    private val predictionCoordinator: PredictionCoordinator,
    private val keyboardView: Keyboard2View,
    private val keyboard2: Keyboard2
) {
    /**
     * Initialize prediction components if enabled.
     *
     * Checks configuration and:
     * 1. Initializes PredictionCoordinator if predictions/swipe enabled
     * 2. Sets swipe typing components on keyboard view if swipe is available
     */
    fun initializeIfEnabled() {
        if (config.word_prediction_enabled || config.swipe_typing_enabled) {
            predictionCoordinator.initialize()

            // Set swipe typing components on keyboard view if swipe is enabled
            if (config.swipe_typing_enabled && predictionCoordinator.isSwipeTypingAvailable()) {
                android.util.Log.d(
                    "Keyboard2",
                    "Neural engine initialized - dimensions and key positions will be set after layout"
                )
                keyboardView.setSwipeTypingComponents(
                    predictionCoordinator.getWordPredictor(),
                    keyboard2
                )
            }
        }
    }

    companion object {
        /**
         * Create a PredictionInitializer.
         *
         * @param config The configuration
         * @param predictionCoordinator The prediction coordinator
         * @param keyboardView The keyboard view
         * @param keyboard2 The Keyboard2 service
         * @return A new PredictionInitializer instance
         */
        @JvmStatic
        fun create(
            config: Config,
            predictionCoordinator: PredictionCoordinator,
            keyboardView: Keyboard2View,
            keyboard2: Keyboard2
        ): PredictionInitializer {
            return PredictionInitializer(
                config,
                predictionCoordinator,
                keyboardView,
                keyboard2
            )
        }
    }
}
