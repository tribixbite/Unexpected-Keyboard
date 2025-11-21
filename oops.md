# Performance Analysis and Optimization Recommendations for Neural Swipe Prediction

This document summarizes the investigation into performance bottlenecks in the neural swipe prediction system, based on the user's observations and an in-depth code review of `OnnxSwipePredictor.java`, `SwipeTrajectoryProcessor.java`, and `OptimizedVocabulary.java`.

## Summary of Findings

The core neural prediction pipeline exhibits a well-architected and highly optimized design in many aspects. Critical optimizations already in place include:

*   **Singleton Pattern**: `OnnxSwipePredictor` is a singleton, ensuring that ONNX Runtime sessions for the encoder and decoder models are loaded only once and persist in memory, significantly reducing model loading overhead between predictions and keyboard loads.
*   **Asynchronous Initialization**: Model loading is correctly offloaded to a dedicated background thread (`_onnxExecutor`) via `initializeAsync()`, preventing UI blocking and ensuring a responsive user experience during keyboard startup.
*   **ONNX Runtime Optimizations**: Session options are configured to leverage NNAPI (for hardware acceleration on NPUs/DSPs/GPUs), XNNPACK (for optimized ARM CPU inference), and graph optimizations (operator fusion, memory pattern optimization). Optimized model graphs are also cached to disk for faster subsequent loads.
*   **Efficient Beam Search**: The `runBeamSearch` method incorporates advanced techniques such as batched processing (when `_config.neural_batch_beams` is enabled and models are broadcast-enabled) and score-gap early stopping, both crucial for reducing inference time.
*   **Optimized Vocabulary Loading**: `OptimizedVocabulary` prioritizes loading a binary cache (claiming a 5ms load time vs. 500ms for JSON parsing and sorting), and uses efficient hash-map based data structures for word lookups.

Despite these strengths, several areas were identified as potential sources of the observed performance lag, particularly concerning repeated work, unnecessary allocations, and inefficient data access in hot paths.

## Bottlenecks and Recommendations

Below is a breakdown of potential bottlenecks, correlating with the user's points, and detailed recommendations for optimization.

### 1. Model Not Persisting & Not Loaded Natively (User Points #1 & #8)

**Status**: Generally good. The `OnnxSwipePredictor` is designed to be a singleton, load models asynchronously on a background thread, and keep them in memory. Hardware acceleration is configured.

**Potential Issue**: The user observes non-persistence. The `OnnxSwipePredictor.setConfig()` method explicitly re-initializes and reloads models if `_currentModelVersion`, `_currentEncoderPath`, or `_currentDecoderPath` change. While this logic is sound for configuration changes, frequent, unnecessary calls to `setConfig` with `Config` objects that appear "different" (even if their model-related parameters are effectively the same) could force reloads.

**Recommendations**:
*   **Investigate `Config` Object Stability**: Trace the lifecycle of the `Config` object passed to `OnnxSwipePredictor.setConfig()`. Ensure that a *new* `Config` object is only created, or its relevant fields are updated, when actual user preferences or system settings pertaining to the model have genuinely changed.
*   **Monitor `setConfig` Callers**: Review classes that call `setConfig` on `NeuralSwipeTypingEngine` (which then calls `OnnxSwipePredictor.setConfig`), specifically `ConfigPropagator.kt` and `PredictionCoordinator.java`, to confirm `setConfig` is not being triggered excessively.

### 2. Unnecessary Logging (User Point #2)

**Status**: Significant logging (`Log.d`, `Log.i`, `Log.w`, `Log.e`, and `sendDebugLog`) is present in `OnnxSwipePredictor.java` and `OptimizedVocabulary.java`, often in performance-critical sections. Some performance timing logs were removed during this investigation. `sendDebugLog` involves creating and broadcasting an `Intent`, which can be expensive.

**Recommendations**:
*   **Strict Logging Control**: Implement a robust compile-time flag (e.g., `BuildConfig.DEBUG`) or ensure the `_config.swipe_debug_detailed_logging` `SharedPreferences` setting is reliably `false` in production/release builds.
*   **Remove Verbose Debug Logs**: For hot paths like `predict()`, `runBeamSearch()`, `filterPredictions()`, and trajectory processing, eliminate or comment out all `Log.d` and `sendDebugLog` statements once debugging is complete. Even disabled `Log.d` calls with complex string formatting can incur minor overhead.
*   **Conditional Logging**: If certain logs are crucial for runtime diagnostics, ensure they are guarded by `if (Log.isLoggable(TAG, Log.VERBOSE))` or similar checks that evaluate cheaply.

### 3. Unoptimized Loading of Vocab/Dict/Frequencies (User Point #3)

**Status**: The initial loading of the main vocabulary is highly optimized by using a binary cache. `OptimizedVocabulary` uses efficient `HashMap`s for lookups. However, there are areas for improvement related to dynamic word lists.

**Potential Issue**:
*   `OptimizedVocabulary.loadCustomAndUserWords()` reads `custom_words` from `SharedPreferences` as a JSON string and parses it into a `JSONObject` *every time* `loadVocabulary()` is called. If `loadVocabulary()` is triggered (e.g., due to cache invalidation or other settings changes), this parsing adds overhead.
*   `OptimizedVocabulary.filterPredictions()` retrieves various configuration parameters (scoring weights, autocorrect settings) from `SharedPreferences` on *every single swipe*. While `get_shared_preferences()` might be cached, repeated `prefs.get*()` calls can still add overhead.

**Recommendations**:
*   **Custom Words JSON Parsing**: Parse the `custom_words` JSON string *only once*. Store the parsed custom words in a dedicated data structure within `OptimizedVocabulary`. Re-parse only if the raw JSON string retrieved from `SharedPreferences` explicitly changes.
*   **Cache `SharedPreferences` Values**: Cache the scoring weights and autocorrect configuration parameters (e.g., `confidenceWeight`, `frequencyWeight`, `swipeAutocorrectEnabled`) when `OnnxSwipePredictor.setConfig()` or `OptimizedVocabulary.setConfig()` is called. `filterPredictions()` should then access these pre-cached values directly, eliminating `SharedPreferences` reads on every swipe.

### 4. Inefficient Beam Corrections & Not Stopping Early for Bad Beams (User Points #4 & #5)

**Status**: `OnnxSwipePredictor.runBeamSearch()` includes a score-gap early stopping mechanism and uses `_config.neural_batch_beams` for potentially faster batched processing. Fuzzy matching with pruning is also implemented in `OptimizedVocabulary.filterPredictions()`.

**Potential Issue**:
*   **Sequential Beam Search Overhead**: The `runBeamSearch()` method's sequential processing path (`else` block when `_config.neural_batch_beams` is `false`) creates new `OnnxTensor` objects within its loop on each step for each beam. This leads to increased GC pressure and slower performance compared to the batched mode.
*   **Fuzzy Matching Cost**: Although fuzzy matching (`calculateLevenshteinDistance`) is valuable for correction, if applied too broadly (e.g., to too many candidates or against too large a dictionary subset), it can be computationally intensive.

**Recommendations**:
*   **Enable Batched Beam Search**: Ensure that `_config.neural_batch_beams` is enabled in production, as the new models (`assets/models/bs2/`) are broadcast-enabled and designed for this more efficient processing.
*   **Optimize Sequential Beam Search (if necessary)**: If batched mode cannot be universally used, refactor the sequential beam search loop to reuse `OnnxTensor` objects. This could involve pre-allocating the tensor and updating its content with `IntBuffer.put()` on each iteration.
*   **Tune Fuzzy Matching Pruning**: Review and potentially fine-tune the parameters for fuzzy matching (e.g., `maxLengthDiff`, `prefixLength`, `minWordLength`) in `OptimizedVocabulary.java` to minimize the number of comparisons while maintaining quality. Consider more aggressive pruning if quality trade-offs are acceptable.

### 5. Excessive Object Creation in Trajectory Processing (Part of User Point #3)

**Status**: `SwipeTrajectoryProcessor.extractFeatures()` performs multiple steps (normalization, resampling, key detection, feature calculation, padding) and creates numerous `ArrayList`s and `PointF`/`TrajectoryPoint` objects on every swipe.

**Potential Issue**: High object allocation rates in `extractFeatures()` contribute to Garbage Collection (GC) pressure, which can cause micro-stutters or pauses in a real-time system like a keyboard.

**Recommendations**:
*   **Object Pooling**: Implement object pooling for `TrajectoryPoint` and `PointF` objects used within the `SwipeTrajectoryProcessor`. Instead of creating new instances in loops, retrieve them from a pool and return them after use.
*   **Reuse `ArrayList`s**: Instead of creating new `ArrayList` instances repeatedly, clear and reuse existing ones where feasible. For example, `TrajectoryFeatures` could have its internal lists cleared and repopulated on each call.
*   **Streamline Loops**: Investigate combining multiple processing steps into fewer loops to reduce redundant iterations and temporary object creation. This is a more complex refactoring but can yield significant gains.

### 6. Poorly Drawing Prediction Output UI (User Point #7)

**Status**: `OnnxSwipePredictor` and `OptimizedVocabulary` are concerned with prediction logic, not UI rendering. The `InputCoordinator.java` is responsible for consuming prediction results and initiating UI updates.

**Recommendations**:
*   **Investigate UI Update Path**: Examine `InputCoordinator.java` and related UI components (e.g., `SuggestionBarView`, `KeyboardView`) that display prediction results.
*   **Profile UI Rendering**: Use Android Studio's profiler to identify specific UI rendering bottlenecks, such as:
    *   **Overdraw**: Too many layers being drawn on top of each other.
    *   **Layout Invalidation**: Frequent and unnecessary layout passes.
    *   **Complex View Hierarchies**: Deeply nested layouts.
    *   **Main Thread Blocking**: Any operations on the main thread that block UI rendering, especially when processing prediction results.

### General Todo Recommendations:

*   **Continuous Profiling**: Regularly profile the application (especially on target devices like the Samsung S25U) with Android Studio's CPU, Memory, and Jank profilers to catch regressions and new bottlenecks.
*   **Configurable Performance Settings**: Introduce more granular configuration options for performance (e.g., beam width, early stopping thresholds) that users can adjust (perhaps in an advanced settings menu) to balance speed and accuracy based on their device capabilities and preferences.
*   **A/B Testing**: For significant optimizations, consider A/B testing with a subset of users to measure real-world impact on perceived latency and prediction quality.
*   **Cleanup `TestOnnxDirect.java`**: After verifying the new models, the standalone test file `TestOnnxDirect.java` should be updated or removed as it does not reflect the current application's model loading and input preparation.

This comprehensive set of recommendations should provide a clear roadmap for addressing the observed performance issues and further optimizing the neural swipe prediction system.
