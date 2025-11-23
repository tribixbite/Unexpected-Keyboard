# Performance Analysis: Startup Lag & Prediction Latency

**Date**: 2025-11-22
**Status**: Analysis Complete

## Top 5 Relevant Files Review

1.  `srcs/juloo.keyboard2/Keyboard2.java`: Handles lifecycle. Initializes managers via `ManagerInitializer`. Calls `PredictionViewSetup`.
2.  `srcs/juloo.keyboard2/PredictionCoordinator.java`: Manages `OnnxSwipePredictor` singleton and `WordPredictor`. Handles locking.
3.  `srcs/juloo.keyboard2/OnnxSwipePredictor.java`: Loads ONNX models, runs inference. Uses `synchronized` initialization.
4.  `srcs/juloo.keyboard2/SwipeTrajectoryProcessor.java`: Preprocesses swipe points. Uses object pooling.
5.  `srcs/juloo.keyboard2/NeuralSwipeTypingEngine.java`: Wrapper for `OnnxSwipePredictor`.

## Findings & Alternative Hypotheses

### H1: Lock Contention on Startup (High Probability)
**File:** `PredictionCoordinator.java` vs `OnnxSwipePredictor.java`
**Analysis:**
- `PredictionCoordinator.initializeNeuralEngine` is `synchronized`.
- `OnnxSwipePredictor.initialize` is `synchronized`.
- `PredictionViewSetup` calls `predictionCoordinator.ensureInitialized()` on a background thread.
- `ensureInitialized` calls `initializeNeuralEngine`.
- This spawns a thread that holds the `PredictionCoordinator` lock while calling `_neuralEngine.initialize()`.
- `_neuralEngine.initialize()` calls `_neuralPredictor.initialize()`, which grabs the `OnnxSwipePredictor` lock.
- **The Problem:** If the Main Thread (UI) calls *any* synchronized method on `PredictionCoordinator` (e.g. `setConfig` via `refresh_config` on start) or `OnnxSwipePredictor` (e.g. `setConfig` again) *while the background thread is loading*, the **Main Thread will block** until the loading finishes.
- **Evidence:** `OnnxSwipePredictor.initialize` takes ~300ms. If the main thread blocks on `setConfig`, that's a 300ms frame drop. If `DictionaryManager` (initialized in the same `ensureInitialized` path via `initializeWordPredictor`) takes longer (loading 50k words), the lock is held longer.

### H2: SharedPreferences I/O Blocking Main Thread (High Probability)
**File:** `Keyboard2.java` -> `Config.java`
**Analysis:**
- `Keyboard2.onStartInputView` calls `refresh_config()`.
- `Config` constructor reads **all** preferences from SharedPreferences.
- While `SharedPreferences` are cached in memory, if the `custom_words` string is large (user dictionary), reading/parsing it might be slow.
- **Critical:** `OptimizedVocabulary.updateConfig` (called from `OnnxSwipePredictor.setConfig` on main thread) reads the `custom_words` JSON string from SharedPreferences *again* to check for changes.
- I optimized `OptimizedVocabulary` to check string equality, but `prefs.getString("custom_words", ...)` still has to fetch the string. If the string is huge, this allocation/copy on the main thread during every app switch is costly.

### H3: `SwipeTrajectoryProcessor` Resampling Overhead (Medium Probability)
**File:** `SwipeTrajectoryProcessor.java`
**Analysis:**
- `extractFeatures` normalizes coordinates.
- If `normalizedCoords.size() > maxSequenceLength` (250), it triggers resampling.
- `SwipeResampler.resample` (not shown, but inferred) likely iterates.
- `extractFeatures` copies data to `coordArray` (float[][]), resamples, then converts back to `PointF` list.
- **Inefficiency:** The 2D array conversion (`new float[size][2]`) allocates a new array every time resampling triggers. Even though `PointF`s are pooled, the array is not.
- **Latency Impact:** For long swipes (many points), this allocation + copy happens on the *critical path* of `predict()`.

### H4: `WordPredictor` Dictionary Loading Contention (Medium Probability)
**File:** `PredictionCoordinator.java`
**Analysis:**
- `initializeWordPredictor` loads the main dictionary (`loadDictionary`).
- This calls `BinaryDictionaryLoader` (likely).
- This is called inside `ensureInitialized`.
- If `ensureInitialized` runs on a background thread, fine.
- **BUT**, `PredictionCoordinator.getWordPredictor()` returns the object.
- If `Keyboard2View` calls `getWordPredictor()` (e.g. for `setSwipeTypingComponents`) before it's ready?
- `PredictionCoordinator` doesn't block getters. It returns null or uninitialized object?
- `WordPredictor` might block on its own internal locks when `getPredictions` is called if it's still loading.

## Proposed Solutions

1.  **Fix Lock Contention:** Remove `synchronized` from `PredictionCoordinator.initializeNeuralEngine` (use double-checked locking or `AtomicBoolean`) to prevent blocking the main thread's `setConfig` calls.
2.  **Optimize Config Refresh:** In `Keyboard2`, only call `refresh_config` if necessary, or make `Config` update lighter. Ensure `custom_words` isn't read on main thread unless dirty.
3.  **Optimize Resampling:** Use a pre-allocated `float[]` buffer in `SwipeTrajectoryProcessor` instead of allocating `float[][]` every time.
4.  **Async Dictionary:** Ensure `WordPredictor` loading doesn't hold the same lock as `NeuralEngine` loading.

