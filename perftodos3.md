# Performance Todos v3: Final Integration and Polish

This document reflects a detailed re-assessment of the performance optimizations. While many underlying components were built, they were not fully integrated into the application's logic. This list contains the final, critical steps required to make those optimizations active.

## I. Summary of Verification

A deep-dive review revealed that while performant components like `AsyncDictionaryLoader` and `UserDictionaryObserver` were created, **they are not currently being used by the `DictionaryManager`**. The application is still using slow, synchronous loading for dictionaries and is not listening for automatic updates.

The critical runtime latency issue from `perftodos2.md` (caused by logging) has been resolved. The focus of this document is to correctly enable the loading-time optimizations.

---

## II. Outstanding Tasks

### Todo 1 (Critical): Integrate Asynchronous Dictionary Loading

**Problem:** `DictionaryManager` still calls the blocking, synchronous `WordPredictor.loadDictionary()` when a language is changed. This will freeze the UI thread. The entire `AsyncDictionaryLoader` is currently unused ("dead code").

**Solution:** Modify `DictionaryManager` to use `WordPredictor.loadDictionaryAsync` and handle its asynchronous callbacks.

**Action Items:**

1.  **Modify `DictionaryManager.setLanguage`:**
    *   When a `WordPredictor` is created for the first time, call `_currentPredictor.loadDictionaryAsync()`.
    *   The UI needs to be able to handle a state where the predictor is loading. For example, `getPredictions` should return an empty list, and the suggestion bar should perhaps show a loading state if `_currentPredictor.isLoading()` is true.
2.  **Implement Callbacks:**
    *   The callback passed to `loadDictionaryAsync` is where the logic to enable the predictor and start the observer should live.

    **Example `DictionaryManager.java` modification:**
    ```java
    // In setLanguage method...
    if (_currentPredictor == null) {
        _currentPredictor = new WordPredictor();
        _currentPredictor.setContext(_context);
        
        // Show loading state in UI here if possible
        
        final WordPredictor predictorToLoad = _currentPredictor;
        predictorToLoad.loadDictionaryAsync(_context, languageCode, new Runnable() {
            @Override
            public void run() {
                // This runs on the main thread when loading is complete
                if (predictorToLoad.isReady()) {
                    // NOW the predictor is ready.
                    // Activate the observer (see Todo 2).
                    predictorToLoad.startObservingDictionaryChanges();
                    
                    // Hide loading state in UI here
                } else {
                    // Handle load failure
                }
            }
        });
        
        _predictors.put(languageCode, predictorToLoad);
    }
    ```
3.  **Update `preloadLanguages`:** This method should also be updated to use the asynchronous loader.

**Benefit:** This will finally eliminate the UI freeze when switching languages or on initial app startup.

---

### Todo 2 (Critical): Activate the User Dictionary Observer

**Problem:** The `UserDictionaryObserver` is correctly implemented, but it is never started because `WordPredictor.startObservingDictionaryChanges()` is never called. User-added words will not appear in predictions until the app is restarted.

**Solution:** Call `startObservingDictionaryChanges()` after a dictionary has been successfully loaded.

**Action Item:**

*   As shown in the example for **Todo 1**, the ideal place to activate the observer is inside the success callback of `loadDictionaryAsync`. This ensures that you only start observing for changes on a fully loaded and ready dictionary.

**Benefit:** User and custom dictionary changes will be reflected in predictions almost instantly, without requiring an app restart or a manual refresh.

---

### Todo 3 (Recommended): Introduce Proper Profiling Hooks

**Problem:** The project still uses a custom `PerformanceProfiler` class instead of the standard Android `Trace` API.

**Solution:** Replace all calls to `PerformanceProfiler` with `android.os.Trace` to enable deep, system-level profiling with tools like Perfetto.

**Action Items:**

*   Replace `PerformanceProfiler.start(...)` with `android.os.Trace.beginSection(...)`.
*   Replace `PerformanceProfiler.end(...)` with `android.os.Trace.endSection()`.
*   Wrap the traced code in a `try/finally` block to guarantee that `endSection()` is always called.

**Benefit:** Enables high-precision diagnosis of any future performance bottlenecks.