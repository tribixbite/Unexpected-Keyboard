# Performance Todos v3: Final Tasks

This document lists the final remaining performance optimization tasks. The critical latency issues and major loading-time bottlenecks identified in `perftodos.md` and `perftodos2.md` have been successfully addressed.

---

## I. Summary of Completed Work

*   **Prediction Latency:** The critical swipe prediction latency has been resolved by removing expensive logging from the main prediction loop.
*   **Dictionary Loading:** The main dictionary loading is now significantly faster, using an optimized binary format (`.bin`) and asynchronous execution.
*   **Incremental Updates:** The dictionary loading process now correctly performs incremental updates for custom/user words, avoiding slow, full index rebuilds.
*   **Contraction Loading:** The contraction data is now loaded from an optimized binary format, improving startup time.

---

## II. Outstanding Tasks

There is one remaining low-priority task from the previous optimization lists.

### Todo 1 (Recommended): Introduce Proper Profiling Hooks

**Problem:** The project uses a custom `PerformanceProfiler` class, which appears to just log to `logcat`. This is not a standard or effective way to analyze performance.

**Solution:** Use the standard `android.os.Trace` API for profiling. This is the Android-native way to trace performance, and it integrates with powerful system-level tools like **Perfetto** and the **Android Studio Profiler**. This provides a much more accurate and detailed view of performance, including thread states, CPU time, and interactions with the rest of the system.

**Action Items:**

*   In `WordPredictor.java` (and any other classes that use it), replace all calls to `PerformanceProfiler` with the `Trace` API.

    **Before:**
    ```java
    PerformanceProfiler.start("Type.predictWordsWithScores");
    // ... code to measure ...
    PerformanceProfiler.end("Type.predictWordsWithScores");
    ```

    **After:**
    ```java
    android.os.Trace.beginSection("WordPredictor.predictInternal");
    try {
        // ... code to measure ...
    } finally {
        android.os.Trace.endSection();
    }
    ```
    *(Note: Using a `try`/`finally` block is crucial to ensure that `endSection()` is always called, even if an exception occurs.)*

*   Apply this to key methods that are critical for performance, such as:
    *   `WordPredictor.predictInternal`
    *   `AsyncDictionaryLoader.loadDictionaryAsync`
    *   `BinaryDictionaryLoader.loadDictionaryWithPrefixIndex`

**Benefit:** Enables deep, system-level performance analysis to identify and diagnose any future bottlenecks with high precision.
