# Performance Todos v2: Addressing Swipe Prediction Latency

This document outlines the investigation into the regression of swipe prediction latency (from <100ms to ~600ms) and provides a clear action plan to resolve it.

## I. Summary of Findings

The previous optimizations successfully improved dictionary *loading times* by introducing a binary format and asynchronous loading. However, prediction *runtime latency* has severely degraded.

The investigation has concluded that the root cause is **not** the new binary format or data structures. The in-memory data structures (`HashMap`) are the same as before.

The primary cause of the latency increase is **excessive and expensive logging within the performance-critical prediction loop.**

### Root Cause Analysis

In `WordPredictor.java`, the `predictInternal` method loops through every potential candidate word for a given swipe. Inside this loop, the following line is executed for every valid candidate:

```java
android.util.Log.d("WordPredictor", "Candidate: " + word + " (score=" + score + ")");
```

A single swipe can generate hundreds of candidates. This results in hundreds of calls to `Log.d`. The cost of these calls is high because:
1.  **String Concatenation:** A new string `"Candidate: " + word + " (score=" + score + ")"` is created for every single candidate. This churns memory and takes significant CPU time.
2.  **Logging Overhead:** The `Log.d` method itself has overhead, even if the logs are not being displayed.

This intense logging within a tight, frequently executed loop is the definitive cause of the observed latency increase from under 100ms to over 600ms.

A secondary performance issue was also noted in the dictionary loading process, where a full prefix index rebuild is triggered unnecessarily.

---

## II. Optimization Plan

### Todo 1 (Critical): Eliminate Runtime Logging

**Problem:** `Log.d` calls inside the `predictInternal` loop are destroying runtime performance.

**Solution:** All diagnostic logging must be completely removed from production builds. This can be achieved by wrapping log calls in a static `DEBUG` flag that can be toggled by the build system.

**Action Items:**

*   **Introduce a `DEBUG` flag:** In `WordPredictor.java` (or a global `Config` class), add a static final boolean flag.
    ```java
    public class Config {
        public static final boolean DEBUG = false; // Set to true for development builds only
    }
    ```
    (Ensure your build system, e.g., `build.gradle`, can set this flag automatically based on the build type, `debug` vs `release`).

*   **Wrap ALL diagnostic logs:** Go through `WordPredictor.java` and wrap every single `Log.d` call in a conditional check.

    **Example (Before):**
    ```java
    android.util.Log.d("WordPredictor", "Candidate: " + word + " (score=" + score + ")");
    ```

    **Example (After):**
    ```java
    if (Config.DEBUG) {
        android.util.Log.d("WordPredictor", "Candidate: " + word + " (score=" + score + ")");
    }
    ```
*   **Apply this to all `Log.d` calls** in the `predictInternal`, `calculateUnifiedScore`, and other performance-sensitive methods to ensure zero logging overhead in release builds.

**Benefit:** This will immediately bring the prediction latency back to its expected sub-100ms level.

---

### Todo 2 (High Priority): Fix Incremental Loading of Custom Words

**Problem:** The async dictionary loading logic (`loadDictionaryAsync` and `loadDictionary`) is inefficiently rebuilding the entire prefix index with `buildPrefixIndex()` just to add custom/user words, negating some of the benefit of loading the pre-built index from the binary file.

**Solution:** Use the already-existing `addToPrefixIndex()` method for an incremental update.

**Action Items:**

*   **Modify `loadDictionaryAsync`:**
    *   After loading the main dictionary and the user words, collect only the `userWords` into a `Set`.
    *   Call `addToPrefixIndex(userWords)` instead of `buildPrefixIndex()`.
*   **Modify `loadDictionary` (for the binary path):**
    *   Apply the same logic: after loading the binary dictionary and user words, call `addToPrefixIndex` with only the set of user/custom words.

    **Example Snippet (for `loadDictionaryAsync`):**
    ```java
    // ... after _dictionary and _prefixIndex are populated from binary
    Set<String> customWords = new HashSet<>();
    // The following method internally adds words to _dictionary
    loadCustomAndUserWords(context, customWords); 
    
    // Now, incrementally update the prefix index ONLY with the new words
    addToPrefixIndex(customWords); 
    
    // NOT buildPrefixIndex();
    ```
    *(This may require a small modification to `loadCustomAndUserWords` to make it return the set of words it loaded.)*

**Benefit:** This will significantly speed up the initial dictionary load time, making the app start faster and language switching more responsive.

---

### Todo 3 (Recommended): Introduce Proper Profiling Hooks

**Problem:** The custom `PerformanceProfiler` is not standard and likely just logs to `logcat`, which is not ideal for performance analysis.

**Solution:** Use the standard Android `Trace` API for profiling. This integrates with system tools like Perfetto and the Android Studio Profiler, giving a much more accurate and detailed view of performance.

**Action Items:**
*   Replace `PerformanceProfiler.start(...)` with `android.os.Trace.beginSection(...)`.
*   Replace `PerformanceProfiler.end(...)` with `android.os.Trace.endSection()`.
*   This should be done for key operations like `predictInternal` and `loadDictionary`.

**Benefit:** Enables deep, system-level performance analysis to find any future bottlenecks.

---
### Todo 4 (Outstanding from v1): Optimize Contraction Loading

**Problem:** The `ContractionManager.java` was not optimized in the first round of changes. It still reads JSON files into memory and parses them at startup, which is a slow and memory-intensive process.

**Original Task:** Apply the same binary-format optimization to the contraction files that was successfully applied to the main dictionary.

**Action Items:**
*   **Create a conversion script:** Add functionality to the dictionary conversion script (or create a new one) to process `contractions_non_paired.json` and `contraction_pairings.json` into a single, efficient binary file.
*   **Update `ContractionManager.java`:** Modify the `loadMappings` method to load the new binary file instead of the JSON files. This will involve using a `ByteBuffer` to read the data, similar to how `BinaryDictionaryLoader` works.
*   **Remove old code:** Delete the now-unused JSON parsing logic (`readStream`, `JSONObject`, etc.) from `ContractionManager.java`.

**Benefit:** This will improve application startup time and reduce memory usage by eliminating another source of slow file parsing.
