# Resampling and Caching Deep Dive

**Date**: 2025-11-22
**Status**: Analysis Complete

## 1. Resampling Mechanism (SwipeResampler.java)

**Purpose**:
The neural network expects a fixed input size of `250` points (or whatever `_maxSequenceLength` is configured to). Swipes can be shorter or longer. Resampling normalizes this.

**How it works:**
The class supports three modes: `TRUNCATE`, `DISCARD`, `MERGE`. The default is `DISCARD`.

**DISCARD Mode Algorithm:**
1.  **Input**: A trajectory of `N` points (`float[][]`).
2.  **Target**: A fixed length `T` (e.g., 250).
3.  **Short Swipes (`N <= T`):** If the input is shorter than or equal to the target, **no resampling occurs**. The raw data is returned as-is. It is later padded with zeros in `SwipeTrajectoryProcessor`.
4.  **Long Swipes (`N > T`):**
    *   **Always Keep:** First point (start) and Last point (end).
    *   **Middle Points:** The algorithm calculates indices to keep based on a weighted distribution.
    *   **Weighting:** It splits the swipe into 3 zones:
        *   **Start Zone (30% of input):** Allocates 35% of output capacity.
        *   **Middle Zone (40% of input):** Allocates 30% of output capacity.
        *   **End Zone (30% of input):** Allocates 35% of output capacity.
    *   **Result:** Higher sampling density at the start and end of the swipe, lower density in the middle. This preserves the critical initial trajectory (first letter) and final trajectory (last letter) while thinning out the middle where speed is highest and precision is lower.

**TRUNCATE Mode:**
Simply takes the first `T` points. Good for latency, bad for accuracy (loses the end of the word).

**MERGE Mode:**
Averages neighboring points. Smoother, but heavier computation.

**Key Takeaway for User:**
For your swipes, if they are under 250 points (sampled at ~120Hz = ~2 seconds), **no resampling happens**. The model sees the exact raw points. If you swipe for 5 seconds, `DISCARD` mode intelligently drops middle points to fit 250, prioritizing your start/stop precision.

## 2. Caching Investigation

### A. Vocabulary Caching (`OptimizedVocabulary.java`)
**Mechanism:**
1.  **Binary Cache (`vocab_cache.bin`):**
    *   The 50k word dictionary is parsed from JSON *once* and written to a binary file.
    *   **Format:** Custom V2 binary format (Magic "VOCB", Word + Freq + Tier).
    *   **Loading:** `tryLoadBinaryCache()` reads this file using `BufferedInputStream`.
    *   **Speed:** Parsing binary is ~100x faster than JSON.
    *   **Validity:** It checks a Magic Number (`0x564F4342`) and Version (`2`). If these match, it assumes the cache is valid.
2.  **Updates:**
    *   If `loadVocabulary` fails to find the cache, it falls back to JSON/Text parsing and *writes* a new cache.
    *   There is **no timestamp check** against the asset file. If you update the app with a new `en_enhanced.json`, the old binary cache might persist until app data is cleared or version bump forces a rewrite. **Potential Issue.**

### B. SharedPreferences Caching (`Config.java`)
**Mechanism:**
1.  **Android Native:** `SharedPreferences` are loaded into memory by the Android framework on process start. Accessing them (`getString`, `getInt`) is effectively a map lookup (very fast).
2.  **Initialization:** `Config` constructor reads all preferences into public fields (`neural_beam_width`, etc.).
3.  **Refresh:** `Keyboard2.onStartInputView` calls `refresh_config` -> `_configManager.refresh()`. This re-reads all prefs from the in-memory map.
4.  **Impact:** This is generally efficient. The only "heavy" object is the `custom_words` string, which I optimized in `OptimizedVocabulary` to only re-parse if the string content changed.

### C. `SwipeMLDataStore` (SQLite)
**Mechanism:**
1.  **SQLite:** Uses standard `SQLiteOpenHelper`.
2.  **Async:** `storeSwipeData` runs on a single-threaded `ExecutorService`. It does **not** block the main thread.
3.  **Loading:** `loadAllData` runs on the caller's thread. If called for export, it might be slow, but it's not on the critical path of typing.

## Conclusion on Caching
*   **Vocabulary:** Highly optimized with binary caching. **Action Item:** Ensure cache versioning strategy handles app updates (currently relies on `DATABASE_VERSION` or similar? No, uses internal version byte).
*   **Prefs:** Efficiently handled by Android and our local optimization.
*   **ML Store:** Writes are async and safe.

**Verdict:** Caching is being done properly for the 50k dict and prefs. The latency issues were almost certainly the locking and layout race conditions we fixed.
