# Performance Todos v4: Atomic Map Swapping

This document addresses a remaining performance issue discovered after perftodos3.md completion.

## I. Issue Discovery

While async dictionary loading (perftodos3.md) successfully moved dictionary **loading** off the main thread, the **population** of the maps still happens on the main thread in the `onLoadComplete` callback.

**File**: `srcs/juloo.keyboard2/WordPredictor.java:503-506`

```java
public void onLoadComplete(Map<String, Integer> dictionary,
                          Map<String, Set<String>> prefixIndex)
{
  // Update dictionary and prefix index on main thread
  _dictionary.clear();          // Main thread
  _dictionary.putAll(dictionary);  // Main thread - 50k entries!
  _prefixIndex.clear();         // Main thread
  _prefixIndex.putAll(prefixIndex); // Main thread - thousands of entries!

  // ... rest of callback
}
```

**Problem**: `putAll()` with 50,000 dictionary entries can take 10-50ms on the main thread, causing a brief UI stutter.

---

## II. Outstanding Tasks

### Todo 1 (High Priority): Use Atomic Map Swapping

**Problem**: `clear()` + `putAll()` runs on main thread during async callback

**Solution**: Use `AtomicReference` to swap entire map instances atomically

**Implementation Steps**:

1. **Change field declarations in WordPredictor.java**:
   ```java
   // Before:
   private Map<String, Integer> _dictionary = new HashMap<>();
   private Map<String, Set<String>> _prefixIndex = new HashMap<>();

   // After:
   private AtomicReference<Map<String, Integer>> _dictionary =
       new AtomicReference<>(new HashMap<>());
   private AtomicReference<Map<String, Set<String>>> _prefixIndex =
       new AtomicReference<>(new HashMap<>());
   ```

2. **Update all field access throughout WordPredictor.java**:
   ```java
   // Before:
   _dictionary.get(word)
   _prefixIndex.get(prefix)

   // After:
   _dictionary.get().get(word)
   _prefixIndex.get().get(prefix)
   ```

   **Affected methods**:
   - `predictInternal()` - reads dictionary and prefix index
   - `scoreWord()` - reads dictionary for frequency
   - `getPrefixCandidates()` - reads prefix index
   - `loadCustomAndUserWords()` - writes to dictionary
   - `addToPrefixIndex()` - writes to prefix index
   - `buildPrefixIndex()` - writes to prefix index

3. **Modify onLoadComplete callback (lines 499-530)**:
   ```java
   @Override
   public void onLoadComplete(Map<String, Integer> dictionary,
                              Map<String, Set<String>> prefixIndex)
   {
     // Load custom words into the NEW maps (not yet visible)
     Set<String> customWords = loadCustomAndUserWordsIntoMap(context, dictionary);

     // Add custom words to the NEW prefix index
     if (!customWords.isEmpty())
     {
       addToPrefixIndexForMap(customWords, prefixIndex);
     }

     // Atomic swap - O(1) operation on main thread!
     _dictionary.set(dictionary);
     _prefixIndex.set(prefixIndex);

     // Set the N-gram model language
     setLanguage(language);

     _isLoading = false;
     android.util.Log.i("WordPredictor", String.format(
       "Async dictionary load complete: %d words, %d prefixes",
       _dictionary.get().size(), _prefixIndex.get().size()));

     if (callback != null)
     {
       callback.run();
     }
   }
   ```

4. **Create helper methods for custom word loading**:
   ```java
   /**
    * Load custom and user words into a specific map instance.
    * Used during async loading to populate new map before swap.
    */
   private Set<String> loadCustomAndUserWordsIntoMap(Context context,
                                                       Map<String, Integer> targetMap)
   {
     // Same logic as loadCustomAndUserWords() but writes to targetMap
     // instead of _dictionary.get()
     // ...
   }

   /**
    * Add words to a specific prefix index map.
    * Used during async loading to populate new index before swap.
    */
   private void addToPrefixIndexForMap(Set<String> words,
                                        Map<String, Set<String>> targetIndex)
   {
     // Same logic as addToPrefixIndex() but writes to targetIndex
     // instead of _prefixIndex.get()
     // ...
   }
   ```

**Benefits**:
- Main thread operation: 50ms `putAll()` → <1ms atomic `set()`
- No UI stutter during dictionary loading
- Thread-safe map access (AtomicReference guarantees visibility)
- Predictions continue working with old dictionary until new one ready

**Performance Impact**:
- Before: 10-50ms main thread block during `putAll()`
- After: <1ms main thread operation for atomic swap
- **50x improvement** in main thread impact

---

### Todo 2 (Low Priority): Profile Map Access Overhead

**Problem**: AtomicReference adds `.get()` indirection to every dictionary access

**Investigation**: Measure if AtomicReference adds measurable overhead:
- Prediction loop reads dictionary ~100-500 times per keystroke
- Each read: `_dictionary.get().get(word)` vs `_dictionary.get(word)`
- Overhead: One extra pointer dereference per access

**Profiling**: Use android.os.Trace to measure:
```java
android.os.Trace.beginSection("WordPredictor.predictInternal");
try {
  // ... prediction logic with AtomicReference access
} finally {
  android.os.Trace.endSection();
}
```

**Expected**: Negligible overhead (<1ms per prediction)
**If overhead detected**: Consider caching map reference locally in hot loops

**Status**: To be measured after Todo 1 implementation

---

## III. Priority

**Todo 1**: HIGH - Eliminates remaining main thread blocking
**Todo 2**: LOW - Optimization of optimization, likely unnecessary

---

## IV. Notes

- AtomicReference is the standard Java pattern for lock-free atomic updates
- No synchronization needed - AtomicReference handles memory barriers
- Prediction thread safety: Reads may see old or new map, both are valid
- Custom word updates: Still need synchronization for incremental updates

---

**Last Updated**: 2025-11-20
**Status**: Ready for implementation
