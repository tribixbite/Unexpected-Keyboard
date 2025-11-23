# Investigation: 3-Second Delay & App Switch Crash

**Date**: 2025-11-21
**Status**: In Progress

## 1. 3-Second Startup Delay on App Switch

**Observations:**
- User reports "1-3 second startup delay when switching apps... seems to wait for first swipe to load nn".
- The delay happens despite the fix in `PredictionViewSetup.kt` (checking availability).
- If `PredictionCoordinator` survives app switches (singleton-like lifecycle in Service), `_neuralEngine` and `OnnxSwipePredictor` should remain initialized.

**Analysis:**
- **WordPredictor Initialization:** `PredictionCoordinator.initializeWordPredictor()` loads the main 50k dictionary via `BinaryDictionaryLoader`. This is synchronous. If `onCreate` runs, this adds latency. But service usually survives.
- **Neural Initialization:** `OnnxSwipePredictor.initialize()` returns fast if already loaded.
- **Config Updates:** `OnnxSwipePredictor.setConfig` was patched to avoid reloading on harmless changes.
- **Vocabulary Loading:** `OptimizedVocabulary.loadVocabulary()` loads 50k words. It uses a binary cache (V2 format). `tryLoadBinaryCache()` uses buffered I/O (~5ms). This is fast.

**New Hypothesis: `PredictionViewSetup` causing Layout/Measure Pass**
- `PredictionViewSetup.kt` adds a `GlobalLayoutListener`.
- Inside the listener:
  ```kotlin
  keyboardView.getViewTreeObserver().addOnGlobalLayoutListener(object : ... {
      override fun onGlobalLayout() {
          // ...
          predictionCoordinator.getNeuralEngine().setKeyboardDimensions(...)
          neuralLayoutHelper?.setNeuralKeyboardLayout()
          // ... remove listener ...
      }
  })
  ```
- `setNeuralKeyboardLayout` iterates over all keys to map positions. This is fast (~1-2ms).
- BUT, if the *first swipe* is what triggers the "load", maybe the "load" isn't the delay, but the *swipe processing* is waiting for something?
- If `ensureInitialized` is NOT called (because we skip it), and `_neuralEngine` is somehow not ready?
- Wait, if `isSwipeTypingAvailable()` returns true, it means `_isModelLoaded` is true.

**Alternative: The delay is visual.**
- If the keyboard takes 3s to *appear*.
- This implies `onCreateInputView` or `onStartInputView` is slow.
- `onStartInputView` calls `PredictionViewSetup`.
- `SuggestionBarInitializer.initialize` creates views.
- `setInputView` is called.

**Wait! The "3-second delay" was the original bug.**
- The original bug was `ensureInitialized` loading models on the main thread.
- We moved it to a thread.
- Now user says it "seems to wait for first swipe".
- This implies the thread IS running, but maybe not fast enough?
- Or maybe the thread is *skipped* correctly, but `predict()` hits a snag?

**Hypothesis: `OnnxSwipePredictor.predict()` lazy initialization?**
- `predict()` checks `if (!_isModelLoaded)`. If not loaded, it returns empty.
- It does NOT trigger initialization.
- So if it's not loaded, user gets nothing.
- If user says "wait for first swipe to load nn", maybe they mean "first swipe fails/is slow, then it works".
- If so, it means the model wasn't loaded when they swiped.
- Why wasn't it loaded? Because `PredictionViewSetup` skipped the thread?
- No, if it skipped, it means `isSwipeTypingAvailable` was true.
- So it WAS loaded.

**Hypothesis: `OptimizedVocabulary` is the bottleneck?**
- `filterPredictions` calls `vocabulary.get(word)`.
- `vocabulary` is a `HashMap`. Fast.

**Hypothesis: `ImprovedSwipeGestureRecognizer` GC?**
- We fixed the allocations.

**User Insight:** "if any user setting doesnt match default it will have to entirely reload the nn every time".
- They are convinced of this.
- I explained `beam_width` doesn't trigger it.
- BUT what if `updateConfig` triggers `_vocabulary.updateConfig`.
- `OptimizedVocabulary.updateConfig` parses custom words JSON *every time* config updates?
- `onStartInputView` -> `refresh_config` -> `onConfigChanged` -> `updateConfig`.
- `OptimizedVocabulary.updateConfig`:
  ```java
    // OPTIMIZATION Phase 2: Parse and cache custom words here
    try {
      SharedPreferences prefs = ...;
      String customWordsJson = prefs.getString("custom_words", "{}");
      JSONObject jsonObj = new JSONObject(customWordsJson);
      // ... loop ...
    }
  ```
- Parsing JSON on *every app switch* might be noticeable if the JSON is huge.
- But typical custom words list is small.
- **However**, `OnnxSwipePredictor.setConfig` calls `updateConfig`.
- Is `updateConfig` called unnecessarily?
- `Keyboard2.onStartInputView` calls `refresh_config` **only if `restarting` is true or config is null**.
- `refresh_config` triggers `onConfigChanged`.
- `onStartInputView` is called with `restarting=false` on normal app switch?
- Documentation says `restarting` is true if the input view is being restarted in the *same* window.
- On app switch, `restarting` is usually false.
- So `refresh_config` runs.
- `ConfigManager` reloads prefs.
- `onConfigChanged` fires.
- `PredictionCoordinator.setConfig` -> `NeuralEngine.setConfig` -> `OnnxSwipePredictor.setConfig` -> `updateConfig` -> `Vocabulary.updateConfig` -> **JSON Parse**.

**Potential Fix:** Only parse custom words if the *preference string* changed?
`OptimizedVocabulary` reads `custom_words` string.
We can cache the string and compare.

## 2. Crash on App Switch with Predictions

**Analysis:**
- Crash happens when switching apps if prediction bar has items.
- `SuggestionBar` retains state.
- `TextView`s in `SuggestionBar` might be holding onto an old `Context` or `EditorInfo` related resources?
- My fix `_suggestionBar.clearSuggestions()` in `onFinishInputView` removes the views.
- **User says "both issues persist".**
- This means the crash still happens.
- Maybe `onFinishInputView` is NOT called on app switch?
- On Android, switching apps triggers `onFinishInputView` then `onStartInputView`.
- BUT, sometimes `onDestroy` happens?
- Or maybe the crash is in `SuggestionHandler` updating the bar *after* `onFinishInputView`?
- If `AsyncPredictionHandler` returns a result after `onFinishInputView`.
- It calls `showSuggestions`.
- `SuggestionBar` tries to add views.
- If `SuggestionBar` is detached?
- `addView` on detached parent might not crash, but interaction might.

**New Fix for Crash:**
- In `SuggestionBridge.handlePredictionResults`: Check if input view is active/started?
- Or in `SuggestionBar.setSuggestions`: Check if attached?
- `SuggestionBar.setSuggestions` calls `removeAllViews`.
- If `_suggestionBar` was cleared in `onFinish`, it's empty.
- Then async result comes in.
- It calls `setSuggestions`.
- It adds views.
- If the user has already switched apps, these views are added to the *old* `SuggestionBar` (if reused) or a *new* one?
- `_suggestionBar` is reused in `PredictionViewSetup`.
- If async result comes in for the *previous* app's context...
- We need to **cancel** pending predictions on `onFinishInputView`.

**Action Plan:**
1.  **Fix 3-Second Delay:** Optimize `OptimizedVocabulary.updateConfig` to check if custom words JSON string actually changed before parsing.
2.  **Fix Crash:** Cancel pending predictions in `onFinishInputView` to prevent stale updates.

