# Main Thread Optimization Plan

## Objective
Fix "Application Not Responding" (ANR) crashes and UI lag caused by heavy computation and excessive logging on the main thread.

## Root Causes
1.  **Synchronous Prediction:** `WordPredictor` runs complex dictionary scoring algorithms directly on the main thread during user typing (`handleRegularTyping`).
2.  **Excessive Logging:** `Keyboard2View` emits high-frequency logs (`KeyDetection`, `onMeasure`) during touch events and layout passes, flooding the main thread.
3.  **Blocking Startup:** `onStartInputView` performs synchronous initialization of prediction views.

## Execution Plan

### 1. Clean up Logging (Immediate Impact)
- **Target:** `srcs/juloo.keyboard2/Keyboard2View.java`
- **Action:** Remove or guard `Log.e("KeyDetection", ...)` and `Log.d` calls in critical paths (`onTouch`, `getKeyAtPosition`, `onMeasure`).
- **Benefit:** Reduces IO overhead and logcat spam during fast typing/swiping.

### 2. Asynchronous Prediction (Architecture Fix)
- **Target:** `srcs/juloo.keyboard2/SuggestionHandler.java`
- **Action:**
    - Introduce `java.util.concurrent.ExecutorService` (SingleThreadExecutor) to `SuggestionHandler`.
    - Refactor `updatePredictionsForCurrentWord` to submit prediction tasks to the executor.
    - Implement task cancellation (`Future.cancel`) to discard stale predictions when the user types quickly.
    - Post results back to the Main Thread via `_suggestionBar.post(...)` to update the UI.
- **Benefit:** Moves the O(N) dictionary scan off the UI thread, ensuring keystrokes remain responsive even with large dictionaries.

### 3. Verify & Test
- **Verification:**
    - Build the project.
    - Check `logcat` to ensure `KeyDetection` logs are gone.
    - Verify typing responsiveness.
    - Ensure suggestions still appear (albeit asynchronously).

## Technical Details

### `SuggestionHandler.java` Changes
```java
// New Members
private final ExecutorService _predictionExecutor = Executors.newSingleThreadExecutor();
private Future<?> _currentPredictionTask;

// Modified Logic
private void updatePredictionsForCurrentWord() {
    if (_currentPredictionTask != null) {
        _currentPredictionTask.cancel(true); // Cancel previous inflight prediction
    }
    
    // Capture state for background thread
    final String partial = _contextTracker.getCurrentWord();
    final List<String> context = new ArrayList<>(_contextTracker.getContextWords());
    
    _currentPredictionTask = _predictionExecutor.submit(() -> {
        if (Thread.currentThread().isInterrupted()) return;
        
        // Heavy calculation
        final WordPredictor.PredictionResult result = 
            _predictionCoordinator.getWordPredictor().predictWordsWithContext(partial, context);
            
        // Update UI on Main Thread
        if (_suggestionBar != null && !Thread.currentThread().isInterrupted()) {
            _suggestionBar.post(() -> {
                 // Update logic...
            });
        }
    });
}
```
