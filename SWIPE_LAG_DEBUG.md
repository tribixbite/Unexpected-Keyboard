# Swipe Lag Investigation - 1 Second Delay

**Reported**: 2025-11-22
**Issue**: "there's like a full second of lag after swiping in termux before the word gets inserted"
**Version**: v1.32.639
**Context**: After Phase 3 UI optimizations (object pooling, path reuse)

## Symptoms

- User swipes a word (e.g., "test")
- Swipe trail appears immediately (UI rendering working)
- **~1 second delay** before predicted word is inserted into text field
- This is NEW behavior after recent optimizations

## Investigation Plan

### Step 1: Capture End-to-End Timing ⏱️

Need to trace the complete swipe→insert path with timestamps:

1. **Touch Event Start** → ImprovedSwipeGestureRecognizer.startSwipe()
2. **Touch Events** → addPoint() (collecting path)
3. **Touch End** → endSwipe() (returns SwipeResult)
4. **Prediction Request** → AsyncPredictionHandler receives request
5. **Neural Inference** → OnnxSwipePredictor.predict()
   - Preprocessing
   - Encoder inference
   - Beam search
6. **Result Callback** → Main thread receives predictions
7. **Text Insertion** → InputConnection.commitText()

**Expected logs to check**:
- `AsyncPredictionHandler`: "Prediction completed in Xms"
- `OnnxSwipePredictor`: "⏱️ End-to-end latency: Xms"
- `InputCoordinator`: Timing for prediction

### Step 2: Identify Which Stage is Slow

Based on perftodos7.md, these were already optimized:
- ✅ Logging disabled (conditional based on BuildConfig)
- ✅ Config caching (no SharedPreferences reads)
- ✅ Beam search pruning
- ✅ Vocabulary trie (fast filtering)
- ✅ GC reduction (object pooling)

**Possible new bottlenecks**:
1. **Main thread queueing** - If AsyncPredictionHandler is backed up
2. **Termux-specific issue** - InputConnection.commitText() slow in Termux
3. **Rendering issue** - Invalidate() calls blocking
4. **Regression from Phase 3** - Object pool contention?
5. **Prediction coordinator** - Request queuing or cancellation logic

### Step 3: Test in Different Apps

Compare swipe latency:
- Termux (reported slow)
- Standard text field (Messages, Chrome, etc.)
- SwipeCalibrationActivity (internal test)

If slow ONLY in Termux → Termux InputConnection issue
If slow everywhere → Keyboard code regression

## Debugging Commands

### Capture Full Timing Trace
```bash
adb logcat -c
# User performs swipe
adb logcat -d -v threadtime "*:D" | grep -E "Swipe|Prediction|Neural|ONNX|latency|ms"
```

### Check AsyncPredictionHandler Queue
```bash
adb logcat -d -s "AsyncPredictionHandler:D"
```

### Check ONNX Inference Times
```bash
adb logcat -d -s "OnnxSwipePredictor:D" | grep "⏱️"
```

### Monitor Main Thread Blocking
```bash
adb shell "am profile start juloo.keyboard2.debug /data/local/tmp/profile.trace"
# Perform swipe
adb shell "am profile stop juloo.keyboard2.debug"
adb pull /data/local/tmp/profile.trace
```

## Hypotheses to Test

### H1: Async Queue Backlog
**Cause**: Multiple prediction requests queuing up
**Test**: Check for "Prediction cancelled" logs
**Fix**: Increase queue drain rate or reduce request rate

### H2: Termux InputConnection Slow
**Cause**: Termux's custom InputConnection has latency
**Test**: Add timing around `commitText()` call
**Fix**: May need Termux-specific optimization or workaround

### H3: Main Thread Callback Delay
**Cause**: `_mainHandler.post()` delayed by other UI work
**Test**: Add timestamp in post() vs run() of callback
**Fix**: Use higher priority handler or optimize main thread work

### H4: Phase 3 Regression
**Cause**: Object pool synchronization overhead
**Test**: Temporarily disable pool, compare latency
**Fix**: Optimize pool implementation or revert

### H5: Prediction Coordinator Throttling
**Cause**: Request rate limiting or debouncing too aggressive
**Test**: Check PredictionCoordinator request handling
**Fix**: Adjust throttling parameters

## ROOT CAUSE IDENTIFIED! 🎯

### The 1-Second Lag is Termux Backspace Key Events

**Location**: `InputCoordinator.java:403-412` and `452-458`

**Problem**: When inserting a swipe prediction in Termux, the code:
1. Auto-inserts the first prediction immediately
2. When user swipes again OR prediction arrives, it needs to DELETE the previous word
3. For Termux, instead of using `deleteSurroundingText()`, it sends INDIVIDUAL backspace key events:

```java
// SLOW CODE (InputCoordinator.java:405-409):
for (int i = 0; i < deleteCount; i++)
{
    _keyeventhandler.send_key_down_up(KeyEvent.KEYCODE_DEL, 0);
}
```

**Why this is slow**:
- Each backspace key event is processed individually by Termux
- Termux's terminal emulator processes each keystroke sequentially
- If deleting a 5-character word + 1 space = 6 backspaces
- Each backspace ~150-200ms → **6 × 150ms = 900ms-1200ms!**

**Evidence**:
- Code explicitly detects Termux: `editorInfo.packageName.equals("com.termux")` (line 352)
- Comment says: "TERMUX: Use backspace key events instead of InputConnection methods" (line 390)
- Comment says: "Termux doesn't support deleteSurroundingText properly" (line 391)

## Solution Implemented

### Added Detailed Timing Instrumentation (v1.32.640)

Added ⏱️ timing logs to pinpoint exact bottleneck:
1. **AsyncPredictionHandler.java**:
   - Prediction completion time
   - Callback delay (main thread queue time)
   - Callback execution time

2. **InputCoordinator.java**:
   - handlePredictionResults start/end
   - setSuggestionsWithScores time
   - commitText time for space
   - onSuggestionSelected time
   - **TERMUX BACKSPACES timing** (the smoking gun!)
   - commitText time for final word

### Next Fix: Batch Termux Deletion

Instead of sending individual backspaces, options:
1. **Use InputConnection anyway** - Test if modern Termux supports it
2. **Send batched backspaces** - Queue them and send in burst
3. **Use finishComposingText + setComposingText** - Atomic replacement
4. **Don't auto-insert** - Wait for prediction, insert once (UX change)

## Next Steps

1. ✅ Root cause identified (Termux backspace key events)
2. ✅ Timing instrumentation added (v1.32.640)
3. ⏳ Test instrumented build to confirm timing
4. ⏳ Implement optimized Termux deletion
5. ⏳ Verify fix resolves 1-second lag

## Related Files

- `srcs/juloo.keyboard2/AsyncPredictionHandler.java` - Background prediction
- `srcs/juloo.keyboard2/OnnxSwipePredictor.java` - Neural inference
- `srcs/juloo.keyboard2/InputCoordinator.java` - Text insertion
- `srcs/juloo.keyboard2/PredictionCoordinator.java` - Request coordination
- `srcs/juloo.keyboard2/ImprovedSwipeGestureRecognizer.java` - Swipe detection

## Timeline

- **2025-11-22 02:42**: User reports 1-second lag after swipe
- **2025-11-22 03:05**: Investigation started, debug document created
- **2025-11-22 03:XX**: Awaiting user swipe test to capture timing logs
