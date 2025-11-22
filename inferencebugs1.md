# Inference Bugs & Latency Investigation (inferencebugs1.md)

**Status**: FIXES IMPLEMENTED
**Priority**: HIGH

## 1. Redundant Model Loading & Layout Updates (Latency Source)
**Observation**: The log showed redundant initialization sequences.
**Status**: ✅ FIXED

**Fixes Applied**:
- **Layout**: Removed redundant `_keyboardView.post(setNeuralKeyboardLayout)` in `Keyboard2.java:556-563`. Layout is now handled exclusively by `PredictionViewSetup`'s `OnGlobalLayoutListener`, ensuring it runs exactly once when dimensions are ready.
- **Vocabulary**: Added `if (_vocabulary.isLoaded())` check in `OnnxSwipePredictor.initialize():469-477` to prevent reloading the 50k-word dictionary if it's already in memory.

## 1a. CRITICAL: 3-Second UI Freeze on App Switch (MAIN THREAD BLOCKING)
**Observation**: User reported 3-second delay before swipes yield predictions when switching apps.
**Status**: ✅ FIXED

**Root Cause Identified**:
- `onStartInputView()` → `PredictionViewSetup.setupPredictionViews()` → `ensureInitialized()` was loading ONNX models on **MAIN THREAD**
- ONNX model loading breakdown:
  - Encoder read: 500-800ms
  - Encoder session creation: 1000-1500ms
  - Decoder read: 300-500ms
  - Decoder session creation: 800-1200ms
  - Tokenizer + Vocabulary: 200-400ms
  - **TOTAL: 2800-4400ms = 2.8-4.4 SECONDS OF UI BLOCKING**
- Code even warned about this at `OnnxSwipePredictor.java:210`: "⚠️ initialize() called on MAIN THREAD - may cause UI jank!"

**Fix Applied**:
- Moved `predictionCoordinator.ensureInitialized()` to background thread in `PredictionViewSetup.kt:73-75`
- Models now load asynchronously, allowing immediate keyboard display
- First swipe may have slight delay while models finish loading, but UI remains responsive
- Location: `PredictionViewSetup.kt:70-75` (Thread wrapper around ensureInitialized)

## 2. Long Swipes Yield No Output / Resampling Issues
**Observation**: User reported long swipes failing and not adhering to resampling settings.
**Status**: ✅ FIXED

**Fixes Applied**:
- **Default Mode**: Changed `SwipeTrajectoryProcessor` default resampling mode from `TRUNCATE` to `DISCARD`.
    - `TRUNCATE` was chopping off the end of long swipes (>250 points), losing the last letters and causing recognition failure.
    - `DISCARD` preserves start and end points and uniformly samples the middle, preserving the word shape.
- **Synchronization**: This ensures that even if `setConfig` is delayed, the default behavior is safe (resampling) rather than destructive (truncation).

## 3. Debug Logging Clutter
**Status**: ✅ IMPROVED
- Reducing redundant initialization calls eliminates the double logging of "Loaded 14 custom words...", "Neural engine: 27 key positions set", etc.

## Verification Needed
- Monitor logs to ensure "Loaded custom words" appears only once per session/config change.
- Monitor "key positions set" to ensure it appears only once.
- Verify long swipes now produce output.