# Crash Analysis: App Switching with Active Predictions

**Date**: 2025-11-21
**Status**: Investigation Complete - Fix Identified

## Problem
The application crashes when switching to Discord (or potentially other apps) while the prediction bar is populated with suggestions.

## Root Cause Analysis

1.  **Lifecycle & View Re-use**:
    *   `Keyboard2.onStartInputView` calls `PredictionViewSetup.setupPredictionViews`.
    *   This method reuses the existing `_suggestionBar` if it's not null.
    *   The `SuggestionBar` contains `TextView` children that were created with the *previous* `Context` (the Service context is long-lived, but views might hold references to resources that become stale or conflict with the new EditorInfo attributes).

2.  **The Specific Crash Scenario**:
    *   When switching apps, `onFinishInputView` is called, then `onStartInputView`.
    *   `SuggestionBar` retains its state (`_currentSuggestions`, `_suggestionViews`).
    *   If the new app triggers a layout pass or theme update incompatible with the existing views, or if the views try to access resources that are invalid for the new input connection state, a crash can occur.
    *   Crucially, `Keyboard2.onFinishInputView` calls `_keyboardView.reset()` but **DOES NOT** clear the `SuggestionBar`.

3.  **"Populations" (Predictions)**:
    *   The user noted "prediction bar has populations". This confirms the state is persisting across the app switch.
    *   Old predictions are likely invalid for the new context (e.g., different text field type).

## Solution

We need to ensure the `SuggestionBar` is cleared when the input view finishes. This prevents stale views and state from persisting into the new app session.

**Proposed Fix**:
Modify `Keyboard2.onFinishInputView` to explicitly clear the `SuggestionBar`.

```java
  @Override
  public void onFinishInputView(boolean finishingInput)
  {
    super.onFinishInputView(finishingInput);
    _keyboardView.reset();
    
    // FIX: Clear suggestion bar to prevent crashes on app switch
    if (_suggestionBar != null) {
        _suggestionBar.clearSuggestions();
    }
  }
```

## Verification of Reverted Changes
The user mentioned "0,0 coordinates" bugs were reverted.
*   Checked `ImprovedSwipeGestureRecognizer.java`: It **DOES** use `TrajectoryObjectPool.INSTANCE.obtainPointF(x, y)`. This means the optimization is **ACTIVE**.
*   If the user *manually* reverted changes, the file content would show `new PointF()`.
*   Since the file content shows the optimized version, and the user says "those changes were causing 0,0 coordinates and were reverted", there is a conflict.
*   **Hypothesis**: The "0,0 coordinates" bug might be caused by the object pool returning dirty objects (reset to 0,0) or being reused incorrectly.
*   **Correction**: I should ensure `TrajectoryObjectPool` recycles correctly.
*   **However**, the user asked about the *crash*. The crash fix (clearing suggestions) is the priority.

**Wait**, if the user *says* they reverted it, but my read shows the optimization, maybe they reverted it *locally* and I'm seeing my last write? No, `read_file` shows current state.
**Actually**, `ImprovedSwipeGestureRecognizer.java` currently shows:
```java
    // Add raw point (using object pool)
    _rawPath.add(TrajectoryObjectPool.INSTANCE.obtainPointF(x, y));
```
So the optimization IS present. If the user thinks it was reverted, or if they experienced 0,0 bugs, the pool might be the culprit.
**But for the crash**, the fix is in `Keyboard2.java`.

## Implementation Plan
1.  Modify `Keyboard2.java` to clear suggestions in `onFinishInputView`.
