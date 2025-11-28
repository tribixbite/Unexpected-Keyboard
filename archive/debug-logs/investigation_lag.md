# Investigation: "6 Circle Lag" (Swipe Trace Delay)

**Date**: 2025-11-22
**Status**: Deep Dive In Progress

## Problem Description
After switching apps, swiping continuously in circles produces **no trace** and **no predictions** for approximately 6 laps (estimated 1-3 seconds). Suddenly, the trace appears and works normally thereafter.

## Knowns & Eliminations
1.  **NN Model Reload**: Logs confirm the model is **not** reloading. Initialization is fast (269ms) and one-time.
2.  **Config Updates**: Config is updated but cheap.
3.  **Key Detection (Width 0)**: `Keyboard2View.setKeyboard` now pre-calculates `_keyWidth` and `onMeasure` preserves it. This *should* have fixed the "0 width" issue.
4.  **Gesture Thresholds**: `MIN_SWIPE_DISTANCE` reduced to 50px. Circles are much larger.

## New Hypotheses

### 1. `ProbabilisticKeyDetector` Initialization Lag
`ImprovedSwipeGestureRecognizer` uses a `ProbabilisticKeyDetector` if a keyboard is set.
```java
  public void setKeyboard(KeyboardData keyboard, float width, float height)
  {
    _currentKeyboard = keyboard;
    if (keyboard != null)
    {
      _probabilisticDetector = new ProbabilisticKeyDetector(keyboard, width, height);
    }
  }
```
- **When is `setKeyboard` called on the recognizer?**
- It's called from `PredictionViewSetup`'s `OnGlobalLayoutListener`.
- **Crucially:** If the layout listener takes 1-3 seconds to fire (e.g., waiting for `Keyboard2View` to be fully attached and laid out by the system), `_probabilisticDetector` might be null or uninitialized?
- **BUT** `ImprovedSwipeGestureRecognizer` falls back to "traditional detection" (simple bounds check) if probabilistic is missing.
- **However**, does `Keyboard2View` pass the key to `addPoint`?
- `Keyboard2View.onSwipeMove`:
  ```java
  KeyboardData.Key key = getKeyAtPosition(x, y);
  recognizer.addPoint(x, y, key);
  ```
- `getKeyAtPosition` uses `_keyboard` and `_tc`.
- If `_tc` (Theme.Computed) is slow to generate? It caches it.

### 2. The "6th Circle" Clue: Sample Count or Distance Accumulation?
- `SwipeGestureRecognizer` (or Improved) has buffers.
- Is there a buffer that needs to fill up?
- `ImprovedSwipeGestureRecognizer.addPoint`:
  ```java
    // Check if this should be considered swipe typing
    if (!_isSwipeTyping && _totalDistance > MIN_SWIPE_DISTANCE)
    {
      _isSwipeTyping = shouldConsiderSwipeTyping();
    }
  ```
- `shouldConsiderSwipeTyping` requires **2 alphabetic keys**.
- If you swipe 6 circles, you cover huge distance.
- **Why are keys not detected?**
- If `getKeyAtPosition` returns null.
- We fixed `_keyWidth`.
- **What else does `getKeyAtPosition` need?**
- `_config.marginTop`. `_marginLeft`.
- `_tc.row_height`.
- `_tc` is calculated in `onMeasure` (or my fix in `setKeyboard`).
- **Is it possible `_tc` calculation fails or returns bad values initially?**
- `Theme.Computed` constructor reads resources.
- If resources are not ready? Unlikely in Service.

### 3. `MotionEvent` coordinate reference frame?
- When `Keyboard2View` is first created/added, its (0,0) might be relative to... where?
- `event.getX()` and `getY()` are relative to the view.
- If the view is shifting/animating during the first few seconds (e.g. sliding up)?
- If the view is animating, `getKeyAtPosition` (static map) might mismatch the actual touch points relative to the moving view?
- **Correction**: `onStartInputView` usually slides the keyboard up.
- If the user starts swiping *while* it's sliding up?
- The touch events might be delivered, but coordinate mapping might be off?
- But `onLayout` should update positions.

### 4. The "Swipe Trail" Drawing Path
- `Keyboard2View.onDraw` calls `drawSwipeTrail`.
- `drawSwipeTrail` iterates `_swipeRecognizer.getSwipePath()`.
- `getSwipePath` returns `_smoothedPath`.
- `addPoint` adds to `_smoothedPath`.
- **Is `invalidate()` being called?**
- `Keyboard2View.onSwipeMove` calls `invalidate()`.
- `invalidate()` triggers `onDraw`.
- **If `invalidate()` is called but `onDraw` doesn't run?**
- This happens if the view is not visible or not attached.
- **Or if `onDraw` runs but `drawSwipeTrail` is skipped?**
- `if (_config.swipe_typing_enabled && ... && _swipeRecognizer.isSwipeTyping())`
- `isSwipeTyping` must be true.
- We know `isSwipeTyping` requires 2 keys.

**Deep Dive into `getKeyAtPosition` Logic:**
```java
  private KeyboardData.Key getKeyAtPosition(float tx, float ty)
  {
    KeyboardData.Row row = getRowAtPosition(ty);
    // ...
  }

  private KeyboardData.Row getRowAtPosition(float ty)
  {
    float y = _config.marginTop;
    // ... checks rows ...
  }
```
- `_config.marginTop` comes from resources.
- `_tc.row_height` comes from `Theme.Computed`.
- **If `_tc.row_height` is wrong?**
- `_tc.row_height` is calculated based on `_config.keyboardHeightPercent`.
- If `Config` hasn't loaded the correct height percent yet?
- `Config.refresh` loads prefs.
- **Wait, `PredictionViewSetup` sets dimensions on the NEURAL engine via listener.**
- `Keyboard2View` uses `_tc` which is updated in `onMeasure` and `setKeyboard`.
- **Is it possible `_tc` is using defaults (0) initially?**
- `setKeyboard` creates `_tc`.
- `Config` is passed.
- **What if `Config.screenHeightPixels` is 0?**
- `Config.refresh` sets `screenHeightPixels = dm.heightPixels`.
- `Theme.Computed` uses it.

### 5. The "6 Circle" Specifics
- 6 circles is a lot of time/distance.
- If it was 1 circle, I'd say "initialization".
- 6 circles suggests a **state change** or **buffer overflow** or **timeout**.
- **Loop Detector?**
- `SwipeGestureRecognizer` has `LoopGestureDetector`.
- `Improved` (which we use) does NOT seem to use it directly?
- Wait, `EnhancedSwipeGestureRecognizer` extends `Improved`.
- `Improved` logic:
  ```java
    // Check if this should be considered swipe typing
    if (!_isSwipeTyping && _totalDistance > MIN_SWIPE_DISTANCE)
    {
      _isSwipeTyping = shouldConsiderSwipeTyping();
    }
  ```
- `shouldConsiderSwipeTyping` iterates `_touchedKeys`.
- If `_touchedKeys` is empty (because `getKeyAtPosition` fails), it stays false.
- **Why does `getKeyAtPosition` start working after 6 circles?**
- Maybe because `onGlobalLayout` finally fires?
- Or `onMeasure` finally runs with `width > 0`?
- **My fix handled `onMeasure` receiving 0.**
- **BUT** did it handle `_keyWidth` being calculated *wrong* initially?
- In `setKeyboard`, I used `dm.widthPixels`.
- If `Keyboard2View` is only *half* the screen width (e.g. tablet, split keyboard)?
- Then `_keyWidth` (based on full screen) would be **huge**.
- If keys are huge, `getKeyAtPosition` might still find *some* key?
- Actually, if keys are huge, coordinate mapping would be way off.
- If you swipe on the *actual* view (small), but the logic thinks keys are huge (screen width)...
- `xLeft` would be huge. `tx` would be small.
- `tx < xLeft` check would fail immediately.
- Result: **No keys detected.**
- **This explains it.**
- `getKeyAtPosition` fails because my "fix" (using screen width) is wrong for the actual layout context?
- **Hypothesis:** The "Pre-calculated keyWidth" is incorrect because `dm.widthPixels` != `Keyboard2View.width`.
- **When does it get fixed?**
- When the *real* `onMeasure` runs with the *real* width.
- Why does that take 6 circles?
- Maybe `onMeasure` is delayed? Or maybe `onMeasure` is running with 0/wrong values repeatedly?

### The Fix for the Fix
I need a better estimate for `_keyWidth` than `dm.widthPixels`.
Or, I need to accept that `getKeyAtPosition` relies on `onMeasure`.
**But** `onMeasure` should run *before* the user can swipe 6 circles.
Unless... the view is `GONE` or `INVISIBLE`?
If `Keyboard2View` is added to `inputViewContainer`.
And `inputViewContainer` is waiting for something?

**Wait, `PredictionViewSetup` creates `inputViewContainer` (LinearLayout).**
It adds `Keyboard2View`.
Then `Keyboard2.setInputView` attaches it to the window.
The layout pass should happen immediately (next frame).
Why would it delay 3 seconds?

**Is it possible the `Config` itself is wrong?**
If `Config.orientation_landscape` is wrong initially?
`Keyboard2View` uses `_keyboard`. `_keyboard` is loaded based on orientation?
No, `setKeyboard` is passed a specific `KeyboardData`.

**Let's look at `Keyboard2View.onTouch`.**
It calls `getKeyAtPosition`.
I added debug logs there in `Keyboard2View.java`.
`android.util.Log.v("KeyDetection", "🎯 Touch at (" + tx + "," + ty + ") ...")`.
If I can see those logs, I'll know if it's running and what it's finding.

**Plan:**
1.  **Verify `Keyboard2View` logic again.**
2.  **Check `Config` refresh.**
3.  **Investigate `PredictionViewSetup` again.**

**One more thing:**
The user mentioned "app switch".
When switching apps, the keyboard might be in "extracted mode" or "fullscreen mode"?
`onEvaluateFullscreenMode` returns false.

**Wait.**
If `width` is `MeasureSpec.getSize(wSpec)`.
If `wSpec` is `EXACTLY 0`?
Then my fix keeps the `dm.widthPixels` version.
If `dm.widthPixels` is wrong (e.g. landscape vs portrait mismatch?), then keys aren't detected.
After 6 circles (~3s), maybe a rotation update or config refresh happens?
Or `onMeasure` finally gets the right width?

**Let's verify `dm.widthPixels` usage.**
In `setKeyboard`:
```java
        DisplayMetrics dm = getResources().getDisplayMetrics();
        int screenWidth = dm.widthPixels;
```
On app switch, orientation might change.
If `dm` reflects the *old* app's orientation?
But `getResources()` should be current.

**Let's assume the `_keyWidth` estimate is the culprit.**
If I remove the estimate, we are back to "ignored swipes" (width=0).
If I keep it, we have "lag" (bad keys).
The goal is to get the *correct* width ASAP.

**Idea:** `onSizeChanged`?
`onSizeChanged` is called when size changes.
`onMeasure` is called before that.

**Let's look for `onMeasure` logs.**
If I add logs to `onMeasure` to see what `width` it gets.

**Also:** `PredictionViewSetup`'s `OnGlobalLayoutListener`.
It sets `setNeuralKeyboardLayout`.
This updates the **Neural Engine's** key positions.
It does **NOT** update `Keyboard2View`'s key positions (used for `SwipeGestureRecognizer`).
`SwipeGestureRecognizer` uses `Keyboard2View`'s internal `_keyboard` and `_tc`.

**Wait!**
`ImprovedSwipeGestureRecognizer` has `setKeyboard(KeyboardData, width, height)`.
And `_probabilisticDetector`.
`Keyboard2View.setKeyboard` calls:
```java
    if (_swipeRecognizer != null && _keyboard != null)
    {
      // Parent class handles keyboard setup - no need for setKeyboardDimensions
    }
```
Wait, `Keyboard2View` doesn't call `setKeyboard` on the recognizer?
The code I read earlier:
```java
    // Set keyboard for swipe recognizer's probabilistic detection  
    if (_swipeRecognizer != null && _keyboard != null)
    {
      DisplayMetrics dm = getContext().getResources().getDisplayMetrics();
      // Parent class handles keyboard setup - no need for setKeyboardDimensions
    }
```
This comment suggests it does *nothing*.
So `ImprovedSwipeGestureRecognizer` **doesn't know the keyboard layout**?
If so, it uses **Traditional Detection**.
Traditional Detection uses `Keyboard2View.getKeyAtPosition` passing the key into `addPoint`.
So `Keyboard2View` determines the key.
So `_probabilisticDetector` is irrelevant (it's null).

So it comes back to `Keyboard2View.getKeyAtPosition`.
Which comes back to `_keyWidth`.

**If `dm.widthPixels` is wrong...**
Why would it be wrong?
Padding? `_marginLeft`?
In `setKeyboard`:
```java
        float marginLeft = Math.max(_config.horizontal_margin, _insets_left);
        // ...
        _keyWidth = (screenWidth - marginLeft - marginRight) / _keyboard.keysWidth;
```
`_insets_left` is updated in `onApplyWindowInsets`.
If `onApplyWindowInsets` hasn't run yet? `_insets_left` is 0.
If there *are* insets (cutout?), `_keyWidth` will be slightly off.
Slightly off width = keys shifted.
`getKeyAtPosition` might miss.

**But 6 circles?**
If you swipe *continuously*, `MotionEvent` stream continues.
Does `_swipeRecognizer` ever reset?
Only on `UP` or `CANCEL`.
If you never lift your finger, `isSwipeTyping` waits for 2 keys.
If `_keyWidth` is wrong, you hit 0 keys.
Then, suddenly, `onMeasure` runs (maybe due to a relayout triggered by something else?).
`_keyWidth` updates **mid-swipe**?
If `_keyWidth` updates, `_tc` updates.
Next `addPoint` calls `getKeyAtPosition`.
Now it works!
`touchedKeys` increments.
Threshold passed.
Trace appears.

**So `onMeasure` is delayed.**
Why?
Because `Keyboard2View` is inside `inputViewContainer`.
Maybe `inputViewContainer` (LinearLayout) is delaying measure?
Or `SuggestionBar` initialization is slow?

**Wait, I added `inputViewContainer.addView(keyboardView)` in `PredictionViewSetup`.**
If `inputViewContainer` is already attached, `addView` triggers layout.
If `SuggestionBar` takes time to load?
It shouldn't.

**Hypothesis:** The `DisplayMetrics` width is actually **correct enough** for 99% of cases.
The issue might be `_keyboard` itself.
Is `setKeyboard` called with the *wrong* layout initially?
`Keyboard2.onStartInputView` calls `current_layout()`.
`current_layout()` depends on `wide_screen` flag in `Config`.
If `Config.wide_screen` is wrong initially?
Then it loads `portrait` layout in `landscape` (or vice versa).
Widths would be wildly wrong.
Then, `onConfigurationChanged` fires?
Updating config -> `refresh_config`.
Updating `wide_screen`.
Updating `setKeyboard`.
Fixing the width.

**This fits the "delay" symptom.**
Configuration update lag.

**How to fix:** Ensure `Config` is up-to-date immediately in `onStartInputView`.
`onStartInputView` calls `refresh_config()`.
It calls `_configManager.refresh(getResources())`.
`Config.refresh(res)` updates `orientation_landscape` and `wide_screen`.
It *should* be correct.

**But... `getResources()` might return old configuration?**
In `InputMethodService`, `getResources()` should be updated.

**Let's verify `Keyboard2View.java`'s `setKeyboard` modification.**
I added the `_keyWidth` calculation.
Did I break something?
I checked `if (_keyWidth == 0)`.
If `_keyWidth` is already set (reused view), I skip calculation.
If reused view has old `_keyWidth` (e.g. from Portrait), and we are now Landscape?
`setKeyboard` is called *after* layout change?
`Keyboard2.onStartInputView` calls `_keyboardView.setKeyboard(current_layout())`.
If `current_layout()` changed (different object), `setKeyboard` runs.
`_keyWidth` is member variable.
It persists.
So `if (_keyWidth == 0)` is **false**.
We **skip** recalculation.
We rely on `onMeasure`.
If `onMeasure` is delayed...
We use the **OLD** `_keyWidth` (from Portrait).
On Landscape screen.
Keys are detected at Portrait positions (left side of screen).
You swipe on the right side. No keys detected.
After 6 circles, `onMeasure` finally runs.
Updates `_keyWidth`.
Trace appears.

**THE BUG IS IN MY FIX!**
I assumed `setKeyboard` meant "fresh start".
But `Keyboard2View` is reused.
If I switch orientation (or just app switch that triggers layout change), `_keyWidth` might need update.
But I only update it `if (_keyWidth == 0)`.
I should update it **always** if I want to rely on it before `onMeasure`.
**BUT**, `DisplayMetrics` might be unreliable if `onMeasure` hasn't run (e.g. multi-window mode).
However, `DisplayMetrics` is better than "Old Stale Value".

**Improved Fix:**
In `setKeyboard`:
Always calculate a "temporary" `_keyWidth` using `DisplayMetrics`, **even if** `_keyWidth` was non-zero.
Because if `setKeyboard` is called, the layout (`kw`) might have changed, so `_keyWidth` (pixels per key unit) *must* change if the number of keys changed or screen width changed.
Actually, `_keyWidth` depends on `width`.
If `width` changed, `_keyWidth` must change.
`setKeyboard` doesn't know if `width` changed.
But `DisplayMetrics` knows the screen width.

**Correct Logic:**
In `setKeyboard`:
Always estimate `_keyWidth` from `DisplayMetrics`.
Overwriting the old value.
Then `onMeasure` will refine it (handling margins/padding/multi-window correctly).
My previous `if (_keyWidth == 0)` check prevented the estimate from updating on orientation changes or view reuse!

**Action:**
Remove `if (_keyWidth == 0)` check in `setKeyboard`.
Always pre-calculate.

