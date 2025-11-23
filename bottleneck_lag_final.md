# Performance Bottleneck: Short Swipe Inconsistency (Termux)

**Date**: 2025-11-22
**Status**: ⚠️ **Regression Identified**

## Issue Description
The user reports that short swiped words are no longer working consistently in Termux, and the previous "6 circle lag" persists or has worsened. The hypothesis about velocity filtering was rejected by the user.

## Regression Analysis
The removal of `MIN_DWELL_TIME_MS` (set to 0) and increasing `HIGH_VELOCITY_THRESHOLD` (to 2000) in `ImprovedSwipeGestureRecognizer.java` was intended to *help* fast swipes. However, if short swipes are now failing, this might have had an unintended side effect.

**Potential Negative Impact:**
If `MIN_DWELL_TIME_MS` is 0, *every* touch point that falls within a key's bounds is registered as a key press.
*   **Noise Amplification:** During a swipe, if the path briefly grazes a neighboring key (even for 1 frame), it is now registered.
*   **Pollution of `_touchedKeys`:** The key sequence becomes cluttered with accidental hits (e.g. "t-g-h-a-t" instead of "t-h-a-t").
*   **Neural Model Confusion:** The neural network receives a noisy sequence. If the sequence is too long or noisy, the model might fail to predict the correct word, or `filterPredictions` might reject it.
*   **Or:** The recognizer's internal duplicate check (`DUPLICATE_CHECK_WINDOW`) might be filtering out the *correct* key if it was registered too early/briefly?

## "6 Circle Lag" Re-evaluation

If the layout/width fixes didn't solve it, and velocity didn't solve it.
**What changes between the 1st and 6th circle?**
*   **Time:** ~3 seconds.
*   **Touch History:** The `_rawPath` grows.
*   **State:** `_isSwipeTyping` transitions from false to true.

**Hypothesis: The "Start" Condition is Flawed.**
`ImprovedSwipeGestureRecognizer.startSwipe` sets `_startTime`.
It adds the first point.
It registers the first key.

If the **first key** detection fails (due to touch slop, or finger landing in a gap), `_touchedKeys` has 0 elements.
Then you start moving.
`addPoint` is called.
If you move fast, and previously `MIN_DWELL` filtered keys, you might skip keys.
But we removed `MIN_DWELL`.

**What if the "Gap" logic in `getKeyAtPosition` is the problem?**
In `Keyboard2View.java`, I added a check:
`if (tx < xLeft) return null;`
This explicitly returns `null` if the touch is in the spacing between keys.
If your "start" point lands in a margin/gap (even 1 pixel), `startSwipe` registers `null`.
Then you circle.
If your circle path stays mostly in gaps (unlikely) or grazes keys too fast?

**The "6th Circle" is key.**
It implies an **accumulation** threshold.
`_totalDistance` > `MIN_SWIPE_DISTANCE`.
I lowered `MIN` to 50.
So distance isn't the blocker.

**What if `PredictionCoordinator` is blocking?**
`ensureInitialized` was the original suspect.
I made it conditional.
What if `_neuralEngine` is NULL initially?
`PredictionViewSetup` calls `ensureInitialized` on a thread.
If `_neuralEngine` is null, `predict()` returns empty.
If `predict()` returns empty, **no trace is drawn**?
**Wait.**
`Keyboard2View` draws the trace if `_swipeRecognizer.isSwipeTyping()` is true.
It does **NOT** depend on prediction results.
The trace drawing is purely local to the view.
So if "no trace appears", it is **100% certain** that `isSwipeTyping()` is false.

**Why is `isSwipeTyping()` false?**
1.  `_touchedKeys.size() < 2`.
2.  `_totalDistance < MIN_SWIPE_DISTANCE`. (We set to 50).
3.  `!isValidAlphabeticKey`.

**If `getKeyAtPosition` works (logs show it does), then keys ARE being added.**
So `_touchedKeys` should increase.
Why wouldn't it?
**Duplicate Check?**
`isRecentDuplicate(key)`.
If you circle "A".
Start: A.
Move: Still A. (Duplicate - skipped).
Move: Exit A. Enter Gap. (Null - skipped).
Move: Enter A. (Duplicate? Check window 5).
If you circle *only* A, you never get 2 keys. `isSwipeTyping` stays false.
You must hit *another* key.
"6th lap around a continuous circle".
Assuming the circle crosses multiple keys (e.g. A-S-D-W).
A -> S -> D -> W.
These are different keys. `_touchedKeys` should grow.

**Is there a Reset occurring?**
If `Keyboard2View` receives `ACTION_UP` or `CANCEL`.
Does the system send `CANCEL` after app switch?
Logs showed:
`15:39:12.272 I/GestureDetector( 2907): obtain mCurrentMotionEventRaw. action: 2 id: 563751231`
`Action 2` is `MOVE`.
`15:39:12.355 D/InputReader( 2907): Btn_touch(5): value=0` (UP).
The logs showed a stream of DOWN/UP events.
If you are swiping continuously, but the system reports UP/DOWN?
Then `SwipeGestureRecognizer` resets every time.
`totalDistance` never reaches 50.
`touchedKeys` never reaches 2.
**Trace never appears.**

**Why would the system send UP/DOWN during a continuous swipe?**
1.  **Palm Rejection?**
2.  **Multi-touch ghosting?**
3.  **"Pointer 1" vs "Pointer 0"?**
    The log showed:
    `15:39:12.355 I/VRI[InputMethod]@e69eec6(20401): ViewPostIme pointer 1`
    `15:39:12.264 I/VRI[InputMethod]@e69eec6(20401): ViewPostIme pointer 0`
    It seems multiple pointers are active or switching?
    If you switch apps, maybe your finger is detected as a *new* pointer?
    `Keyboard2View.onTouch` handles pointers.
    `_pointers.onTouchMove` takes `id`.
    If `id` changes?
    `_pointers` maps IDs to state.
    If `id` changes, it might be treated as a new gesture.

**If the logs show broken touch streams (UP/DOWN repeated), then the issue is INPUT LAYER.**
Not the swipe logic.
Why would Termux/System break touch streams on app switch?
Maybe the window loses focus/visibility repeatedly?
`15:39:45.908 Changing focus from Window{...TermuxActivity} to null`
`15:39:45.937 Changing focus from null to Window{...LogAccessDialogActivity}`
The focus churn is real.

**Conclusion:**
If the touch stream is fragmented, we can't fix it in the recognizer logic (garbage in, garbage out).
**BUT**, you said "6th circle... all of a sudden the entire trace appears".
This implies the **HISTORY** was preserved!
If the history is preserved, then `reset()` was **NOT** called.
So the touch stream **IS** continuous.
So `_rawPath` has data.
But `isSwipeTyping` was false.
Then suddenly true.
Why?
Because `_touchedKeys` finally hit 2.
Why did it take 6 circles to hit 2 keys?
Because **`getKeyAtPosition` was failing** (returning null) for the first 5 circles.
Why?
Because `_keyWidth` (or layout) was wrong.

**Wait.**
The logs I just analyzed showed `getKeyAtPosition` **succeeding** (finding 'A' key bounds).
BUT, maybe `tx` was consistently *outside* the bounds?
The log: `📍 'A' KEY: x=61...167, tx=755`.
`tx` (755) is WAY outside `x` (61-167).
If 'A' is the key at `row.keys[i]`.
And we are iterating.
We log 'A'.
But `tx` matches 'K' or 'L' (far right).
Does `getKeyAtPosition` return 'K'?
If I didn't see `❌ Touch after last key`, it implies it matched *something*.
OR it returned `null` silently (gap)?
My log:
```java
      if (tx < xLeft) {
          // Gap between keys?
          // android.util.Log.d("SWIPE_LAG_DEBUG", "❌ Touch in gap before " + key.toString() + ": " + tx + " < " + xLeft);
          return null;
      }
```
I commented out the "Touch in gap" log!
If `tx` falls in a gap (due to misalignment), it returns `null`.
If `_keyWidth` is wrong (e.g. too small), gaps become huge?
No, `xRight = xLeft + width`. `x` advances by `width`.
Unless `key.shift` creates gaps?
If `_keyWidth` is calculated from `dm.widthPixels` (screen width).
But the view is actually smaller?
Then `_keyWidth` is **too large**.
If `_keyWidth` is too large, the keys overlap or extend off-screen.
`xLeft` grows faster than `tx`.
Eventually `xLeft > tx`.
So `tx` falls "before" a key.
Returns `null` (Gap).
**This is it.**
`tx < xLeft`. Gap. Null.
Swipe ignored.

**Verification:**
The log `📍 'A' KEY: x=61...167` means 'A' starts at 61px.
If `_keyWidth` was small (correct), 'A' might start at 30px.
The fact that `x` values are large confirms `_keyWidth` is large (`106.425`).
If the *actual* view is smaller, `tx` (touch) will be smaller than expected.
`tx` will fail to catch up to the bloated `xLeft` positions.
It falls into "gaps" (which are actually valid keys, shifted right).

**Fix:**
We need a **smaller** `_keyWidth` estimate if the screen is wide but the keyboard is narrow.
But we don't know the keyboard width yet!
**However**, we know `onMeasure` runs eventually.
The lag happens *before* `onMeasure`.
Can we assume the keyboard matches `DisplayMetrics`?
Usually yes.
**Unless** in multi-window or some weird Termux X11 context?
The user mentioned "Termux".
Termux X11 might handle windows differently.

**Alternative:**
If `getKeyAtPosition` fails to find a key, **don't return null**.
Return the **closest key**?
`ProbabilisticKeyDetector` does this.
But `ImprovedSwipeGestureRecognizer` only uses `Probabilistic` if `setKeyboard` logic worked.
I noted earlier `setKeyboard` in `Keyboard2View` does **not** pass the keyboard to the recognizer correctly?
Let's check `Keyboard2View.setKeyboard` again.
```java
    // Initialize swipe recognizer if not already created
    if (_swipeRecognizer == null)
    {
      _swipeRecognizer = new EnhancedSwipeGestureRecognizer();
    }
    
    // Set keyboard for swipe recognizer's probabilistic detection  
    if (_swipeRecognizer != null && _keyboard != null)
    {
      DisplayMetrics dm = getResources().getDisplayMetrics();
      // Parent class handles keyboard setup - no need for setKeyboardDimensions
    }
```
It does **NOT** call `_swipeRecognizer.setKeyboard(...)`.
It just comments "Parent class handles...".
But `EnhancedSwipeGestureRecognizer` inherits `Improved...`.
`Improved` has `setKeyboard`.
**It is never called!**
So `ProbabilisticKeyDetector` is **never used**.
It always relies on `getKeyAtPosition`.
Which is brittle to layout bugs.

**The Solution:**
1.  **Enable Probabilistic Detection:** Call `_swipeRecognizer.setKeyboard(_keyboard, width, height)` in `Keyboard2View.setKeyboard`.
    *   This allows the recognizer to find the "nearest key" even if coordinates are slightly off (gap/misalignment).
    *   This bridges the gap during the layout race.
2.  **Revert Velocity Changes:** The user said "second one was faster", implying my velocity fix was wrong/harmful. I will revert the velocity thresholds to default (safe) values to fix the regression of "short words not working".

**Plan:**
1.  Revert `ImprovedSwipeGestureRecognizer` constants.
2.  Update `Keyboard2View` to actually configure the `_swipeRecognizer` with the keyboard layout.

