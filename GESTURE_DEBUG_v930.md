# Gesture Debug Report - v1.32.930

**Date**: 2025-11-27
**Version Installed**: v1.32.930
**Issue**: Backspace NW, Ctrl SW, and Fn gestures not working

---

## ❌ Problem Confirmed

Manual testing shows gestures on backspace, ctrl, and fn keys are **NOT working** even after installing v1.32.930 with the gesture fix.

---

## 🔍 Investigation Findings

### Code Analysis

The gesture detection logic in Pointers.kt:213-218 is **CORRECT**:

```kotlin
val isCharKey = ptr.value != null && ptr.value!!.getKind() == KeyValue.Kind.Char
val shouldBlockGesture = isCharKey && ptr.modifiers.size() > 0

if (_config.short_gestures_enabled && !ptr.hasLeftStartingKey &&
    swipePath != null && swipePath.size >= 1 &&
    !shouldBlockGesture  // Only blocks on CHAR keys with modifiers
) {
    // Allow gesture detection
}
```

### Key Type Verification (KeyValue.kt)

- **backspace**: `Kind.Keyevent` (line 571)
- **ctrl**: `Kind.Modifier` (line 464)
- **fn**: `Kind.Modifier` (line 491)

None of these are `Kind.Char`, so `isCharKey = false`, and gestures should **NOT** be blocked.

### Logcat Analysis

From `~/gesture-debug-live.log`:

```
11-27 22:50:24.500 D Pointers: Gesture classified as: TAP (hasLeftKey=false distance=0.0 time=91ms)
11-27 22:50:24.500 D Pointers: TAP path: short_gestures=true hasLeftKey=false pathSize=1 modifiers=0
11-27 22:50:24.500 D Pointers: Short gesture check: distance=0.0 minDistance=53.21485 (50% of 106.4297)
```

**Key Observations**:
1. ✅ `short_gestures=true` - Short gestures are enabled
2. ✅ `modifiers=0` - No modifiers active during gesture attempt
3. ❌ `distance=0.0` - **CRITICAL**: Gesture distance is ZERO

The swipe distance is 0.0 pixels, which is LESS than the minimum required distance (53.21px), so the gesture is rejected.

---

## 🐛 Root Cause Hypothesis

There are **TWO POSSIBLE ISSUES**:

### Hypothesis 1: Touch Event Collection Failure
The `swipePath` may not be collecting touch points properly. Looking at the logs, every gesture shows `pathSize=1`, which means only the DOWN event is being recorded, not the MOVE events.

**Code Location**: Pointers.kt:200-230 (gesture classification)

The path collection might be failing in `onTouchMove()` or the swipe recognizer.

### Hypothesis 2: onTouchMove() Not Being Called
The keyboard's touch handling may not be receiving `MotionEvent.ACTION_MOVE` events for some reason.

**Potential causes**:
- Android system intercepting touch events
- View touch handling configuration issue
- Event dispatching problem

---

## 🔧 Debugging Steps Needed

### Step 1: Check onTouchMove() Logging

Add more verbose logging to see if `onTouchMove()` is being called:

```kotlin
// In Pointers.kt onTouchMove()
Log.d("Pointers", "onTouchMove ENTRY: id=$id x=$x y=$y")
```

### Step 2: Check Swipe Path Collection

Verify that touch points are being added to the swipe path:

```kotlin
// After collecting swipe point
Log.d("Pointers", "Swipe path size after collection: ${swipePath?.size}")
```

### Step 3: Manual Testing Protocol

**IMPORTANT**: Test manually on the device (not via ADB):

1. Open any text input field
2. Press and **HOLD** on backspace key
3. While holding, **SWIPE** diagonally up-left (NW direction)
4. Release finger
5. Check if "delete last word" occurred

**Expected behavior**: Last word should be deleted

**Actual behavior**: ?

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Code logic | ✅ CORRECT | Should allow gestures on non-CHAR keys |
| Short gestures enabled | ✅ YES | Config confirms enabled |
| Key type detection | ✅ CORRECT | backspace/ctrl/fn are NOT Char type |
| Touch event collection | ❌ FAILING | distance=0.0, pathSize=1 |
| Actual gesture behavior | ❌ NOT WORKING | User confirmed |

---

## 🎯 Next Steps

###Option 1: Add Debug Logging

1. Edit `srcs/juloo.keyboard2/Pointers.kt`
2. Add detailed logging to `onTouchMove()` (around line 300-350)
3. Add logging to swipe path collection
4. Rebuild and test

### Option 2: Check Touch Handling in Keyboard2View

The touch event handling might be configured incorrectly in `Keyboard2View.kt`.

### Option 3: Test with Different Short Gesture Settings

Try adjusting the "Short Gesture Minimum Distance" setting:
- Settings → Unexpected Keyboard → Short gestures
- Try setting to lower value (e.g., 30% instead of 50%)

---

## 📝 User Testing Instructions

**CRITICAL**: The gestures **MUST** be tested manually, not via ADB.

ADB's `input swipe` command generates synthetic events that may not properly trigger the keyboard's touch event collection.

### Manual Test Procedure:

1. Open Messaging app or Notes app
2. Tap in text field to open keyboard
3. Type: "test word here"
4. **Test Backspace NW Gesture**:
   - Press and hold on backspace key (bottom-right)
   - While holding, swipe diagonally up and to the left
   - Release
   - **Expected**: Word "here" deleted
5. **Test Ctrl SW Gesture**:
   - Press and hold on ctrl key
   - While holding, swipe diagonally down and to the left
   - Release
   - **Expected**: Clipboard switcher opens

---

## 🔬 Logs Available for Analysis

- `~/gesture-debug-live.log` - Full gesture debugging log
- `~/kb-test.log` - Continuous keyboard monitoring
- `~/gesture-test-v923.log` - Previous v923 gesture tests

To search for specific events:
```bash
# Check gesture classifications
cat ~/gesture-debug-live.log | grep "Gesture classified"

# Check short gesture attempts
cat ~/gesture-debug-live.log | grep "Short gesture check"

# Check touch move events
cat ~/gesture-debug-live.log | grep "onTouchMove"
```

---

## 🚨 CRITICAL QUESTION

**Did the gestures work in an earlier version (before v1.32.925)?**

If YES: When did they stop working?
- v1.32.921? ✅ (before shift+c fix)
- v1.32.923? (first attempt at fix)
- v1.32.925? (refined fix for shift+c)
- v1.32.927? (shift+swipe ALL CAPS feature)
- v1.32.929? (gesture regression fix attempt)

This will help determine if the issue is:
1. Introduced by the v1.32.925 fix (regression)
2. A pre-existing issue that was never working
3. A device-specific or configuration issue

---

**Report Generated**: 2025-11-27 22:51
**Next Action Required**: User manual testing to verify gesture behavior
