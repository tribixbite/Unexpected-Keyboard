# Resampling Debug Test Plan

**Version**: v1.32.671
**Date**: 2025-11-23
**Status**: Ready for testing

## Test Build Installed

Debug logging added to trace resampling execution in `SwipeTrajectoryProcessor.java`.

## What to Test

Perform the following swipes to generate different point counts:

1. **Short swipe**: "iffy" (expect ~81 points, no resampling needed)
2. **Medium swipe**: "empathy" (expect ~182 points, no resampling needed)
3. **Long swipe**: "oxidizing" (expect ~283 points, **SHOULD trigger resampling**)

## What We're Looking For

The logcat monitor is running and will capture:

### 1. Pre-Resampling Check (Line 133-136)
```
🔍 Resampling check: size=X, max=250, mode=DISCARD, needsResample=true/false
```

**Expected for "oxidizing" (283 points)**:
```
🔍 Resampling check: size=283, max=250, mode=DISCARD, needsResample=true
```

### 2. Resampling Execution (Line 174-175)
```
🔄 Resampled trajectory: X → 250 points (mode: DISCARD)
```

**Should appear** if resampling works correctly.

### 3. Resampling Verification (Line 178-182)
```
❌ RESAMPLING FAILED! Still have X points after resampling, expected max 250
```

**Should NOT appear** if resampling works correctly.

## Current Hypothesis

**If we see "🔍 Resampling check" with needsResample=false**:
- The condition `size > maxSequenceLength && mode != TRUNCATE` is not matching
- Possible causes:
  - `size` is not > 250 (but logs show 283)
  - `mode` is TRUNCATE (but line 37 sets DISCARD)
  - Logic error in condition

**If we DON'T see any "🔍" logs at all**:
- The code path isn't being reached
- Processing might be failing earlier
- Exception might be thrown

**If we see "🔍" but no "🔄"**:
- Condition evaluated to false
- OR exception thrown inside if block
- OR SwipeResampler.resample() is failing silently

## How to Capture Results

Logcat is being monitored in background (shell ID: ebaa11).

After swiping, check output:
```bash
# Check background monitoring output
# (Will be available via BashOutput tool)
```

Or manually:
```bash
adb logcat -d | grep -E "🔍 Resampling|🔄 Resampled|❌ RESAMPLING|SwipeTrajectoryProcessor"
```

## Next Steps After Testing

1. If resampling IS working:
   - Investigate why "empathy" (182 points) fails despite being < 250
   - Focus on trajectory noise/quality

2. If resampling NOT working:
   - Fix the DISCARD implementation
   - Add emergency fallback to guarantee resampling

---

**Ready for testing!** Please swipe "iffy", "empathy", and "oxidizing" to generate the debug logs.
