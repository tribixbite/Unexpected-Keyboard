# SMOOTHING_WINDOW Analysis - Keep at 2 or Revert to 3?

**Date**: 2025-11-23
**Version**: v1.32.671
**Status**: Awaiting user decision

## Summary

User requested changing SMOOTHING_WINDOW from 3 → 2 to test impact. Results show **increased point density causes prediction failures**.

## Test Results with SMOOTHING_WINDOW=2

### ✅ Short Swipes Work Well
```
✅ "log" (34 points)   → 7 keys  → Prediction: "log" (0.646 confidence)
✅ "check" (62 points) → 17 keys → Prediction: "check" (0.625 confidence)
```

### ❌ Long Swipes Fail Completely
```
❌ 261 points (resampled→250) → 33 keys → 0 candidates (too noisy)
❌ 279 points (resampled→250) → 41 keys → 0 candidates (too noisy)
```

## Root Cause: Over-Detection of Keys

| Swipe Length | Keys Detected | Typical Word | Over-detection |
|--------------|---------------|--------------|----------------|
| 34 pts | 7 keys | 3 letters | 2.3x |
| 62 pts | 17 keys | 5 letters | 3.4x |
| 261 pts | 33 keys | 8 letters | **4.1x** |
| 279 pts | 41 keys | 10 letters | **4.1x** |

**Problem**: Encoder trained on cleaner trajectories - noisy sequences produce 0.0000 confidence.

---

## Option 1: REVERT to SMOOTHING_WINDOW=3 (RECOMMENDED)

### Pros:
- ✅ Proven to work (previous versions used this)
- ✅ Fewer points → cleaner key sequences
- ✅ Better encoder confidence
- ✅ More successful predictions
- ✅ NO code changes needed (just revert constant)

### Cons:
- ⚠️ Slightly more smoothing (but velocity/accel already smooth!)
- ⚠️ User originally wanted to test window=2

### Implementation:
```java
// ImprovedSwipeGestureRecognizer.java:34
private static final int SMOOTHING_WINDOW = 3; // Revert from 2
```

**Build and test**: 1 line change, ~60s build time

---

## Option 2: KEEP SMOOTHING_WINDOW=2, Tune Detection

### Pros:
- ✅ Honors user's original request to test window=2
- ✅ More granular trajectory data (higher resolution)
- ✅ May improve accuracy for very short swipes

### Cons:
- ❌ Requires tuning multiple thresholds
- ❌ Unknown if thresholds can fully compensate
- ❌ More testing iterations needed
- ❌ May not fully solve noise problem

### Implementation:
```java
// ImprovedSwipeGestureRecognizer.java
private static final float MIN_KEY_DISTANCE = 50.0f; // From 30.0f
private static final long MIN_DWELL_TIME_MS = 20; // From 10ms
private static final float HIGH_VELOCITY_THRESHOLD = 800.0f; // From 1000.0f
```

**Testing required**: Build → test → adjust → repeat

---

## Option 3: KEEP SMOOTHING_WINDOW=2, Add Noise Filter

### Pros:
- ✅ Addresses root cause (noisy key sequences)
- ✅ Could improve ALL swipe quality
- ✅ Retains high-resolution trajectory data

### Cons:
- ❌ Requires new filtering algorithm
- ❌ Development time: ~30-60 minutes
- ❌ Risk of over-filtering (removing valid keys)
- ❌ More testing needed

### Implementation:
New `filterNoisyKeys()` method to remove:
- Zigzag patterns (A→B→A)
- Very short dwell times
- Keys detected during high-velocity passes

**Complexity**: Medium - requires algorithm design + testing

---

## Recommendation

### ⭐ REVERT to SMOOTHING_WINDOW=3

**Rationale**:
1. **Proven solution** - worked in previous versions
2. **Immediate fix** - 1 line change, 60s build
3. **Low risk** - reverting to known-good state
4. **User can test** - compare before/after immediately

**Velocity/acceleration already provide smoothing** - the moving average window may be redundant.

### Alternative: Test Both

1. Build with SMOOTHING_WINDOW=3 (call it v1.32.672)
2. Do 10 test swipes, record success rate
3. Compare to SMOOTHING_WINDOW=2 results
4. User decides based on empirical data

---

## Key Insight

**SMOOTHING_WINDOW doesn't reduce point count - it smooths coordinates!**

The original assumption "less smoothing = less computation" is wrong:
- Both window=2 and window=3 collect same raw points
- Window affects coordinate accuracy, not count
- Less smoothing → **noisier** trajectories → worse predictions

**The "computation savings" are negligible compared to prediction failures.**

---

## User Decision Required

@User: What do you want to do?

1. ✅ **REVERT to SMOOTHING_WINDOW=3** (recommended - quick fix)
2. ⚠️ **TUNE thresholds with SMOOTHING_WINDOW=2** (experimental)
3. 🔧 **ADD noise filter with SMOOTHING_WINDOW=2** (development work)
4. 📊 **TEST BOTH and compare** (data-driven decision)

Let me know and I'll proceed with the chosen option.
