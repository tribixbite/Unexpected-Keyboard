# Analysis: overzealous & genealogy Prediction Failures

**Date**: 2025-11-23
**Version**: v1.32.673 (SMOOTHING_WINDOW=1)
**Issue**: No predictions for "overzealous" and "genealogy" despite words being in vocabulary

## Test Results

### Swipe 1: "overzealous" (11 letters)
```
Points collected: 198
Detected keys: "ijhgvcfrertrdszsewasdfghjkloiuytrds" (36 keys)
Over-detection ratio: 36/11 = 3.3x
Beam search: Ran for 100ms
Predictions: 0 (Postprocessing: 0ms)
```

### Swipe 2: "genealogy" (9 letters)
```
Points collected: 161
Detected keys: "gfrerfghbnbhgfdrertyuioklkoiuhgfgy" (34 keys)
Over-detection ratio: 34/9 = 3.8x
Beam search: Ran for 91ms
Predictions: 0 (Postprocessing: 0ms)
```

## Root Cause: Trajectory Noise → Confidence Pruning

### The Problem Chain

1. **SMOOTHING_WINDOW=1** → Raw touch points (no averaging)
2. **Too many points** → More key transitions detected
3. **Noisy key sequences** → 36-34 keys for 9-11 letter words (3-4x over-detection)
4. **Encoder confusion** → Low-confidence embeddings produced
5. **Beam search pruning** → All candidates have confidence < 0.1
6. **Line 2256 filter** → `if (candidate.confidence >= 0.1)` rejects ALL candidates
7. **Result**: Empty predictions (Postprocessing: 0ms)

### Evidence

**No "Raw NN Beam Search" logs** = All candidates filtered by confidence threshold before logging.

This is **NOT a vocabulary issue** - both words are present:
```json
"genealogy": 177,
"overzealous": 137
```

This is **NOT an encoder failure** - beam search ran successfully (91-100ms).

This is **beam search confidence pruning** - all candidates below 0.1 threshold.

---

## Solution Options

### Option 1: Increase Key Detection Thresholds ✅ RECOMMENDED

**Problem**: Too many spurious keys detected during fast swipes.

**Fix**: Make key registration more strict in `ImprovedSwipeGestureRecognizer.java`:

```java
// Current (too lenient):
private static final float MIN_KEY_DISTANCE = 30.0f;
private static final long MIN_DWELL_TIME_MS = 10;
private static final float HIGH_VELOCITY_THRESHOLD = 1000.0f;

// Proposed (more strict):
private static final float MIN_KEY_DISTANCE = 45.0f;  // From 30.0 → require 50% more movement
private static final long MIN_DWELL_TIME_MS = 15;    // From 10ms → require 50% longer dwell
private static final float HIGH_VELOCITY_THRESHOLD = 750.0f; // From 1000 → filter faster passes
```

**Expected impact**:
- Fewer spurious key detections (36 keys → ~20 keys for 11-letter word)
- Cleaner trajectories → better encoder confidence
- Predictions above 0.1 threshold → results appear

**Risk**: May miss some valid keys on very fast swipes.

---

### Option 2: Lower Confidence Threshold

**Fix**: Allow lower-confidence predictions in `OnnxSwipePredictor.java`:

```java
// Line 63
private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.05f; // From 0.1 → 0.05
```

**Expected impact**:
- More predictions pass threshold
- May include more incorrect guesses

**Risk**: Lower quality predictions, possible garbage results.

---

### Option 3: Revert SMOOTHING_WINDOW to 3

**Fix**: Go back to known-good smoothing:

```java
// ImprovedSwipeGestureRecognizer.java:34
private static final int SMOOTHING_WINDOW = 3; // From 1 → 3
```

**Expected impact**:
- Fewer points collected (198 → ~140, 161 → ~115)
- Cleaner trajectories from averaging
- Fewer key detections
- Better encoder confidence

**Risk**: May lose some trajectory detail.

---

### Option 4: Hybrid Approach (BEST)

Combine Options 1 + 3:

1. **Revert SMOOTHING_WINDOW to 3** (proven to work)
2. **Slightly increase MIN_KEY_DISTANCE** to 40.0f (moderate tightening)
3. **Keep MIN_DWELL_TIME_MS at 10** (don't penalize fast swipes too much)

**Expected impact**:
- Cleaner trajectories from smoothing
- Fewer false key detections from higher distance threshold
- Best balance of quality and responsiveness

---

## Comparison: SMOOTHING_WINDOW Impact

| Window | Points (overzealous) | Keys Detected | Over-detection | Predictions |
|--------|---------------------|---------------|----------------|-------------|
| 1 (current) | 198 | 36 | 3.3x | 0 ❌ |
| 2 (previous) | 261-279 | 33-41 | 3-4x | 0 ❌ |
| 3 (original) | ~140 est | ~20 est | 2x est | ? ✓ |

**Hypothesis**: Window=3 with 2x over-detection is within encoder's tolerance, but 3-4x is too noisy.

---

## Recommended Action

**REVERT SMOOTHING_WINDOW to 3** + moderate key detection tuning:

```java
// ImprovedSwipeGestureRecognizer.java
private static final int SMOOTHING_WINDOW = 3;
private static final float MIN_KEY_DISTANCE = 40.0f; // From 30.0
```

**Why**:
- Window=1 and Window=2 both produce too many points → too noisy
- Window=3 is proven (previous builds worked)
- Modest MIN_KEY_DISTANCE increase reduces noise without breaking fast swipes

---

## Test Plan

After implementing fix:
1. Swipe "overzealous" → expect 2-3 predictions
2. Swipe "genealogy" → expect 2-3 predictions
3. Swipe "check" → expect 2-3 predictions (verify short words still work)
4. Swipe "obviously" → expect 2-3 predictions (verify medium words)

**Success criteria**: All swipes produce at least 1 prediction with confidence > 0.1.
