# SMOOTHING_WINDOW=1 Test Results

**Date**: 2025-11-23
**Version**: v1.32.673
**Change**: SMOOTHING_WINDOW changed from 2 → 1

## Hypothesis

**User's insight**: Beam search pruning (line 2256 in OnnxSwipePredictor.java) filters out candidates with `confidence < 0.1`, causing 0 results for noisy trajectories.

**Test**: SMOOTHING_WINDOW=1 should produce **EVEN MORE points** than window=2:
- Window=3: Averages 3 points → moderate decimation
- Window=2: Averages 2 points → more points
- **Window=1: NO AVERAGING → maximum point density**

## Expected Outcomes

### Scenario A: More Points = Worse (Trajectory Noise Theory)
If trajectory noise is the problem:
- Window=1 produces MOST points
- MOST spurious key detections
- LOWEST encoder confidence
- MOST predictions pruned by confidence threshold
- **Result**: Even fewer successful predictions than window=2

### Scenario B: More Points = Better (Information Theory)
If more data helps encoder:
- Window=1 provides highest-resolution trajectory
- Encoder can learn from denser signal
- Better confidence scores
- Fewer predictions pruned
- **Result**: More successful predictions

### Scenario C: Beam Search Pruning is the Real Problem
If pruning threshold (0.1) is too aggressive:
- Regardless of window size, noisy trajectories produce low confidence
- All candidates get pruned before vocabulary filtering
- **Solution needed**: Lower confidence threshold OR improve trajectory quality

## Beam Search Pruning Analysis

**Found in OnnxSwipePredictor.java**:

### Line 63: Default Threshold
```java
private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
```

### Line 1994: Early Beam Pruning
```java
candidates.removeIf(beam -> Math.exp(-beam.score) < 1e-6); // Prune prob < 0.0001%
```

### Line 2256: Final Confidence Filter
```java
if (candidate.confidence >= _confidenceThreshold) // Must be >= 0.1
{
    words.add(candidate.word);
}
```

**This is why we see 0 predictions**: All candidates have confidence < 0.1 → filtered out → empty result.

## Test Plan

### Test Swipes (same as before):
1. "check" - short word (~60 points expected)
2. "empathy" - medium word (~180 points expected)
3. "oxidizing" - long word (~280 points expected)

### What to Compare:

| Metric | Window=2 | Window=1 | Better? |
|--------|----------|----------|---------|
| Points for "check" | 62 | ? | |
| Keys for "check" | 17 | ? | |
| Predictions for "check" | 2 | ? | |
| Points for long swipe | 261-279 | ? | |
| Keys for long swipe | 33-41 | ? | |
| Predictions for long swipe | 0 | ? | |

---

## Install Status

✅ **v1.32.673 installed with SMOOTHING_WINDOW=1**

Ready for test swipes!

---

## Alternative Fix: Lower Confidence Threshold

If SMOOTHING_WINDOW=1 doesn't help, the real solution may be:

```java
// Config.java or Settings
neural_confidence_threshold = 0.01f; // From 0.1 → allow lower confidence
```

**Trade-off**: May produce more incorrect predictions, but at least SOME predictions vs ZERO.
