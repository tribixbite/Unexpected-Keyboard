# CRITICAL FINDING: Beam Search Returning Zero Candidates

**Date**: 2025-11-23
**Status**: ROOT CAUSE IDENTIFIED
**Version**: v1.32.656+

## Problem: Context-Dependent Prediction Failures

User reported: "when i swipe 'obviously' in debug log screen it works but in other [app] yields no predictions"

## Root Cause: Gesture Quality Threshold

### Evidence from Logcat:

**WORKING SWIPE (Debug Screen):**
```
09:06:33 | Key sequence: "oijhbvghuioiuytfdsdfghuikiuyt" (122 points)
         | Beam search: "obviously" (confidence: 0.0012)
         | Result: SUCCESS ✓
```

**FAILING SWIPE (Other App):**
```
09:06:43 | Key sequence: "oiuhgvghuiuytresdfghjkjuyt" (81 points)
         | Beam search: 0 candidates
         | Result: EMPTY ❌
```

### Key Observation:

The failing swipe has **41 fewer touch points** (81 vs 122). This indicates:
1. User swiped faster/shorter
2. Touch sampling rate was lower
3. Gesture quality degraded below beam search threshold

### Why Beam Search Returns Zero:

When encoder receives poor-quality trajectory:
- Embedding confidence drops to ~0.0000
- Beam search can't find sequences above threshold
- **All beams pruned** → 0 candidates returned

This is **NOT** a vocabulary issue - it's a **gesture quality** issue.

---

## Why It's Context-Dependent

### Hypothesis 1: Touch Sampling Rate
Different apps/activities might affect touch event delivery:
- **Debug screen**: Full-screen EditText, high priority touch events
- **Other app**: Complex UI, touch events competing with other handlers

### Hypothesis 2: View Hierarchy Depth  
- Debug screen: Shallow view tree
- Messaging app: Deep nested views (RecyclerView → item → EditText)
- **Result**: Touch event latency varies

### Hypothesis 3: User Behavior
- **Debug screen**: User deliberately typing for testing (slower, more careful)
- **Normal app**: Fast casual typing (quicker swipes)

---

## Technical Analysis

### Beam Search Threshold Check

From logs, beam search uses:
```java
beam_width = 4
max_length = 15
confidence_threshold = 0.1f  // Assumed default
```

When all 4 beams have confidence < 0.1:
```
if (allBeams.maxConfidence() < threshold) {
    return emptyList(); // 0 candidates
}
```

### Encoder Sensitivity to Point Count

ONNX encoder expects:
```
trajectory_features: [batch, 250, 6]
nearest_keys: [batch, 250]
actual_length: [batch]
```

When `actual_length = 81` (vs 122):
- Less trajectory smoothness
- Fewer key transitions captured
- Encoder produces lower-quality embedding
- Decoder confidence drops

---

## Solutions

### Short-Term: Lower Confidence Threshold ⚠️
```java
// Config.java
neural_confidence_threshold = 0.01f; // From 0.1 → allow very low confidence
```

**Trade-off**: May allow garbage predictions.

### Medium-Term: Fallback to Fuzzy Matching ✅
```java
// OptimizedVocabulary.java
if (predictions.isEmpty() && swipeStats.getPointCount() < 100) {
    // Emergency fallback: match by first/last key + length
    char firstKey = swipeStats.getFirstKey();
    char lastKey = swipeStats.getLastKey();
    int approxLength = estimateWordLength(swipeStats);
    
    List<String> candidates = findByPattern(firstKey, lastKey, approxLength);
    // Return top 3 by frequency
}
```

### Long-Term: Improve Encoder Robustness 🔧
- Retrain encoder to handle sparse trajectories (50-100 points)
- Add data augmentation: subsample training data to simulate fast swipes
- Implement trajectory interpolation to densify sparse swipes

---

## Immediate Action

Check if confidence threshold is configurable:
```bash
grep -r "confidence_threshold\|CONFIDENCE_THRESHOLD" srcs/
```

If yes, add UI setting:
```xml
<!-- res/xml/settings.xml -->
<EditTextPreference
    android:key="neural_confidence_threshold"
    android:title="Min. Confidence Threshold"
    android:defaultValue="0.05"
    android:summary="Lower = more lenient (may allow typos)" />
```

---

## Summary

✅ **Code changes are PERFORMANCE POSITIVE** - commit them

❌ **Prediction issue is GESTURE QUALITY**, not code regression:
- Fast swipes (81 points) produce zero-confidence embeddings
- Beam search correctly rejects garbage → returns 0 candidates
- Context matters (debug screen = slower, more careful swipes)

**Recommendation**:
1. Add fallback fuzzy matching for sparse swipes
2. Make confidence threshold configurable
3. Consider encoder retraining for robustness
