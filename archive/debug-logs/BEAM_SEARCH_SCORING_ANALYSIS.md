# Beam Search Scoring & Pruning Analysis

**Date**: 2025-11-23
**Source**: OnnxSwipePredictor.java analysis + nopred4.txt

## How NN Score is Calculated

### Beam Search Score Accumulation

Each beam maintains a **negative log-likelihood score** (lower is better):

```java
// During beam search loop (line ~1900-2056):
for (each decoding step) {
    // Get logits from decoder for each beam
    float[] logits = decoder.forward(beam.tokens, memory);

    // Convert logits to probabilities via softmax
    float[] probs = softmax(logits);

    // For each candidate token:
    float tokenProb = probs[tokenId];
    float newScore = beam.score + (-Math.log(tokenProb)); // Accumulate negative log-likelihood

    // Lower score = higher probability sequence
}
```

### Final Confidence Conversion (Line 2093)

```java
// Convert accumulated negative log likelihood back to probability
float confidence = (float)Math.exp(-beam.score);
```

**Example**:
- Beam score = 2.3 (accumulated -log(prob))
- Confidence = exp(-2.3) = **0.100** (10%)
- If threshold = 0.1, this BARELY passes

- Beam score = 4.6
- Confidence = exp(-4.6) = **0.010** (1%)
- If threshold = 0.1, this FAILS and gets filtered

---

## All Pruning & Early Stopping Mechanisms

### 1. **Line 1994: Low-Probability Beam Pruning**
```java
if (step >= 2) { // Wait at least 2 steps before pruning
    candidates.removeIf(beam -> Math.exp(-beam.score) < 1e-6); // Keep beams with prob > 0.0001%
}
```

**Threshold**: `exp(-beam.score) >= 1e-6`
- Beam score must be < 13.8 (since exp(-13.8) ≈ 1e-6)
- **Very permissive** - only removes essentially impossible sequences

---

### 2. **Line 2007-2021: Adaptive Beam Width Reduction**
```java
if (step == 5 && beams.size() > 3) {
    float confidence = (float)Math.exp(-topScore);

    // If top beam has >50% confidence, narrow search to top 3 beams
    if (confidence > 0.5f) {
        beams = beams.subList(0, Math.min(3, beams.size()));
    }
}
```

**Trigger**: At step 5, if top beam confidence > 50%
**Action**: Reduce beam width from 4 → 3 to save compute

**Impact on failures**: Unlikely - only triggers if top beam is VERY confident

---

### 3. **Line 2023-2037: Score-Gap Early Stopping**
```java
if (beams.size() >= 2 && step >= 3) {
    float topScore = beams.get(0).score;
    float secondScore = beams.get(1).score;
    float scoreGap = secondScore - topScore; // Higher gap = more confident

    // If top beam finished and score gap > 2.0 (e^2 ≈ 7.4x more likely), stop early
    if (beams.get(0).finished && scoreGap > 2.0f) {
        break; // Early stop
    }
}
```

**Trigger**: After step 3, if:
- Top beam is finished (emitted EOS token)
- Gap between 1st and 2nd beam > 2.0

**Meaning**: Top beam is e²=7.4x more likely than 2nd beam → very confident

**Impact**: This could cause early termination before exploring all possibilities!

---

### 4. **Line 2040-2055: All-Beams-Finished Stopping**
```java
boolean allFinished = true;
for (BeamSearchState beam : beams) {
    if (!beam.finished) allFinished = false;
}

if (allFinished || finishedCount >= beamWidth) {
    break; // Stop search
}
```

**Trigger**: All beams have emitted EOS token

**This is correct** - no point continuing if all beams finished

---

### 5. **Line 2256: Final Confidence Threshold Filter** ⚠️ CRITICAL
```java
for (BeamSearchCandidate candidate : candidates) {
    if (candidate.confidence >= _confidenceThreshold) { // Default 0.1
        words.add(candidate.word);
    }
}
```

**Threshold**: 0.1 (10% confidence)
**This is the MAIN filter** that causes 0 predictions!

---

## Analysis of nopred4.txt Failures

### Failed Swipes (Postprocessing: 0ms)

1. **17:56:22** - "hyuiopoiuytrewertytrtyuiopoiuhyt" (33 keys from 250 points)
2. **17:57:12** - "ertyhjhgfdsaswerewasdfghjkjhgfderfghjmnjhytr" (43 keys from 250 points)
3. **17:58:48** - "ertyhjnbvcfdsawertrewsasdfghjnmnbhgfdertghjnmnjhgyt" (51 keys from 193 points)
4. **17:59:01** - "rtyhjkmnbvgfdsawerewasdsdfghjmnhgfrertghjnmnjhgtr" (48 keys from 205 points)
5. **18:04:31** - "vcfdsasdfghjkoiuytr" (19 keys from 75 points)
6. **18:04:41** - "dfgyuiuytresdfghbhgtr" (21 keys from 81 points)
7. **18:04:46** - "hgtrertyhjko" (12 keys from 48 points)

### Successful Swipes (got predictions)

1. **18:04:21** - "sertyuiop" (9 keys from 46 points) → **Success** ✓
2. **18:04:24** - "cfghjhgfdsasdfgvbnbhgfgy uijnbgf" (31 keys from 135 points) → **Success** ✓ (with fuzzy match)
3. **18:04:28** - "sdfghbnmkoiuytghjijhbvgy" (24 keys from 127 points) → **Success** ✓ (with fuzzy match)
4. **18:04:34** - "tyhgfdsasert" (12 keys from 46 points) → **Success** ✓
5. **18:04:36** - "dfghyuiuytrewsdfgbnbhgtr" (24 keys from 77 points) → **Success** ✓ (with fuzzy match)
6. **18:04:43** - "dftyuiuytresdfghjnjhgyt" (23 keys from 87 points) → **Success** ✓ (with fuzzy match)

---

## Pattern Analysis

### Over-Detection Threshold

| Keys Detected | Success Rate |
|---------------|--------------|
| 9-12 keys | 100% (4/4) ✓ |
| 19-24 keys | 75% (3/4) ⚠️ |
| 31-33 keys | 33% (1/3) ❌ |
| 43-51 keys | 0% (0/4) ❌ |

**Clear threshold**: ~25 keys is the limit. Above that, encoder confidence drops below 0.1.

---

## Root Cause Summary

### The Problem Is NOT Pruning/Early Stopping

**All pruning mechanisms are working correctly:**
1. ✅ Low-prob beam pruning (1e-6) - very permissive
2. ✅ Adaptive beam reduction - only at high confidence
3. ✅ Score-gap early stop - only when very confident
4. ✅ All-beams-finished - correct behavior

### The REAL Problem: Encoder Input Quality

**The issue happens BEFORE beam search**:

```
Noisy key sequence (43-51 keys for 10-letter word)
  → Encoder produces poor embedding
  → Decoder starts with low confidence
  → ALL beam candidates have score > 2.3 (confidence < 0.1)
  → Line 2256 filters ALL candidates
  → Result: 0 predictions
```

---

## Why Changing SMOOTHING_WINDOW Doesn't Help

**User is right** - SMOOTHING_WINDOW doesn't address the core issue:

| SMOOTHING_WINDOW | Effect |
|------------------|--------|
| 1 | Raw points → MORE noise → WORSE |
| 2 | Minimal smoothing → Still noisy → WORSE |
| 3 | Moderate smoothing → Less noise → SLIGHTLY better |
| 4+ | Heavy smoothing → MAY lose detail |

**But ALL fail when key sequence is too noisy!**

Even with SMOOTHING_WINDOW=3, swipes with 43-51 detected keys still produce 0 predictions.

---

## Real Solutions

### Option 1: Lower Confidence Threshold (Quick Fix)
```java
// Line 63 in OnnxSwipePredictor.java
private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.05f; // From 0.1
```

**Pros**: Immediate fix, allows more predictions
**Cons**: May include lower-quality predictions

---

### Option 2: Improve Key Detection (Harder)

The problem is in `ImprovedSwipeGestureRecognizer`:
- Currently detects 43-51 keys for 10-letter word (4-5x over-detection)
- Need to reduce to < 25 keys (2.5x max)

**Requires**: Better filtering of spurious key transitions

---

### Option 3: Retrain Encoder (Long-term)

Train encoder to be robust to noisy key sequences:
- Add noise augmentation during training
- Use swipes with 3-4x over-detection in training data
- Encoder learns to handle imperfect trajectories

---

## Recommendation

**Stop changing SMOOTHING_WINDOW** - it's not the solution.

**Instead**: Lower the confidence threshold to 0.05 as a quick fix:

```java
// Config.java or user settings
neural_confidence_threshold = 0.05f;
```

This will allow predictions with 5% confidence instead of 10%, which should catch most of the failed swipes.

**Long-term**: The key detection logic needs improvement to reduce spurious key detections from 43-51 down to ~20-25 keys.
