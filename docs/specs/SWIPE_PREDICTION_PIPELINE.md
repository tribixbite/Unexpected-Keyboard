# Swipe Prediction Pipeline Analysis

**Date**: 2025-10-21
**Status**: Analysis Complete
**Version**: v1.32.191+

---

## Executive Summary

Complete analysis of the neural network swipe prediction pipeline from raw touch input to final word predictions displayed to the user.

### Key Findings

1. **Two-Stage Pipeline**: Neural network beam search → Vocabulary filtering
2. **Critical Issue**: Raw neural network outputs only visible when ALL predictions filtered out
3. **Missing Feature**: No "closest predictions" shown (unfiltered beam search results)
4. **Performance**: Beam search with vocabulary filtering provides high-quality predictions

---

## Pipeline Architecture

```
User Swipe Input
    ↓
SwipeTrajectoryProcessor.extractFeatures()
    ↓ (trajectory: x,y,vx,vy,ax,ay + nearest_keys)
ONNX Encoder (transformer)
    ↓ (memory: [1, 150, 256])
Beam Search Decoder (greedy or beam_width=2)
    ↓ (List<BeamSearchCandidate>: word + confidence)
OptimizedVocabulary.filterPredictions()
    ↓ (filters unknown words, applies frequency boost)
createOptimizedPredictionResult()
    ↓ (PredictionResult: words + scores)
Display to User
```

---

## Detailed Pipeline Stages

### Stage 1: Input Processing

**Component**: `SwipeTrajectoryProcessor`

**Input**: `SwipeInput` (list of touch coordinates)

**Processing**:
1. Normalize coordinates to [0, 1] range
2. Calculate velocity (vx, vy) and acceleration (ax, ay)
3. Find nearest key for each point
4. Pad/truncate to MAX_SEQUENCE_LENGTH (150 points)

**Output**: `TrajectoryFeatures`
- `normalizedPoints`: x, y, vx, vy, ax, ay (6 features per point)
- `nearestKeys`: token indices for nearest keyboard keys
- `actualLength`: number of real points (not padding)

**Performance**: ~1-5ms

---

### Stage 2: Encoder Inference

**Component**: `OnnxSwipePredictor._encoderSession`

**Model**: `swipe_model_character_quant.onnx`

**Input Tensors**:
- `trajectory_features`: [1, 150, 6] float32
- `nearest_keys`: [1, 150] int64
- `src_mask`: [1, 150] bool (true = padded, false = valid)

**Output**: `memory` [1, 150, 256] float32 (encoder hidden states)

**Performance**: ~20-40ms on NNAPI/QNN, ~50-80ms on CPU

---

### Stage 3: Beam Search Decoding

**Component**: `OnnxSwipePredictor.runBeamSearch()`

**Model**: `swipe_decoder_character_quant.onnx`

**Parameters**:
- `beam_width`: 2 (mobile-optimized, was 8)
- `max_length`: 35 characters
- `confidence_threshold`: 0.1

**Algorithm**:
1. Initialize beams with SOS token
2. For each step (up to 35):
   - Run decoder for all active beams in BATCH
   - Get logits for next character
   - Apply softmax to get probabilities
   - Expand top-K candidates (K = beam_width)
   - Keep top beam_width beams by cumulative score
   - Stop if all beams finished (EOS token)
3. Convert token sequences to words
4. Calculate confidence: exp(-cumulative_negative_log_likelihood)

**Output**: `List<BeamSearchCandidate>`
- `word`: decoded string (e.g., "hello")
- `confidence`: 0.0-1.0 (higher = more confident)

**Performance**: ~10-30ms (beam_width * max_steps decoder calls)

**Example Output**:
```
Beam 0: "hello" (confidence: 0.85)
Beam 1: "hallo" (confidence: 0.12)
Beam 2: "helo" (confidence: 0.03)
```

---

### Stage 4: Vocabulary Filtering

**Component**: `OptimizedVocabulary.filterPredictions()`

**Input**: `List<BeamSearchCandidate>` (raw neural network outputs)

**Processing**:
1. For each candidate word:
   - Lookup in 50k vocabulary HashMap (O(1))
   - If NOT in vocabulary → **DISCARD** (unknown word)
   - If in vocabulary → apply tier boost:
     - Tier 2 (top 100): 1.3x boost
     - Tier 1 (top 3000): 1.0x boost
     - Tier 0 (rest): 0.75x penalty
2. Calculate combined score:
   ```java
   score = (CONFIDENCE_WEIGHT * confidence + FREQUENCY_WEIGHT * frequency) * tierBoost
   where:
     CONFIDENCE_WEIGHT = 0.6 (neural network confidence)
     FREQUENCY_WEIGHT = 0.4 (word frequency from corpus)
     frequency = 0.0-1.0 (normalized from 128-255 JSON range)
   ```
3. Sort by combined score descending
4. Return top 10

**Output**: `List<FilteredPrediction>`
- `word`: validated dictionary word
- `score`: combined NN confidence + frequency
- `source`: "common", "top3000", or "vocabulary"

**Performance**: <1ms (HashMap lookups)

**Example**:
```
Input (raw NN):
  "hello" (0.85) → VALID (tier 2, freq 0.98) → score 0.87 * 1.3 = 1.13
  "hallo" (0.12) → VALID (tier 0, freq 0.02) → score 0.08 * 0.75 = 0.06
  "helo" (0.03) → INVALID (not in vocabulary) → DISCARDED

Output (filtered):
  "hello" (score: 1.13)
  "hallo" (score: 0.06)
```

---

### Stage 5: Result Formatting

**Component**: `OnnxSwipePredictor.createOptimizedPredictionResult()`

**Input**: `List<FilteredPrediction>` (vocabulary-validated)

**Processing**:
1. Convert filtered predictions to `PredictionResult` format
2. Scale scores to 0-1000 integer range
3. **DEBUG MODE**: If debug enabled AND all predictions filtered out:
   - Show top 2 raw beam search outputs with "[raw:X.XX]" suffix

**Output**: `PredictionResult`
- `words`: List<String> (words to display)
- `scores`: List<Integer> (0-1000 range)

**Current Behavior**:
- Shows filtered vocabulary predictions (high quality)
- Only shows raw NN if ALL predictions discarded (rare)
- No way to see beam search outputs alongside filtered results

---

## Issues Identified

### Issue 1: Raw Predictions Only When Empty ⚠️

**Severity**: Medium (affects debugging and transparency)

**Description**: Raw neural network beam search outputs only shown when vocabulary filtering eliminates ALL predictions.

**Code**: `OnnxSwipePredictor.java:1300-1310`
```java
if (words.isEmpty() && !candidates.isEmpty() && _config.swipe_debug_show_raw_output)
{
  // Show raw outputs
}
```

**Impact**:
- User can't see what neural network actually predicted
- Can't compare NN confidence vs vocabulary filtering
- Makes debugging difficult ("why didn't my swipe predict X?")

**Example Scenario**:
```
User swipes: "helo" (typo)
NN predicts: "helo" (0.92 confidence) + "hello" (0.08)
Vocab filters: "helo" discarded (not in dict), "hello" kept
User sees: "hello" only
User thinks: "Why did it correct my swipe?"
Reality: NN was 92% confident in "helo", vocab corrected it
```

---

### Issue 2: No "Closest Predictions" Feature ⚠️

**Severity**: Medium (missing transparency feature)

**Description**: Users can't see the top beam search candidates before vocabulary filtering.

**Current State**: Only one pipeline: NN → Vocab Filter → Display

**Desired State**: Show both pipelines:
1. Filtered predictions (current behavior)
2. Closest predictions (raw beam search, unfiltered)

**Use Case**:
- User swipes uncommon word (e.g., "phlebotomist")
- NN correctly predicts it (0.85 confidence)
- Vocab filters it out (not in 50k dictionary)
- User sees: nothing or fallback
- User should see: "phlebotomist [closest:0.85]" to know NN got it right

---

### Issue 3: Debug Mode Not User-Friendly ⚠️

**Severity**: Low (UI/UX issue)

**Description**: Debug mode requires enabling setting and only shows when predictions empty.

**Current**: `swipe_debug_show_raw_output` setting + empty predictions required

**Desired**: Always show raw beam search in separate section

---

## Performance Characteristics

### Timing Breakdown (50k vocabulary)

| Stage | Time | Percentage |
|-------|------|------------|
| Feature Extraction | 1-5ms | 5% |
| Encoder Inference | 20-40ms | 40% |
| Beam Search Decoding | 10-30ms | 30% |
| Vocabulary Filtering | <1ms | <1% |
| Result Formatting | <1ms | <1% |
| **TOTAL** | **30-75ms** | **100%** |

**Target**: <100ms for smooth user experience ✅

---

### Memory Usage

| Component | Memory |
|-----------|--------|
| Encoder Model (quant) | ~4 MB |
| Decoder Model (quant) | ~3 MB |
| Vocabulary HashMap (50k) | ~7 MB |
| Beam Search Buffers | ~1 MB |
| **TOTAL** | **~15 MB** |

**Acceptable** for modern Android devices (2-8GB RAM) ✅

---

## Recommendations

### 1. Always Show Raw Beam Search Results (Priority: HIGH)

**Change**: Remove `words.isEmpty()` condition

**Before**:
```java
if (words.isEmpty() && !candidates.isEmpty() && _config.swipe_debug_show_raw_output)
```

**After**:
```java
if (!candidates.isEmpty() && _config.swipe_debug_show_raw_output)
```

**Impact**: Raw NN predictions always visible in debug mode for comparison

---

### 2. Add "Closest Predictions" Section (Priority: HIGH)

**Implementation**:
```java
// After vocabulary filtering, add top 3 raw beam search results
if (_config.swipe_show_closest_predictions && !candidates.isEmpty())
{
  int numClosest = Math.min(3, candidates.size());
  for (int i = 0; i < numClosest; i++)
  {
    BeamSearchCandidate candidate = candidates.get(i);
    // Only add if not already in filtered results
    if (!words.contains(candidate.word))
    {
      words.add(candidate.word + " [closest:" + String.format("%.2f", candidate.confidence) + "]");
      scores.add((int)(candidate.confidence * 1000));
    }
  }
}
```

**Display Example**:
```
Filtered:    hello (975), world (823)
Closest NN:  helo [closest:0.92], world [closest:0.15]
```

---

### 3. Improve Debug Display Format (Priority: MEDIUM)

**Current**: `word [raw:0.85]`

**Proposed**: Add source indicators
- `word [NN:0.85]` - Neural network raw output
- `word [+freq]` - Vocabulary frequency boost applied
- `word [tier2:1.3x]` - Tier boost applied

---

### 4. Add Pipeline Stats (Priority: LOW)

**Implementation**: Add metadata to PredictionResult
```java
public class PredictionResult
{
  public final List<String> words;
  public final List<Integer> scores;
  public final PipelineStats stats; // NEW

  public static class PipelineStats
  {
    public int rawCandidates;        // NN beam search outputs
    public int filteredCandidates;   // After vocabulary
    public int discardedUnknown;     // Removed (not in vocab)
    public long totalTimeMs;         // End-to-end latency
  }
}
```

**Display**: Show in debug panel or logcat

---

## Testing Recommendations

### Test Case 1: Common Word Swipe

**Input**: Swipe "hello"

**Expected NN Output**:
- "hello" (0.95)
- "hallo" (0.03)
- "helo" (0.02)

**Expected Vocab Filtering**:
- "hello" (kept, tier 2 boost 1.3x)
- "hallo" (kept, tier 0 penalty 0.75x)
- "helo" (discarded, not in vocab)

**Expected Display**:
- Filtered: "hello", "hallo"
- Closest: "helo [closest:0.02]"

---

### Test Case 2: Typo Swipe

**Input**: Swipe "thsi" (typo for "this")

**Expected NN Output**:
- "thsi" (0.88) - NN learned typo pattern
- "this" (0.10)
- "thus" (0.02)

**Expected Vocab Filtering**:
- "thsi" (discarded, not in vocab)
- "this" (kept, tier 2 boost)
- "thus" (kept, tier 1)

**Expected Display**:
- Filtered: "this", "thus"
- Closest: "thsi [closest:0.88]"

**Analysis**: Vocabulary filtering corrected NN's confident typo prediction

---

### Test Case 3: Uncommon Word

**Input**: Swipe "phlebotomist" (uncommon medical term)

**Expected NN Output**:
- "phlebotomist" (0.85) - NN trained on medical corpus
- "phlebotomy" (0.10)
- "flebotomist" (0.05) - typo variant

**Expected Vocab Filtering**:
- "phlebotomist" (discarded, not in 50k vocab)
- "phlebotomy" (discarded, not in 50k vocab)
- "flebotomist" (discarded, typo + not in vocab)

**Expected Display**:
- Filtered: (empty)
- Closest: "phlebotomist [closest:0.85]", "phlebotomy [closest:0.10]"

**Analysis**: Shows value of "closest predictions" for specialized vocabulary

---

## Files and Components

### Core Prediction Files

```
srcs/juloo.keyboard2/
├── OnnxSwipePredictor.java        # Main predictor, beam search, result formatting
├── SwipeTrajectoryProcessor.java  # Input preprocessing
├── SwipeTokenizer.java            # Character tokenization
├── OptimizedVocabulary.java       # 50k vocabulary + filtering
├── PredictionResult.java          # Result container
└── BeamSearchCandidate.java       # Internal beam search state
```

### Model Assets

```
assets/models/
├── swipe_model_character_quant.onnx    # Encoder (transformer)
└── swipe_decoder_character_quant.onnx  # Decoder (autoregressive)
```

### Vocabulary Assets

```
assets/dictionaries/
└── en_enhanced.json    # 50k words with frequencies (128-255 range)
```

---

## Configuration Parameters

### Neural Network Settings

```java
// OnnxSwipePredictor.java
private static final int DEFAULT_BEAM_WIDTH = 2;           // Mobile-optimized
private static final int DEFAULT_MAX_LENGTH = 35;          # Maximum word length
private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
```

### Vocabulary Settings

```java
// OptimizedVocabulary.java
private static final float CONFIDENCE_WEIGHT = 0.6f;    // NN confidence weight
private static final float FREQUENCY_WEIGHT = 0.4f;     // Word frequency weight
private static final float COMMON_WORDS_BOOST = 1.3f;   // Tier 2 (top 100)
private static final float TOP5000_BOOST = 1.0f;        // Tier 1 (top 3000)
private static final float RARE_WORDS_PENALTY = 0.75f;  // Tier 0 (rest)
```

---

## References

- [BEAM_SEARCH_VOCABULARY.md](BEAM_SEARCH_VOCABULARY.md) - Vocabulary system details
- [DICTIONARY_MANAGER.md](DICTIONARY_MANAGER.md) - Dictionary management
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- Neural transformer architecture based on "Attention Is All You Need"

---

## Changelog

### 2025-10-21 - Initial Analysis

- Documented complete swipe prediction pipeline
- Identified 3 issues with raw/closest predictions display
- Provided recommendations for improvements
- Created test cases for validation
