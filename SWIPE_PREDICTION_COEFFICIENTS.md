# Complete Swipe Prediction Coefficients & Constants

**Date**: 2025-11-25
**Purpose**: Comprehensive list of ALL weights, thresholds, coefficients, and constants used in the neural swipe typing pipeline from touch input to word prediction.

---

## Table of Contents
1. [Touch Input Collection](#1-touch-input-collection)
2. [Coordinate Normalization](#2-coordinate-normalization)
3. [Trajectory Resampling](#3-trajectory-resampling)
4. [Feature Calculation](#4-feature-calculation)
5. [Neural Network Input](#5-neural-network-input)
6. [Beam Search Decoding](#6-beam-search-decoding)
7. [Post-Processing & Filtering](#7-post-processing--filtering)

---

## 1. Touch Input Collection

### Raw Touch Processing
**File**: `Pointers.java` / `Keyboard2View.java`

No coefficients - raw (x, y, timestamp) coordinates collected directly from Android MotionEvent.

---

## 2. Coordinate Normalization

### File: `SwipeTrajectoryProcessor.java`

#### Keyboard Layout Detection
- **Grid**: QWERTY 3-row layout
- **Key Width**: `0.1` (1/10 of normalized width)
- **Row Height**: `1.0 / 3.0 = 0.333...` (normalized height)

#### Row Offsets (Horizontal Centering)
```java
row0_x0 = 0.0f    // "qwertyuiop" - 10 keys, starts at left edge
row1_x0 = 0.05f   // "asdfghjkl"  - 9 keys, offset 0.5 key width
row2_x0 = 0.15f   // "zxcvbnm"    - 7 keys, offset 1.5 key widths
```

#### QWERTY Area Bounds (v1.32.463+)
Used for proper Y-coordinate normalization:
```java
_qwertyAreaTop    // Y pixel offset where QWERTY keys start
_qwertyAreaHeight // Height in pixels of QWERTY key area only
```

**Calculation**:
```java
// X normalization (over full keyboard width)
x_normalized = x_raw / _keyboardWidth

// Y normalization (over QWERTY area only if bounds set)
y_adjusted = y_raw + _touchYOffset  // Apply fat finger correction
if (usingQwertyBounds) {
    y_normalized = (y_adjusted - _qwertyAreaTop) / _qwertyAreaHeight
} else {
    y_normalized = y_adjusted / _keyboardHeight
}

// Clamp to [0, 1]
x_normalized = max(0.0, min(1.0, x_normalized))
y_normalized = max(0.0, min(1.0, y_normalized))
```

#### Touch Y-Offset Compensation (v1.32.466+)
**Fat Finger Effect Correction**:
```java
_touchYOffset = 74.0f  // Default: 74 pixels (user touches above key center)
```
**Configurable**: Can be set via `setTouchYOffset()` based on calibration.

**Purpose**: Users typically touch ~74 pixels above the visual key center due to finger geometry. This offset shifts coordinates down to match the actual intended key.

---

## 3. Trajectory Resampling

### File: `SwipeResampler.java`

#### Maximum Sequence Length
```java
maxSequenceLength = 150  // Model input constraint
```

#### Resampling Modes
**Default Mode**: `DISCARD` (matching Config.java default)

##### DISCARD Mode Weights
When `originalLength > 150`, use weighted selection:

```java
// Zone distribution (of middle points)
startZoneEnd   = 1 + (availableRange * 0.3)   // First 30% of trajectory
endZoneStart   = originalLength - 1 - (availableRange * 0.3)  // Last 30%

// Point allocation across zones
pointsInStart  = numMiddle * 0.35   // 35% of selected points from start
pointsInEnd    = numMiddle * 0.35   // 35% of selected points from end
pointsInMiddle = numMiddle * 0.30   // 30% of selected points from middle
```

**Strategy**: Preserve more detail at start/end of swipe (critical for word recognition).

##### TRUNCATE Mode
```java
// Simply keep first maxSequenceLength points
result = trajectory[0:maxSequenceLength]
```

##### MERGE Mode
```java
mergeRatio = originalLength / targetLength
groupSize = ceil(mergeRatio)
// Average consecutive groups of points
```

---

## 4. Feature Calculation

### File: `TrajectoryFeatureCalculator.kt`

#### Feature Vector: `[x, y, vx, vy, ax, ay]`

##### Minimum Time Delta (Prevent Division by Zero)
```kotlin
dt_min = 1e-6f  // 0.000001 milliseconds
dt[i] = max(dt[i], dt_min)
```

##### Velocity Calculation
```kotlin
vx[0] = 0.0f  // First point has zero velocity
vy[0] = 0.0f

for (i in 1 until n) {
    dt[i] = max(timestamps[i] - timestamps[i-1], 1e-6f)
    vx[i] = (x[i] - x[i-1]) / dt[i]
    vy[i] = (y[i] - y[i-1]) / dt[i]
}
```

##### Acceleration Calculation
```kotlin
ax[0] = 0.0f  // First point has zero acceleration
ay[0] = 0.0f

for (i in 1 until n) {
    ax[i] = (vx[i] - vx[i-1]) / dt[i]
    ay[i] = (vy[i] - vy[i-1]) / dt[i]
}
```

##### Velocity & Acceleration Clipping
```kotlin
vx_clip_min = -10.0f
vx_clip_max =  10.0f
vy_clip_min = -10.0f
vy_clip_max =  10.0f
ax_clip_min = -10.0f
ax_clip_max =  10.0f
ay_clip_min = -10.0f
ay_clip_max =  10.0f

vx[i] = vx[i].coerceIn(-10f, 10f)
vy[i] = vy[i].coerceIn(-10f, 10f)
ax[i] = ax[i].coerceIn(-10f, 10f)
ay[i] = ay[i].coerceIn(-10f, 10f)
```

**Rationale**: Prevents extreme outliers from dominating model input, matches training data preprocessing.

---

## 5. Neural Network Input

### Vocabulary & Tokenization
**File**: `SwipeTokenizer.java`

#### Token Indices
```java
PAD_IDX = 0   // Padding token
UNK_IDX = 1   // Unknown character
SOS_IDX = 2   // Start of sequence
EOS_IDX = 3   // End of sequence

// Letter tokens: a-z
'a' -> 4
'b' -> 5
...
'z' -> 29

VOCAB_SIZE = 30
```

#### Sequence Padding
```java
MAX_SEQUENCE_LENGTH = 150

// Trajectory features padded with zeros
while (features.size < 150) {
    features.add([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

// Nearest keys padded with PAD_IDX (0)
while (nearestKeys.size < 150) {
    nearestKeys.add(0)
}
```

### Model Architecture Constants
**File**: `OnnxSwipePredictor.java` / `BeamSearchEngine.kt`

#### Encoder Input Shapes
```kotlin
traj_features:     [1, 150, 6]    // [batch, seq_len, features]
nearest_keys:      [1, 150]       // [batch, seq_len]
actual_src_length: [1]            // scalar int (true length before padding)
```

#### Decoder Input Shapes
```kotlin
memory:            [1, 150, 256]  // Encoder output [batch, src_len, d_model]
actual_src_length: [1]            // scalar int
target_tokens:     [1, 20]        // [batch, tgt_len] - decoder sequence
```

#### Model Dimensions
```kotlin
ENCODER_SEQ_LEN = 150   // Max trajectory length
DECODER_SEQ_LEN = 20    // Max output word length
D_MODEL         = 256   // Transformer hidden dimension
VOCAB_SIZE      = 30    // Output vocabulary size
```

---

## 6. Beam Search Decoding

### File: `BeamSearchEngine.kt`

#### Core Beam Search Parameters
```kotlin
beamWidth               = 5       // Number of parallel hypotheses (default)
maxLength               = 20      // Maximum word length (matches DECODER_SEQ_LEN)
confidenceThreshold     = 0.05f   // Min confidence to accept prediction (5%)
lengthPenaltyAlpha      = 1.2f    // Length normalization exponent
```

#### Scoring Formula
```kotlin
// Raw score accumulation (negative log-likelihood)
score = 0.0f
for each step:
    logProbs = logSoftmax(logits)  // Convert to log probabilities
    score += -logProbs[selectedToken]  // Accumulate NLL

// Length-normalized score (for ranking)
normalized_score = score / (length ^ lengthPenaltyAlpha)

// Final confidence (for filtering)
confidence = exp(-score)  // Convert NLL back to probability
```

**Length Penalty Formula**:
```
normalized_score = score / len^α

where:
  α = 1.2  (lengthPenaltyAlpha)

Examples:
  α = 1.0  → no penalty (score / len)
  α = 1.2  → moderate penalty favoring longer words
  α = 1.5  → strong penalty favoring longer words
```

#### Pruning & Early Stopping
```kotlin
// Adaptive pruning (reduce beam width dynamically)
adaptiveWidthConfidence = 0.8f         // Confidence threshold for pruning
ADAPTIVE_WIDTH_STEP     = 5            // Start adaptive pruning at step 5

if (step >= ADAPTIVE_WIDTH_STEP && topBeamConfidence > adaptiveWidthConfidence) {
    // Reduce beam width by factor of 2
    pruned_beam_width = beamWidth / 2
}

// Score gap early stopping
scoreGapThreshold = 5.0f               // Score difference threshold
SCORE_GAP_STEP    = 3                  // Start checking gap at step 3

if (step >= SCORE_GAP_STEP && (bestScore - worstScore) > scoreGapThreshold) {
    // Stop decoding early - winner is clear
    break
}

// Low probability pruning
LOG_PROB_THRESHOLD = -13.8f            // approx ln(1e-6)
PRUNE_STEP_THRESHOLD = 2               // Start pruning at step 2

if (step >= PRUNE_STEP_THRESHOLD && logProb < LOG_PROB_THRESHOLD) {
    // Discard extremely unlikely tokens
    continue
}
```

#### Diversity Promotion (4D Beam Search)
```kotlin
DIVERSITY_LAMBDA = 0.5f  // Penalty weight for similar beams

diversity_penalty = DIVERSITY_LAMBDA * sibling_similarity
adjusted_score = original_score + diversity_penalty
```

**Purpose**: Prevent beam collapse where all hypotheses converge to same prefix.

#### Log-Softmax Numerical Stability
```kotlin
// CRITICAL FIX: Use NEGATIVE_INFINITY for proper initialization
maxLogit = Float.NEGATIVE_INFINITY  // NOT 0.0f!

for (logit in logits) {
    if (logit > maxLogit) maxLogit = logit
}

// Numerically stable log-softmax
sumExp = 0.0f
for (logit in logits) {
    sumExp += exp(logit - maxLogit)
}
logSumExp = maxLogit + ln(sumExp)

for (i in 0 until vocabSize) {
    logProbs[i] = logits[i] - logSumExp
}
```

**Why this matters**:
- Old bug: `maxLogit = 0.0f` caused NaN when ALL logits were negative
- Fix: `maxLogit = Float.NEGATIVE_INFINITY` handles all cases correctly

#### Special Token Handling
```kotlin
SOS_IDX = 2  // Start token (only at position 0)
EOS_IDX = 3  // End token (triggers beam termination)
PAD_IDX = 0  // Padding token (ignored)
UNK_IDX = 1  // Unknown token (rarely used)

// Beam finishes when EOS or PAD or SOS is generated
if (token == SOS_IDX || token == EOS_IDX || token == PAD_IDX) {
    beam.finished = true
}
```

---

## 7. Post-Processing & Filtering

### File: `PredictionPostProcessor.kt`

#### Confidence Threshold
```kotlin
MIN_CONFIDENCE = 0.05f  // 5% minimum (lowered from 0.1)

// Filter candidates
if (confidence < MIN_CONFIDENCE) {
    discard_candidate
}
```

#### Length Filtering
```kotlin
MIN_WORD_LENGTH = 2   // Discard single-letter predictions
MAX_WORD_LENGTH = 20  // Model constraint (DECODER_SEQ_LEN)

if (word.length < MIN_WORD_LENGTH || word.length > MAX_WORD_LENGTH) {
    discard_candidate
}
```

#### Duplicate Removal
```kotlin
// Keep only unique words (case-insensitive)
uniqueWords = candidates.distinctBy { it.word.lowercase() }
```

#### Score Conversion (Display)
```kotlin
// Convert NLL score to display score (0-100 scale)
displayScore = (confidence * 100).toInt()

// Examples:
confidence = 0.9  → displayScore = 90
confidence = 0.5  → displayScore = 50
confidence = 0.05 → displayScore = 5
```

---

## 8. Frequency Weighting & Scoring

### File: `VocabularyUtils.kt` / `Config.java`

#### Combined Score Formula
```kotlin
// Weighted combination of neural network confidence and dictionary frequency
combinedScore = (confidenceWeight * confidence + frequencyWeight * frequency) * boost

where:
  confidence       = Neural network confidence [0.0, 1.0]
  frequency        = Dictionary frequency (normalized) [0.0, 1.0]
  confidenceWeight = Neural confidence weight (default: 0.6)
  frequencyWeight  = Dictionary frequency weight (default: 0.4)
  boost            = Frequency tier boost multiplier
```

#### Default Weight Configuration (v1.33+)
**User-Configurable via "Prediction Source" Slider (0-100)**:

```java
// Config.java:299-303
predictionSource = 60  // Default: Balanced (60% AI, 40% Dictionary)

// Derived weights:
confidenceWeight = predictionSource / 100.0f       // = 0.60
frequencyWeight  = 1.0f - confidenceWeight         // = 0.40

// Slider positions:
//   0  = 100% Dictionary (conf=0.0, freq=1.0) - Pure frequency-based
//  60  = Balanced (conf=0.6, freq=0.4) - Default
// 100  = 100% AI (conf=1.0, freq=0.0) - Pure neural confidence
```

#### Frequency Tier Boosts (v1.33+)
**3-Tier System Based on Word Popularity**:

```java
// Config.java:305-307
swipe_common_words_boost  = 1.3f   // Tier 2: Top 100 most common words
swipe_top5000_boost       = 1.0f   // Tier 1: Top 3000 words (baseline)
swipe_rare_words_penalty  = 0.75f  // Tier 0: Rare words (penalized)
```

**Boost Application**:
```kotlin
// Determine boost based on word frequency rank
boost = if (word in top100) {
    1.3f    // +30% boost for ultra-common words
} else if (word in top3000) {
    1.0f    // Baseline (no boost or penalty)
} else {
    0.75f   // -25% penalty for rare words
}

finalScore = (0.6 * confidence + 0.4 * frequency) * boost
```

**Example Scores**:
```
Word: "the" (top100, freq=1.0, conf=0.8)
  → score = (0.6*0.8 + 0.4*1.0) * 1.3 = (0.48 + 0.40) * 1.3 = 1.144

Word: "hello" (top3000, freq=0.5, conf=0.8)
  → score = (0.6*0.8 + 0.4*0.5) * 1.0 = (0.48 + 0.20) * 1.0 = 0.680

Word: "zephyr" (rare, freq=0.1, conf=0.8)
  → score = (0.6*0.8 + 0.4*0.1) * 0.75 = (0.48 + 0.04) * 0.75 = 0.390
```

---

## 9. Smoothing Algorithms

### Touch Input Smoothing
**File**: `Pointers.java:823-874`

#### Exponential Smoothing for Slider Speed
```java
SPEED_SMOOTHING = 0.7f     // Smoothing factor (0-1)
SPEED_MAX       = 4.0f     // Maximum speed cap
SPEED_VERTICAL_MULT = 0.5f // Vertical movement multiplier

// Exponential moving average formula
instant_speed = min(SPEED_MAX, travelled / elapsed_time + 1.0f)
speed = speed + (instant_speed - speed) * SPEED_SMOOTHING

// Equivalent form: EMA with α = 0.7
// speed_new = α * instant_speed + (1-α) * speed_old
//           = 0.7 * instant_speed + 0.3 * speed_old
```

**Purpose**: Smooth out noise from jittery touch input, prevent erratic slider movement.

**Effect**:
- `α = 0.7` → 70% weight on new data, 30% on history (responsive but smooth)
- `α = 1.0` → No smoothing (instant jitter)
- `α = 0.0` → Complete smoothing (no response)

### Language Model Smoothing
**File**: `NgramModel.java:20`, `BigramModel.java:473`

#### Add-k Smoothing (Laplace Smoothing)
```java
// NgramModel.java:20
SMOOTHING_FACTOR = 0.001f  // k = 0.001 (additive smoothing)

// Applied to all probability lookups:
bigramProb  = _bigramProbs.getOrDefault(bigram, SMOOTHING_FACTOR)
trigramProb = _trigramProbs.getOrDefault(trigram, SMOOTHING_FACTOR)
startCharProb = _startCharProbs.getOrDefault(c, SMOOTHING_FACTOR)
endCharProb   = _endCharProbs.getOrDefault(c, SMOOTHING_FACTOR)
```

**Purpose**: Assign non-zero probability to unseen n-grams (avoid division by zero).

#### Exponential Smoothing for Adaptation
```java
// BigramModel.java:473
// Adaptive learning from user patterns
newProb = 0.9f * currentProb + 0.1f * weight

// EMA formula: P_new = 0.9 * P_old + 0.1 * observed
```

**Purpose**: Gradually adapt bigram probabilities based on user typing patterns without sudden jumps.

**Effect**:
- New observation gets 10% weight
- Historical probability retains 90% weight
- Converges slowly (~10 observations to substantially change probability)

---

## Summary Table: Critical Coefficients

| **Stage** | **Parameter** | **Value** | **Purpose** |
|-----------|--------------|-----------|-------------|
| **Normalization** | QWERTY Row Height | `1/3 = 0.333` | 3-row keyboard grid |
| | Key Width | `0.1` | QWERTY key spacing |
| | Touch Y-Offset | `74.0 px` | Fat finger correction |
| **Resampling** | Max Sequence Length | `150` | Model input constraint |
| | DISCARD Start Zone | `0.3` (30%) | Preserve swipe beginning |
| | DISCARD End Zone | `0.3` (30%) | Preserve swipe ending |
| **Features** | Min Time Delta | `1e-6 ms` | Avoid division by zero |
| | Velocity Clip Range | `[-10, 10]` | Prevent outliers |
| | Acceleration Clip Range | `[-10, 10]` | Prevent outliers |
| **Tokenizer** | Vocabulary Size | `30` | 26 letters + 4 special |
| | PAD/UNK/SOS/EOS | `0, 1, 2, 3` | Special token indices |
| **Beam Search** | Beam Width | `5` | Parallel hypotheses |
| | Max Word Length | `20` | Decoder constraint |
| | Length Penalty Alpha | `1.2` | Favor longer words |
| | Confidence Threshold | `0.05` (5%) | Min acceptance |
| | Adaptive Width Conf | `0.8` (80%) | Pruning trigger |
| | Score Gap Threshold | `5.0` | Early stopping |
| | Diversity Lambda | `0.5` | Sibling penalty |
| | Log Prob Threshold | `-13.8` | Extremely unlikely cutoff |
| **Post-Processing** | Min Word Length | `2` | No single letters |
| | Display Score Scale | `0-100` | UI presentation |
| **Frequency Weights** | Confidence Weight | `0.6` (60%) | Neural network importance |
| | Frequency Weight | `0.4` (40%) | Dictionary frequency importance |
| | Common Words Boost | `1.3` (+30%) | Top 100 words boost |
| | Top 5000 Baseline | `1.0` (±0%) | Top 3000 words (neutral) |
| | Rare Words Penalty | `0.75` (-25%) | Uncommon words penalty |
| **Smoothing** | Speed Smoothing (EMA) | `0.7` (α) | Touch input noise reduction |
| | Speed Max Cap | `4.0` | Maximum slider speed |
| | N-gram Smoothing | `0.001` (k) | Add-k (Laplace) smoothing |
| | Bigram Adaptation | `0.9/0.1` | Learning rate (90% old, 10% new) |

---

## Modification Impact Analysis

### High-Impact Parameters (Tune Carefully)
1. **`confidenceThreshold = 0.05`** - Too high = no predictions, too low = garbage predictions
2. **`lengthPenaltyAlpha = 1.2`** - Affects short vs long word preference
3. **`beamWidth = 5`** - Higher = better quality but slower (5-10 recommended)
4. **`touchYOffset = 74`** - Device-specific, may need calibration per user
5. **`confidenceWeight = 0.6 / frequencyWeight = 0.4`** - Balance between AI and dictionary
6. **Frequency tier boosts** (1.3, 1.0, 0.75) - Dramatically affects common word ranking

### Medium-Impact Parameters
7. **`scoreGapThreshold = 5.0`** - Affects early stopping frequency
8. **`adaptiveWidthConfidence = 0.8`** - Controls aggressive pruning
9. **Resampling weights** (0.3, 0.35, 0.3) - Affects long swipe quality
10. **`SPEED_SMOOTHING = 0.7`** - Slider responsiveness vs stability tradeoff
11. **`SMOOTHING_FACTOR = 0.001`** - Language model probability floor

### Low-Impact Parameters (Safe to Modify)
12. **`DIVERSITY_LAMBDA = 0.5`** - Only affects duplicate prevention
13. **`MIN_WORD_LENGTH = 2`** - Post-processing cosmetic
14. **Clip ranges `[-10, 10]`** - Matches training, rarely need changes
15. **`SPEED_MAX = 4.0`** - Slider speed cap (cosmetic)
16. **Bigram adaptation rate** (0.9/0.1) - Learning speed (already conservative)

---

## Configuration Access

### Runtime Configuration
Most parameters can be adjusted via:
```java
SwipePredictorOrchestrator.getInstance(context)
    .setBeamWidth(10)
    .setConfidenceThreshold(0.08f)
    .setLengthPenaltyAlpha(1.5f);
```

### Compile-Time Constants
Constants in `BeamSearchEngine.kt` require rebuild:
- `DECODER_SEQ_LEN = 20` (matches model export)
- `VOCAB_SIZE = 30` (matches vocabulary)
- `D_MODEL = 256` (transformer dimension)

---

## References

**Implementation Files**:
- `SwipeTrajectoryProcessor.java` - Lines 1-532 (Normalization, resampling)
- `TrajectoryFeatureCalculator.kt` - Lines 1-189 (Feature calculation, clipping)
- `SwipeResampler.java` - Lines 1-250 (Resampling algorithms)
- `BeamSearchEngine.kt` - Lines 1-392 (Beam search, pruning, scoring)
- `SwipeTokenizer.java` - Token definitions
- `PredictionPostProcessor.kt` - Filtering logic
- `VocabularyUtils.kt` - Lines 1-129 (Combined scoring, fuzzy matching)
- `Config.java` - Lines 95-134 (User-configurable weights)
- `Pointers.java` - Lines 823-879 (Touch input smoothing)
- `NgramModel.java` - Line 20 (Language model smoothing)
- `BigramModel.java` - Line 473 (Adaptive learning)

**Documentation**:
- `BEAM_SEARCH_AUDIT.md` - Original audit findings
- `AUDIT_VERIFICATION.md` - Fix verification
- `TEST_SUITE_SUMMARY.md` - Test coverage
- `DISCORD_ANR_ANALYSIS.md` - Performance investigation

---

**Last Updated**: 2025-11-25
**Status**: ✅ All coefficients, weights, and smoothing parameters verified and documented
