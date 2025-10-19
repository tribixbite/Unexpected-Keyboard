# Advanced Word Prediction Settings

## Overview

There are **TWO prediction systems** in Unexpected Keyboard:

1. **Neural Network (ONNX)** - Primary system for swipe typing (v1.32+)
2. **WordPredictor** - Fallback for swipes + primary for regular typing predictions

The "Advanced Word Prediction" settings control the **WordPredictor fallback system only**. The neural network has its own settings.

## System Architecture

### When Each System is Used

**Neural Network (ONNX)**:
- ✅ Primary for swipe typing
- ✅ Uses LSTM + beam search
- ✅ Trained on millions of real swipe gestures
- ❌ Does NOT use the 8 advanced settings below

**WordPredictor** (uses the 8 settings):
- ✅ Fallback when neural network fails or is disabled
- ✅ Primary for regular typing predictions (prefix matching)
- ✅ Dictionary-based with heuristic scoring
- ✅ Uses some of the 8 advanced settings for endpoint bonuses

## The 8 Settings - Actual Current Usage

### ⚠️ NOT USED (Removed from Code)

These 4 settings are loaded from Config but **NEVER ACTUALLY USED** anywhere in WordPredictor:

#### 1. **Shape Weight** (`swipe_confidence_shape_weight`)
- **Default**: 90 (0.90)
- **Range**: 0-200
- **Original Purpose**: Weight for geometric path similarity
- **Current Status**: ❌ **NOT USED** - No shape matching algorithm exists in current code
- **Why Removed**: Neural network handles shape intrinsically

#### 2. **Location Weight** (`swipe_confidence_location_weight`)
- **Default**: 130 (1.30)
- **Range**: 0-200
- **Original Purpose**: Weight for spatial proximity to keys
- **Current Status**: ❌ **NOT USED** - No location-based scoring in current code
- **Why Removed**: Neural network handles spatial relationships

#### 3. **Frequency Weight** (`swipe_confidence_frequency_weight`)
- **Default**: 80 (0.80)
- **Range**: 0-200
- **Original Purpose**: Boost common words
- **Current Status**: ❌ **NOT USED** - Frequency applied directly without weighting
- **Why Removed**: Dictionary frequency used as simple multiplier instead

#### 4. **Velocity Weight** (`swipe_confidence_velocity_weight`)
- **Default**: 60 (0.60)
- **Range**: 0-200
- **Original Purpose**: Reward smooth swipes
- **Current Status**: ❌ **NOT USED** - No velocity analysis in current code
- **Why Removed**: Neural network handles temporal patterns

### ✅ ACTIVELY USED (WordPredictor Endpoint Bonuses)

These 4 settings ARE used when WordPredictor is matching swipe patterns:

#### 5. **First Letter Weight** (`swipe_first_letter_weight`)
- **Default**: 150 (1.50)
- **Range**: 0-300
- **Purpose**: Multiplier when swipe starts on correct letter
- **Current Usage**: ✅ **WordPredictor.java:387, 422** - Applied to endpoint matches
- **How it Works**:
  ```java
  if (firstChar == seqFirst) {
    baseScore = (int)(baseScore * _config.swipe_first_letter_weight);
  }
  ```
- **Effect**: Boosts predictions that start with the right letter (1.5x by default)
- **When to Adjust**:
  - Increase (2.0-3.0): If first letters are always accurate in your swipes
  - Decrease (1.0-1.2): If you want more flexibility in starting position

#### 6. **Last Letter Weight** (`swipe_last_letter_weight`)
- **Default**: 150 (1.50)
- **Range**: 0-300
- **Purpose**: Multiplier when swipe ends on correct letter
- **Current Usage**: ✅ **WordPredictor.java:389, 424** - Applied to endpoint matches
- **How it Works**:
  ```java
  if (lastChar == seqLast) {
    baseScore = (int)(baseScore * _config.swipe_last_letter_weight);
  }
  ```
- **Effect**: Boosts predictions that end with the right letter (1.5x by default)
- **When to Adjust**:
  - Increase (2.0-3.0): If last letters are always accurate
  - Decrease (1.0-1.2): If you tend to not finish swipes precisely

#### 7. **Endpoint Bonus Weight** (`swipe_endpoint_bonus_weight`)
- **Default**: 200 (2.00)
- **Range**: 0-400
- **Purpose**: Extra multiplier when BOTH endpoints match
- **Current Usage**: ✅ **WordPredictor.java:391** - Applied to priority matches
- **How it Works**:
  ```java
  if (firstChar == seqFirst && lastChar == seqLast) {
    baseScore = (int)(baseScore * _config.swipe_endpoint_bonus_weight);
  }
  ```
- **Effect**: Massive boost (2.0x on top of first/last weights) for perfect endpoint matches
- **Combined Power**: With defaults, perfect endpoints get **4.5x boost** (1.5 × 1.5 × 2.0)
- **When to Adjust**:
  - Increase (3.0-4.0): If your endpoints are extremely accurate
  - Decrease (1.0-1.5): If endpoints often don't match but middle is correct

#### 8. **Require Endpoints** (`swipe_require_endpoints`)
- **Default**: false
- **Type**: Boolean
- **Purpose**: Filter out words where endpoints don't match
- **Current Usage**: ✅ **WordPredictor.java:408, 439** - Hard filter
- **How it Works**:
  ```java
  if (_config != null && _config.swipe_require_endpoints) {
    continue; // Skip words without matching endpoints
  }
  ```
- **Effect**: When enabled, ONLY shows words where first AND last letters match
- **When to Enable**:
  - ✅ Enable: If you want strictest, most precise predictions
  - ❌ Disable: If you want flexibility (e.g., swiping close but not exactly on endpoints)

## WordPredictor Scoring Algorithm (Actually Used)

For swipe pattern matching in WordPredictor:

```java
// Step 1: Priority matches (both endpoints correct)
if (firstChar == seqFirst && lastChar == seqLast) {
  baseScore = 10000 + (innerMatches * 100);
  baseScore *= swipe_first_letter_weight;     // 1.5x
  baseScore *= swipe_last_letter_weight;      // 1.5x
  baseScore *= swipe_endpoint_bonus_weight;   // 2.0x
  // Total multiplier: 4.5x for perfect endpoints
}

// Step 2: Partial matches (one endpoint correct)
else if (firstChar == seqFirst || lastChar == seqLast) {
  if (swipe_require_endpoints) {
    skip; // Reject if strict mode enabled
  }
  baseScore = 1000 + (innerMatches * 50);
  if (firstChar == seqFirst) baseScore *= swipe_first_letter_weight;
  if (lastChar == seqLast) baseScore *= swipe_last_letter_weight;
  baseScore *= frequency; // Dictionary frequency applied here
}

// Step 3: Other matches (no endpoint match)
else {
  if (swipe_require_endpoints) {
    skip; // Reject if strict mode enabled
  }
  // Simple match ratio * gap penalty scoring
  matchRatio = matchedChars / wordLength;
  gapPenalty = 1.0 / (1.0 + totalGaps/10.0);
  score = matchRatio * gapPenalty * 1000;
}
```

**Key Insight**: The shape/location/frequency/velocity weights were REMOVED. Only endpoint weights remain.

## Neural Network Settings (Actually Control Swipe Typing)

These settings in **Settings → Advanced** control the **primary swipe typing system**:

### Active Neural Settings

1. **Neural Prediction Enabled** (`neural_prediction_enabled`)
   - Default: true
   - Controls: Enable/disable ONNX model
   - Effect: When disabled, falls back to WordPredictor

2. **Beam Width** (`neural_beam_width`)
   - Default: 2
   - Range: 1-16
   - Controls: Number of parallel prediction paths
   - Effect: Higher = more accurate but slower (2 beams = 4x faster than 8)

3. **Max Word Length** (`neural_max_length`)
   - Default: 35
   - Range: 10-50
   - Controls: Maximum characters to predict
   - Effect: Allows prediction of long words

4. **Confidence Threshold** (`neural_confidence_threshold`)
   - Default: 0.1
   - Range: 0.0-1.0
   - Controls: Minimum NN confidence to show prediction
   - Effect: Higher = fewer but more confident predictions

5. **Detailed Logging** (`swipe_debug_detailed_logging`)
   - Default: false
   - Controls: Verbose swipe trajectory/NN logging
   - Effect: Debugging only

6. **Show Raw Output** (`swipe_debug_show_raw_output`)
   - Default: true
   - Controls: Show at least 2 raw NN outputs with scores
   - Effect: Helps see what neural network is thinking

## Recommendation

### For Best Swipe Typing (Neural Network)
- **Use defaults** - Neural network handles everything automatically
- **Ignore** the 8 "Advanced Word Prediction" settings (they don't affect neural predictions)
- **Adjust only neural settings** if needed (beam width, confidence threshold)

### For Word Predictions (Regular Typing via WordPredictor)
- **Endpoint weights don't matter** - Regular typing uses prefix matching, not swipe scoring
- WordPredictor for typing only cares about dictionary frequency and prefix match length

### For Fallback Swipe Mode (WordPredictor when NN disabled)
- **Increase endpoint weights (2.0-3.0)** if your swipe endpoints are very accurate
- **Enable Require Endpoints** if you want strict, precise matching only
- **Decrease endpoint weights (1.0-1.2)** if you want more flexibility

## Why Keep Unused Settings?

1. **Backwards Compatibility**: Old Config code still loads them
2. **Future Hybrid System**: May combine neural + heuristic approaches
3. **Debugging**: Useful for comparing approaches
4. **User Expectations**: Some users may have customized these historically

## Technical Details

- **Location in Code**:
  - Config.java:77-84 (field declarations)
  - Config.java:223-231 (loading from preferences)
  - WordPredictor.java:387-391, 408, 422-424, 439 (actual usage - endpoints only)
- **Storage**: SharedPreferences as integers (0-400), divided by 100 for float weights
- **Neural Code**: OnnxSwipePredictor.java, NeuralSwipeTypingEngine.java
- **Fallback Code**: WordPredictor.java (uses only endpoint weights, not shape/location/frequency/velocity)
