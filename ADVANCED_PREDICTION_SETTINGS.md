# Advanced Word Prediction Settings

## Overview

The Advanced Word Prediction settings control **legacy word prediction parameters** that were designed for the original swipe gesture classifier. These settings are **NO LONGER USED** by the neural swipe typing system (ONNX model) introduced in v1.32+.

⚠️ **IMPORTANT**: These settings are retained for backwards compatibility but do not affect the current neural network-based swipe predictions. The neural network uses its own internal confidence calculations.

## The 8 Settings (Legacy - Not Currently Used)

### 1. **Shape Weight** (`swipe_confidence_shape_weight`)
- **Default**: 90 (0.90)
- **Range**: 0-200 (stored as percentage, divided by 100)
- **Purpose**: Weight given to how well the swipe path matches the expected shape between keys
- **Calculation**: Measures geometric similarity between user's swipe curve and ideal path
- **When Used**: During word candidate scoring in legacy WordPredictor
- **Current Status**: ❌ Not used by neural network

### 2. **Location Weight** (`swipe_confidence_location_weight`)
- **Default**: 130 (1.30)
- **Range**: 0-200
- **Purpose**: Weight given to spatial proximity - how close swipe points are to target keys
- **Calculation**: Distance-based scoring using Euclidean distance from swipe points to key centers
- **When Used**: Penalizes candidates where swipe path is far from expected keys
- **Current Status**: ❌ Not used by neural network

### 3. **Frequency Weight** (`swipe_confidence_frequency_weight`)
- **Default**: 80 (0.80)
- **Range**: 0-200
- **Purpose**: Weight given to word frequency/popularity in English language
- **Calculation**: Uses frequency data from word dictionary
- **When Used**: Boosts common words like "the", "and", "you" over rare words
- **Current Status**: ❌ Not used by neural network (NN has frequency embedded in training)

### 4. **Velocity Weight** (`swipe_confidence_velocity_weight`)
- **Default**: 60 (0.60)
- **Range**: 0-200
- **Purpose**: Weight given to swipe speed consistency
- **Calculation**: Analyzes velocity changes during swipe gesture
- **When Used**: Penalizes erratic/jerky swipes, rewards smooth gestures
- **Current Status**: ❌ Not used by neural network

### 5. **First Letter Weight** (`swipe_first_letter_weight`)
- **Default**: 150 (1.50)
- **Range**: 0-300
- **Purpose**: Extra weight/bonus for correctly matching the first letter of the word
- **Calculation**: Strong boost if swipe starts on the correct key for first letter
- **When Used**: Critical for disambiguating words with similar swipe paths
- **Current Status**: ❌ Not used by neural network (NN detects endpoints automatically)

### 6. **Last Letter Weight** (`swipe_last_letter_weight`)
- **Default**: 150 (1.50)
- **Range**: 0-300
- **Purpose**: Extra weight/bonus for correctly matching the last letter
- **Calculation**: Strong boost if swipe ends on the correct key for last letter
- **When Used**: Helps distinguish words with same beginning but different endings
- **Current Status**: ❌ Not used by neural network (NN detects endpoints automatically)

### 7. **Endpoint Bonus Weight** (`swipe_endpoint_bonus_weight`)
- **Default**: 200 (2.00)
- **Range**: 0-400
- **Purpose**: Multiplier applied when BOTH first and last letters match correctly
- **Calculation**: Applied on top of first_letter_weight + last_letter_weight
- **When Used**: Maximum confidence boost for words where endpoints are certain
- **Current Status**: ❌ Not used by neural network

### 8. **Require Endpoints** (`swipe_require_endpoints`)
- **Default**: false
- **Type**: Boolean checkbox
- **Purpose**: If enabled, ONLY show predictions where first/last letters match swipe endpoints
- **Calculation**: Hard filter - removes candidates that don't match endpoints
- **When Used**: Strict mode for users who want only precise endpoint matches
- **Current Status**: ❌ Not used by neural network

## Legacy Scoring Formula

The original WordPredictor calculated confidence scores using:

```
total_score =
  (shape_similarity * shape_weight) +
  (location_proximity * location_weight) +
  (word_frequency * frequency_weight) +
  (velocity_consistency * velocity_weight) +
  (first_letter_match ? first_letter_weight : 0) +
  (last_letter_match ? last_letter_weight : 0) +
  (both_endpoints_match ? endpoint_bonus_weight : 0)

if (require_endpoints && !both_endpoints_match) {
  reject_candidate()
}
```

## Current Neural Network Approach

The ONNX neural network (v1.32+) uses:

1. **Character-level LSTM** trained on millions of swipe gestures
2. **Beam search** with configurable width (default: 2 beams for mobile performance)
3. **Internal confidence scores** based on model training, not manual weights
4. **Automatic endpoint detection** without requiring explicit configuration
5. **Contextual predictions** using previous words in sentence

The neural approach is **significantly more accurate** because it learned patterns from real data rather than using hand-tuned weights.

## Neural Settings (Currently Active)

These settings in **Settings → Advanced** control the **actual neural network** behavior:

- **Neural Prediction Enabled**: Enable/disable ONNX model (default: true)
- **Beam Width**: Number of parallel prediction paths (default: 2, range: 1-16)
  - Higher = more accurate but slower
  - 2 beams = 4x faster than 8 beams with minimal accuracy loss
- **Max Word Length**: Maximum characters to predict (default: 35, range: 10-50)
- **Confidence Threshold**: Minimum NN confidence to show prediction (default: 0.1, range: 0.0-1.0)
- **Detailed Logging**: Enable verbose swipe trajectory/NN logging for debugging
- **Show Raw Output**: Always show at least 2 raw NN outputs with confidence scores

## Why Keep Legacy Settings?

1. **Backwards Compatibility**: Old code still references them in Config.java
2. **Future Hybrid System**: May combine neural predictions with heuristic scoring
3. **Debugging**: Useful for comparing neural vs. heuristic approaches
4. **User Expectations**: Existing users may have customized these settings

## Recommendation

**For best swipe typing experience**, use default neural settings and ignore the legacy weight parameters. The neural network handles all aspects of prediction quality automatically.

## Technical Details

- **Location in Code**: `Config.java:77-84` (fields), `Config.java:223-231` (loading)
- **Storage**: SharedPreferences, loaded via `safeGetInt()` helper
- **Format**: Stored as integers (0-400), divided by 100 to get float weights
- **Used By**: Legacy `WordPredictor.java` (not actively used for swipe typing)
- **Neural Code**: `OnnxSwipePredictor.java`, `NeuralSwipeTypingEngine.java`
