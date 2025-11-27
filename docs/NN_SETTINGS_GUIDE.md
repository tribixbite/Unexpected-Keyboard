# Neural Network Settings Guide
**Version**: 1.32.340+
**Last Updated**: 2025-11-13

## 🎯 Quick Start

The Neural Prediction system has **17 working settings** that significantly affect swipe typing accuracy and behavior. This guide explains what each setting does and how to optimize them.

---

## ✅ Core Settings (Essential)

### Prediction Source (0-100)
**Location**: Settings → Swipe Corrections → Advanced Swipe Tuning
**Default**: 60
**Status**: ✅ **NOW WORKS** (fixed in v1.32.340)

Controls the balance between neural network confidence and dictionary frequency:
- **0**: Pure dictionary (ignores NN confidence completely)
- **50**: Balanced (equal weight to NN and dictionary)
- **60**: Slightly favor NN (recommended default)
- **100**: Pure AI (ignores dictionary frequency)

**When to adjust**:
- **Increase (70-90)**: If you have good swipe technique and want the NN to dominate
- **Decrease (30-50)**: If you have many custom words or prefer dictionary-based suggestions
- **Set to 100**: For testing NN raw output (not recommended for daily use)

---

## 🧠 Neural Core Settings

### Beam Width (1-16)
**Location**: Settings → Neural Prediction → Beam Search
**Default**: 2
**Status**: ✅ WORKS

Controls how many prediction paths are explored simultaneously.

**Trade-offs**:
- **Width = 1**: Greedy search (fastest, lowest accuracy)
- **Width = 2**: Mobile-optimized (balanced speed/accuracy) ⭐ **RECOMMENDED**
- **Width = 8**: Desktop/web quality (slower, higher accuracy)
- **Width = 16**: Maximum quality (very slow on mobile)

**Performance impact**:
- Beam width × max length = decoder calls per swipe
- Width 2 × Length 35 = 70 calls (fast)
- Width 8 × Length 35 = 280 calls (slow)

**When to adjust**:
- **Increase to 4-8**: If you have a fast device and want better long-word accuracy
- **Keep at 2**: For smooth experience on most devices
- **Decrease to 1**: Only if experiencing severe lag (not recommended)

### Maximum Length (10-50)
**Location**: Settings → Neural Prediction → Beam Search
**Default**: 35
**Status**: ✅ WORKS

Maximum word length the neural network will predict.

**When to adjust**:
- **Increase to 40-50**: If you frequently swipe long words (e.g., "internationally")
- **Decrease to 25-30**: Slight performance improvement for shorter words only
- **Keep at 35**: Handles most English words well ⭐ **RECOMMENDED**

### Model Version
**Location**: Settings → Neural Prediction → Model Configuration
**Default**: v2 (builtin)
**Status**: ✅ WORKS (fixed in v1.32.339)

Select which ONNX model to use:
- **v2**: Builtin 250-point model (80.6% accuracy) ⭐ **RECOMMENDED**
- **v1**: External 150-point model (65% accuracy, legacy)
- **v3**: External 250-point model (72.1% accuracy, if available)
- **custom**: Your own trained model

**Using external models**:
1. Tap "📁 Load Encoder Model" → select `swipe_encoder_android.onnx`
2. Tap "📁 Load Decoder Model" → select `swipe_decoder_android.onnx`
3. Change "Model Version" to "custom"
4. Swipe to test (check logcat for "External model loaded from content URI")

**Note**: Keyboard restarts automatically when model changes.

### User Max Sequence Length (0-400)
**Location**: Settings → Neural Prediction → Model Configuration
**Default**: 0 (use model default)
**Status**: ✅ WORKS

Override the model's native sequence length.

**When to set**:
- **0**: Use model's default (250 for v2, 150 for v1) ⭐ **RECOMMENDED**
- **Custom value**: Only if using a custom-trained model with different sequence length
- **Leave at 0** unless you know your model's training configuration

### Resampling Mode
**Location**: Settings → Neural Prediction → Model Configuration
**Default**: discard
**Status**: ✅ WORKS

How to handle swipes with more points than max_sequence_length:
- **discard**: Drop the swipe (show error)
- **truncate**: Cut trajectory to fit (may lose accuracy)
- **merge**: Combine nearby points intelligently ⭐ **RECOMMENDED**

**When to adjust**:
- **Keep on "discard"**: For debugging (you'll know if trajectories are too long)
- **Use "merge"**: For production use (handles edge cases gracefully)

---

## 🎨 Scoring Weights

### Common Words Boost (0.5-2.0)
**Default**: 1.3
**Status**: ✅ WORKS

Multiplier for top 100 most common words (Tier 2: "the", "and", "you", etc.)

**When to adjust**:
- **Increase to 1.5-2.0**: If common words aren't appearing enough
- **Decrease to 1.0**: If you want rarer words to compete equally
- **Keep at 1.3**: Balanced preference for common words ⭐ **RECOMMENDED**

### Frequent Words Boost (0.5-2.0)
**Default**: 1.0
**Status**: ✅ WORKS

Multiplier for top 3000 frequent words (Tier 1: everyday vocabulary)

**When to adjust**:
- **Increase to 1.2-1.5**: If you use standard vocabulary heavily
- **Keep at 1.0**: Neutral (no boost or penalty) ⭐ **RECOMMENDED**
- **Decrease to 0.8**: If you prefer technical/uncommon words

### Rare Words Penalty (0.0-1.5)
**Default**: 0.75
**Status**: ✅ WORKS

Multiplier for infrequent words (Tier 0: rest of dictionary)

**When to adjust**:
- **Decrease to 0.5**: Strongly prefer common words
- **Increase to 1.0**: No penalty (all words equal weight)
- **Keep at 0.75**: Moderate penalty for rare words ⭐ **RECOMMENDED**

---

## ✨ Autocorrect Settings

### Enable Beam Search Corrections
**Default**: ✅ Enabled
**Status**: ✅ WORKS

Master switch for fuzzy matching during prediction.

**What it does**:
- Matches custom words against NN predictions (e.g., "Jhn" → "John")
- Applies dictionary fuzzy matching (e.g., "teh" → "the")
- Works during beam search (before final selection)

**When to disable**:
- If you want only exact NN predictions (no autocorrection)
- For debugging raw NN output

### Typo Forgiveness (0-5)
**Setting**: `autocorrect_max_length_diff`
**Default**: 2
**Status**: ✅ WORKS

Maximum length difference between swipe and correction.

**Examples**:
- **Diff = 2**: "btw" (3) can match "between" (7) ✅
- **Diff = 1**: "btw" (3) can only match up to length 4
- **Diff = 0**: Exact length match only

**When to adjust**:
- **Increase to 3-4**: If you swipe imprecisely and need aggressive correction
- **Decrease to 1**: If you're getting unwanted long-word corrections
- **Keep at 2**: Handles most typos well ⭐ **RECOMMENDED**

### Starting Letter Accuracy (0-4)
**Setting**: `autocorrect_prefix_length`
**Default**: 2
**Status**: ✅ WORKS

How many initial letters must match for correction.

**Examples**:
- **Prefix = 2**: "teh" can match "the" (both start with "th") ✅
- **Prefix = 3**: "teh" cannot match "the" (different 3rd letter)
- **Prefix = 0**: Any starting letter allowed (very aggressive)

**When to adjust**:
- **Increase to 3**: For more conservative corrections
- **Decrease to 1**: If you often miss the first letter
- **Keep at 2**: Good balance ⭐ **RECOMMENDED**

### Correction Search Depth (1-10)
**Setting**: `autocorrect_max_beam_candidates`
**Default**: 3
**Status**: ✅ WORKS

How many top NN predictions to attempt autocorrect on.

**Trade-offs**:
- **Depth = 1**: Fast, only corrects top prediction
- **Depth = 3**: Balanced (checks top 3) ⭐ **RECOMMENDED**
- **Depth = 10**: Slow, exhaustive search

**When to adjust**:
- **Increase to 5-10**: If correct word is never showing up
- **Keep at 3**: Optimal for speed/coverage balance

### Character Match Threshold (0.5-0.9)
**Default**: 0.67 (2/3 of characters)
**Status**: ✅ WORKS

Minimum character similarity ratio for fuzzy matching.

**Examples**:
- **0.67**: "teh" (3 chars) needs 2 matching → "the" ✅
- **0.75**: "teh" needs 2.25 matching → stricter
- **0.5**: "teh" needs 1.5 matching → very loose

**When to adjust**:
- **Increase to 0.75-0.8**: For stricter, more conservative corrections
- **Decrease to 0.5-0.6**: For aggressive autocorrect
- **Keep at 0.67**: Good middle ground ⭐ **RECOMMENDED**

### Matching Algorithm
**Setting**: `swipe_fuzzy_match_mode`
**Default**: edit_distance
**Status**: ✅ WORKS

Algorithm for calculating word similarity:
- **edit_distance**: Levenshtein distance (character insertions/deletions/substitutions) ⭐ **RECOMMENDED**
- **positional**: Based on key positions on keyboard (considers QWERTY layout)

**When to adjust**:
- **Keep "edit_distance"**: More accurate for most cases
- **Use "positional"**: If you make systematic finger-placement errors

---

## ⚠️ Partially Working Settings

### Confidence Threshold (0.0-1.0)
**Default**: 0.1
**Status**: ⚠️ PARTIAL (only used in fallback path)

Filters predictions below confidence threshold.

**Current behavior**: Only applies if OptimizedVocabulary fails to load. In normal operation, vocabulary filtering handles quality control instead.

**Recommendation**: Leave at 0.1 (default)

---

## ❌ Not Implemented Settings

These settings exist in the UI but **don't affect swipe typing** (they're for different pipelines):

### Regular Typing Autocorrect
- `autocorrect_enabled` - Global autocorrect for tap typing
- `prediction_context_boost` - N-gram context for typing predictions
- `prediction_frequency_scale` - Typing frequency scaling
- `word_prediction_enabled` - Word prediction bar for typing

**Status**: ❌ NOT IMPLEMENTED (separate prediction engine not provided)

### Post-Swipe Autocorrect
- `swipe_final_autocorrect_enabled` - Autocorrect word after selection

**Status**: ❌ NOT IMPLEMENTED (future feature)

---

## 🎮 Recommended Presets

### Balanced (Default)
```
Prediction Source: 60
Beam Width: 2
Max Length: 35
Common Words Boost: 1.3
Frequent Words Boost: 1.0
Rare Words Penalty: 0.75
Beam Autocorrect: Enabled
Max Length Diff: 2
Prefix Length: 2
Max Beam Candidates: 3
Character Match: 0.67
Fuzzy Mode: edit_distance
```

### Accuracy-Focused (Slower but better)
```
Prediction Source: 70
Beam Width: 8
Max Length: 40
Common Words Boost: 1.5
Frequent Words Boost: 1.2
Rare Words Penalty: 0.5
Beam Autocorrect: Enabled
Max Length Diff: 3
Prefix Length: 2
Max Beam Candidates: 5
Character Match: 0.6
Fuzzy Mode: edit_distance
```

### Speed-Focused (Faster but simpler)
```
Prediction Source: 60
Beam Width: 2
Max Length: 30
Common Words Boost: 1.5
Frequent Words Boost: 1.0
Rare Words Penalty: 0.7
Beam Autocorrect: Enabled
Max Length Diff: 1
Prefix Length: 2
Max Beam Candidates: 1
Character Match: 0.75
Fuzzy Mode: edit_distance
```

### Custom Vocabulary (Technical/Uncommon words)
```
Prediction Source: 80 (favor NN over dictionary)
Beam Width: 4
Max Length: 40
Common Words Boost: 1.0 (no boost)
Frequent Words Boost: 1.0
Rare Words Penalty: 1.0 (no penalty)
Beam Autocorrect: Disabled (or with loose settings)
Max Length Diff: 3
Prefix Length: 1
Max Beam Candidates: 5
Character Match: 0.5
Fuzzy Mode: edit_distance
```

---

## 🐛 Testing & Debugging

### Enable Detailed Logging
**Location**: Settings → Swipe Debug Log

Enable these for troubleshooting:
- **Detailed Pipeline Logging**: See trajectory processing, NN internals
- **Show Raw NN Output**: See at least 2 raw predictions before filtering
- **Show Raw Beam Predictions**: See all beam search outputs (labeled with "raw:")

### Checking Logcat
```bash
# Monitor keyboard logs
adb logcat | grep -E "OnnxSwipePredictor|OptimizedVocabulary|NeuralSwipe"

# Check if settings are applied
adb logcat | grep "confidence.*weight\|prediction.*source"

# Check model loading
adb logcat | grep "model.*loaded\|External model"
```

### Verify Settings Work
1. **Prediction Source**: Change to 0 (should prefer dictionary) vs 100 (should prefer NN)
2. **Beam Width**: Change to 1 (faster, fewer options) vs 8 (slower, more options)
3. **Common Words Boost**: Change to 2.0 (should strongly prefer "the", "and", "you")
4. **Beam Autocorrect**: Disable (should see raw NN output without fuzzy matching)

---

## 📊 Performance Impact Chart

| Setting | Low Impact | Medium Impact | High Impact |
|---------|------------|---------------|-------------|
| Prediction Source | ✅ | - | - |
| Beam Width | - | - | ✅ |
| Max Length | ✅ | - | - |
| Word Boosts | ✅ | - | - |
| Autocorrect Enable | - | ✅ | - |
| Max Beam Candidates | - | ✅ | - |
| Fuzzy Match Mode | ✅ | - | - |

**Legend**:
- ✅ Low: Negligible performance impact
- ⚠️ Medium: Noticeable on older devices
- 🔴 High: Significant impact, adjust carefully

---

## 🔧 Troubleshooting

### Predictions seem random/don't match swipe
- Check **Prediction Source** (should be 50-70)
- Enable **Beam Autocorrect**
- Increase **Max Beam Candidates** to 5

### Common words not appearing
- Increase **Common Words Boost** to 1.5-2.0
- Decrease **Prediction Source** to 40-50

### Rare/technical words not appearing
- Increase **Prediction Source** to 80-100
- Increase **Rare Words Penalty** to 1.0
- Decrease **Common Words Boost** to 1.0

### Lag during swiping
- Decrease **Beam Width** to 2 or 1
- Decrease **Max Length** to 30
- Decrease **Max Beam Candidates** to 1

### External models not loading
- Check Settings → Neural Prediction → Model Configuration
- Ensure both encoder AND decoder are selected
- Change "Model Version" to "custom" AFTER loading files
- Check logcat for "External model loaded" or permission errors

---

## 📝 Version History

- **v1.32.340**: Fixed Prediction Source slider (was completely broken)
- **v1.32.339**: Fixed external model file picker (was not loading)
- **v1.32.337**: Added clipboard date filter
- **v1.32.281**: Fixed src_mask in beam search
- **v1.32.280**: Fixed trajectory preprocessing (calculate features before padding)

---

## 🤝 Credits

- Neural network training and architecture by FUTO
- Settings audit and bug fixes by Claude (Anthropic) + Gemini 2.5 Pro (Google)
- Unexpected Keyboard by Jules Aguillon

---

**Need more help?** Check the [GitHub issues](https://github.com/Julow/Unexpected-Keyboard/issues) or ask in discussions.
