# Configuration Audit - User-Settable Values

## Summary

Audit of all swipe/neural/short gesture configuration values to verify they are actually used in the code.

## Neural Prediction Settings ✅ ALL VERIFIED

| Setting | Default | Used? | Location |
|---------|---------|-------|----------|
| `neural_prediction_enabled` | `true` | ✅ YES | Config check in multiple places |
| `neural_beam_width` | `2` | ✅ YES | `OnnxSwipePredictor.java:738` |
| `neural_max_length` | `20` | ✅ YES | `OnnxSwipePredictor.java:739` |
| `neural_confidence_threshold` | `0.1f` | ✅ YES | `OnnxSwipePredictor.java:740-741` |

**Status**: All neural settings are properly used. Hardcoded defaults in `OnnxSwipePredictor.java` match Config.java defaults.

```java
// OnnxSwipePredictor.java lines 45-47
private static final int DEFAULT_BEAM_WIDTH = 2; // ✅ Matches Config default
private static final int DEFAULT_MAX_LENGTH = 20; // ✅ Matches Config default
private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f; // ✅ Matches Config default
```

## Short Gesture Settings ✅ VERIFIED

| Setting | Default | Used? | Location |
|---------|---------|-------|----------|
| `short_gestures_enabled` | `true` | ✅ YES | `Pointers.java:226` (condition check) |
| `short_gesture_min_distance` | `20%` | ✅ YES | `Pointers.java:233` (distance calculation) |

**Status**: Short gesture settings are properly used.

```java
// Pointers.java line 233
float minDistance = keyHypotenuse * (_config.short_gesture_min_distance / 100.0f);
```

**Recent Change**: Default was `30%` → changed to `20%` in v1.32.64 for better reliability.

## Legacy Swipe Weights ⚠️ HARDCODED (WordPredictor only)

| Setting | Default | Used? | Notes |
|---------|---------|-------|-------|
| `swipe_confidence_shape_weight` | `90%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_confidence_location_weight` | `130%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_confidence_frequency_weight` | `80%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_confidence_velocity_weight` | `60%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_first_letter_weight` | `150%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_last_letter_weight` | `150%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_endpoint_bonus_weight` | `200%` | ✅ YES | WordPredictor (non-swipe suggestions) |
| `swipe_require_endpoints` | `false` | ✅ YES | WordPredictor (non-swipe suggestions) |

**Status**: These settings ARE used by `WordPredictor.java` for **non-swipe word suggestions** (the suggestion bar when typing normally). They are NOT used by the neural swipe system (`OnnxSwipePredictor`).

**Resolution**: Hardcoded in Config.java with default values. **NO UI SETTINGS** - not exposed to users since:
1. WordPredictor is legacy code for non-swipe suggestions only
2. Neural swipe system (OnnxSwipePredictor) doesn't use these
3. Keeping code compatibility without cluttering UI

**Evidence**:
```bash
$ grep -rn "swipe_first_letter_weight" srcs/juloo.keyboard2/ --include="*.java"
WordPredictor.java:387: baseScore = (int)(baseScore * _config.swipe_first_letter_weight);
WordPredictor.java:422: baseScore = (int)(baseScore * _config.swipe_first_letter_weight);
# Used in Keyboard2.java:1117 for predictWordsWithContext() - non-swipe suggestions
```

**Two Prediction Systems**:
- **WordPredictor** (legacy): Pattern matching for non-swipe typing suggestions
- **OnnxSwipePredictor** (neural): ONNX transformer for swipe gesture recognition

## Other Swipe Settings

| Setting | Default | Used? | Notes |
|---------|---------|-------|-------|
| `swipe_typing_enabled` | `false` | ✅ YES | Master toggle for swipe typing |
| `swipe_show_debug_scores` | `false` | ⚠️ TBD | Need to verify |
| `word_prediction_enabled` | `false` | ✅ YES | Master toggle for predictions |
| `suggestion_bar_opacity` | `90` | ✅ YES | UI rendering |

## Action Items

### Completed ✅
1. ✅ **Legacy swipe weights** - Hardcoded in Config.java, removed from UI settings
2. ✅ **Neural settings verified** - All properly used by OnnxSwipePredictor
3. ✅ **"Add keys to keyboard" verified** - ExtraKeysPreference/CustomExtraKeysPreference unaffected by short gesture refactor
4. ✅ **short_gesture_min_distance = 20%** - Documented change from 30% in v1.32.64
5. ✅ **Hardcoded defaults match Config** - Verified for all neural and short gesture settings
6. ✅ **GitHub Actions CI/CD** - Created automated build pipeline with #RELEASE detection

### Still TODO
1. ⚠️ **Verify swipe_show_debug_scores** - Check if debug score display still works with neural system
2. ⚠️ **Check for other hardcoded values** - Search codebase for magic numbers that should be configurable

## Files Checked

### Configuration
- `srcs/juloo.keyboard2/Config.java` (lines 74-96, 217-242)

### Neural Prediction
- `srcs/juloo.keyboard2/OnnxSwipePredictor.java` (lines 45-47, 738-741)
- `srcs/juloo.keyboard2/NeuralSwipeTypingEngine.java` (no legacy weight usage)
- `srcs/juloo.keyboard2/SwipeCalibrationActivity.java` (UI for neural settings)

### Short Gestures
- `srcs/juloo.keyboard2/Pointers.java` (lines 226, 233, 236)

### Legacy Systems (now unused)
- No files found using legacy swipe confidence weights
- Old `SwipeTypingEngine.java` was replaced by `NeuralSwipeTypingEngine.java`
- Old DTW/Bayesian system completely removed

## Questions for User

1. **"Add keys to keyboard" logic** - What specific functionality does this refer to?
   - Extra keys system?
   - Compose/dead keys?
   - Context menu actions?
   - Something in short gesture handling?

2. **Legacy weights removal** - Should we:
   - Remove completely from code and UI?
   - Mark as deprecated but keep for compatibility?
   - Convert to new neural-equivalent settings?

3. **Debug score display** - Is `swipe_show_debug_scores` still functional with ONNX system?

## Verification Commands

```bash
# Check if setting is used
grep -rn "SETTING_NAME" srcs/juloo.keyboard2/ --include="*.java" | grep -v "Config.java"

# Check neural settings
grep -rn "neural_beam_width\|neural_max_length\|neural_confidence_threshold" srcs/juloo.keyboard2/ --include="*.java"

# Check legacy weights
grep -rn "swipe_confidence_shape_weight\|swipe_confidence_location_weight" srcs/juloo.keyboard2/ --include="*.java"

# Check short gesture settings
grep -rn "short_gesture" srcs/juloo.keyboard2/ --include="*.java"
```

## Resolution Summary (v1.32.79)

**Problem**: 8 legacy swipe weight config values appeared unused after neural system implementation.

**Investigation**:
- Confirmed neural swipe (OnnxSwipePredictor) does NOT use these weights
- Discovered WordPredictor (non-swipe suggestions) DOES still use them
- Two separate prediction systems coexist:
  - **NeuralSwipeTypingEngine** → **OnnxSwipePredictor** (neural ONNX for swipe gestures)
  - **WordPredictor** (pattern matching for non-swipe typing suggestions)

**Resolution**:
- Hardcoded legacy weights in Config.java with default values (no SharedPreferences loading)
- Removed UI settings exposure (cleaner settings screen)
- Maintained backward compatibility with WordPredictor
- Neural swipe uses only: `neural_beam_width`, `neural_max_length`, `neural_confidence_threshold`

**Files Modified**:
- `srcs/juloo.keyboard2/Config.java` - Hardcoded 8 legacy weights
- `memory/config_audit.md` - Updated documentation

**Build**: v1.32.79 - All changes compile successfully

## Audit Date

2025-10-17 (v1.32.65-v1.32.79)
