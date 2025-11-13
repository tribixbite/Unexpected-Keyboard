# Testing Checklist for NN Fixes (v1.32.339-340)

## Pre-Test Setup
- [ ] Install APK: `/storage/emulated/0/unexpected/debug-kb.apk`
- [ ] Enable keyboard in Android Settings → System → Languages & input
- [ ] Set as default keyboard
- [ ] Enable logcat monitoring: `adb logcat | grep -E "Keyboard2|OnnxSwipe|OptimizedVocabulary"`

---

## Test 1: External Model File Picker (v1.32.339)

### Setup
- [ ] Go to Settings → Neural Prediction → Model Configuration
- [ ] Tap "📁 Load Encoder Model"
- [ ] Select `swipe_encoder_android.onnx` file
- [ ] Verify toast: "✅ Encoder loaded: swipe_encoder_android.onnx"
- [ ] Tap "📁 Load Decoder Model"
- [ ] Select `swipe_decoder_android.onnx` file
- [ ] Verify toast: "✅ Decoder loaded: swipe_decoder_android.onnx"
- [ ] **Expected toast**: "✅ Files loaded. Now, change 'Model Version' to 'custom' to use them."

### Test Custom Model Loading
- [ ] Change "Model Version" dropdown to "custom"
- [ ] Keyboard should restart automatically
- [ ] Open any text field and perform a swipe
- [ ] Check logcat for: `"Loading external ONNX model from URI: content://"`
- [ ] Check logcat for: `"External model loaded from content URI"`
- [ ] Check logcat for: `"Encoder model loaded: custom"`
- [ ] Check logcat for: `"Decoder model loaded: custom"`
- [ ] **Should NOT see**: "External model files not configured. Using builtin v2 model."

### Test Fallback (Without Files)
- [ ] Change "Model Version" to "v1" or "v3" WITHOUT loading files
- [ ] **Expected**: Toast "External model files not configured. Using builtin v2 model."
- [ ] Verify fallback to v2 in logcat

### Test Path Change Detection
- [ ] With custom model loaded, select DIFFERENT encoder file
- [ ] Perform swipe
- [ ] **Expected**: Model reloads (check logcat for "Model config changed: pathsChanged=true")

**Status**: ⬜ Pass / ⬜ Fail
**Notes**:

---

## Test 2: Prediction Source Slider (v1.32.340)

### Setup
- [ ] Go to Settings → Swipe Corrections → Advanced Swipe Tuning
- [ ] Enable "Detailed Pipeline Logging" in Settings → Swipe Debug Log
- [ ] Find "Prediction Source" slider (currently shows "Balance between dictionary and AI")

### Test Slider Values

#### Test A: Pure Dictionary (0)
- [ ] Set "Prediction Source" to 0
- [ ] Swipe a common word (e.g., "hello")
- [ ] Check logcat for scoring: `"confidenceWeight"`
- [ ] **Expected in logcat**: `confidenceWeight = 0.0, frequencyWeight = 1.0`
- [ ] Predictions should heavily favor dictionary frequency

#### Test B: Balanced (50)
- [ ] Set "Prediction Source" to 50
- [ ] Swipe the same word
- [ ] **Expected in logcat**: `confidenceWeight = 0.5, frequencyWeight = 0.5`
- [ ] Equal weight to NN and dictionary

#### Test C: Pure AI (100)
- [ ] Set "Prediction Source" to 100
- [ ] Swipe the same word
- [ ] **Expected in logcat**: `confidenceWeight = 1.0, frequencyWeight = 0.0`
- [ ] Should show raw NN output, ignoring dictionary frequency

#### Test D: Default (60)
- [ ] Set "Prediction Source" to 60 (default)
- [ ] Swipe the same word
- [ ] **Expected in logcat**: `confidenceWeight = 0.6, frequencyWeight = 0.4`
- [ ] Slightly favors NN over dictionary

### Verification
- [ ] Try swiping a rare word with high NN confidence
- [ ] At 0 (dict): Should rank low or not appear
- [ ] At 100 (AI): Should rank high if NN is confident
- [ ] **Behavior should clearly differ between 0 and 100**

**Status**: ⬜ Pass / ⬜ Fail
**Notes**:

---

## Test 3: Working Settings Verification

### Test Beam Width
- [ ] Settings → Neural Prediction → Beam Search → Beam Width
- [ ] Set to 1 (greedy)
- [ ] Swipe "hello" - note predictions
- [ ] Set to 8
- [ ] Swipe "hello" again
- [ ] **Expected**: More diverse predictions with width=8

**Status**: ⬜ Pass / ⬜ Fail

### Test Common Words Boost
- [ ] Settings → Swipe Corrections → Advanced Swipe Tuning → Common Words Boost
- [ ] Set to 2.0
- [ ] Swipe imprecise "the"
- [ ] **Expected**: "the" appears at top despite imprecision
- [ ] Set to 0.5
- [ ] Swipe again
- [ ] **Expected**: Other words may outrank "the"

**Status**: ⬜ Pass / ⬜ Fail

### Test Beam Autocorrect
- [ ] Settings → Swipe Corrections → Enable Beam Search Corrections
- [ ] Disable it
- [ ] Swipe "teh"
- [ ] **Expected**: Raw NN output (may show "teh" or similar)
- [ ] Enable it
- [ ] Swipe "teh" again
- [ ] **Expected**: Should autocorrect to "the"

**Status**: ⬜ Pass / ⬜ Fail

### Test Typo Forgiveness (Max Length Diff)
- [ ] Settings → Swipe Corrections → Fuzzy Matching → Typo Forgiveness
- [ ] Set to 0
- [ ] Swipe "btw" (3 chars)
- [ ] **Expected**: Cannot match "between" (7 chars, diff > 0)
- [ ] Set to 4
- [ ] Swipe "btw" again
- [ ] **Expected**: Can now match "between"

**Status**: ⬜ Pass / ⬜ Fail

---

## Test 4: Performance Check

### Baseline (Default Settings)
- [ ] Reset all settings to defaults
- [ ] Beam Width: 2, Max Length: 35, Prediction Source: 60
- [ ] Swipe 10 common words quickly
- [ ] Note responsiveness (should be smooth)

**Responsiveness**: ⬜ Smooth / ⬜ Laggy
**Average time**: _____ ms per swipe

### High Performance Test
- [ ] Beam Width: 8, Max Length: 50, Max Beam Candidates: 10
- [ ] Swipe the same 10 words
- [ ] Note if lag increases

**Responsiveness**: ⬜ Smooth / ⬜ Laggy
**Average time**: _____ ms per swipe
**Lag increased**: ⬜ Yes / ⬜ No

---

## Test 5: Edge Cases

### Very Long Words
- [ ] Set Max Length: 50
- [ ] Swipe "internationally" or "consciousness"
- [ ] **Expected**: Word appears in predictions

### Very Short Swipes
- [ ] Swipe very quickly (3-5 points only)
- [ ] Check if resampling mode handles it
- [ ] **Expected**: Either prediction or graceful error

### Rapid Swiping
- [ ] Swipe 5 words rapidly in succession
- [ ] **Expected**: No crashes, predictions appear for each

### Custom Words
- [ ] Add custom word to dictionary: "mytestword123"
- [ ] Swipe it (should match via autocorrect)
- [ ] Enable "Beam Autocorrect"
- [ ] **Expected**: Custom word appears in suggestions

---

## Test 6: Config Reload (Critical)

### Test Live Updates
- [ ] Open keyboard in a text field
- [ ] While keyboard is open, go to Settings
- [ ] Change "Prediction Source" from 60 to 0
- [ ] Return to text field (keep keyboard open)
- [ ] Swipe a word
- [ ] Check logcat for: "Neural model setting changed: swipe_prediction_source - engine config updated"
- [ ] **Expected**: New setting applied WITHOUT keyboard restart

**Status**: ⬜ Pass / ⬜ Fail
**Critical**: This tests the OnSharedPreferenceChangeListener fix

---

## Logcat Commands

```bash
# Monitor all keyboard activity
adb logcat -s Keyboard2:D OnnxSwipePredictor:D OptimizedVocabulary:D NeuralSwipeTypingEngine:D

# Check config changes
adb logcat | grep "setting changed\|config updated\|reinitialization"

# Check scoring weights
adb logcat | grep -E "confidenceWeight|frequencyWeight|predictionSource"

# Check model loading
adb logcat | grep -E "model loaded|External model|fallback"

# Check performance
adb logcat | grep -E "prediction took|inference time"
```

---

## Known Issues to Verify Fixed

- [x] **Fixed v1.32.339**: External models not loading (always fell back to v2)
- [x] **Fixed v1.32.340**: Prediction Source slider had no effect (always used 0.6/0.4)
- [ ] **Test both**: Verify fixes work as documented

---

## Success Criteria

✅ **Test 1 passes**: External models load correctly, no fallback toast
✅ **Test 2 passes**: Slider changes confidenceWeight values in logcat
✅ **Test 3 passes**: At least 3 of 4 settings show different behavior when changed
✅ **Test 4 passes**: Performance acceptable with default settings
✅ **Test 6 passes**: Config changes apply WITHOUT keyboard restart

---

## Test Results Summary

**Date**: _____________
**Tester**: _____________
**Device**: _____________
**Android Version**: _____________

**Overall Status**: ⬜ All Pass / ⬜ Some Fail / ⬜ Major Issues

**Critical Bugs Found**:
-
-
-

**Performance Notes**:
-
-

**Recommendations**:
-
-
