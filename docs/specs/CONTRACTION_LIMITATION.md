# Contraction and Apostrophe Limitation

**Date**: 2025-11-02
**Version**: v1.32.232
**Status**: ⚠️ **CRITICAL DESIGN LIMITATION**

---

## Executive Summary

The swipe typing system **cannot predict contractions** (don't, can't, it's, etc.) despite the dictionary containing them. This is a fundamental model architecture limitation: the neural network tokenizer only supports 26 letters (a-z), with no apostrophe token.

**Impact**:
- **40+ common contractions** in dictionary are **unpredictable**
- **Calibration page misleads users** by including contractions
- **User experience degraded** for natural English typing

---

## The Problem

### Dictionary Contains Contractions

The 50k vocabulary (`assets/dictionaries/en_enhanced.json`) includes **40+ contractions**:

```json
"it's": 253,
"don't": 253,
"i'm": 253,
"you're": 250,
"can't": 250,
"that's": 250,
"we're": 243,
"won't": 242,
"i'll": 244,
"he's": 247,
"she's": 241,
"wasn't": 241,
"isn't": 243,
"didn't": 248,
"wouldn't": 239,
"hasn't": ...,
"haven't": 237,
... (40+ total)
```

**Frequency Scores**: 239-253 (very high - these are common words!)

---

### Tokenizer Only Supports a-z

**File**: `assets/models/tokenizer.json`

```json
{
  "vocab_size": 30,
  "char_to_idx": {
    "<pad>": 0,
    "<unk>": 1,
    "<sos>": 2,
    "<eos>": 3,
    "a": 4, "b": 5, ..., "z": 29
  }
}
```

**Total Tokens**: 30 (4 special + 26 letters)
**Apostrophe Token**: ❌ **DOES NOT EXIST**

---

### Neural Network Cannot Output Apostrophes

**Component**: `SwipeTokenizer.java` (lines 122-127)

```java
// Alphabet (a-z) - exactly matching web demo
int idx = 4;
for (char c = 'a'; c <= 'z'; c++)
{
  addMapping(idx++, c);
}
// No extra symbols - web demo only uses 4 special tokens + 26 letters = 30 total
```

**Beam Search Decoder Output**: Can only produce sequences of tokens 4-29 (a-z)

**Result**: Neural network physically incapable of generating "don't" or any word with apostrophe

---

### Calibration Page Includes Contractions

**File**: `SwipeCalibrationActivity.java` (lines 241-258)

```java
String[] dictFiles = {"dictionaries/en.txt", "dictionaries/en_enhanced.txt"};

for (String dictFile : dictFiles)
{
  while ((line = reader.readLine()) != null)
  {
    line = line.trim().toLowerCase();
    if (!line.isEmpty() && uniqueWords.add(line))
    {
      fullVocabulary.add(line);  // Includes "don't", "can't", etc.
    }
  }
}
```

**Random Selection**: Picks 20 words from full vocabulary including contractions

**User Experience**:
1. User swipes "don't" (calibration prompt)
2. Neural network predicts "dont" (no apostrophe possible)
3. Vocabulary filtering: "dont" ❌ NOT IN DICTIONARY
4. Result: **NO PREDICTIONS** or raw output only
5. Calibration shows **0% accuracy** for contractions
6. User confused: "Why can't it predict this common word?"

---

## Root Cause Analysis

### Why Tokenizer Excludes Apostrophe

Looking at the code comments:

```java
// No extra symbols - web demo only uses 4 special tokens + 26 letters = 30 total
```

**Reason**: The model was trained using a **web demo** that only supported lowercase a-z.

**Design Choice**: Simplicity over completeness
- Fewer tokens = smaller model
- Faster inference
- But sacrifices punctuation support

---

### Model Training Data

The ONNX models (`swipe_model.onnx`) were trained with:
- **Input**: Swipe trajectories
- **Output**: Character sequences from vocab_size=30
- **Character set**: a-z only

**Retraining Required**: Adding apostrophe token requires:
1. New tokenizer with token 30 = "'"
2. Retrain encoder model
3. Retrain decoder model
4. Re-collect training data including contractions

---

## Current Behavior

### What Happens When User Swipes Contraction

**Example**: User swipes "don't"

```
Input: Swipe trajectory over d-o-n-t letters
    ↓
Trajectory Processor: Detects nearest keys: [7, 18, 17, 23] (d,o,n,t)
    ↓
Neural Network Beam Search:
    Beam 0: "dont" (confidence: 0.92)  ← No apostrophe!
    Beam 1: "done" (confidence: 0.05)
    Beam 2: "dent" (confidence: 0.03)
    ↓
Vocabulary Filtering:
    "dont" → ❌ NOT IN DICTIONARY (discarded)
    "done" → ✅ IN DICTIONARY (kept)
    "dent" → ✅ IN DICTIONARY (kept)
    ↓
Final Result:
    #1: "done" (score: 678)
    #2: "dent" (score: 245)

User Expected: "don't"
User Got: "done" (wrong word!)
```

---

### Calibration Accuracy Impact

**Scenario**: Calibration session with contractions

```
Session Words (20 random):
  - "hello" → ✅ Predicted correctly
  - "world" → ✅ Predicted correctly
  - "don't" → ❌ Predicted "done" (WRONG)
  - "it's" → ❌ Predicted "its" (WRONG - different meaning!)
  - "can't" → ❌ Predicted "cant" (NOT IN DICT → no prediction)
  - "you're" → ❌ Predicted "your" (WRONG - different meaning!)
  ...

Accuracy: 60% (would be 90% without contractions)
User perception: "This system doesn't work"
```

---

## Evidence of Dictionary Mismatch

### Contractions in Dictionary (40+ entries)

| Contraction | Frequency | Predictable? | Alternative |
|-------------|-----------|--------------|-------------|
| don't | 253 | ❌ | done, dont (not in dict) |
| it's | 253 | ❌ | its (different meaning!) |
| can't | 250 | ❌ | cant (archaic word) |
| won't | 242 | ❌ | wont (archaic/not in dict) |
| you're | 250 | ❌ | your (different meaning!) |
| we're | 243 | ❌ | were (different meaning!) |
| they're | 245 | ❌ | their/there (wrong!) |
| i'm | 253 | ❌ | im (not in dict) |
| i'll | 244 | ❌ | ill (sick) |
| i've | 248 | ❌ | ive (not in dict) |
| didn't | 248 | ❌ | didnt (not in dict) |
| wasn't | 241 | ❌ | wasnt (not in dict) |
| isn't | 243 | ❌ | isnt (not in dict) |
| doesn't | 245 | ❌ | doesnt (not in dict) |
| haven't | 237 | ❌ | havent (not in dict) |
| wouldn't | 239 | ❌ | wouldnt (not in dict) |
| couldn't | 236 | ❌ | couldnt (not in dict) |
| shouldn't | 230 | ❌ | shouldnt (not in dict) |
| let's | 239 | ❌ | lets (allows) |
| that's | 250 | ❌ | thats (not in dict) |

**Total Impact**: ~40 high-frequency words are unpredictable

---

## Attempted Workarounds (None Exist)

### Checked for Current Solutions

**OptimizedVocabulary.java**: ❌ No apostrophe handling
**SwipeTokenizer.java**: ❌ No apostrophe mapping
**Config.java**: ❌ No contraction settings
**Post-processing**: ❌ No text replacement

**Conclusion**: Zero current workarounds exist.

---

## Proposed Solutions

### Solution 1: Remove Contractions from Dictionary ⚠️ NOT RECOMMENDED

**Approach**: Filter contractions out of calibration and main dictionary

**Implementation**:
```java
// In SwipeCalibrationActivity.java
if (!line.isEmpty() && !line.contains("'") && uniqueWords.add(line))
{
  fullVocabulary.add(line);
}

// In OptimizedVocabulary.java (loadVocabulary)
if (!word.contains("'"))
{
  vocabulary.put(word, frequency);
}
```

**Pros**:
- ✅ Quick fix (1 hour)
- ✅ No model changes
- ✅ Stops misleading users

**Cons**:
- ❌ Removes 40+ high-frequency words
- ❌ English feels unnatural without contractions
- ❌ User can't type common phrases naturally
- ❌ Doesn't solve the fundamental problem

**Verdict**: ⚠️ **Band-aid fix, not recommended**

---

### Solution 2: Apostrophe-Free Alternatives (Mapping) ⚠️ PARTIAL

**Approach**: Map swipe output "dont" → dictionary "don't"

**Implementation**:
```java
// Add to OptimizedVocabulary.java
private static final Map<String, String> CONTRACTION_MAPPING = new HashMap<String, String>() {{
  put("dont", "don't");
  put("cant", "can't");
  put("wont", "won't");
  put("im", "i'm");
  put("ill", "i'll");
  put("youre", "you're");
  put("theyre", "they're");
  put("were", "we're");  // CONFLICT with "were" (past tense)!
  put("its", "it's");    // CONFLICT with "its" (possessive)!
  put("lets", "let's");  // CONFLICT with "lets" (allows)!
  // ... 40+ mappings
}};

// In filterPredictions()
for (BeamSearchCandidate candidate : candidates)
{
  String word = candidate.word.toLowerCase();

  // Check if this is a contraction without apostrophe
  if (CONTRACTION_MAPPING.containsKey(word))
  {
    String withApostrophe = CONTRACTION_MAPPING.get(word);
    // Add contraction variant with same NN confidence
    filteredPredictions.add(new FilteredPrediction(
      withApostrophe, confidence, frequency, "contraction_mapped"
    ));
  }
}
```

**Pros**:
- ✅ No model retraining required
- ✅ Can implement in ~4 hours
- ✅ Works for some contractions

**Cons**:
- ❌ **Ambiguity problems**:
  - "its" → "it's" or "its" (possessive)?
  - "were" → "we're" or "were" (past tense)?
  - "lets" → "let's" or "lets" (allows)?
- ❌ Requires maintaining large mapping table
- ❌ Doesn't work for all cases (user intent unclear)
- ❌ Still can't distinguish "your" vs "you're"

**Verdict**: ⚠️ **Partial solution with ambiguity issues**

---

### Solution 3: Post-Prediction Autocorrect 🟡 RECOMMENDED (SHORT-TERM)

**Approach**: Use existing autocorrect to suggest contractions

**Implementation**:
```java
// Extend existing autocorrect in OptimizedVocabulary.java
// After beam search autocorrect (line 307) and final autocorrect (Keyboard2.java:928)

private List<FilteredPrediction> suggestContractions(List<FilteredPrediction> predictions)
{
  List<FilteredPrediction> enhanced = new ArrayList<>(predictions);

  // For each prediction, check if a contraction exists
  for (FilteredPrediction pred : predictions)
  {
    String word = pred.word;

    // Check common patterns
    List<String> contractionCandidates = new ArrayList<>();

    if (word.equals("do") || word.equals("dont"))
      contractionCandidates.add("don't");
    if (word.equals("can") || word.equals("cant"))
      contractionCandidates.add("can't");
    if (word.equals("its"))
      contractionCandidates.addAll(Arrays.asList("it's", "its"));  // Both!
    if (word.equals("im"))
      contractionCandidates.add("i'm");
    if (word.equals("youre") || word.equals("your"))
      contractionCandidates.add("you're");
    if (word.equals("were"))
      contractionCandidates.addAll(Arrays.asList("we're", "were"));  // Both!

    // Add contraction candidates with slightly lower score
    for (String contraction : contractionCandidates)
    {
      if (vocabulary.containsKey(contraction))
      {
        float contractionScore = pred.score * 0.9f;  // Slightly penalize
        enhanced.add(new FilteredPrediction(
          contraction, pred.confidence,
          vocabulary.get(contraction).frequency,
          "contraction_suggestion"
        ));
      }
    }
  }

  // Re-sort by score
  Collections.sort(enhanced, (a, b) -> Float.compare(b.score, a.score));
  return enhanced;
}
```

**Pros**:
- ✅ Works with existing autocorrect system
- ✅ Can suggest BOTH "its" and "it's" (user chooses)
- ✅ Gradual degradation (slightly lower score)
- ✅ No model changes required
- ✅ Implement in ~6 hours

**Cons**:
- ❌ Heuristic-based (not comprehensive)
- ❌ Requires pattern matching for each contraction
- ❌ Still doesn't "predict" contractions directly

**Verdict**: 🟡 **RECOMMENDED for immediate implementation** (band-aid until model retrained)

---

### Solution 4: Retrain Model with Apostrophe Token ✅ RECOMMENDED (LONG-TERM)

**Approach**: Add apostrophe as token 30, retrain models

**Implementation**:

#### Step 1: Update Tokenizer

```json
{
  "vocab_size": 31,  // Increased from 30
  "char_to_idx": {
    "<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3,
    "a": 4, "b": 5, ..., "z": 29,
    "'": 30  // NEW
  },
  "idx_to_char": {
    "0": "<pad>", "1": "<unk>", "2": "<sos>", "3": "<eos>",
    "4": "a", ..., "29": "z",
    "30": "'"  // NEW
  }
}
```

#### Step 2: Retrain Models

```bash
# Training pipeline (Python/PyTorch)
1. Update tokenizer vocab_size = 31
2. Re-collect training data with contractions:
   - Include "don't", "can't", "it's", etc. in training set
   - Ensure apostrophe appears in trajectories
3. Retrain encoder:
   - Update embedding layer: nn.Embedding(31, embed_dim)  # Was 30
   - Retrain on new data
4. Retrain decoder:
   - Update output layer: nn.Linear(hidden_dim, 31)  # Was 30
   - Retrain on new data
5. Export to ONNX with vocab_size=31
6. Test predictions include apostrophe
```

#### Step 3: Update Android Code

```java
// SwipeTokenizer.java
private void initializeDefaultMapping()
{
  _charToIdx = new HashMap<>();
  _idxToChar = new HashMap<>();

  addMapping(PAD_IDX, '\0');
  addMapping(UNK_IDX, '?');
  addMapping(SOS_IDX, '^');
  addMapping(EOS_IDX, '$');

  int idx = 4;
  for (char c = 'a'; c <= 'z'; c++)
  {
    addMapping(idx++, c);
  }

  // NEW: Add apostrophe
  addMapping(30, '\'');

  Log.d(TAG, String.format("Tokenizer initialized with %d tokens (including apostrophe)",
    _charToIdx.size()));
}
```

**Pros**:
- ✅ **Proper solution** - fixes root cause
- ✅ Neural network can predict "don't", "it's", etc. directly
- ✅ No ambiguity (network learns when to use apostrophe)
- ✅ Future-proof for other punctuation (periods, commas)

**Cons**:
- ❌ Requires model retraining (~2-4 weeks work)
- ❌ Requires training data with contractions
- ❌ Larger model (31 tokens vs 30)
- ❌ May need more training data for accuracy

**Verdict**: ✅ **RECOMMENDED LONG-TERM SOLUTION**

---

## Implementation Roadmap

### Phase 1: Immediate Fix (1-2 days)

1. **Remove contractions from calibration word list**
   ```java
   // SwipeCalibrationActivity.java line 256
   if (!line.isEmpty() && !line.contains("'") && uniqueWords.add(line))
   ```

2. **Add warning in calibration UI**
   ```java
   logToResults("ℹ️ Note: Contractions (don't, can't, etc.) not yet supported by neural model");
   ```

3. **Document limitation in STATUS.md**
   - Add to "Known Issues" section
   - Explain why contractions don't work
   - Set user expectations

**Timeline**: 1 day
**Priority**: HIGH (stops misleading users)

---

### Phase 2: Heuristic Workaround (1 week)

1. **Implement contraction suggestion system** (Solution 3)
   - Add `suggestContractions()` method to OptimizedVocabulary
   - Integrate with existing autocorrect pipeline
   - Handle ambiguous cases (suggest both variants)

2. **Test common contractions**
   - "don't" → suggests "don't" after predicting "done"
   - "it's" → suggests both "its" and "it's"
   - "you're" → suggests "you're" after predicting "your"

3. **Add config toggle**
   ```xml
   <SwitchPreference
     android:key="swipe_suggest_contractions"
     android:title="Suggest Contractions"
     android:summary="Auto-suggest apostrophe variants (don't, it's, etc.)"
     android:defaultValue="true"/>
   ```

**Timeline**: 1 week
**Priority**: MEDIUM (improves UX significantly)

---

### Phase 3: Model Retraining (2-4 months)

1. **Update tokenizer** (1 day)
   - vocab_size: 30 → 31
   - Add apostrophe token
   - Update SwipeTokenizer.java

2. **Collect training data** (2-4 weeks)
   - Add contractions to training vocabulary
   - Ensure apostrophe trajectories represented
   - Balance dataset (contractions should be proportional to frequency)

3. **Retrain encoder/decoder** (2-6 weeks depending on compute)
   - Update embedding dimensions
   - Train until convergence
   - Validate on test set

4. **Export and integrate** (1 week)
   - Export to ONNX
   - Test on Android
   - Verify apostrophe predictions work
   - Benchmark performance (ensure no degradation)

5. **A/B testing** (2 weeks)
   - Test with beta users
   - Compare accuracy: old model vs new
   - Measure user satisfaction

**Timeline**: 2-4 months
**Priority**: HIGH (proper solution)

---

## Testing Strategy

### Immediate Testing (Phase 1)

```bash
# Verify contractions removed from calibration
1. Launch calibration
2. Complete 3 sessions (60 words)
3. Verify NO contractions appear
4. Accuracy should improve vs current state
```

### Heuristic Testing (Phase 2)

```bash
# Test contraction suggestions
1. Swipe "don't" → expect "done" + "don't" (suggestion)
2. Swipe "it's" → expect "its" + "it's" (both suggested)
3. Swipe "you're" → expect "your" + "you're"
4. Swipe "they're" → expect "their" + "they're"

# Verify no false positives
5. Swipe "done" (not contraction) → expect "done" only
6. Swipe "your" (possessive) → expect "your" + "you're"
```

### Model Retraining Testing (Phase 3)

```bash
# Test apostrophe predictions
1. Swipe "don't" → #1 prediction should be "don't" (with apostrophe)
2. Swipe "it's" → distinguish from "its" based on trajectory
3. Swipe "can't" → predict "can't" not "cant"

# Regression testing
4. Swipe "hello" → still works
5. Swipe "world" → still works
6. Accuracy on non-contractions: ≥95% (same as before)
7. Accuracy on contractions: ≥85% (new capability)
```

---

## Metrics for Success

### Phase 1 Success Criteria

- ✅ Zero contractions in calibration sessions
- ✅ Calibration accuracy improves by 10-20% (no impossible words)
- ✅ User documentation updated
- ✅ No user complaints about "can't predict don't"

### Phase 2 Success Criteria

- ✅ 80%+ of swipes for "don't" result in "don't" being suggested
- ✅ Ambiguous cases ("its"/"it's") show both options
- ✅ <5% false positive suggestions
- ✅ User can disable feature if they prefer

### Phase 3 Success Criteria

- ✅ Neural network directly predicts contractions
- ✅ "don't" swipe → "don't" prediction with ≥85% accuracy
- ✅ No accuracy regression on non-contractions
- ✅ <100ms latency (same as before)
- ✅ APK size increase <5MB

---

## User Communication

### What to Tell Users Now

**In Settings/Documentation**:
```
⚠️ Contractions Not Supported

The current neural model cannot predict words with apostrophes
(don't, can't, it's, etc.). This is a known limitation.

Workaround: Type the full form ("do not" instead of "don't")
or use tap-typing for contractions.

We're working on adding apostrophe support in a future update.
```

### What to Tell Users After Phase 2

```
✅ Contraction Suggestions Enabled

When you swipe words like "don't" or "it's", the keyboard will
now suggest the contraction form automatically. For ambiguous
cases (its/it's), both options are shown.

This is a temporary solution while we train a new model with
full apostrophe support.
```

### What to Tell Users After Phase 3

```
✅ Full Contraction Support

The keyboard now fully supports contractions! Swipe "don't",
"can't", "it's" and other common contractions just like any
other word.

This required retraining the neural network with an expanded
character set. Enjoy more natural typing!
```

---

## Related Issues

### Other Punctuation Missing

This limitation affects more than just contractions:

- ❌ **Periods** - Can't predict end of sentence
- ❌ **Commas** - Can't predict "hello, world"
- ❌ **Hyphens** - Can't predict "co-op", "re-enter"
- ❌ **Numbers** - Can't predict "COVID-19", "9pm"

**Future Enhancement**: Expand tokenizer to vocab_size=50+
- a-z (26)
- Apostrophe (1)
- Space (1) - for multi-word predictions
- Period, comma, hyphen, question mark, exclamation (5)
- Numbers 0-9 (10)
- Total: ~43 tokens

This would enable:
- "don't worry, it's fine!" (full punctuation)
- "COVID-19" (numbers + hyphen)
- "hello world" (multi-word predictions)

---

## Conclusion

**Current State**: Dictionary contains 40+ contractions that are **impossible to predict** due to tokenizer limitation.

**Immediate Action**: Remove contractions from calibration (Phase 1) to stop misleading users.

**Short-Term Fix**: Implement contraction suggestions (Phase 2) as autocorrect heuristic.

**Long-Term Solution**: Retrain model with apostrophe token (Phase 3) for proper support.

**Timeline**:
- Phase 1: 1 day
- Phase 2: 1 week
- Phase 3: 2-4 months

**User Impact**: High-frequency words like "don't", "it's", "can't" currently don't work, degrading user experience significantly. Fixing this is **high priority**.

---

## References

- `SwipeTokenizer.java` - Character tokenization (lines 111-132)
- `assets/models/tokenizer.json` - Vocabulary definition
- `assets/dictionaries/en_enhanced.json` - 50k dictionary with contractions
- `SwipeCalibrationActivity.java` - Calibration word loading (lines 237-272)
- [TRAJECTORY_PREPROCESSING.md](TRAJECTORY_PREPROCESSING.md) - Swipe processing pipeline
- [SWIPE_PREDICTION_PIPELINE.md](SWIPE_PREDICTION_PIPELINE.md) - Neural prediction flow

---

## Changelog

### v1.32.232 (2025-11-02)
- Created comprehensive analysis of contraction limitation
- Identified 40+ unpredictable contractions in dictionary
- Documented root cause (tokenizer vocab_size=30, no apostrophe)
- Proposed 4 solution approaches with trade-offs
- Created 3-phase implementation roadmap
- Defined success metrics and testing strategy
