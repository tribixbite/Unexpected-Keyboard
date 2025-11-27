# Contraction System Specification

**Final Version**: v1.32.264
**Status**: Production
**Last Updated**: 2025-11-02

## Overview

The contraction system handles English contractions and possessives in swipe typing predictions. It solves three key challenges:

1. **Swipe gesture limitation**: Apostrophes can't be swiped, so neural network predicts apostrophe-free forms
2. **Semantic disambiguation**: Filter contraction variants based on what the neural network actually predicted
3. **Autocorrect conflicts**: Invalid apostrophe-free forms must be replaced before autocorrect runs

## Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Dictionary: Apostrophe-Free Forms                           │
│    - Contains both valid words AND invalid apostrophe-free     │
│    - Valid: well, were, id, hell, ill, shed, shell             │
│    - Invalid: cant, dont, wholl, whatd (not real words)        │
│    - Purpose: Allow NN predictions to pass vocabulary filter   │
│    - Count: 49,296 words (includes 62 apostrophe-free forms)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Paired Contractions (Create Variants)                       │
│    - Base word → multiple contraction forms                    │
│    - Example: "what" → what'd, what'll, what're, what's, etc.  │
│    - Filter by NN predictions: Only create variants NN output  │
│    - Count: 1,744 base words → multiple variants each          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Non-Paired Contractions (Replace Invalid Forms)             │
│    - Apostrophe-free → proper contraction                      │
│    - Example: "cant" → "can't", "wholl" → "who'll"             │
│    - REPLACES prediction (not variant)                         │
│    - Count: 62 apostrophe-free forms                           │
└─────────────────────────────────────────────────────────────────┘
```

## File Specifications

### Dictionary: `assets/dictionaries/en_enhanced.json`

**Format**: JSON object with word→frequency mappings
```json
{
  "what": 254,
  "whatd": 200,
  "well": 252,
  "cant": 200,
  "hell": 238
}
```

**Contains**:
- Valid words: 49,234 standard English words
- Invalid apostrophe-free forms: 62 words (cant, dont, wholl, etc.)
  - Needed for NN predictions to pass vocabulary filter
  - Will be replaced by non-paired contraction system

**Current Count**: 49,296 words

### Paired Contractions: `assets/dictionaries/contraction_pairings.json`

**Format**: Base word → array of contraction variants with frequencies
```json
{
  "what": [
    {"contraction": "what'd", "frequency": 135},
    {"contraction": "what'll", "frequency": 140},
    {"contraction": "what're", "frequency": 135},
    {"contraction": "what's", "frequency": 196},
    {"contraction": "what've", "frequency": 130}
  ],
  "could": [
    {"contraction": "couldn't", "frequency": 175},
    {"contraction": "could've", "frequency": 165}
  ],
  "well": [
    {"contraction": "we'll", "frequency": 252}
  ]
}
```

**Purpose**:
- Maps base words to all possible contraction variants
- Creates VARIANTS (both base and contraction appear)
- Filtered by NN predictions (only create variants NN actually predicted)

**Current Count**: 1,744 base words

**Examples**:
- Pronouns: i→(i'd, i'll, i'm, i've), he→(he'd, he'll, he's), etc.
- Question words: what→(what'd, what'll, what're, what's, what've), etc.
- Modals: could→(couldn't, could've), should→(shouldn't, should've), etc.
- Valid words with different meanings: well→(we'll), were→(we're, weren't), etc.

### Non-Paired Contractions: `assets/dictionaries/contractions_non_paired.json`

**Format**: Apostrophe-free → proper contraction
```json
{
  "cant": "can't",
  "dont": "don't",
  "wholl": "who'll",
  "whatd": "what'd",
  "im": "i'm",
  "yall": "y'all"
}
```

**Purpose**:
- Maps invalid apostrophe-free forms to proper contractions
- REPLACES prediction (doesn't create variant)
- Only for forms that aren't valid English words

**Current Count**: 62 apostrophe-free forms

**Patterns**:
- Negatives: n't (can't, won't, shouldn't, couldn't, wouldn't, etc.)
- Am: 'm (i'm)
- Are: 're (you're, we're, they're)
- Have: 've (i've, you've, could've, should've, would've, might've)
- Will: 'll (i'll, you'll, he'll, she'll, who'll, what'll, that'll, there'll, it'll)
- Would/Had: 'd (i'd, you'd, he'd, she'd, what'd, who'd, there'd)
- Is/Has: 's (it's, that's, what's, here's, there's, let's, who's)
- Special: y'all, shan't, mightn't, mustn't, n't

## Code Flow

### 1. Neural Network Prediction

```java
// OnnxSwipePredictor.java
// User swipes "what'd" gesture (no apostrophe possible)
// NN predicts: ["what", "whatd", "that", "had", ...]
```

**Critical**: NN cannot predict apostrophes, so outputs apostrophe-free forms like "whatd", "cant", "wholl"

### 2. Vocabulary Filtering (OptimizedVocabulary.java)

```java
// Lines 193-200: Build set of raw NN predictions
Set<String> rawPredictionWords = new HashSet<>();
for (CandidateWord candidate : rawPredictions) {
    rawPredictionWords.add(candidate.word.toLowerCase().trim());
}
// Example: {"what", "whatd", "that", "had"}
```

**Purpose**: Track what NN actually predicted for filtering contraction variants

### 3. Paired Contractions - NN-Based Filtering (OptimizedVocabulary.java)

```java
// Lines 488-537: Filter paired contractions by NN predictions
if (contractionPairings.containsKey(word)) {  // "what" found
    List<String> contractions = contractionPairings.get(word);
    // contractions = ["what'd", "what'll", "what're", "what's", "what've"]

    for (String contraction : contractions) {
        // Get apostrophe-free form: "what'd" → "whatd"
        String apostropheFree = contraction.replace("'", "").toLowerCase();

        // Only create variant if NN predicted this specific form
        if (!rawPredictionWords.contains(apostropheFree)) {
            continue;  // Skip - NN didn't predict "whatd"
        }

        // Create variant (both "what" and "what'd" appear)
        contractionVariants.add(new FilteredPrediction(
            contraction,     // word for insertion: "what'd"
            contraction,     // displayText for UI: "what'd"
            variantScore,
            pred.confidence,
            pred.frequency,
            pred.source + "-contraction"
        ));
    }
}
```

**Key Insight**: Only create "what'd" variant if raw predictions contain "whatd"

**Example Flow**:
1. NN predicts: ["what", "whatd"]
2. "what" is in paired contractions
3. Check all variants: what'd→"whatd"✓, what'll→"whatll"✗, what's→"whats"✗
4. Only create "what'd" variant (NN predicted "whatd")
5. Result: User sees "what" and "what'd" (not all 5 variants)

### 4. Non-Paired Contractions - Replacement (OptimizedVocabulary.java)

```java
// Lines 539-556: Replace invalid apostrophe-free forms
if (nonPairedContractions.containsKey(word)) {  // "cant" found
    String contraction = nonPairedContractions.get(word);  // "can't"

    // REPLACE the current prediction (don't create variant)
    validPredictions.set(i, new FilteredPrediction(
        contraction,     // word: "can't"
        contraction,     // displayText: "can't"
        pred.score,      // Same score (not a variant)
        pred.confidence,
        pred.frequency,
        pred.source + "-contraction"
    ));
}
```

**Key Difference**: REPLACEMENT vs VARIANT
- Paired: Creates variant (both "well" and "we'll" appear)
- Non-paired: Replaces prediction ("cant" → only "can't" appears)

### 5. Autocorrect Protection (Keyboard2.java)

```java
// Lines 931-974: Skip autocorrect for known contractions
boolean isKnownContraction = _knownContractions.contains(word.toLowerCase());

if (isKnownContraction) {
    // Skip autocorrect - insert contraction as-is
    Log.d("Keyboard2", "KNOWN CONTRACTION: " + word + " - skipping autocorrect");
} else {
    // Run autocorrect for unknown words
    if (_config.swipe_final_autocorrect_enabled && _wordPredictor != null) {
        word = _wordPredictor.autoCorrect(word);
    }
}
```

**Purpose**: Prevent fuzzy matching (who'll → wholly, don't → donut)

## Complete Contraction Coverage

### Negatives (n't)
- aren't, can't, couldn't, didn't, doesn't, don't
- hadn't, hasn't, haven't, isn't, mightn't, mustn't
- shan't, shouldn't, wasn't, weren't, won't, wouldn't

### Pronoun Contractions
**I**: I'd, I'll, I'm, I've
**You**: you'd, you'll, you're, you've
**He**: he'd, he'll, he's
**She**: she'd, she'll, she's
**We**: we'd, we'll, we're, we've
**They**: they'd, they'll, they're, they've
**It**: it'll

### Question Words
**What**: what'd, what'll, what're, what's, what've
**Who**: who'd, who'll, who're, who's, who've
**Where**: where'd, where's
**When**: when'd, when's
**Why**: why'd
**How**: how'd, how's

### Demonstratives
**That**: that'll, that's
**There**: there'd, there'll, there's

### Modal Verbs
**Could**: couldn't, could've
**Should**: shouldn't, should've
**Would**: wouldn't, would've
**Might**: mightn't, might've

### Other
- let's, here's, n't, y'all

**Total**: 66 distinct contractions (excluding possessives)

## Problem Cases Solved

### Before: All Variants Showing

| User Swipes | NN Predicts | Old System | Problem |
|-------------|-------------|------------|---------|
| "whatd" | "what", "whatd" | Shows all 5: what'd, what'll, what're, what's, what've | User wanted "what'd" but sees 4 irrelevant options |

### After: NN-Based Filtering

| User Swipes | NN Predicts | New System | Result |
|-------------|-------------|------------|--------|
| "whatd" | "what", "whatd" | Only "what'd" (filtered by "whatd" in raw predictions) | ✓ Only relevant contraction |
| "whatll" | "what", "whatll" | Only "what'll" | ✓ Correct filtering |
| "whats" | "what", "whats" | Only "what's" | ✓ Smart filtering |

### Before: Invalid Forms Appearing

| User Swipes | NN Predicts | Old System | Problem |
|-------------|-------------|------------|---------|
| "cant" | "cant" | Shows "cant" AND "can't" | Invalid word "cant" visible |

### After: Replacement Logic

| User Swipes | NN Predicts | New System | Result |
|-------------|-------------|------------|--------|
| "cant" | "cant" | REPLACES with "can't" | ✓ Only "can't" appears (no invalid "cant") |
| "wholl" | "wholl" | REPLACES with "who'll" | ✓ Only valid contraction |

## Testing Checklist

### Negatives
- [x] can't, don't (basic negatives)
- [x] couldn't, wouldn't, shouldn't (modal negatives)
- [x] doesn't, hasn't, hadn't (verb negatives)
- [x] mustn't, mightn't (rare negatives)

### Pronoun Contractions
- [x] I'm, I've, I'd, I'll
- [x] you're, you've, you'd, you'll
- [x] he's, he'd, he'll
- [x] she's, she'd, she'll
- [x] we're, we've, we'd, we'll
- [x] they're, they've, they'd, they'll
- [x] it'll

### Question Words
- [x] what'd, what'll, what's (NN filtering)
- [x] who'll, who'd, who's
- [x] where'd, where's
- [x] when'd, when's
- [x] why'd
- [x] how'd, how's

### Have Contractions
- [x] could've, would've, should've, might've

### Demonstratives
- [x] that'll, that's
- [x] there'd, there'll, there's

### Special
- [x] y'all
- [x] let's, here's

### NN-Based Filtering
- [x] Swipe "whatd" → only "what'd" (not what'll, what's, etc.)
- [x] Swipe "whatll" → only "what'll" (not what'd, what's, etc.)
- [x] Swipe "whats" → only "what's" (not what'd, what'll, etc.)

### No False Positives
- [x] Swipe "hell" → "hell" (not he'll)
- [x] Swipe "well" → both "well" and "we'll" (valid word creates variant)
- [x] Invalid forms replaced: "cant" → "can't" (not both)

## Maintenance

### Adding New Contractions

1. **Determine type**:
   - If base word exists and is valid: Add to `contraction_pairings.json`
   - If apostrophe-free form is invalid: Add to `contractions_non_paired.json`

2. **Update files**:
   ```bash
   # Add apostrophe-free form to dictionary
   # Add to appropriate contraction file
   # Rebuild app
   ```

3. **Test**:
   - Swipe the contraction pattern
   - Verify NN filtering works
   - Confirm no invalid forms appear

### Recently Added (v1.32.264)

- could've, should've, would've, might've (*ve contractions)
- there'd, there'll (demonstrative future/conditional)
- that'll (demonstrative future)
- it'll (pronoun future)
- y'all (special colloquial)

## Version History

- **v1.32.235**: Initial deduplication, removed 42 words
- **v1.32.236**: First fix attempt (FAILED - autocorrect conflicts)
- **v1.32.241**: Second fix (FAILED - UI + autocorrect issues)
- **v1.32.245**: Skip autocorrect for contractions ✓
- **v1.32.250**: Semantic categorization (possessives vs contractions)
- **v1.32.252**: Remove invalid apostrophe-free forms (9 words)
- **v1.32.253**: Complete cleanup + base word mappings (28 more removed, 25 added)
- **v1.32.256**: Comprehensive contraction mappings (1,735 paired)
- **v1.32.257**: Remove 16 additional invalid forms (49,240 words)
- **v1.32.259**: Add apostrophe-free forms back + replacement logic (49,293 words)
- **v1.32.261**: Smart filtering by swipe path lastChar (SUPERSEDED)
- **v1.32.263**: NN-based filtering (use raw predictions) ✓
- **v1.32.264**: Add 9 missing contractions (*ve forms, there'll, that'll, it'll, y'all) ✓

## Key Insights

1. **NN-based filtering is superior**: Use what NN predicted, not reconstructed swipe path
2. **Two-tier system needed**:
   - Paired for variants (both base and contraction)
   - Non-paired for replacement (only contraction)
3. **Invalid forms must be in dictionary**: So NN predictions pass vocabulary filter
4. **Replacement prevents invalid words**: "cant" → "can't" (user never sees "cant")
5. **Autocorrect must skip contractions**: Prevents fuzzy matching to wrong words

## Related Files

- `OptimizedVocabulary.java` - NN-based filtering, paired/non-paired logic
- `OnnxSwipePredictor.java` - Passes SwipeInput for NN prediction tracking
- `Keyboard2.java` - Loads _knownContractions set, skips autocorrect
- `assets/dictionaries/en_enhanced.json` - 49,296 words (includes 62 apostrophe-free)
- `assets/dictionaries/contraction_pairings.json` - 1,744 base words → multiple variants
- `assets/dictionaries/contractions_non_paired.json` - 62 apostrophe-free → contraction mappings
