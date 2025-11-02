# Contraction System Specification

## Overview

The contraction system handles English contractions and possessives in swipe typing predictions. It solves two key challenges:

1. **Swipe gesture ambiguity**: Words ending in apostrophe-s (e.g., "jesus's", "who'll") have identical swipe patterns to their apostrophe-free forms
2. **Autocorrect conflicts**: Dictionary contains both apostrophe-free forms and similar words that cause fuzzy matching errors

## Architecture

### Three-Component System

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Dictionary Cleanup                                           │
│    - Remove all apostrophes from en_enhanced.json              │
│    - Remove spurious double-s words (jesuss → removed)         │
│    - Remove orphaned possessives with no base word             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Contraction Mappings (Two Files)                            │
│    A. contraction_pairings.json                                │
│       Base word → possessive/contraction variants              │
│       Example: "jesus" → ["jesus's"]                           │
│                                                                 │
│    B. contractions_non_paired.json                             │
│       Apostrophe-free → proper contraction                     │
│       Example: "wholl" → "who'll"                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Insertion-Time Mapping                                      │
│    - Predictions contain apostrophe-free forms                 │
│    - Keyboard2.java applies mapping before autocorrect         │
│    - Prevents fuzzy matching to wrong words                    │
└─────────────────────────────────────────────────────────────────┘
```

## File Specifications

### Dictionary: `assets/dictionaries/en_enhanced.json`

**Format**: JSON array of word-frequency objects
```json
[
  {"word": "jesus", "frequency": 129},
  {"word": "obama", "frequency": 231},
  {"word": "wholl", "frequency": 170},
  {"word": "dont", "frequency": 228}
]
```

**Rules**:
- ZERO words with apostrophes
- No spurious double-s words (jesuss, jamess, etc.)
- No orphaned possessives without base words
- Current count: 49,293 words

### Paired Contractions: `assets/dictionaries/contraction_pairings.json`

**Format**: Base word → array of variants with frequencies
```json
{
  "jesus": [{"contraction": "jesus's", "frequency": 129}],
  "obama": [{"contraction": "obama's", "frequency": 231}],
  "well": [{"contraction": "we'll", "frequency": 243}],
  "its": [{"contraction": "it's", "frequency": 253}]
}
```

**Purpose**: Maps possessives and some contractions where base word exists in dictionary

**Count**: 1,752 base words → 1,754 variants

**Usage**:
- OptimizedVocabulary.java loads this file
- Creates variant predictions with apostrophes for UI display
- Base word (apostrophe-free) used for insertion

### Non-Paired Contractions: `assets/dictionaries/contractions_non_paired.json`

**Format**: Apostrophe-free → proper contraction
```json
{
  "aint": "ain't",
  "cant": "can't",
  "dont": "don't",
  "wholl": "who'll",
  "im": "i'm",
  "youre": "you're"
}
```

**Purpose**: Real contractions where base word doesn't exist

**Count**: 74 contractions

**Patterns**:
- Negatives: n't (can't, won't, shouldn't)
- Am: 'm (i'm)
- Are: 're (you're, we're, they're)
- Have: 've (could've, should've, i've)
- Will: 'll (i'll, you'll, who'll)
- Would/Had: 'd (i'd, you'd, he'd)
- Is/Has: 's (it's, that's, what's)

**Usage**:
- Keyboard2.java loads this file at startup
- Applied during word insertion BEFORE autocorrect
- Converts prediction to proper contraction

## Code Flow

### 1. Prediction Generation (OptimizedVocabulary.java)

```java
// Lines 480-542: Load and apply contraction mappings
private Map<String, List<String>> contractionPairings = new HashMap<>();
private Map<String, String> nonPairedContractions = new HashMap<>();

// For each prediction:
if (contractionPairings.containsKey(word)) {
    // Create variant with apostrophe for UI display
    // But keep base word (apostrophe-free) for insertion
    List<String> contractions = contractionPairings.get(word);
    for (String contraction : contractions) {
        contractionVariants.add(new FilteredPrediction(
            word,                    // word for insertion
            contraction,             // displayText for UI
            variantScore,
            pred.confidence,
            pred.frequency,
            "contraction"
        ));
    }
}
```

### 2. UI Display (OnnxSwipePredictor.java)

```java
// Lines 1328-1337: Build prediction lists with displayText
// Use displayText to show proper contractions with apostrophes
for (Map.Entry<String, WordDisplayPair> entry : wordScoreMap.entrySet()) {
    words.add(entry.getValue().displayText);  // Shows "who'll" not "wholl"
    scores.add(entry.getValue().score);
}
```

**Critical**: Use `displayText` for proper UI display with apostrophes. Keyboard2 will recognize and protect contractions from autocorrect.

### 3. Insertion (Keyboard2.java)

```java
// Lines 931-960: Skip autocorrect for known contractions
word = word.replaceAll("^raw:", "");  // Strip prefix

// Check if this is a known contraction (from _knownContractions set)
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

**Critical Logic**:
1. Strip "raw:" prefix
2. **Check if word is known contraction** ← Key insight
3. If known: Skip autocorrect (insert as-is)
4. If unknown: Run autocorrect normally

## Problem Cases Solved

### Before Fix: Autocorrect Conflicts

| Prediction | Inserted | Why Wrong |
|------------|----------|-----------|
| wholl | wholly | Final autocorrect fuzzy-matched "who'll" to "wholly" (both in dict) |
| dont | donut | Final autocorrect fuzzy-matched "don't" to "donut" |
| hell | shell | Final autocorrect fuzzy-matched "he'll" to "shell" |
| whos | whose | Final autocorrect fuzzy-matched "who's" to "whose" |

**Root cause**: Passing contractions with apostrophes to autocorrect caused fuzzy matching to similar dictionary words.

### After Fix: Skip Autocorrect for Contractions

| Swipe | Prediction (UI) | Known Contraction? | Autocorrect? | Final Insert |
|-------|-----------------|-------------------|--------------|--------------|
| who'll pattern | who'll | YES | SKIP | who'll ✓ |
| don't pattern | don't | YES | SKIP | don't ✓ |
| he'll pattern | he'll | YES | SKIP | he'll ✓ |
| i'm pattern | i'm | YES | SKIP | i'm ✓ |
| hell pattern | hell | NO | RUN | hell ✓ |

## Swipe Gesture Ambiguity

### The Problem

Swipe gestures ending in 's' are **physically identical** to swipes ending in 'ss':

```
jesus's swipe:  j-e-s-u-s-[apostrophe]-s
jesus swipe:    j-e-s-u-s
                ↑ IDENTICAL pattern ↑

Result: Both swipes predict "jesus"
```

### The Solution

**Paired contractions**: Map base word to both forms
```json
{
  "jesus": [{"contraction": "jesus's", "frequency": 129}]
}
```

Prediction logic:
1. Swipe pattern matches "jesus" (base word)
2. System generates TWO predictions:
   - "jesus" (base form)
   - "jesus's" (possessive variant)
3. User taps desired form
4. Both insert correctly (no autocorrect conflict)

## Deduplication Process

Automated script: `deduplicate_contractions.py`

### Categorization Logic

```python
real_contraction_indicators = {
    "n't",  # negatives: can't, won't, shouldn't
    "'m",   # am: i'm
    "'re",  # are: you're, we're, they're
    "'ve",  # have: could've, should've
    "'ll",  # will: i'll, you'll, who'll
    "'d"    # would/had: i'd, you'd, he'd
}

is_has_contractions = {
    "it's", "that's", "what's", "he's", "she's", ...
}
```

### Spurious Double-S Detection

```python
if contraction.endswith("'s"):
    base_candidate = contraction[:-2]  # Remove 's

    if base_candidate.endswith('s') and base_candidate in current_dict:
        # Spurious: jesuss (base=jesus exists in dict)
        spurious_ss_words.add(contraction_no_apostrophe)
```

**Removed**: jesuss, jamess, chriss, bosss, thomass, joness, rosss, lewiss, daviss, harriss, uss

### Results

| Category | Count | Destination |
|----------|-------|-------------|
| Real contractions | 74 | contractions_non_paired.json |
| Possessives with base | 1,108 | contraction_pairings.json |
| Spurious ss-words | 11 | REMOVED from dictionary |
| Orphaned possessives | 31 | REMOVED from dictionary |

## Testing Checklist

### Test Contractions
- [ ] Swipe "don't" → Predicts "dont" → Inserts "don't"
- [ ] Swipe "can't" → Predicts "cant" → Inserts "can't"
- [ ] Swipe "who'll" → Predicts "wholl" → Inserts "who'll"
- [ ] Swipe "i'm" → Predicts "im" → Inserts "i'm"
- [ ] Swipe "you're" → Predicts "youre" → Inserts "you're"

### Test Possessives
- [ ] Swipe "jesus" → Predicts both "jesus" and "jesus's"
- [ ] Tap "jesus's" → Inserts "jesus's"
- [ ] Swipe "obama" → Predicts both "obama" and "obama's"

### Test No False Positives
- [ ] Swipe "hell" → Predicts "hell" → Inserts "hell" (NOT shell)
- [ ] Swipe "whose" → Inserts "whose" (NOT who's)
- [ ] Swipe "its" → Distinguishes "its" vs "it's"

## Maintenance

### Adding New Contractions

1. Identify if paired or non-paired
2. Add to appropriate JSON file
3. Rebuild app
4. Test insertion

### Rebuilding Mappings

Run deduplication script when dictionary changes:
```bash
python deduplicate_contractions.py
```

This regenerates:
- `contraction_pairings.json`
- `contractions_non_paired.json`
- Updated `en_enhanced.json`

## Version History

- **v1.32.235**: Initial deduplication, removed 42 words, categorized 1,213 contractions
- **v1.32.236**: First fix attempt (FAILED - displayText still passed to autocorrect)
- **v1.32.241**: Second fix attempt (FAILED - apostrophe-free UI + autocorrect ran on mapped contractions)
- **v1.32.245**: Final fix (displayText for UI + skip autocorrect for known contractions) ✓

## Key Insights

1. **Autocorrect must never process contractions**: Contractions not in dictionary will fuzzy match to wrong words
2. **Skip autocorrect for known contractions**: Check if word is in known set, bypass autocorrect entirely
3. **Swipe ambiguity requires variants**: Base word must generate both standard and apostrophe forms
4. **Dictionary purity**: No apostrophes in main dictionary prevents conflicts
5. **Use displayText for UI**: Show proper contractions with apostrophes in suggestion bar
6. **Protect contractions from autocorrect**: The fundamental solution to insertion bugs

## Related Files

- `OptimizedVocabulary.java` - Loads mappings, generates prediction variants
- `OnnxSwipePredictor.java` - Builds final prediction list (apostrophe-free)
- `Keyboard2.java` - Applies contraction mapping during insertion
- `deduplicate_contractions.py` - Automated categorization and deduplication
- `assets/dictionaries/en_enhanced.json` - Main dictionary (49,293 words)
- `assets/dictionaries/contraction_pairings.json` - Paired mappings (1,752 base → 1,754 variants)
- `assets/dictionaries/contractions_non_paired.json` - Non-paired mappings (74 contractions)
