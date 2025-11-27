# Typing Prediction System Specification

**Version**: 1.0
**Status**: Implemented (Previously Undocumented)
**Platform**: Android API 21+
**Last Updated**: 2025-10-22

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Scoring Algorithm](#scoring-algorithm)
4. [Configuration](#configuration)
5. [Bigram Model Integration](#bigram-model-integration)
6. [Performance](#performance)
7. [Known Issues](#known-issues)

---

## Overview

The typing prediction system provides word suggestions as the user types, using prefix matching, word frequency, context (bigram model), and user adaptation. This is separate from swipe predictions (neural network-based).

### Key Features

- **Prefix Matching**: Fast O(1) lookup using prefix index (1-3 char prefixes)
- **Frequency Scoring**: Logarithmic scaling balances common vs uncommon words
- **Context Boost**: Bigram model amplifies contextually likely words
- **User Adaptation**: Learns frequently typed words (not yet implemented)
- **Configurable Weights**: Context boost and frequency scaling user-adjustable

---

## Architecture

### Components

1. **WordPredictor.java** - Core prediction algorithm
2. **BigramModel.java** - Context probability model
3. **AdaptationManager.java** - User vocabulary learning (stub)
4. **Config.java** - Settings storage

### Data Flow

```
User types characters
    ↓
Keyboard2.handleRegularTyping()
    ↓
WordPredictor.predictInternal()
    ↓
getPrefixCandidates(prefix)  // O(1) prefix index lookup
    ↓
For each candidate word:
    1. calculatePrefixScore()      // Prefix match quality
    2. getAdaptationMultiplier()   // User typing frequency
    3. getContextMultiplier()      // Bigram probability
    4. calculateFrequencyFactor()  // Log-scaled corpus frequency
    5. Combine all signals → final score
    ↓
Sort by score, return top 10
    ↓
Display in suggestion bar
```

---

## Scoring Algorithm

**File**: `srcs/juloo.keyboard2/WordPredictor.java:616-651`

### Formula

```java
finalScore = prefixScore
           × adaptationMultiplier
           × (1.0 + (contextMultiplier - 1.0) × contextBoost)
           × frequencyFactor

where:
  prefixScore         = 0-100 (exact match=100, prefix match=50-99)
  adaptationMultiplier = 1.0 (not yet implemented, always 1.0)
  contextMultiplier   = 1.0-10.0 (from bigram model, 1.0=no context)
  contextBoost        = config.prediction_context_boost (default: 2.0)
  frequencyFactor     = 1.0 + log1p(frequency / frequencyScale)
  frequencyScale      = config.prediction_frequency_scale (default: 1000.0)
  frequency           = 100-10000 (scaled from raw 128-255 or custom 1-10000)
```

**NOTE**: ⚠️ **Bigram model integration not yet validated**

### Implementation

**File**: `srcs/juloo.keyboard2/WordPredictor.java:616-651`

```java
private int scoreWord(String word, String keySequence, String context)
{
  // 1. Base prefix matching score (0-100)
  int prefixScore = calculatePrefixScore(word, keySequence);
  if (prefixScore == 0) return 0;

  // 2. Get word frequency from dictionary
  Integer freq = _dictionary.get(word);
  if (freq == null) return 0;
  int frequency = freq.intValue();

  // 3. Adaptation multiplier (user typing history)
  float adaptationMultiplier = 1.0f;
  if (_adaptationManager != null)
  {
    adaptationMultiplier = _adaptationManager.getAdaptationMultiplier(word);
  }

  // 4. Context multiplier (bigram probability)
  float contextMultiplier = 1.0f;
  if (_bigramModel != null && context != null && !context.isEmpty())
  {
    contextMultiplier = _bigramModel.getContextMultiplier(word, context);
  }

  // 5. Frequency scaling (log to prevent common words from dominating)
  // Using log1p helps balance: "the" (freq ~10000) vs "think" (freq ~100)
  // Without log: "the" would always win. With log: context can override frequency
  // Scale factor is configurable (default: 1000.0)
  float frequencyScale = (_config != null) ? _config.prediction_frequency_scale : 1000.0f;
  float frequencyFactor = 1.0f + (float)Math.log1p(frequency / frequencyScale);

  // COMBINE ALL SIGNALS
  // Formula: prefixScore × adaptation × (1 + boosted_context) × freq_factor
  // Context boost is configurable (default: 2.0)
  // Higher boost = context has more influence on predictions
  float contextBoost = (_config != null) ? _config.prediction_context_boost : 2.0f;
  float finalScore = prefixScore
      * adaptationMultiplier
      * (1.0f + (contextMultiplier - 1.0f) * contextBoost)  // Configurable context boost
      * frequencyFactor;

  return (int)finalScore;
}
```

### Component Details

#### 1. Prefix Score

**File**: `srcs/juloo.keyboard2/WordPredictor.java:656-682`

```java
private int calculatePrefixScore(String word, String keySequence)
{
  if (word.equals(keySequence))
  {
    return 100;  // Exact match
  }

  if (word.startsWith(keySequence))
  {
    // Prefix match: longer word = lower score
    // "hello" with prefix "hel" → 50 + (3/5) × 50 = 80
    float matchRatio = (float)keySequence.length() / word.length();
    return (int)(50 + matchRatio * 50);
  }

  return 0;  // No match
}
```

**Range**: 0-100
- Exact match: 100
- Prefix match: 50-99 (longer prefix = higher score)
- No match: 0

#### 2. Adaptation Multiplier

**Status**: Not yet implemented (always returns 1.0)

**Purpose**: Boost words user types frequently

**Planned**: Track user's typing history, increase multiplier for commonly typed words

#### 3. Context Multiplier

**File**: `srcs/juloo.keyboard2/BigramModel.java:45-68`

```java
public float getContextMultiplier(String word, String context)
{
  if (context == null || context.isEmpty()) return 1.0f;

  String bigram = context.toLowerCase() + " " + word.toLowerCase();
  Integer prob = _bigrams.get(bigram);

  if (prob == null) return 1.0f;  // No bigram data

  // Bigram probability stored as 0-1000, convert to 1.0-10.0 range
  // Higher probability = more likely word follows context
  return 1.0f + (prob / 100.0f);  // 0 → 1.0, 1000 → 11.0
}
```

**Range**: 1.0-11.0
- No context or no bigram data: 1.0
- Weak bigram: 1.0-3.0
- Strong bigram: 3.0-11.0

**Example**: "Happy" → "New" has high bigram probability, multiplier ~5.0

**NOTE**: ⚠️ **Bigram model not yet validated** - may need tuning

#### 4. Context Boost

**File**: `srcs/juloo.keyboard2/Config.java:80`

```java
public float prediction_context_boost;  // How strongly context influences predictions (default: 2.0)
```

**Purpose**: Amplify or dampen bigram model influence

**Formula**: `1.0 + (contextMultiplier - 1.0) × contextBoost`

**Examples**:
```
contextMultiplier=3.0, contextBoost=2.0 → 1.0 + (3.0 - 1.0) × 2.0 = 5.0
contextMultiplier=3.0, contextBoost=0.5 → 1.0 + (3.0 - 1.0) × 0.5 = 2.0
contextMultiplier=3.0, contextBoost=4.0 → 1.0 + (3.0 - 1.0) × 4.0 = 9.0
```

**NOTE**: Higher boost = context has MORE influence on predictions

#### 5. Frequency Factor

**File**: `srcs/juloo.keyboard2/WordPredictor.java:637-638`

```java
float frequencyScale = (_config != null) ? _config.prediction_frequency_scale : 1000.0f;
float frequencyFactor = 1.0f + (float)Math.log1p(frequency / frequencyScale);
```

**Purpose**: Logarithmic scaling prevents common words from dominating

**Why log scaling?**
- Without log: "the" (freq=10000) would ALWAYS beat "think" (freq=100)
- With log: context can override frequency differences

**Formula**: `1.0 + log₁₀(1 + frequency / frequencyScale)`

**Examples** (frequencyScale=1000):
```
frequency=100    → 1.0 + log1p(0.1)   = 1.095  (uncommon word)
frequency=1000   → 1.0 + log1p(1.0)   = 1.693  (common word)
frequency=10000  → 1.0 + log1p(10.0)  = 2.398  (very common word)
```

**Ratio**: Very common / uncommon = 2.398 / 1.095 = **2.2x** (manageable, not 100x!)

---

## Configuration

**File**: `srcs/juloo.keyboard2/Config.java:79-81`

```java
// Word prediction scoring weights (for regular typing)
public float prediction_context_boost;     // How strongly context influences predictions (default: 2.0)
public float prediction_frequency_scale;   // Balance common vs uncommon words (default: 1000.0)
```

### Settings UI

**File**: `res/xml/settings.xml` (section needed for typing prediction settings)

**Currently**: These settings exist in Config.java but have NO UI controls!

**TODO**: Add UI sliders for:
- Context Boost (0.0-5.0, default 2.0)
- Frequency Scale (100-5000, default 1000)

---

## Bigram Model Integration

**File**: `srcs/juloo.keyboard2/BigramModel.java`

### Data Source

**File**: `assets/bigrams/en.txt`

**Format**: `word1 word2 probability`
```
happy new 850
new year 920
thank you 780
```

**Probability Range**: 0-1000 (higher = more likely)

### Loading

**File**: `srcs/juloo.keyboard2/BigramModel.java:20-42`

```java
private void loadBigrams(Context context, String language)
{
  try
  {
    BufferedReader reader = new BufferedReader(
      new InputStreamReader(context.getAssets().open("bigrams/" + language + ".txt")));

    String line;
    while ((line = reader.readLine()) != null)
    {
      String[] parts = line.split(" ");
      if (parts.length == 3)
      {
        String bigram = parts[0] + " " + parts[1];
        int probability = Integer.parseInt(parts[2]);
        _bigrams.put(bigram, probability);
      }
    }
    reader.close();
  }
  catch (IOException e)
  {
    android.util.Log.e("BigramModel", "Failed to load bigrams: " + e.getMessage());
  }
}
```

**Memory**: ~50 KB for 5000 bigrams

**Lookup**: O(1) HashMap

**NOTE**: ⚠️ **Bigram probabilities not yet validated** - may need tuning or retraining

---

## Performance

### Dictionary Loading Performance (v1.32.537-539)

**Asynchronous Loading** (perftodos3.md Todo 1):
- **Binary Format**: BinaryDictionaryLoader 5-10x faster than JSON
  - 50k words + prefix index load in ~30-60ms
  - Pre-built prefix index eliminates runtime indexing
  - Memory-efficient ByteBuffer with direct asset access
- **Background Thread**: AsyncDictionaryLoader with ExecutorService
  - NO UI freeze during language switching ✅
  - NO UI freeze during app startup ✅
  - Callback-based completion on main thread
- **Loading State**: `WordPredictor.isLoading()` API
  - Returns true while dictionary loading asynchronously
  - getPredictions() returns empty list during load
  - UI can show loading indicator

**Auto-Update Observers** (perftodos3.md Todo 2):
- **UserDictionaryObserver**: Monitors both system and custom dictionaries
  - ContentObserver for UserDictionary.Words (system-level changes)
  - SharedPreferences.OnSharedPreferenceChangeListener for custom words
  - Instant updates: NO app restart required ✅
- **Activation**: `startObservingDictionaryChanges()` called after dictionary loads
  - DictionaryManager calls this in async load callback (v1.32.539)
  - Previously: Observer built but never started (dead code)
  - Now: Active for all loaded dictionaries

### Prefix Index Optimization

**File**: `srcs/juloo.keyboard2/WordPredictor.java:330-364`

**Before (v1.32.186)**: Iterated ALL 50k words on every keystroke
**After (v1.32.187)**: Prefix index reduces to ~100-500 words per keystroke

**Implementation**:
```java
// Build prefix index during dictionary loading
private void buildPrefixIndex()
{
  _prefixIndex.clear();

  for (String word : _dictionary.keySet())
  {
    int maxLen = Math.min(PREFIX_INDEX_MAX_LENGTH, word.length());
    for (int len = 1; len <= maxLen; len++)
    {
      String prefix = word.substring(0, len);
      _prefixIndex.computeIfAbsent(prefix, k -> new HashSet<>()).add(word);
    }
  }
}

// Use prefix index in predictInternal()
private Set<String> getPrefixCandidates(String prefix)
{
  if (prefix.isEmpty()) return _dictionary.keySet();

  String lookupPrefix = prefix.length() <= PREFIX_INDEX_MAX_LENGTH
      ? prefix : prefix.substring(0, PREFIX_INDEX_MAX_LENGTH);

  Set<String> candidates = _prefixIndex.get(lookupPrefix);
  if (candidates == null) return Collections.emptySet();

  // Filter if prefix longer than indexed
  if (prefix.length() > PREFIX_INDEX_MAX_LENGTH)
  {
    Set<String> filtered = new HashSet<>();
    for (String word : candidates)
    {
      if (word.startsWith(prefix)) filtered.add(word);
    }
    return filtered;
  }
  return candidates;
}
```

**Memory Cost**: +2 MB (prefix index for 50k words)

**Performance Gain**: 100x speedup (50k → 200 iterations per keystroke)

### Timing Breakdown

| Operation | Time |
|-----------|------|
| Prefix index lookup | <1ms |
| Iterate candidates (~200) | 1-2ms |
| Score calculation per word | <0.01ms |
| Sort top 10 | <1ms |
| **TOTAL** | **2-4ms** |

**Target**: <10ms for smooth typing ✅

---

## Known Issues

### Issue 1: Bigram Model Not Validated

**Severity**: ⚠️ Medium

**Description**: Bigram probabilities loaded from static file, not validated or tuned

**Impact**: Context predictions may not be optimal

**Fix**: Need to:
1. Validate bigram probabilities against test corpus
2. Tune probability ranges
3. Consider retraining on larger corpus

**Status**: Functional but unvalidated

---

### Issue 2: Adaptation Manager Not Implemented

**Severity**: ⚠️ Low

**Description**: User adaptation always returns 1.0 multiplier (no learning)

**Impact**: Doesn't learn user's frequently typed words

**Fix**: Implement user typing history tracking

**Status**: Stub implementation

---

### Issue 3: No UI for Typing Prediction Settings

**Severity**: ⚠️ Low

**Description**: `prediction_context_boost` and `prediction_frequency_scale` exist in Config.java but have no UI sliders

**Impact**: Users can't adjust typing prediction behavior

**Fix**: Add settings UI in `res/xml/settings.xml`

**Status**: Settings exist, UI missing

---

## Future Enhancements

### v1.33+ Planned Features

**NOTE**: 🚧 **To be implemented**:

1. **Add Settings UI** (Priority: HIGH)
   - Context Boost slider (0.0-5.0, default 2.0)
   - Frequency Scale slider (100-5000, default 1000)

2. **Validate Bigram Model** (Priority: MEDIUM)
   - Test against corpus
   - Tune probability ranges
   - Consider retraining

3. **Implement User Adaptation** (Priority: MEDIUM)
   - Track frequently typed words
   - Boost adaptation multiplier over time
   - Persist to SharedPreferences

4. **N-gram Support** (Priority: LOW)
   - Extend beyond bigrams to trigrams
   - Better context understanding

---

## Comparison: Typing vs Swipe Predictions

| Feature | Typing (WordPredictor) | Swipe (OptimizedVocabulary) |
|---------|------------------------|------------------------------|
| **Input** | Prefix characters | Neural network beam search |
| **Lookup** | Prefix index (O(1)) | Vocabulary HashMap (O(1)) |
| **Context** | Bigram model (validated❌) | None currently |
| **Scoring** | Prefix + context + freq | NN confidence + freq + tier |
| **User Words** | Custom + user dict | Custom + user dict + autocorrect |
| **Latency** | 2-4ms | 20-45ms (includes NN inference) |
| **Configurable** | Context boost, freq scale | Will be in v1.33+ |

---

## References

- **Implementation**: `srcs/juloo.keyboard2/WordPredictor.java`
- **Bigram Model**: `srcs/juloo.keyboard2/BigramModel.java`
- **Configuration**: `srcs/juloo.keyboard2/Config.java:79-81`
- **Prefix Indexing**: BEAM_SEARCH_VOCABULARY.md v2.1
- Related specs: `BEAM_SEARCH_VOCABULARY.md`, `AUTO_CORRECTION.md`

---

## Changelog

### v1.0 - Initial Documentation (2025-10-22)

- Documented previously undocumented typing prediction system
- Full scoring formula with all components
- Context boost and frequency scaling explained
- Bigram model integration documented
- Prefix index optimization documented (v1.32.187)
- Identified 3 known issues (bigram validation, adaptation, settings UI)

**NOTE**: This system has been implemented since v1.0 but was never documented until now.
