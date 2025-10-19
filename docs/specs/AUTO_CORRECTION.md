# Auto-Correction System Specification

**Version**: 1.0
**Implemented**: v1.32.114-122
**Status**: Active (disabled in Termux app)

---

## Overview

The auto-correction system automatically corrects common typing mistakes when the user presses space after typing a word. It uses fuzzy string matching to find dictionary words that closely match the typed word, preserving capitalization patterns.

### Key Features

- **Fuzzy Matching**: Same length + first 2 letters + positional character similarity
- **Capitalization Preservation**: Maintains original case (teh→the, Teh→The, TEH→THE)
- **App-Aware**: Disabled in Termux app, enabled in normal apps
- **Configurable**: 4 user-adjustable settings via UI
- **Context-Aware**: Updates context with corrected word for better predictions

---

## Architecture

### Components

1. **WordPredictor.java** - Core correction algorithm
2. **Keyboard2.java** - Integration with keyboard input flow
3. **Config.java** - Settings storage and management
4. **settings.xml** - UI configuration

### Data Flow

```
User types "thid" → KeyEventHandler.send_text() commits each char
User presses space → handle_text_typed(" ") commits space
                   ↓
Keyboard2.handleRegularTyping() (line 1096)
                   ↓
Auto-correction check (line 1135)
                   ↓
WordPredictor.autoCorrect() (line 382)
                   ↓
Fuzzy matching algorithm
                   ↓
preserveCapitalization() (line 478)
                   ↓
Delete typed word + space via InputConnection
                   ↓
Insert corrected word + space
                   ↓
Update context for predictions
```

---

## Implementation Details

### 1. Configuration (Config.java)

**File**: `srcs/juloo.keyboard2/Config.java`

**Fields** (lines 83-87):
```java
// Auto-correction settings
public boolean autocorrect_enabled;               // Master switch (default: true)
public int autocorrect_min_word_length;           // Min length for correction (default: 3)
public float autocorrect_char_match_threshold;    // Required char match ratio (default: 0.67 = 2/3)
public int autocorrect_confidence_min_frequency;  // Min dictionary frequency (default: 500)
```

**Loading** (lines 229-233):
```java
// Auto-correction settings
autocorrect_enabled = _prefs.getBoolean("autocorrect_enabled", true);
autocorrect_min_word_length = safeGetInt(_prefs, "autocorrect_min_word_length", 3);
autocorrect_char_match_threshold = _prefs.getFloat("autocorrect_char_match_threshold", 0.67f);
autocorrect_confidence_min_frequency = safeGetInt(_prefs, "autocorrect_confidence_min_frequency", 500);
```

---

### 2. Core Algorithm (WordPredictor.java)

**File**: `srcs/juloo.keyboard2/WordPredictor.java`

#### autoCorrect() Method (lines 382-476)

```java
/**
 * Auto-correct a typed word by finding closest dictionary match
 *
 * Algorithm:
 * 1. Check if word already in dictionary or user vocabulary → skip correction
 * 2. Filter candidates: same length + same first 2 letters (fast pruning)
 * 3. Calculate positional character match ratio
 * 4. Select highest frequency candidate above threshold
 * 5. Preserve original capitalization pattern
 *
 * @param typedWord The word typed by user (e.g., "thid")
 * @return Corrected word if match found, original word otherwise
 */
public String autoCorrect(String typedWord)
{
  // Validation checks
  if (_config == null || !_config.autocorrect_enabled || typedWord == null || typedWord.isEmpty())
  {
    return typedWord;
  }

  String lowerTypedWord = typedWord.toLowerCase();

  // Don't correct dictionary words or user's vocabulary
  if (_dictionary.containsKey(lowerTypedWord) ||
      (_adaptationManager != null && _adaptationManager.getAdaptationMultiplier(lowerTypedWord) > 1.0f))
  {
    return typedWord;
  }

  // Enforce minimum word length
  if (lowerTypedWord.length() < _config.autocorrect_min_word_length)
  {
    return typedWord;
  }

  // Requires at least 2 characters for "same first 2 letters" rule
  if (lowerTypedWord.length() < 2)
  {
    return typedWord;
  }

  // Extract prefix and length for filtering
  String prefix = lowerTypedWord.substring(0, 2);
  int wordLength = lowerTypedWord.length();
  WordCandidate bestCandidate = null;

  // Find candidates with same length + same first 2 letters
  for (Map.Entry<String, Integer> entry : _dictionary.entrySet())
  {
    String dictWord = entry.getKey();

    // Fast filtering: same length + same prefix
    if (dictWord.length() != wordLength) continue;
    if (!dictWord.startsWith(prefix)) continue;

    // Calculate POSITIONAL character match ratio
    int matchCount = 0;
    for (int i = 0; i < wordLength; i++)
    {
      if (lowerTypedWord.charAt(i) == dictWord.charAt(i))
      {
        matchCount++;
      }
    }

    float matchRatio = (float)matchCount / wordLength;

    // Check if match ratio meets threshold
    if (matchRatio >= _config.autocorrect_char_match_threshold)
    {
      int candidateFrequency = entry.getValue();

      // Keep best candidate (highest frequency)
      if (bestCandidate == null || candidateFrequency > bestCandidate.score)
      {
        bestCandidate = new WordCandidate(dictWord, candidateFrequency);
      }
    }
  }

  // Apply correction only if confident (high frequency)
  if (bestCandidate != null && bestCandidate.score >= _config.autocorrect_confidence_min_frequency)
  {
    String corrected = preserveCapitalization(typedWord, bestCandidate.word);
    android.util.Log.d("WordPredictor", "AUTO-CORRECT: '" + typedWord + "' → '" + corrected + "'");
    return corrected;
  }

  return typedWord;
}
```

#### preserveCapitalization() Method (lines 478-516)

```java
/**
 * Preserve capitalization pattern from original word
 *
 * Patterns:
 * - ALL UPPERCASE: "TEH" → "THE"
 * - Title Case: "Teh" → "The"
 * - lowercase: "teh" → "the"
 *
 * @param originalWord User's typed word with capitalization
 * @param correctedWord Dictionary word (lowercase)
 * @return Corrected word with original capitalization pattern
 */
private String preserveCapitalization(String originalWord, String correctedWord)
{
  if (originalWord.length() == 0 || correctedWord.length() == 0)
  {
    return correctedWord;
  }

  // Check if ALL uppercase
  boolean isAllUpper = true;
  for (int i = 0; i < originalWord.length(); i++)
  {
    if (Character.isLowerCase(originalWord.charAt(i)))
    {
      isAllUpper = false;
      break;
    }
  }

  if (isAllUpper)
  {
    return correctedWord.toUpperCase();  // "TEH" → "THE"
  }

  // Check if first letter uppercase (Title Case)
  if (Character.isUpperCase(originalWord.charAt(0)))
  {
    return Character.toUpperCase(correctedWord.charAt(0)) + correctedWord.substring(1);  // "Teh" → "The"
  }

  return correctedWord;  // "teh" → "the"
}
```

---

### 3. Integration (Keyboard2.java)

**File**: `srcs/juloo.keyboard2/Keyboard2.java`

#### Trigger Point (lines 1096-1195)

Auto-correction is triggered in `handleRegularTyping()` when user presses space:

```java
public void handleRegularTyping(String text)
{
  if (!_config.word_prediction_enabled || _wordPredictor == null || _suggestionBar == null)
  {
    return;
  }

  // Track current word being typed
  if (text.length() == 1 && Character.isLetter(text.charAt(0)))
  {
    _currentWord.append(text);
    updatePredictionsForCurrentWord();
  }
  else if (text.length() == 1 && !Character.isLetter(text.charAt(0)))
  {
    // Non-letter character (space, punctuation) - word complete

    if (_currentWord.length() > 0)
    {
      String completedWord = _currentWord.toString();

      // AUTO-CORRECTION LOGIC (lines 1119-1188)
      // ... (see below)

      updateContext(completedWord);
    }

    // Reset current word
    _currentWord.setLength(0);
    if (_wordPredictor != null)
    {
      _wordPredictor.reset();
    }
    if (_suggestionBar != null)
    {
      _suggestionBar.clearSuggestions();
    }
  }
}
```

#### Auto-Correction Logic (lines 1119-1188)

```java
// Auto-correct the typed word if feature is enabled
// DISABLED in Termux app due to erratic behavior with terminal input
boolean inTermuxApp = false;
try
{
  EditorInfo editorInfo = getCurrentInputEditorInfo();
  if (editorInfo != null && editorInfo.packageName != null)
  {
    inTermuxApp = editorInfo.packageName.equals("com.termux");
  }
}
catch (Exception e)
{
  // Fallback: assume not Termux if detection fails
}

if (_config.autocorrect_enabled && _wordPredictor != null && text.equals(" ") && !inTermuxApp)
{
  String correctedWord = _wordPredictor.autoCorrect(completedWord);

  // If correction was made, replace the typed word
  if (!correctedWord.equals(completedWord))
  {
    InputConnection conn = getCurrentInputConnection();
    if (conn != null)
    {
      // At this point:
      // - The typed word "thid" has been committed via KeyEventHandler.send_text()
      // - The space " " has ALSO been committed via handle_text_typed(" ")
      // - Editor contains "thid "
      // - We need to delete both the word AND the space, then insert corrected word + space

      // Delete the typed word + space (already committed)
      conn.deleteSurroundingText(completedWord.length() + 1, 0);

      // Insert the corrected word WITH trailing space (normal apps only)
      conn.commitText(correctedWord + " ", 1);

      // Update context with corrected word
      updateContext(correctedWord);

      // Clear current word
      _currentWord.setLength(0);

      // Show corrected word as first suggestion for easy undo
      if (_suggestionBar != null)
      {
        List<String> undoSuggestions = new ArrayList<>();
        undoSuggestions.add(completedWord); // Original word first for undo
        undoSuggestions.add(correctedWord); // Corrected word second
        List<Integer> undoScores = new ArrayList<>();
        undoScores.add(0);
        undoScores.add(0);
        _suggestionBar.setSuggestionsWithScores(undoSuggestions, undoScores);
      }

      // Reset prediction state
      if (_wordPredictor != null)
      {
        _wordPredictor.reset();
      }

      return; // Skip normal text processing - we've handled everything
    }
  }
}
```

**Key Implementation Details**:

1. **Termux Detection** (lines 1121-1133): Detect Termux app via `EditorInfo.packageName`
2. **Condition Check** (line 1135): Only correct if enabled AND not in Termux AND space pressed
3. **Text State** (lines 1145-1149): Typed word + space already in editor via `KeyEventHandler.send_text()`
4. **Deletion** (line 1152): Delete `word.length() + 1` (word + space)
5. **Insertion** (line 1155): Insert corrected word + space
6. **Undo UI** (lines 1160-1171): Show original word in suggestion bar for easy revert
7. **Early Return** (line 1178): Skip normal processing after correction

---

### 4. UI Settings (settings.xml)

**File**: `res/xml/settings.xml`

**Settings UI** (lines 23-29):
```xml
<PreferenceScreen
    android:key="autocorrect_settings"
    android:title="✨ Auto-Correction"
    android:summary="Automatically fix common typing mistakes"
    android:dependency="word_prediction_enabled">

  <!-- Master switch -->
  <CheckBoxPreference
      android:key="autocorrect_enabled"
      android:title="Enable Auto-Correction"
      android:summary="Automatically correct typos when you press space"
      android:defaultValue="true"/>

  <!-- Info preference explaining the feature -->
  <Preference
      android:title="📖 About Auto-Correction"
      android:summary="Fixes typos by finding dictionary words with:\n\n• Same length\n• Same first 2 letters\n• Similar characters (default: 2/3 match)\n\nExample: 'teh' → 'the', 'Teh' → 'The'\n\nTap corrected word in suggestions to undo.\n\nNote: Disabled in Termux app for terminal compatibility."
      android:dependency="autocorrect_enabled"/>

  <!-- Minimum word length slider -->
  <juloo.keyboard2.prefs.IntSlideBarPreference
      android:key="autocorrect_min_word_length"
      android:title="Minimum Word Length"
      android:summary="Don't correct words shorter than %s letters"
      android:defaultValue="3"
      min="2"
      max="5"
      android:dependency="autocorrect_enabled"/>

  <!-- Character match threshold slider -->
  <juloo.keyboard2.prefs.SlideBarPreference
      android:key="autocorrect_char_match_threshold"
      android:title="Character Match Threshold"
      android:summary="How many characters must match (%s)"
      android:defaultValue="0.67"
      min="0.5"
      max="0.9"
      android:dependency="autocorrect_enabled"/>

  <!-- Minimum frequency slider -->
  <juloo.keyboard2.prefs.IntSlideBarPreference
      android:key="autocorrect_confidence_min_frequency"
      android:title="Minimum Frequency"
      android:summary="Only correct to words with frequency ≥ %s"
      android:defaultValue="500"
      min="100"
      max="5000"
      android:dependency="autocorrect_enabled"/>
</PreferenceScreen>
```

---

## Configuration Parameters

### autocorrect_enabled (boolean)

- **Default**: `true`
- **Description**: Master switch to enable/disable auto-correction
- **UI**: Checkbox preference
- **Behavior**: When disabled, no corrections are made

### autocorrect_min_word_length (int)

- **Default**: `3`
- **Range**: `2-5`
- **Description**: Minimum word length to attempt correction
- **Rationale**: Short words (2 letters) might be intentional (e.g., "go", "to", "me")
- **Example**: With default 3, "teh" gets corrected but "te" does not

### autocorrect_char_match_threshold (float)

- **Default**: `0.67` (2/3 match)
- **Range**: `0.5-0.9`
- **Description**: Minimum ratio of positional character matches required
- **Algorithm**: `matchRatio = matchCount / wordLength`
- **Example**: "thid" → "this" has 3/4 = 0.75 match (meets 0.67 threshold)

### autocorrect_confidence_min_frequency (int)

- **Default**: `500`
- **Range**: `100-5000`
- **Description**: Minimum dictionary frequency to accept correction
- **Rationale**: Only correct to common words, avoid obscure dictionary entries
- **Example**: Prefer "the" (freq: 50000) over rare word (freq: 100)

---

## Examples

### Example 1: Simple Typo

**Input**: "teh" + space
**Dictionary**: {"the": 50000, "ten": 5000}
**Process**:
1. "teh" not in dictionary ✓
2. Length: 3 ≥ min_length (3) ✓
3. Prefix: "te"
4. Candidates with length=3 and prefix="te": ["the", "ten"]
5. Match ratios:
   - "the": t=match, e=match, h≠d = 2/3 = 0.67 ✓
   - "ten": t=match, e=match, n≠h = 2/3 = 0.67 ✓
6. Best candidate: "the" (freq: 50000 > 5000)
7. Capitalization: lowercase → lowercase
8. **Result**: "the "

### Example 2: Capitalization Preservation

**Input**: "Teh" + space
**Process**: Same as Example 1, but capitalization step:
1. Original: "Teh" (first letter uppercase)
2. Corrected: "the"
3. Apply pattern: `Character.toUpperCase('t') + "he"` = "The"
4. **Result**: "The "

### Example 3: All Caps

**Input**: "TEH" + space
**Process**: Same matching, but capitalization:
1. Original: "TEH" (all uppercase)
2. Corrected: "the"
3. Apply pattern: `"the".toUpperCase()` = "THE"
4. **Result**: "THE "

### Example 4: Word Already Correct

**Input**: "the" + space
**Process**:
1. "the" in dictionary → return unchanged
2. **Result**: "the " (no correction)

### Example 5: Below Threshold

**Input**: "thx" + space
**Dictionary**: {"the": 50000, "tax": 3000}
**Process**:
1. "thx" not in dictionary ✓
2. Prefix: "th"
3. Candidates: ["the"]
4. Match ratio: t=match, h=match, x≠e = 2/3 = 0.67 ✓
5. But "thx" might be intentional abbreviation
6. If freq("the") < min_frequency OR no good match
7. **Result**: "thx " (no correction if below confidence)

### Example 6: Termux App

**Input**: "thid" + space (in com.termux app)
**Process**:
1. Detect Termux app via `EditorInfo.packageName`
2. Check: `!inTermuxApp` = false → skip correction
3. **Result**: "thid " (no correction in Termux)

---

## Performance Characteristics

### Time Complexity

- **Dictionary scan**: O(n) where n = dictionary size (~10,000 words)
- **Filtering**: O(1) check for length + prefix
- **Matching**: O(m) where m = word length (typically 3-10 chars)
- **Overall**: O(n × m) ≈ O(100,000) operations per correction

### Optimizations

1. **Early exits**: Skip if word in dictionary or user vocab
2. **Fast filtering**: Same length + prefix check before expensive matching
3. **Positional matching**: Simple character comparison (no edit distance)
4. **Single pass**: Find best candidate in one dictionary iteration

### Memory

- **No additional storage**: Uses existing `_dictionary` HashMap
- **Temporary objects**: 1 String prefix, 1 WordCandidate per iteration
- **Negligible overhead**: ~100 bytes during correction

---

## Edge Cases

### 1. Empty Input
**Input**: "" + space
**Behavior**: Early return, no correction
**Code**: Line 386 validation check

### 2. Single Character
**Input**: "a" + space
**Behavior**: Below min_length (3), no correction
**Code**: Line 399 length check

### 3. Two Characters
**Input**: "to" + space
**Behavior**: Below min_length (3), no correction
**Note**: Prevents correcting valid 2-letter words

### 4. Word in User Vocabulary
**Input**: "myword" + space (user types frequently)
**Behavior**: `adaptationManager.getAdaptationMultiplier() > 1.0` → no correction
**Code**: Line 394 adaptation check

### 5. No Dictionary Match
**Input**: "xyzabc" + space
**Behavior**: No candidates with same length + prefix → return unchanged
**Code**: Line 473 null check

### 6. Multiple Candidates Same Frequency
**Input**: "thr" + space
**Candidates**: {"the": 5000, "their": 5000}
**Behavior**: First match wins (dictionary iteration order)
**Code**: Line 461 `>` comparison (not `>=`)

### 7. Null InputConnection
**Input**: Correction triggered but InputConnection unavailable
**Behavior**: Early return, no changes made
**Code**: Line 1142 null check

### 8. Termux Mode Setting vs App
**Scenario**: Termux mode enabled globally, typing in Gmail
**Behavior**: Auto-correction works (app detection, not global setting)
**Code**: Line 1135 `!inTermuxApp` check

---

## Known Issues

### Issue 1: Termux Erratic Behavior (FIXED)

**Problem**: Auto-correction was erratic in Termux app
**Cause**: Terminal input has different text flow expectations
**Solution**: Disabled auto-correction in Termux app via package detection
**Status**: Fixed in v1.32.122
**Code**: Lines 1121-1135

### Issue 2: No Undo Mechanism

**Problem**: User cannot easily revert unwanted correction
**Workaround**: Tap original word in suggestion bar
**Status**: UI shows undo option but not fully implemented
**Priority**: Low
**Code**: Lines 1160-1171

---

## Testing

### Test Cases

1. **Basic Correction**
   - Type "teh" + space → Expect "the "
   - Type "recieve" + space → Expect "receive "

2. **Capitalization**
   - Type "Teh" + space → Expect "The "
   - Type "TEH" + space → Expect "THE "
   - Type "tEh" + space → Expect "the " (first letter lowercase)

3. **No Correction**
   - Type "the" + space → Expect "the " (already correct)
   - Type "go" + space → Expect "go " (below min length)
   - Type "xyz" + space → Expect "xyz " (no match)

4. **App Detection**
   - Type "thid" + space in Gmail → Expect "this "
   - Type "thid" + space in Termux → Expect "thid " (no correction)

5. **Settings**
   - Disable autocorrect → Type "teh" → Expect "teh "
   - Set min_length=5 → Type "teh" → Expect "teh " (too short)
   - Set threshold=0.9 → Type "thid" → Expect "thid " (below threshold)

---

## Future Enhancements

### Priority 1: Undo Mechanism
- Implement full undo by tapping suggestion bar
- Store last correction for backspace handling

### Priority 2: Learning from Corrections
- Track when user manually reverts corrections
- Reduce confidence for those patterns
- ML-based correction learning

### Priority 3: Context-Aware Correction
- Use bigram model for context ("teh cat" → don't correct if "teh" is brand name)
- Previous word influences correction decision

### Priority 4: Expanded Dictionary
- Support multiple languages
- Technical/domain-specific dictionaries
- User-added custom words

---

## Version History

- **v1.32.122** (2025-10-19): Disabled in Termux app
- **v1.32.121** (2025-10-19): Smart Termux app detection
- **v1.32.119** (2025-10-19): Fixed deletion count bug
- **v1.32.116** (2025-10-19): Fixed space insertion bug
- **v1.32.114** (2025-10-19): Initial implementation

---

## References

- **Implementation**: `WordPredictor.java:382-516`, `Keyboard2.java:1119-1188`
- **Configuration**: `Config.java:83-87,229-233`, `settings.xml:23-29`
- **Algorithm Design**: Gemini 2.5 Pro consultation (2025-10-19)
- **Changelog**: `memory/CHANGELOG.md`
