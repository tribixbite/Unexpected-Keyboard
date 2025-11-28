# Prediction Systems Analysis - Conflicting Behaviors

## 🚨 User-Reported Issues

1. **Issue 1**: Selecting a prediction from "non nn non onnx prediction list" deletes recently swiped input
2. **Issue 2**: Conflicting and battling prediction logic between multiple systems
3. **Issue 3**: Short two-character word swipes (e.g., 'me') exhibit bizarre behavior:
   - An 'm' gets inserted (shouldn't be possible)
   - Prediction bar shows ~5 predictions including 'm' and 'me'
   - 'me' is highlighted as top suggestion (correct), but 'm' is inserted

## 📊 Discovery: THREE Prediction Systems

### System 1: Neural/ONNX Swipe Typing (Primary)
**File**: `NeuralSwipeTypingEngine.java`
**Trigger**: Long swipes across multiple keys
**Flow**:
1. User swipes across keyboard
2. `ImprovedSwipeGestureRecognizer` collects path
3. `handleSwipeTyping()` (Keyboard2.java:1195-1354) processes gesture
4. `AsyncPredictionHandler` calls `_neuralEngine.predict()` (line 1316-1334)
5. Returns top 5 predictions with scores
6. **Auto-inserts top prediction** (Keyboard2.java:845-860)
7. Sets `_lastAutoInsertedWord` for replacement tracking

**Key Code** (Keyboard2.java:845-860):
```java
// Auto-insert top (highest scoring) prediction immediately after swipe completes
String topPrediction = _suggestionBar.getTopSuggestion();
if (topPrediction != null && !topPrediction.isEmpty())
{
  sendDebugLog(String.format("Auto-inserting top prediction: \"%s\"\n", topPrediction));

  _lastAutoInsertedWord = null;              // Clear BEFORE
  onSuggestionSelected(topPrediction);       // Insert
  _lastAutoInsertedWord = topPrediction;     // Track AFTER

  _suggestionBar.setSuggestionsWithScores(predictions, scores); // Re-display
}
```

### System 2: CGR Template-Based Prediction (Secondary/Fallback?)
**File**: `RealTimeSwipePredictor.java`
**Uses**: `ContinuousSwipeGestureRecognizer` with word templates
**Status**: Initialized but unclear when it's used

**Key Code** (RealTimeSwipePredictor.java:48-98):
```java
public void initialize(Context context)
{
  // Load dictionary
  templateGenerator.loadDictionary(context);

  // Generate templates for 3000 most frequent words
  List<ContinuousGestureRecognizer.Template> templates =
    templateGenerator.generateBalancedWordTemplates(3000);

  // Set templates in gesture recognizer
  gestureRecognizer.setTemplateSet(templates);
}
```

**Question**: Where is this called? Is it still active?

### System 3: Short Gesture System (Within-Key Swipes)
**File**: `Pointers.java` (lines 166-205)
**Trigger**: Swipe within a single key
**Purpose**: Access directional characters (e.g., swipe up on 'e' for '3')

**Key Code** (Pointers.java:166-205):
```java
// Check for short gesture ONLY on touch up (not during movement)
// Short gesture: swipe within a single key to get directional character
if (_config.swipe_typing_enabled && _config.short_gestures_enabled &&
    ptr.gesture == null && !ptr.hasLeftStartingKey &&
    ptr_value != null && ptr_value.getKind() == KeyValue.Kind.Char)
{
  java.util.List<android.graphics.PointF> swipePath = _swipeRecognizer.getSwipePath();

  if (swipePath != null && swipePath.size() > 1)
  {
    android.graphics.PointF lastPoint = swipePath.get(swipePath.size() - 1);
    float dx = lastPoint.x - ptr.downX;
    float dy = lastPoint.y - ptr.downY;
    float distance = (float) Math.sqrt(dx * dx + dy * dy);
    float minDistance = keyHypotenuse * (_config.short_gesture_min_distance / 100.0f);

    if (distance >= minDistance)
    {
      double a = Math.atan2(dy, dx);
      int direction = (int)Math.round(a * 8.0 / Math.PI) & 15;
      KeyValue gestureValue = getKeyAtDirection(ptr.key, direction);

      if (gestureValue != null)
      {
        _handler.onPointerDown(gestureValue, false);
        _handler.onPointerUp(gestureValue, ptr.modifiers);
        _swipeRecognizer.reset();
        return;
      }
    }
  }
}
```

**Issue**: Does NOT call neural prediction but goes straight to character output!

## 🔥 Root Cause Analysis

### Problem 1: Deletion Logic Conflict

**Location**: `Keyboard2.java:943-967` in `onSuggestionSelected()`

```java
// CRITICAL: If we just auto-inserted a word, delete it for replacement
// This allows user to tap a different prediction instead of appending
if (_lastAutoInsertedWord != null && !_lastAutoInsertedWord.isEmpty())
{
  // Calculate how many characters to delete (word + space after it)
  int deleteCount = _lastAutoInsertedWord.length();
  if (!_config.termux_mode_enabled)
  {
    deleteCount += 1; // Delete trailing space in normal mode
  }

  // Delete the auto-inserted word and its space
  ic.deleteSurroundingText(deleteCount, 0);

  // Clear the tracking variable
  _lastAutoInsertedWord = null;
}
```

**The Conflict**:
1. System 1 (Neural) auto-inserts "hello" → sets `_lastAutoInsertedWord = "hello"`
2. User manually taps ANY suggestion → calls `onSuggestionSelected()`
3. Deletion logic **always** deletes `_lastAutoInsertedWord` regardless of source
4. This is WRONG if the suggestion came from a different system!

**Missing Context**: The deletion logic can't distinguish between:
- Tapping a neural prediction (should replace auto-inserted word)
- Tapping a CGR/template prediction (shouldn't delete previous word)
- Selecting after short gesture (context unclear)

### Problem 2: Short Gesture vs Neural Swipe Ambiguity

**Scenario**: User swipes "me" (2 characters, short swipe)

**What Should Happen**:
- Neural system detects short swipe → generates predictions including "me"
- Auto-inserts "me"
- Shows predictions for correction

**What Actually Happens** (based on user report):
- An 'm' gets inserted
- Prediction bar shows 'm' and 'me'
- 'me' is highlighted but 'm' is inserted

**Hypothesis**: Short gesture system is interfering!

**Flow Analysis**:
1. User touches 'm' key → starts tracking
2. User swipes towards 'e' key but path is short
3. System checks short gesture conditions (Pointers.java:168-172):
   - `_config.swipe_typing_enabled` ✅
   - `_config.short_gestures_enabled` ✅
   - `ptr.gesture == null` ✅
   - `!ptr.hasLeftStartingKey` ??? (might be FALSE if path touched 'e')
   - `ptr_value != null` ✅
   - `ptr_value.getKind() == KeyValue.Kind.Char` ✅

4. If short gesture triggers:
   - Outputs directional character (possibly 'm' from a direction?)
   - Calls `_swipeRecognizer.reset()` (line 199)
   - Returns early (line 200)

5. If short gesture doesn't trigger but also not long enough:
   - Falls through to line 207-217
   - Outputs the base character 'm' (line 214)
   - Resets swipe recognizer (line 216)

6. Separately, neural system also processes the path:
   - Generates predictions including 'me'
   - Displays in suggestion bar

**Result**: Character output happens BEFORE neural predictions complete!

### Problem 3: Threshold Ambiguity

**Short Gesture Threshold**: Dynamic based on key size
```java
float minDistance = keyHypotenuse * (_config.short_gesture_min_distance / 100.0f);
```

**Neural Swipe Threshold**: Fixed at 100 pixels
```java
// ImprovedSwipeGestureRecognizer.java:30
private static final float MIN_SWIPE_DISTANCE = 100.0f;
```

**Gap Zone**: Swipes between short gesture threshold and neural threshold!
- Too long to be a short gesture
- Too short to trigger neural predictions
- Falls back to outputting base character (line 214)

## 🎯 System Interaction Matrix

| User Action | System 1 (Neural) | System 2 (CGR) | System 3 (Short) | Result |
|-------------|------------------|----------------|------------------|--------|
| Long swipe (>100px) | ✅ Predicts + Auto-inserts | ❓ Unknown | ❌ Not triggered | Works ✅ |
| Medium swipe (50-100px) | ❌ Too short | ❓ Unknown | ❌ Too long | Outputs base char ⚠️ |
| Short swipe within key | ❌ Too short | ❓ Unknown | ✅ Directional char | Works ✅? |
| Tap neural prediction | ❌ (handled by onSuggestionSelected) | N/A | N/A | Works ✅ |
| Tap CGR prediction | ❌ Deletes last neural word! | ✅ | N/A | BUG 🔥 |

## 🔍 Key Questions for DeepThink

1. **Where is RealTimeSwipePredictor used?**
   - Is it still active?
   - Does it generate predictions shown in the bar?
   - When does it run vs neural system?

2. **What is "non nn non onnx prediction list"?**
   - Is this the CGR system?
   - Or something else entirely?

3. **Short swipe "me" behavior**:
   - Why is 'm' being inserted?
   - Is short gesture triggering incorrectly?
   - Or is fallback char output the culprit?
   - Why do predictions still appear if char already output?

4. **Deletion logic fix needed**:
   - How to distinguish prediction source?
   - Should `_lastAutoInsertedWord` only track neural auto-inserts?
   - Should CGR predictions APPEND instead of REPLACE?

5. **Threshold coordination**:
   - Should medium-length swipes trigger neural or output char?
   - Need unified threshold strategy?

## 📝 Relevant Code Locations

### Keyboard2.java
- **Line 845-860**: Neural auto-insertion logic
- **Line 873-1019**: `onSuggestionSelected()` - handles ALL prediction taps
- **Line 943-967**: Deletion logic for `_lastAutoInsertedWord`
- **Line 1195-1354**: `handleSwipeTyping()` - neural prediction orchestration

### Pointers.java
- **Line 166-205**: Short gesture detection and output
- **Line 207-217**: Fallback character output for non-swipes

### ImprovedSwipeGestureRecognizer.java
- **Line 30**: `MIN_SWIPE_DISTANCE = 100.0f`

### AsyncPredictionHandler.java
- **Line 124**: Only calls `_neuralEngine.predict()` (no CGR)

### RealTimeSwipePredictor.java
- **Line 12-98**: CGR initialization with 3000 word templates
- **Status**: Unknown usage - needs investigation

## ✅ Expert Analysis Results (Gemini 2.5 Pro)

### Validation Summary

**All 3 bugs confirmed**. Expert recommends architectural refactor over tactical patches.

### Core Issue Identified

**State Management Problem**: The system has parallel decision paths that race to handle the same gesture. The neural system runs speculatively while fallback logic also processes the gesture, leading to inconsistent outcomes.

### Expert-Recommended Implementation Plan

#### Priority 1: Remove Dead Code (IMMEDIATE)
Delete all `RealTimeSwipePredictor` code - it's initialized but never used. Reduces confusion and maintenance burden.

**Files to delete**:
- `RealTimeSwipePredictor.java`
- CGR-related code in `EnhancedSwipeGestureRecognizer.java` (lines 16-143)

#### Priority 2: Unified Gesture Classification (ARCHITECTURAL)

**Current Problem**: Multiple parallel checks create race conditions
**Solution**: Single authoritative `GestureClassifier`

**Implementation**:
```java
public enum GestureType {
    TAP,
    SWIPE
}

public class GestureClassifier {
    private final float MIN_SWIPE_DISTANCE; // Dynamic based on screen density
    private final long MAX_TAP_DURATION = 150; // ms

    public GestureType classify(GestureData gesture) {
        boolean hasLeftStartingKey = gesture.hasLeftStartingKey;
        float totalDistance = gesture.totalDistance;
        long timeElapsed = gesture.timeElapsed;

        // Clear criteria: SWIPE if left starting key AND (distance OR time threshold met)
        if (hasLeftStartingKey && (totalDistance > MIN_SWIPE_DISTANCE || timeElapsed > MAX_TAP_DURATION)) {
            return GestureType.SWIPE;
        }
        return GestureType.TAP;
    }
}
```

**Refactor Event Handler**:
```java
// Pointers.java onTouchUp
public void onTouchUp(int pointerId) {
    Pointer ptr = getPtr(pointerId);
    GestureData data = collectGestureData(ptr);
    GestureType type = _gestureClassifier.classify(data);

    if (type == GestureType.TAP) {
        // Commit starting key character
        _handler.onPointerDown(ptr.value, false);
        _handler.onPointerUp(ptr.value, ptr.modifiers);
    } else {
        // Send ONLY to neural predictor
        _handler.onSwipeEnd(_swipeRecognizer);
    }

    removePtr(ptr);
}
```

**Benefits**:
- Eliminates race conditions
- Single source of truth for gesture classification
- Fixes "me" bug cleanly (80px swipe correctly classified as SWIPE)
- No more parallel prediction systems fighting

**Critical**: Use `dp` instead of `px` for MIN_SWIPE_DISTANCE. Recommend starting with key width / 2.

#### Priority 3: Track Prediction Source (MEDIUM COMPLEXITY)

**Current Problem**: Deletion logic can't distinguish between auto-inserted predictions and manual typing
**Solution**: Tag every commit with its source

**Implementation**:
```java
public enum PredictionSource {
    UNKNOWN,
    USER_TYPED_TAP,
    NEURAL_SWIPE,
    AUTOCORRECT,
    CANDIDATE_SELECTION
}

// In Keyboard2.java
private PredictionSource _lastCommitSource = PredictionSource.UNKNOWN;

private void commitText(String text, PredictionSource source) {
    InputConnection ic = getCurrentInputConnection();
    ic.commitText(text, 1);
    _lastCommitSource = source;
}

// In onSuggestionSelected
public void onSuggestionSelected(String word) {
    // Check if we should delete previous auto-insertion
    if (_lastAutoInsertedWord != null && _lastCommitSource == PredictionSource.NEURAL_SWIPE) {
        // Only delete if this is also a neural swipe replacement
        // Check current context to determine if this is a replacement vs new word
        deletePreviousWord();
    }

    commitText(word, PredictionSource.CANDIDATE_SELECTION);
}
```

**Benefits**:
- Fixes cross-system deletion conflicts
- Enables future source-specific behaviors
- Makes deletion logic explicit and testable

#### Priority 4: Testing & Tuning

**Test Suite Requirements**:
1. Short words: "me", "is", "go", "at", "I", "a"
2. Fast tapping: Ensure not misclassified as swipes
3. Deletion scenarios:
   - Swipe "hello" → tap different suggestion → should replace
   - Swipe "hello" → swipe "world" → should append
   - Tap "h" → tap "e" → tap different suggestion → should NOT delete

**Tuning Parameters**:
- `MIN_SWIPE_DISTANCE`: Start with `keyWidth / 2` in dp
- `MAX_TAP_DURATION`: 150ms (adjust based on testing)

## 🛠️ Original Proposed Solutions (Superseded by Expert Analysis)

### Solution 1: Track Prediction Source
Add flag to distinguish auto-insertion source:
```java
private String _lastAutoInsertedWord = null;
private boolean _lastAutoInsertWasNeural = false;

// In handleSwipeTyping auto-insertion:
_lastAutoInsertWasNeural = true;

// In onSuggestionSelected deletion check:
if (_lastAutoInsertedWord != null && _lastAutoInsertWasNeural)
{
  // Only delete if we auto-inserted from neural AND user tapped a neural suggestion
  // Need to also track suggestion source!
}
```

### Solution 2: Unified Threshold Management
```java
// Make thresholds consistent and configurable
private float getSwipeThreshold(KeyboardData.Key key)
{
  if (_config.short_gestures_enabled)
  {
    // Short gesture threshold
    float shortThreshold = getKeyHypotenuse(key) *
                          (_config.short_gesture_min_distance / 100.0f);

    // Neural threshold
    float neuralThreshold = 100.0f;

    // Gap zone: if between thresholds, favor neural
    return Math.min(shortThreshold, neuralThreshold);
  }
  return 100.0f;
}
```

### Solution 3: Disable Fallback Character Output
```java
// Pointers.java:207-217
// Remove or conditionalize the fallback char output
// Let ONLY neural predictions handle swipe gestures
```

## 🎪 Test Cases Needed

1. Swipe "me" (2 chars, ~80px) → should insert "me" not "m"
2. Swipe "hello" → auto-inserts "hello" → tap different prediction → should replace "hello"
3. Swipe "hello" → auto-inserts "hello" → swipe "world" → should append "world"
4. Short gesture on 'e' → should output directional char, no predictions
5. CGR prediction selection → should NOT delete previous neural word

