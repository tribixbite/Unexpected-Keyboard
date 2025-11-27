# Algorithm Documentation - Unexpected Keyboard

## Overview
Complete documentation of swipe typing recognition algorithms, scoring systems, and file locations.

## Core Recognition System

### KeyboardSwipeRecognizer Algorithm
**File**: `srcs/juloo.keyboard2/KeyboardSwipeRecognizer.java`
**Framework**: Bayesian Recognition - P(word | swipe) ∝ P(swipe | word) × P(word)

#### Main Entry Point
```java
public List<RecognitionResult> recognizeSwipe(List<PointF> swipePath, List<String> context)
```
**Location**: Line 139
**Process**:
1. Letter detection with error reporting
2. Candidate generation with error reporting  
3. Word scoring with error reporting
4. Final result ranking and return

### Scoring Components (P(swipe | word))

#### 1. Proximity Score - WORKING ✅
**Function**: `calculateProximityScore(String word, List<PointF> swipePath)`
**Location**: Line 474
**Algorithm**:
```java
// Get word template for key positions
Template wordTemplate = templateGenerator.generateWordTemplate(word);

// For each swipe point, calculate distance to template point
double distance = Math.sqrt(Math.pow(swipePoint.x - templatePoint.x, 2) + 
                           Math.pow(swipePoint.y - templatePoint.y, 2));

// Convert distance to proximity score (closer = higher score)
double proximityScore = Math.exp(-distance / keyZoneRadius);

// Apply start point emphasis (users begin precisely)
double pathPosition = (double)i / swipePath.size();
double positionWeight = startPointWeight * (1.0 - pathPosition) + 1.0;

totalScore += proximityScore * positionWeight;
```

**Parameters**:
- `keyZoneRadius = 180.0` (detection radius)
- `startPointWeight = 3.0` (start point emphasis)

**Output**: Average of all point-to-point proximity scores

#### 2. Sequence Score - PROBLEMATIC ❌
**Function**: `calculateSequenceScore(String word, List<Character> detectedLetters)`
**Location**: Line 529
**Algorithm**:
```java
// Check each required letter in word
for (char requiredLetter : word.toCharArray()) {
    // Look for letter in detected sequence
    if (!found) {
        score *= Math.exp(-missingKeyPenalty); // = exp(-10.05) ≈ 0.000045
    }
}

// Penalty for extra letters
for (Character detectedLetter : detectedLetters) {
    if (word.indexOf(detectedLetter) == -1) {
        score *= Math.exp(-extraKeyPenalty); // = exp(-1.99) ≈ 0.136
    }
}

// Order penalty
if (!isSubsequence(word, detectedSequence)) {
    score *= Math.exp(-orderPenalty); // = exp(-3.05) ≈ 0.047
}
```

**Parameters**:
- `missingKeyPenalty = 10.05` ← **TOO HARSH**
- `extraKeyPenalty = 1.99` ← **TOO HARSH** 
- `orderPenalty = 3.05` ← **TOO HARSH**

**Problem**: For partial matches, multiple missing letters cause score → 0.000000

#### 3. Start Point Score - WORKING ✅
**Function**: `calculateStartPointScore(String word, List<PointF> swipePath)`
**Location**: Line ~580 (need to verify)

#### 4. Language Model Score - WORKING ✅
**Function**: `calculateLanguageModelScore(String word, List<String> context)`
**Location**: Line ~600 (need to verify)

### Letter Detection Pipeline

#### Letter Detection
**Function**: `detectLetterSequence(List<PointF> swipePath)` 
**Location**: Line 233
**Algorithm**:
```java
// Sample strategically across swipe path
int sampleInterval = Math.max(20, swipePath.size() / 8); // Target ~8 letters max
double minDistanceForNewLetter = keyZoneRadius * 0.8; // 80% of radius

// For each sample point
Character nearestKey = getNearestKey(point);

// Filter: Require significant movement between letters
double distanceFromLast = Math.sqrt(Math.pow(point.x - lastValidPoint.x, 2) + 
                                   Math.pow(point.y - lastValidPoint.y, 2));
if (distanceFromLast >= minDistanceForNewLetter) {
    sequence.add(nearestKey);
}
```

**Improvements Made**:
- ✅ Reduced over-detection from 22+ letters to ~9 letters
- ✅ Added movement filtering to prevent noise
- ✅ Strategic sampling targets realistic word length

#### Key Detection
**Function**: `getNearestKey(PointF point)`
**Location**: Line 266
**Algorithm**:
```java
// Check all keyboard letters
String allLetters = "qwertyuiopasdfghjklzxcvbnm";
for (char c : allLetters.toCharArray()) {
    PointF keyCenter = getKeyCenter(c);
    double distance = Math.sqrt(Math.pow(point.x - keyCenter.x, 2) + 
                               Math.pow(point.y - keyCenter.y, 2));
    
    // Only consider if within key zone radius
    if (distance <= keyZoneRadius && distance < minDistance) {
        nearestKey = c;
    }
}
```

#### Template Coordinate System  
**Function**: `getKeyCenter(char letter)`
**Location**: Line 318
**Algorithm**:
```java
ContinuousGestureRecognizer.Point coord = templateGenerator.getCharacterCoordinate(letter);
return coord != null ? new PointF((float)coord.x, (float)coord.y) : null;
```

**Dependencies**:
- Uses same coordinate system as proximity calculation
- Requires `templateGenerator.setKeyboardDimensions(width, height)`

### Template Generation System

#### Word Template Generator
**File**: `srcs/juloo.keyboard2/WordGestureTemplateGenerator.java`
**Key Methods**:
- `generateWordTemplate(String word)` - Creates ideal swipe path for word
- `getCharacterCoordinate(char letter)` - Gets key center position
- `setKeyboardDimensions(float width, float height)` - Sets coordinate system
- `loadDictionary(Context context)` - Loads 10,000 word dictionary (was 5,000)

**Dictionary Limit**: Line 136
```java
while ((line = reader.readLine()) != null && wordCount < 10000) // Expanded from 5000
```

### Calibration vs Main Keyboard Integration

#### Calibration Page Setup
**File**: `srcs/juloo.keyboard2/SwipeCalibrationActivity.java`
**Key Integration Points**:

**Shared Recognizer Initialization** (Line ~505):
```java
_sharedRecognizer = new KeyboardSwipeRecognizer(this);
```

**Dimension Setup** (Line ~2268):
```java
_sharedRecognizer.setKeyboardDimensions(keyboardWidth, keyboardHeight);
```

**Recognition Call** (Line ~2287):
```java
List<RecognitionResult> newResults = _sharedRecognizer.recognizeSwipe(userSwipe, context);
```

#### Main Keyboard Integration - IDENTIFIED ✅
**Files**: 
- `srcs/juloo.keyboard2/Keyboard2.java` - Main service, calls SwipeTypingEngine
- `srcs/juloo.keyboard2/SwipeTypingEngine.java` - Contains KeyboardSwipeRecognizer
- `srcs/juloo.keyboard2/Keyboard2View.java` - Calls handleSwipeTyping

**Flow**: `Keyboard2View.java:306` → `Keyboard2.handleSwipeTyping()` → `AsyncPredictionHandler` → `SwipeTypingEngine.predict()`

**KEY DIFFERENCES FOUND**:
1. **Classification Layer**: `SwipeDetector.classifyInput()` filters swipes first
2. **Algorithm Selection**: `shouldUseDTW()` decides which prediction path to use
3. **Hybrid System**: May use `hybridPredict()` vs `enhancedSequencePredict()`
4. **Async Processing**: Uses AsyncPredictionHandler vs direct synchronous call

## Critical Issues

### 1. Sequence Score Always 0
**Root Cause**: Penalty system too harsh
```java
// 4 missing letters in "CHILDREN": 
score = 1.0 * exp(-10.05)^4 ≈ 0.000000000003
```

**Solutions to Consider**:
- Reduce penalty values (missingKeyPenalty: 10.05 → 2.0)
- Use additive penalties instead of multiplicative
- Implement partial credit for letter presence regardless of order
- Use edit distance (Levenshtein) for more forgiving matching

### 2. Main Keyboard vs Calibration Differences
**Investigation Needed**:
- Does main keyboard use KeyboardSwipeRecognizer or different algorithm?
- Are dimensions set consistently between calibration and main usage?
- Are the same parameters applied in both contexts?
- Is template generation identical in both cases?

### 3. Dictionary Size
**Status**: Expanded to 10,000 words but app update needed
**Location**: `WordGestureTemplateGenerator.java:136`

## Parameter Configuration

### KeyboardSwipeRecognizer Parameters
**File**: `srcs/juloo.keyboard2/KeyboardSwipeRecognizer.java`

**Default Values** (Lines 28-44):
```java
public double proximityWeight = 1.0;      // α - Distance to key centers
public double missingKeyPenalty = 10.0;   // β - Missing required letters ← TOO HARSH
public double extraKeyPenalty = 2.0;      // γ - Passing over wrong letters
public double orderPenalty = 5.0;         // δ - Out-of-order letter sequence  
public double startPointWeight = 3.0;     // ε - Start point accuracy emphasis
public double keyZoneRadius = 180.0;      // Key detection radius
public double pathSampleDistance = 5.0;   // Letter detection sampling frequency
```

### Settings Integration
**File**: `srcs/juloo.keyboard2/SwipeWeightConfig.java` (likely)
**Purpose**: Central weight management and user configuration

## Next Steps

### 1. Document Main Keyboard Integration
- Find where KeyboardSwipeRecognizer is used in main typing
- Compare parameter settings between calibration and main usage
- Identify why predictions differ between contexts

### 2. Fix Sequence Scoring
- Reduce harsh penalties to allow partial matches
- Test with milder penalty values
- Consider alternative scoring approaches

### 3. Ensure Main Keyboard Consistency ✅ SOLUTION IDENTIFIED
**Root Cause**: Main keyboard uses SwipeDetector classification that may bypass KeyboardSwipeRecognizer

**Solutions**:
1. **Force KeyboardSwipeRecognizer Path**: Bypass classification to always use CGR algorithm
2. **Sync Parameters**: Ensure SwipeTypingEngine._cgrRecognizer uses same parameters as calibration
3. **Direct Recognition**: Add option to bypass hybrid system and use KeyboardSwipeRecognizer directly
4. **Debug Classification**: Log why SwipeDetector chooses different algorithms

**Implementation Locations**:
- `SwipeTypingEngine.predict()` line 77 - Add forced CGR path option
- `SwipeDetector.classifyInput()` - Check classification logic
- `SwipeDetector.shouldUseDTW()` - Verify DTW decision logic

## Files to Investigate for Main Keyboard Integration

### Likely Integration Points:
- `srcs/juloo.keyboard2/Keyboard2.java` - Main input service
- `srcs/juloo.keyboard2/Keyboard2View.java` - Touch handling
- `srcs/juloo.keyboard2/Pointers.java` - Gesture processing
- `srcs/juloo.keyboard2/SwipeTypingEngine.java` - Prediction orchestration
- `srcs/juloo.keyboard2/SwipeInput.java` - Swipe input handling

### Questions to Answer:
1. Which class handles swipe recognition in main typing?
2. Are the same scoring parameters used?
3. Is template generation identical?
4. Are keyboard dimensions set the same way?