# Swipe Prediction Architecture - Complete Technical Breakdown

## 🏗️ SYSTEM OVERVIEW

The swipe prediction system uses a **dual-path architecture** with **ONNX transformer models** for neural swipe-to-text conversion. The system supports both **main keyboard usage** and **calibration/training** with unified prediction behavior.

---

## 📊 PREDICTION FLOW ARCHITECTURE

### **PATH 1: Main Keyboard Prediction Flow**

```
Touch Events (Keyboard2View.java)
    ↓
EnhancedSwipeGestureRecognizer.java
    ↓ 
Gesture Recognition Result (keys, path, timestamps)
    ↓
Keyboard2.handleSwipeTyping() [Line 1053]
    ↓
SwipeInput Creation [Line 1184]
    ↓
AsyncPredictionHandler.requestPredictions() [Line 1206]
    ↓ (Background Thread)
NeuralSwipeTypingEngine.predict() [Line 124]
    ↓
OnnxSwipePredictor.predict() [Line 246]
    ↓ 
ONNX Transformer Pipeline
    ↓
PredictionResult → UI Suggestions
```

### **PATH 2: Calibration Activity Flow**

```
Touch Events (SwipeCalibrationActivity.NeuralKeyboardView.onTouchEvent())
    ↓ (ACTION_UP)
recordSwipe() [Line 1101]
    ↓
SwipeInput Creation [Line 1120]
    ↓ (Synchronous)
NeuralSwipeTypingEngine.predict() [Line 1141]
    ↓
OnnxSwipePredictor.predict() [Line 246]
    ↓
ONNX Transformer Pipeline  
    ↓
PredictionResult → Calibration Logging
```

---

## 🧠 NEURAL PREDICTION PIPELINE

### **Core Engine: NeuralSwipeTypingEngine.java**

**Primary Entry Point:**
- `predict(SwipeInput input)` [Line 89]
- **Initialization**: `initialize()` [Line 47] - loads ONNX models
- **Configuration**: `setConfig(Config config)` [Line 115] - updates beam search parameters
- **Debugging**: `setDebugLogger()` [Line 120] - enables detailed logging

**Prediction Process:**
1. **Input Validation**: Checks initialization state
2. **Delegate to ONNX**: Calls `_neuralPredictor.predict(input)` [Line 105]
3. **Result Formatting**: Returns `PredictionResult` with words and scores

### **ONNX Implementation: OnnxSwipePredictor.java**

**Core Architecture:**
- **Transformer Encoder**: Processes trajectory features → memory states
- **Transformer Decoder**: Memory states → word predictions via beam search
- **ONNX Runtime**: Uses Samsung S25U QNN execution provider for NPU acceleration

**Model Loading:**
```java
loadModelFromAssets("models/swipe_encoder.onnx")  // 5.3MB encoder
loadModelFromAssets("models/swipe_decoder.onnx")  // 7.2MB decoder
```

**Feature Extraction Pipeline:**
1. **Trajectory Processing**: `SwipeTrajectoryProcessor.extractFeatures()` 
   - Coordinates → [x,y,vx,vy,ax,ay] features [150×6 tensor]
   - Nearest key detection for each point
   - Source mask creation for valid trajectory points

2. **Encoder Inference**: 
   - Input: `[trajectory_features, nearest_keys, src_mask]`
   - Output: `encoder_output` [150×256 memory tensor]

3. **Beam Search Decoder**:
   - **Algorithm**: Transformer decoder with beam search
   - **Inputs**: `[memory, target_tokens, src_mask, target_mask]`
   - **Output**: `logits` [batch×sequence×vocab_size]

---

## ⚙️ CONFIGURATION PARAMETERS

### **Beam Search Configuration (Dynamic - User Configurable)**

**Source**: `Config.java` [Lines 228-230]
```java
neural_beam_width = safeGetInt(_prefs, "neural_beam_width", 8);      // Default: 8, Range: 1-16
neural_max_length = safeGetInt(_prefs, "neural_max_length", 35);     // Default: 35, Range: 10-50  
neural_confidence_threshold = _prefs.getFloat("neural_confidence_threshold", 0.1f); // Default: 0.1
```

**Persistence**: 
- **Playground Settings**: `SwipeCalibrationActivity.showNeuralPlayground()` [Line 633]
- **Settings Storage**: `SharedPreferences` via playground Apply button [Line 656]
- **Live Updates**: `_neuralEngine.setConfig(_config)` applies changes immediately

### **Model Configuration (Static - Hard-coded)**

**ONNX Parameters**: `OnnxSwipePredictor.java`
```java
MAX_SEQUENCE_LENGTH = 150;        // Maximum trajectory points
TRAJECTORY_FEATURES = 6;          // [x,y,vx,vy,ax,ay] per point
NORMALIZED_WIDTH = 1.0f;          // Coordinate normalization
NORMALIZED_HEIGHT = 1.0f;         // Coordinate normalization
```

**Special Tokens**:
```java
PAD_IDX = 0;    // Padding token
UNK_IDX = 1;    // Unknown token  
SOS_IDX = 2;    // Start of sequence
EOS_IDX = 3;    // End of sequence
```

**Hardware Acceleration**: 
- **QNN Provider**: Samsung S25U Snapdragon NPU support
- **Optimization Level**: `ALL_OPT` for maximum performance
- **Threading**: Auto-detect intra-op threads

---

## 📚 VOCABULARY & WORD LISTS

### **Neural Vocabulary: NeuralVocabulary.java**

**Primary Source**: `assets/dictionaries/`
- **File**: `en.txt` - 9,999 most frequent English words
- **File**: `en_enhanced.txt` - 9,999 additional vocabulary words
- **Combined**: ~20,000 unique words after deduplication

**Vocabulary Loading**: `loadVocabulary()` method
- **Format**: One word per line, lowercase
- **Processing**: Automatic deduplication via `HashSet`
- **Size**: Actual loaded vocabulary: 9,892 words

### **Calibration Word Selection: SwipeCalibrationActivity.java**

**Random Word Generation**: `prepareRandomSessionWords()` [Line 194]
```java
WORDS_PER_SESSION = 20;                    // Fixed session size
fullVocabulary = ~20,000 words;            // Combined dictionaries  
```

**Selection Process**:
1. **Dictionary Loading**: `loadFullVocabulary()` [Line 243]
   - Loads both `en.txt` and `en_enhanced.txt`
   - **No filtering**: Uses all words from dictionaries
   - **Deduplication**: `HashSet` prevents duplicates

2. **Random Sampling**: 
   - **Algorithm**: `random.nextInt(fullVocabulary.size())`
   - **No filtering**: Uses any word from vocabulary
   - **Duplicate prevention**: `Set<String> selectedWords`

**Fallback**: If vocabulary loading fails → empty session (no fallback words)

### **Tokenization: SwipeTokenizer.java**

**Character-to-Token Mapping**:
```java
Vocabulary Size: 30 tokens
Mapping: 'a'→4, 'b'→5, 'c'→6, ..., 'z'→29
Special tokens: PAD(0), UNK(1), SOS(2), EOS(3)
```

---

## 🔀 FALLBACK MECHANISMS

### **Primary Fallback Chain**

1. **Neural Engine Failure**: 
   - **Trigger**: ONNX model loading fails or prediction throws exception
   - **Fallback**: `WordPredictor.java` (legacy DTW/Bayesian system)
   - **Location**: `Keyboard2.handleSwipeTyping()` [Line 1101]

2. **AsyncPredictionHandler Unavailable**:
   - **Trigger**: Background thread initialization fails
   - **Fallback**: Synchronous prediction on UI thread [Line 1218]
   - **Behavior**: Same neural engine, different execution model

3. **Vocabulary Loading Failure**:
   - **Trigger**: Dictionary assets not found or corrupted
   - **Fallback**: Empty vocabulary → no calibration words
   - **Location**: `SwipeCalibrationActivity.loadFullVocabulary()` [Line 274]

4. **ONNX Session Creation Failure**:
   - **Trigger**: Hardware incompatibility or model corruption
   - **Fallback**: Error dialog → activity termination
   - **Location**: `SwipeCalibrationActivity.onCreate()` [Line 143]

### **Configuration Fallbacks**

**Missing Preferences**:
- `neural_beam_width`: defaults to 8
- `neural_max_length`: defaults to 35  
- `neural_confidence_threshold`: defaults to 0.1

**Hardware Fallbacks**:
- **Primary**: QNN execution provider (NPU acceleration)
- **Fallback**: CPU execution provider
- **Detection**: Automatic via ONNX Runtime

---

## 🔧 KEY ARCHITECTURAL COMPONENTS

### **1. Touch Event Processing**

**Main Keyboard**: `Keyboard2View.java`
- **Entry**: `Pointers.onTouchEvent()` → gesture recognition
- **Processing**: `EnhancedSwipeGestureRecognizer` interprets gestures
- **Output**: Processed coordinates + detected keys

**Calibration**: `SwipeCalibrationActivity.NeuralKeyboardView`
- **Entry**: Direct `onTouchEvent()` handling [Line 985]
- **Processing**: Raw coordinate collection in `_swipePoints`
- **Output**: Unprocessed touch coordinates

### **2. SwipeInput Creation - CRITICAL DIFFERENCE**

**Main Keyboard**: `Keyboard2.java` [Line 1184]
```java
SwipeInput swipeInput = new SwipeInput(
    swipePath,           // Processed coordinates from gesture recognizer
    timestamps,          // Processed timestamps  
    new ArrayList<>()    // NOW EMPTY - matches calibration (FIXED)
);
```

**Calibration**: `SwipeCalibrationActivity.java` [Line 1120]
```java
SwipeInput swipeInput = new SwipeInput(
    points,                    // Raw touch coordinates
    _currentSwipeTimestamps,   // Raw touch timestamps
    new ArrayList<>()          // Empty - neural system detects keys
);
```

### **3. Prediction Execution Models**

**Asynchronous (Main Keyboard)**:
- **Handler**: `AsyncPredictionHandler.java` with `HandlerThread`
- **Cancellation**: Request ID system prevents stale predictions
- **Callback**: `PredictionCallback` interface for UI updates
- **Thread Safety**: Background prediction + UI thread callbacks

**Synchronous (Calibration)**:
- **Direct Call**: `_neuralEngine.predict()` on UI thread
- **Blocking**: Waits for complete prediction before UI update
- **Timing**: Precise performance measurement possible
- **Debugging**: Full stack trace and timing analysis

---

## 🎯 PERFORMANCE CHARACTERISTICS

### **Hardware Acceleration**

**Samsung S25U Optimizations**:
- **QNN Execution Provider**: Snapdragon NPU acceleration
- **HTP Burst Mode**: Maximum performance mode enabled
- **Memory Optimization**: Pattern optimization for transformer models
- **Thread Configuration**: Auto-detect optimal thread count

### **Model Performance Metrics**

**Current Performance** (from logs):
- **Encoder Inference**: ~8-15ms
- **Decoder Beam Search**: ~3-16 seconds (varies by beam width/length)
- **Total Prediction**: 3.7-16.7 seconds
- **Memory Usage**: ~43MB APK (includes 12.5MB ONNX models)

**Optimization Targets**:
- **Web Demo Comparison**: Near-instant (<100ms) vs current 3-16 seconds
- **Performance Gap**: 30-160x slower than web demo with same models
- **Investigation Needed**: Hardware acceleration utilization analysis

---

## 🔍 DEBUGGING & LOGGING

### **Debug Infrastructure**

**Neural Engine Logging**: `NeuralSwipeTypingEngine.java`
- **Stack Traces**: Full call stack logging [Line 92]
- **Performance Timing**: Prediction duration measurement
- **Debug Logger**: Configurable logging interface [Line 120]

**Calibration Logging**: `SwipeCalibrationActivity.java`
- **Results Log**: `_resultsLog` with timestamp formatting [Line 725]
- **Copy to Clipboard**: Full debug export capability [Line 747]
- **Performance Tracking**: Accuracy and timing statistics [Line 472]

**ONNX Predictor Logging**: `OnnxSwipePredictor.java`
- **Tensor Shapes**: Input/output tensor debugging
- **Beam Search Steps**: Detailed step-by-step progression
- **Performance Breakdown**: Preprocess/Encoder/Decoder/Postprocess timing

### **Configuration Debugging**

**Playground Interface**: `SwipeCalibrationActivity.showNeuralPlayground()` [Line 633]
- **Beam Width**: Slider control (1-16) [Line 643]
- **Max Length**: Slider control (10-50) [Line 647] 
- **Confidence Threshold**: Float slider (0.0-1.0) [Line 651]
- **Live Preview**: Real-time parameter adjustment
- **Persistence**: Settings saved to SharedPreferences [Line 657]

---

## 🔄 DATA FLOW DETAILS

### **SwipeInput Structure**
```java
public class SwipeInput {
    List<PointF> coordinates;     // Touch trajectory points
    List<Long> timestamps;        // Touch event timing
    List<String> detectedKeys;    // Pre-detected keys (NOW EMPTY for consistency)
    
    // Computed properties:
    String keySequence;           // Derived from coordinates
    float pathLength;             // Total gesture distance
    float duration;               // Total gesture time
}
```

### **Coordinate Processing**

**Raw Touch Data**:
- **Source**: Android `MotionEvent.getX()` / `getY()`
- **Coordinate System**: Screen pixels (device-specific)
- **Frequency**: ~60-120 Hz depending on device
- **Precision**: Sub-pixel accuracy

**Feature Extraction**: `SwipeTrajectoryProcessor.java`
1. **Coordinate Normalization**: Screen space → [0,1] normalized space
2. **Velocity Calculation**: Finite difference approximation
3. **Acceleration**: Second derivative of position
4. **Nearest Key Detection**: Maps coordinates to keyboard layout
5. **Tensor Creation**: [150×6] trajectory features tensor

### **Neural Processing Pipeline**

**Step 1: Encoder Processing**
```java
Input Tensors:
- trajectory_features: [1, 150, 6] (FLOAT32)
- nearest_keys: [1, 150] (INT64)  
- src_mask: [1, 150] (BOOL)

Output:
- encoder_output: [1, 150, 256] (memory tensor)
```

**Step 2: Decoder Beam Search**
```java
Beam Search Parameters:
- beam_width: 4-8 (user configurable, default 8)
- max_length: 22-35 (user configurable, default 35)  
- confidence_threshold: 0.1 (user configurable)

Per-Step Processing:
- Input: [memory, target_tokens, src_mask, target_mask]
- Output: [1, 20, 30] logits (vocabulary probabilities)
- Beam Management: Top-k selection and expansion
```

**Step 3: Post-Processing**
```java
Vocabulary Lookup: Token IDs → String words
Confidence Scoring: Log probability → normalized scores
Result Filtering: Remove non-dictionary words
Quality Ranking: Sort by confidence scores
```

---

## 📈 CONFIGURATION MATRIX

### **User-Configurable Parameters**

| Parameter | Source | Default | Range | Impact |
|-----------|---------|---------|-------|---------|
| `neural_beam_width` | SharedPreferences | 8 | 1-16 | Search breadth vs speed |
| `neural_max_length` | SharedPreferences | 35 | 10-50 | Max word length |
| `neural_confidence_threshold` | SharedPreferences | 0.1 | 0.0-1.0 | Prediction filtering |
| `swipe_typing_enabled` | SharedPreferences | true | bool | Global feature toggle |
| `neural_prediction_enabled` | SharedPreferences | true | bool | Neural vs fallback |

### **Hard-coded Constants**

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `MAX_SEQUENCE_LENGTH` | OnnxSwipePredictor.java:35 | 150 | Trajectory tensor size |
| `TRAJECTORY_FEATURES` | OnnxSwipePredictor.java:37 | 6 | Feature dimensions |
| `WORDS_PER_SESSION` | SwipeCalibrationActivity.java:49 | 20 | Calibration session size |
| `Vocabulary Size` | SwipeTokenizer | 30 | Character tokens + specials |

---

## 🗂️ VOCABULARY MANAGEMENT

### **Dictionary Assets Structure**
```
assets/dictionaries/
├── en.txt                    # 9,999 most frequent words
├── en_enhanced.txt           # 9,999 additional words  
├── fr.txt                    # French vocabulary (58 words)
├── es.txt                    # Spanish vocabulary (58 words)
└── de.txt                    # German vocabulary (58 words)
```

### **Vocabulary Loading Details**

**Neural Vocabulary**: `NeuralVocabulary.java`
- **Primary**: Loads from word frequency lists
- **Size**: 9,892 words (after processing)
- **Format**: Memory-efficient hash maps for fast lookup
- **Usage**: Neural prediction post-processing and validation

**Calibration Vocabulary**: `SwipeCalibrationActivity.loadFullVocabulary()` [Line 243]
- **Sources**: Both `en.txt` and `en_enhanced.txt`
- **Processing**: Automatic deduplication via `HashSet<String>`
- **Size**: ~20,000 unique words total
- **Filtering**: **NONE** - uses all dictionary words
- **Usage**: Random test word selection for calibration sessions

### **Word Selection Algorithms**

**Calibration Random Selection**: `prepareRandomSessionWords()` [Line 194]
```java
Algorithm: Pure random sampling
Input: fullVocabulary (~20,000 words)
Process: 
  1. Generate random indices: random.nextInt(fullVocabulary.size())
  2. Collect unique words: Set<String> prevents duplicates
  3. No filtering applied - any word can be selected
Output: 20 random words for testing session
```

**Neural Vocabulary Access**:
```java
Token-to-Word Mapping: Array-based lookup by token ID
Word-to-Token Mapping: HashMap for reverse lookup
Frequency Data: Statistical weights for prediction scoring
```

---

## 🚨 CRITICAL ARCHITECTURAL DIFFERENCES

### **SwipeInput Creation Inconsistency (RECENTLY FIXED)**

**Previous Issue**:
```java
// Main Keyboard (OLD - WRONG)
SwipeInput(swipePath, timestamps, swipedKeys);  // Pre-detected keys

// Calibration (CORRECT)  
SwipeInput(points, timestamps, new ArrayList<>()); // Empty keys
```

**Current State (FIXED)**:
```java
// Both systems now use empty swipedKeys for consistency
SwipeInput(coordinates, timestamps, new ArrayList<>());
```

### **Coordinate Source Differences (REMAINING ISSUE)**

**Main Keyboard**: 
- **Source**: `EnhancedSwipeGestureRecognizer` processed coordinates
- **Processing**: Gesture recognition, smoothing, filtering
- **Format**: `List<PointF>` from `result.path`

**Calibration**:
- **Source**: Raw `MotionEvent` coordinates  
- **Processing**: Direct collection from touch events
- **Format**: `List<PointF>` from `_swipePoints`

**Impact**: Different coordinate data may affect neural prediction quality

---

## 🔧 ASYNC PREDICTION INFRASTRUCTURE

### **AsyncPredictionHandler.java Architecture**

**Threading Model**:
- **Background Thread**: `HandlerThread` for neural processing
- **Request Queue**: Message-based request handling
- **Cancellation**: Request ID system for stale prediction cleanup
- **UI Callbacks**: Results delivered to main thread

**Request Processing**:
```java
Method: requestPredictions(SwipeInput, PredictionCallback)
Flow:
  1. Generate unique request ID
  2. Cancel pending predictions  
  3. Queue prediction message
  4. Background: Execute _neuralEngine.predict()
  5. UI Thread: Deliver results via callback
```

**Cancellation Logic**:
- **Automatic**: New requests cancel previous ones
- **Manual**: `cancelPendingPredictions()` method
- **Thread-Safe**: Atomic request ID management

---

## 📱 INTEGRATION POINTS

### **Main Keyboard Integration**: `Keyboard2.java`

**Initialization**: `onCreate()` [Line 188]
```java
if (_config.swipe_typing_enabled) {
    _neuralEngine = new NeuralSwipeTypingEngine(this, _config);
    _asyncPredictionHandler = new AsyncPredictionHandler(_neuralEngine);
}
```

**Gesture Processing**: `handleSwipeTyping()` [Line 1053]
- **Input Validation**: Check swipe typing enabled
- **Data Logging**: Extensive coordinate and key debugging
- **Prediction Request**: Via AsyncPredictionHandler
- **Result Handling**: UI suggestion bar updates

**Suggestion Display**: `handlePredictionResults()` [Line 769]
- **UI Update**: `_suggestionBar.setSuggestions()`
- **Logging**: Prediction results and scores
- **Error Handling**: Clear suggestions on failure

### **Configuration Integration**: `Config.java`

**Neural Settings Loading**: [Line 227]
- **Preferences Access**: `DirectBootAwarePreferences`
- **Safe Parsing**: `safeGetInt()` with defaults
- **Global Config**: `Config.globalConfig()` singleton pattern

**Setting Persistence**: Automatic via Android SharedPreferences system

---

## 🎮 CALIBRATION SYSTEM

### **SwipeCalibrationActivity.java Overview**

**Purpose**: Neural training data collection + performance testing
- **Session Management**: 20 random words per session
- **Data Storage**: `SwipeMLDataStore` for training data
- **Performance Metrics**: Accuracy and timing analysis
- **Export Capability**: JSON format training data export

**UI Components**:
- **Custom Keyboard**: `NeuralKeyboardView` with 4-row QWERTY layout
- **Playground**: Live parameter adjustment interface
- **Results Log**: Real-time prediction analysis
- **Progress Tracking**: Session completion status

### **Training Data Storage**: `SwipeMLDataStore.java`

**Database Schema**: SQLite storage for swipe training data
- **Target Words**: Expected prediction results
- **Trace Points**: Raw coordinate trajectories  
- **Screen Dimensions**: Device-specific calibration data
- **Source Tracking**: Calibration vs production data classification

---

## 🚀 PERFORMANCE OPTIMIZATION OPPORTUNITIES

### **Current Bottlenecks** (Identified)

1. **Decoder Beam Search**: 3-16 second prediction times
2. **Sequential Processing**: Single-threaded beam expansion
3. **Tensor Creation**: Per-step tensor allocation overhead
4. **Hardware Utilization**: Potential QNN acceleration underutilization

### **Web Demo Performance Gap**

**Current**: 3.7-16.7 seconds per prediction
**Web Demo**: <100ms for same ONNX models  
**Gap**: 30-160x performance difference

**Investigation Priorities**:
1. Hardware acceleration verification (QNN vs CPU execution)
2. Batch processing implementation (8 beams → single decoder call)
3. Model quantization for NPU optimization
4. Tensor memory management optimization

---

## 📋 ARCHITECTURAL SUMMARY

**Strengths**:
- ✅ Proper ONNX transformer implementation
- ✅ Hardware acceleration support (QNN)
- ✅ Comprehensive debugging infrastructure  
- ✅ User-configurable parameters
- ✅ Robust fallback mechanisms
- ✅ Training data collection system

**Current Issues**:
- ❌ Coordinate processing inconsistency (main KB vs calibration)
- ❌ Massive performance gap vs web demo
- ❌ Complex async handling may be over-engineered
- ❌ Multiple gesture recognition layers causing confusion

**Immediate Priorities**:
1. **Data Pipeline Unification**: Make main keyboard use raw coordinates like calibration
2. **Performance Investigation**: Understand 30-160x slowdown vs web demo
3. **Architecture Simplification**: Reduce gesture processing complexity

This architecture provides a solid foundation but requires optimization for production-ready performance and behavioral consistency.