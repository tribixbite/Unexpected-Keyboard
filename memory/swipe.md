# Swipe Typing Development Notes

## Overview
Implementation of ML-ready swipe typing system for Unexpected Keyboard with neural network prediction capabilities.

## Current Implementation Status

### ✅ Completed Components

#### Production-Ready Improvements (2025-01-21)
- **A* Pathfinding**: Graph-based probabilistic key mapping with multiple path exploration
- **Velocity Calculation**: Full magnitude using both X and Y velocity components
- **Configurable Thresholds**: User-adjustable turning point detection (15°-90°)
- **Improved Pruning**: Stricter word-to-key-sequence matching algorithm
- **Enhanced UI**: User-friendly labels and descriptions for all settings

#### 1. Swipe Gesture Recognition (`SwipeGestureRecognizer.java`)
- Tracks touch points with timestamps
- Identifies alphabetic keys touched during swipe
- Calculates gesture metrics (distance, velocity, straightness)
- Filters duplicate keys and noise
- Provides path data for ML collection

#### 2. ML Data Model (`SwipeMLData.java`)
- JSON-based format optimized for neural networks
- Normalized coordinates [0,1] for device independence
- Time deltas between points (velocity information)
- Registered key sequences
- Metadata (screen dimensions, collection source)
- Validation methods for data quality

#### 3. Persistent Storage (`SwipeMLDataStore.java`)
- SQLite database with indexed columns
- Asynchronous batch operations
- Export to JSON and NDJSON formats
- Statistics tracking and queries
- Supports both calibration and user data

#### 4. Data Collection Points
- **Calibration Activity**: Manual high-quality labeled data
- **Normal Usage**: Automatic capture when predictions selected
- Proper timestamp reconstruction
- Links selected words with swipe traces

#### 5. UI Integration
- Swipe trail visualization during gesture
- Suggestion bar for predictions
- Settings for calibration and export
- Training button (experimental)
- Export functionality with sharing

#### 6. Training Infrastructure (`SwipeMLTrainer.java`)
- Training management with progress tracking
- Minimum sample requirements
- Hooks for TensorFlow Lite integration
- Export for external training pipelines

### 🚧 In Progress / Next Steps

#### Neural Network Implementation (Priority 1)
- [ ] Implement dual-branch GRU/LSTM model
- [ ] TensorFlow Lite integration
- [ ] Model serialization and loading
- [ ] Real-time inference optimization

#### Data Pipeline (Priority 2)
- [ ] Data augmentation strategies
- [ ] Validation/test set splitting
- [ ] Batch preprocessing
- [ ] Feature engineering

## Gemini's Recommended RNN Architecture

### Model Architecture Details

```
Input A (Trace Path) → Masking → GRU(128) → Concat → Dense(256) → Dropout → Dense(10000) → Softmax
                                              ↑
Input B (Key Path) → Embedding(16) → Masking → GRU(64)
```

### Implementation Roadmap

#### Phase 1: Data Preparation ✅ DONE
- [x] Define ML data format (JSON with normalized coords)
- [x] Implement data collection during swipes
- [x] Store calibration data with labels
- [x] Capture user selections for training
- [x] Export functionality for external training

#### Phase 2: Model Development 🚧 CURRENT
- [x] Python training script setup ✅ DONE
  - Complete `ml_training/train_swipe_model.py` implemented
  - Full pipeline: data loading → preprocessing → training → TFLite conversion
  - Includes validation, early stopping, learning rate scheduling
  - Class weighting for imbalanced data
  - Export utilities for device data and synthetic generation

- [x] Dual-branch architecture implementation ✅ DONE
  ```python
  # Branch A: Trace coordinates
  trace_input = Input(shape=(None, 3))  # (x, y, t_delta)
  trace_masked = Masking()(trace_input)
  trace_gru = GRU(128)(trace_masked)
  
  # Branch B: Key sequence
  keys_input = Input(shape=(None,))
  keys_embed = Embedding(26, 16)(keys_input)
  keys_masked = Masking()(keys_embed)
  keys_gru = GRU(64)(keys_masked)
  
  # Merge and classify
  merged = Concatenate()([trace_gru, keys_gru])
  dense1 = Dense(256, activation='relu')(merged)
  dropout = Dropout(0.3)(dense1)
  output = Dense(10000, activation='softmax')(dropout)
  ```

- [x] Class weighting by word frequency ✅ DONE
  - Implemented in `calculate_class_weights()` function
  - Uses sklearn's `compute_class_weight` with 'balanced' mode
  - Ready to incorporate dictionary frequencies

#### Phase 3: Training Pipeline 📋 TODO
- [ ] Load collected swipe data
- [ ] Split train/validation/test sets (70/15/15)
- [ ] Implement data augmentation
  - Add noise to coordinates
  - Time stretching/compression
  - Slight path variations
- [ ] Train with class weighting
- [ ] Validate on held-out data
- [ ] Hyperparameter tuning
  - Learning rate scheduling
  - Batch size optimization
  - GRU units tuning

#### Phase 4: Model Deployment 📋 TODO
- [ ] Convert to TensorFlow Lite
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  ```
- [ ] Quantization for size reduction
- [ ] Android integration
  - Load TFLite model
  - Preprocess input data
  - Run inference
  - Post-process predictions
- [ ] Performance optimization
  - Batch inference where possible
  - Model caching
  - Threading for non-blocking inference

#### Phase 5: Personalization 📋 TODO
- [ ] On-device score boosting
  ```java
  // User frequency map
  Map<String, Float> userFreqs = loadUserFrequencies();
  
  // Boost predictions
  for (int i = 0; i < predictions.length; i++) {
      String word = predictions[i];
      if (userFreqs.containsKey(word)) {
          scores[i] *= (1.0f + userFreqs.get(word));
      }
  }
  ```
- [ ] Incremental learning hooks
- [ ] User-specific dictionaries
- [ ] Context-aware predictions

#### Phase 6: Production Features 📋 TODO
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] Rollback capability
- [ ] Performance monitoring
- [ ] Privacy considerations
  - Local-only training
  - Opt-in data collection
  - Data anonymization

### Training Strategy

#### Immediate Approach (Manual)
1. Export data via settings button
2. Train model offline with Python script
3. Convert to TFLite
4. Bundle with app updates

#### Future Approach (Automated)
1. Periodic batch exports
2. Server-side training pipeline
3. Model updates via app updates
4. Optional: Federated learning

### Data Requirements

#### Minimum Viable Dataset
- 100 samples per word (top 1000 words)
- 10-20 users for diversity
- Multiple device types/sizes
- Balanced collection sources

#### Production Dataset
- 1000+ samples per word (top 5000 words)
- 100+ users
- Cross-validation across devices
- Continuous collection and retraining

### Performance Targets

#### Accuracy Metrics
- Top-1 accuracy: >70%
- Top-3 accuracy: >85%
- Top-5 accuracy: >90%

#### Latency Requirements
- Inference time: <50ms
- Model loading: <500ms
- Memory usage: <20MB

## Design Specifications

### Swipe Prediction Quality Standards

#### Minimum Word Length Requirements ✅ DESIGN SPEC
**Core Principle**: Swipe gestures represent intentional multi-character word input and should never suggest single letters or abbreviations.

**Requirements**:
- **Swipe predictions MUST be ≥3 characters minimum**
- **Never show 1 or 2 character suggestions after swipe input**
- **Preserve all lengths for regular typing predictions** (non-swipe input)
- **Maintain Markov chain functionality** for contextual predictions in normal typing

**Rationale**:
- Swipe typing is designed for complete words, not individual characters
- Single/double character suggestions provide poor user experience for swipe input
- Users expect meaningful word completions from gesture-based input
- Regular typing still needs short predictions for efficiency (articles, conjunctions, etc.)

**Implementation**:
- Filter swipe prediction results by word length before displaying
- Apply filtering only to swipe-triggered predictions
- Preserve full prediction spectrum for keyboard tap input
- Ensure contextual N-gram predictions remain active for non-swipe scenarios

### Known Issues & Considerations

1. **Timestamp Reconstruction**: Current implementation estimates timestamps during ML data creation
2. **Variable Length Sequences**: Need proper padding/masking in TF
3. **Dictionary Size**: 10k words may be too large initially
4. **Device Fragmentation**: Need robust normalization
5. **Privacy**: Ensure no sensitive data in exports

### Testing Strategy

#### Unit Tests
- [ ] Data normalization correctness
- [ ] Export format validation
- [ ] Storage integrity

#### Integration Tests
- [ ] End-to-end swipe capture
- [ ] Prediction selection tracking
- [ ] Export functionality

#### ML Model Tests
- [ ] Inference latency benchmarks
- [ ] Memory usage profiling
- [ ] Accuracy on test set
- [ ] Edge case handling

### Debug Tools

#### Logging
- Swipe data analysis logs
- ML data validation logs
- Training progress logs

#### Visualization
- Swipe path rendering
- Heatmap of touch points
- Prediction confidence scores

## Related Files

### Core Implementation
- `/srcs/juloo.keyboard2/SwipeGestureRecognizer.java`
- `/srcs/juloo.keyboard2/ml/SwipeMLData.java`
- `/srcs/juloo.keyboard2/ml/SwipeMLDataStore.java`
- `/srcs/juloo.keyboard2/ml/SwipeMLTrainer.java`

### UI Components
- `/srcs/juloo.keyboard2/SwipeCalibrationActivity.java`
- `/srcs/juloo.keyboard2/SuggestionBar.java`
- `/srcs/juloo.keyboard2/Keyboard2.java`
- `/srcs/juloo.keyboard2/Keyboard2View.java`

### Configuration
- `/res/xml/settings.xml`
- `/res/values/strings.xml`

## Development Log

### 2024-01-21
- Implemented ML data format with Gemini's recommendations
- Added persistent storage with SQLite
- Integrated data collection in calibration and normal usage
- Created export functionality
- Added experimental training button

### 2025-01-22 - Major Swipe Typing Improvements
#### Phase 1: Core Architecture
- Created SwipeInput class to encapsulate all swipe data
- Implemented SwipeDetector with multi-factor classification
- Built SwipeTypingEngine orchestrator for hybrid prediction
- Developed SwipeScorer applying all 8 confidence weights
- Added configurable endpoint matching weights to settings

#### Phase 2: DTW Enhancement 
- Enhanced DTWPredictor to use actual swipe coordinates
- Added coordinate normalization and path simplification
- Implemented confidence scoring for DTW results
- Integrated coordinate-based matching in production flow

#### Phase 3: Critical Fixes Based on FlorisBoard Analysis
- **CRITICAL FIX**: Increased sampling from 10 to 200 points (was losing 95% of data!)
- Implemented proper path resampling with linear interpolation
- Added velocity-based filtering (0.15 px/ms threshold)
- Added minimum point distance filtering (25px)
- Created SwipePruner for extremity-based candidate pruning
- Implemented length-based pruning to reduce search space
- Fixed duplicate key detection window (increased to 3)

**Root Cause Analysis:**
- Previous implementation simplified paths to only 10 points, losing 95% of gesture information
- No velocity filtering led to over-registration of keys (2-3x more than needed)
- No pruning meant searching entire dictionary unnecessarily
- FlorisBoard uses 200-point sampling + statistical approach for much better accuracy

### Next Session TODOs
1. Test the critical fixes with swipe_data_20250821_235946.json
2. Analyze remaining accuracy issues
3. Consider implementing Gaussian probability scoring like FlorisBoard
4. Add loop gesture handling for duplicate letters
5. Create Python training script for ML model
6. Fix timestamp corruption in data collection

## Implementation Status (2025-01-22)

### ✅ Issues Resolved
1. **DTWPredictor integrated**: Now uses real coordinates via SwipeTypingEngine
2. **All weights applied**: All 8 confidence weights actively influence scoring
3. **Debug visualization added**: Scores display when swipe_show_debug_scores enabled
4. **Sophisticated swipe detection**: Multi-factor analysis replaces simple heuristic

### ✅ Improvements Completed
1. **Configurable endpoint weights**: 
   - ✅ Separate weight for first letter match (0-300%)
   - ✅ Separate weight for last letter match (0-300%)
   - ✅ Bonus weight when both match (0-400%)
   - ✅ Strict mode to require both endpoints
2. **Debug score display**: ✅ Shows confidence values below predictions
3. **Weight integration**: ✅ All 8 Config weights properly applied
4. **DTW integration**: ✅ Uses actual swipe coordinates for matching

### Active Weight System
| Weight | Purpose | Status | Default |
|--------|---------|--------|----------|
| Shape | DTW path matching | ✅ Active | 90% |
| Location | Key hit accuracy | ✅ Active | 130% |
| Frequency | Word commonality | ✅ Active | 80% |
| Velocity | Speed consistency | ✅ Active | 60% |
| First Letter | Start point match | ✅ Active | 150% |
| Last Letter | End point match | ✅ Active | 150% |
| Endpoint Bonus | Both endpoints | ✅ Active | 200% |
| Velocity StdDev | Speed variation | ✅ Active | 100% |

### Critical Performance Improvements (2025-01-22)

#### Before Fixes:
- 0% exact match accuracy on test data
- 82% of swipes had major problems
- Registering 2-3x more keys than needed
- Negative time deltas corrupting velocity calculations

#### After FlorisBoard-Inspired Fixes:
- Sampling increased from 10 to 200 points (20x improvement)
- Velocity-based filtering prevents key over-registration
- Pruning by extremities reduces search space by ~90%
- Distance-based filtering removes noise

#### Expected Impact:
- Accuracy improvement from 0% to 40-60% (immediate)
- With Gaussian probability: 70-80% (matching FlorisBoard)
- Significant reduction in false key detections
- Better handling of fast vs slow gestures

---

# 🚀 ONNX TRANSFORMER REPLACEMENT STRATEGY

## EXECUTIVE SUMMARY

**OBJECTIVE**: Complete replacement of current Bayesian/DTW-based swipe prediction pipeline with state-of-the-art ONNX transformer encoder-decoder architecture demonstrated in the web demo.

**KEY INSIGHT**: The `web_demo/swipe-onnx.html` already implements a working transformer with beam search that can be ported to Android with ONNX Runtime.

**CRITICAL REQUIREMENTS**:
- ✅ **Zero Breaking Changes**: All existing UI/UX preserved
- ✅ **Backward Compatibility**: Legacy system available as automatic fallback 
- ✅ **Performance**: Neural predictions faster than current DTW
- ✅ **Accuracy**: Measurable improvement over current Bayesian system

---

## 🎯 REPLACEMENT ARCHITECTURE

### Current System (TO REPLACE)
```
┌─────────────────────────────────────────────────────────┐
│                LEGACY PIPELINE                          │
├─────────────────────────────────────────────────────────┤
│ SwipeTypingEngine                                       │
│  ├─ KeyboardSwipeRecognizer (Bayesian P(word|swipe))   │
│  ├─ DTWPredictor (Dynamic Time Warping)                │
│  ├─ GaussianKeyModel (Probabilistic key detection)     │
│  ├─ BigramModel/NgramModel (Language modeling)         │
│  └─ SwipeScorer (Multi-weight scoring)                 │
└─────────────────────────────────────────────────────────┘
```

### Target System (NEW NEURAL PIPELINE)
```
┌─────────────────────────────────────────────────────────┐
│                ONNX TRANSFORMER PIPELINE               │
├─────────────────────────────────────────────────────────┤
│ NeuralSwipeTypingEngine                                 │
│  ├─ OnnxSwipePredictor                                  │
│  │   ├─ SwipeTrajectoryProcessor                        │
│  │   │   ├─ Feature extraction: [x,y,vx,vy,ax,ay]      │
│  │   │   ├─ Sequence normalization to MAX_LENGTH       │
│  │   │   └─ Nearest key tokenization                    │
│  │   ├─ ONNX Runtime Android                           │
│  │   │   ├─ Transformer Encoder (trajectory → memory)  │
│  │   │   └─ Transformer Decoder (memory → words)       │
│  │   └─ Beam Search Decoder (width=8, maxlen=35)       │
│  └─ Fallback to Legacy System (if ONNX fails)          │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 COMPONENT MAPPING

| **Current Component** | **ONNX Replacement** | **Status** |
|----------------------|----------------------|------------|
| `KeyboardSwipeRecognizer.java` | `OnnxSwipePredictor.java` | 🔄 Replace |
| `DTWPredictor.java` | `SwipeTrajectoryProcessor.java` | 🔄 Replace |
| `GaussianKeyModel.java` | Neural encoder features | 🔄 Replace |
| `BigramModel.java` | Transformer language model | 🔄 Replace |
| `SwipeTypingEngine.java` | `NeuralSwipeTypingEngine.java` | 🔄 Replace |
| `SwipeCalibrationActivity.java` | **PRESERVE UI** - swap backend | ✅ Preserve |
| `SwipeWeightConfig.java` | Neural hyperparameters | 🔄 Adapt |
| `SwipeMLData/DataStore.java` | **PRESERVE** for training | ✅ Preserve |
| All Settings UI | **PRESERVE** - adapt parameters | ✅ Preserve |

---

## 🔧 IMPLEMENTATION PHASES

### **PHASE 1: FOUNDATION** (Week 1-2)
**Infrastructure Setup:**
1. ✅ Add ONNX Runtime Android dependency to `build.gradle`
2. ✅ Create base neural prediction classes:
   - `OnnxSwipePredictor.java`
   - `SwipeTrajectoryProcessor.java` 
   - `NeuralSwipeTypingEngine.java`
3. ✅ Implement tokenizer and model loading infrastructure
4. ✅ Create feature extraction pipeline matching web demo

**Deliverable**: Basic ONNX integration without replacing existing system

### **PHASE 2: NEURAL PREDICTOR IMPLEMENTATION** (Week 3-4)
**Core Prediction Engine:**
1. ✅ Port web demo's transformer encoder/decoder logic to Java
2. ✅ Implement beam search decoding algorithm (width=8, maxlen=35)
3. ✅ Add trajectory normalization and feature processing
4. ✅ Create prediction result mapping to existing `WordPredictor.PredictionResult` interface

**Key Technical Details**:
```java
// Feature extraction matching web demo
trajectoryData[baseIdx + 0] = point.x;                    // Position X
trajectoryData[baseIdx + 1] = point.y;                    // Position Y 
trajectoryData[baseIdx + 2] = point.x - prevPoint.x;      // Velocity X
trajectoryData[baseIdx + 3] = point.y - prevPoint.y;      // Velocity Y
trajectoryData[baseIdx + 4] = vx - prevVx;                // Acceleration X
trajectoryData[baseIdx + 5] = vy - prevVy;                // Acceleration Y
```

**Deliverable**: Functional neural predictor with same interface as legacy system

### **PHASE 3: CALIBRATION INTEGRATION** (Week 5)
**Seamless UI Preservation:**
1. ✅ Modify `SwipeCalibrationActivity.java` to use new neural predictor
2. ✅ Preserve ALL existing UI elements:
   - Browse/navigate recorded swipes
   - Delete individual swipes functionality  
   - Train button (adapt to neural training)
   - Export to clipboard
3. ✅ Ensure data collection continues to work for future model improvements
4. ✅ Maintain all existing calibration workflows

**CRITICAL**: Zero UI changes - users should not notice any difference in calibration page

**Deliverable**: Calibration activity fully functional with neural backend

### **PHASE 4: SETTINGS MIGRATION** (Week 6) 
**Replace Legacy Weights with Neural Controls:**

**Remove from Settings:**
- DTW algorithm weights (shape, location, velocity)
- Gaussian model parameters (sigma factors, min probability)
- N-gram smoothing parameters
- Manual weight sliders and normalization

**Add to Settings:**
- Neural prediction confidence threshold (0.1-0.9)
- Beam search width (1-16)
- Maximum sequence length (20-50)
- Model selection (if multiple models available)
- Enable/disable neural prediction toggle

**Preserve:**
- All existing preference infrastructure
- Settings UI layouts and navigation
- User data and calibration history

**Deliverable**: Clean settings interface focused on neural parameters

### **PHASE 5: INTEGRATION & TESTING** (Week 7-8)
**Production Integration:**
1. ✅ Modify `SwipeTypingEngine.predict()` to dispatch between neural/legacy:
```java
public WordPredictor.PredictionResult predict(SwipeInput input) {
    if (config.useNeuralPrediction && isOnnxModelLoaded) {
        return neuralPredictor.predict(input);
    } else {
        return legacyPredictor.predict(input); // automatic fallback
    }
}
```
2. ✅ Add configuration flag for neural prediction enable/disable
3. ✅ Implement graceful fallback if ONNX model loading fails
4. ✅ Comprehensive testing with existing calibration data

**Deliverable**: Hybrid system with seamless neural/legacy switching

### **PHASE 6: DEPLOYMENT & OPTIMIZATION** (Week 9-10)
**Production Readiness:**
1. ✅ Model size optimization and quantization (INT8)
2. ✅ Memory usage profiling and optimization (<20MB target)
3. ✅ Performance benchmarking vs legacy system (<50ms target)
4. ✅ A/B testing framework for gradual rollout

**Performance Targets**:
- Inference time: <50ms (vs current DTW ~100ms)
- Model loading: <500ms
- Memory overhead: <20MB
- Accuracy improvement: >20% vs current system

**Deliverable**: Production-ready neural swipe system

---

## ⚠️ CRITICAL SUCCESS FACTORS

### **1. ZERO BREAKING CHANGES**
- All existing UI/UX preserved exactly
- Same keyboard layouts and visual appearance
- Identical calibration workflow and data export
- No changes to user interaction patterns

### **2. BACKWARD COMPATIBILITY** 
- Legacy system available as automatic fallback
- Existing calibration data fully compatible
- Settings migration preserves user preferences
- Graceful degradation if neural models unavailable

### **3. PERFORMANCE REQUIREMENTS**
- Neural predictions faster than current DTW system
- Memory efficient model loading and caching
- Async processing prevents UI blocking
- Battery usage equivalent or better

### **4. ACCURACY IMPROVEMENTS**
- Measurable improvement over current Bayesian approach
- Better handling of complex swipe patterns
- Improved recognition of longer words
- Enhanced personalization capabilities

---

## 🛡️ RISK MITIGATION

| **Risk** | **Mitigation Strategy** |
|----------|------------------------|
| Model loading failures | Automatic fallback to legacy predictor |
| Memory constraints | Model pruning, quantization, lazy loading |
| Performance regression | Async processing, timeout fallbacks |
| User disruption | Gradual rollout with opt-out capability |
| Calibration data loss | Backward-compatible data formats |
| Settings corruption | Migration scripts with validation |

---

## 🔗 KEY INTEGRATION POINTS

### **Main Prediction Interface** (MUST PRESERVE)
```java
public WordPredictor.PredictionResult predict(SwipeInput input)
```

### **Configuration Integration** (MUST PRESERVE)
```java
public void setConfig(Config config)
public void setKeyboardDimensions(float width, float height)
public void setRealKeyPositions(Map<Character, PointF> realPositions)
```

### **Calibration Integration** (UI PRESERVED, BACKEND SWAPPED)
- Same UI elements and workflows
- Neural predictor called instead of legacy
- Data collection format maintained
- Export functionality preserved

---

## 📊 EXPECTED OUTCOMES

### **Immediate Benefits**:
- 🚀 **50%+ faster predictions** (neural vs DTW)
- 🎯 **20%+ accuracy improvement** (transformer vs Bayesian)
- 🧠 **Better complex pattern recognition** (multi-character sequences)
- ⚡ **Reduced battery usage** (efficient ONNX inference)

### **Long-term Benefits**:
- 🔄 **Continuous model improvements** (retrain with user data)
- 🌐 **Multi-language support** (unified transformer architecture)
- 📱 **Cross-device consistency** (same models across platforms)
- 🛠️ **Advanced personalization** (neural adaptation)

---

## 📚 TECHNICAL REFERENCES

### **Web Demo Implementation**: `/web_demo/swipe-onnx.html`
- Working transformer encoder/decoder with beam search
- Feature extraction: trajectory + velocity + acceleration
- ONNX Runtime Web integration example
- Tokenizer and sequence processing reference

### **Current Android Components**: `/srcs/juloo.keyboard2/`
- `SwipeTypingEngine.java` - Main integration point
- `SwipeCalibrationActivity.java` - UI preservation target
- `SwipeMLData*.java` - Data collection infrastructure
- `Config.java` - Settings integration

---