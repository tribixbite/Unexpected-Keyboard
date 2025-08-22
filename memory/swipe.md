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

### Next Session TODOs
1. Create Python training script
2. Implement TensorFlow model
3. Test with collected data
4. Deploy TFLite model
5. Integrate inference engine

## Recent Commit Review (2025-01-22)

### Issues Found
1. **DTWPredictor not integrated**: While DTWPredictor class exists, all predictions currently route through WordPredictor only
2. **Unused confidence weights**: Config has 4 swipe confidence weights that aren't applied in scoring
3. **Missing debug visualization**: No way to see prediction scores for testing weight effectiveness
4. **Calibration data not used**: SwipeCalibrationActivity collects data but doesn't affect predictions yet

### Positive Findings
1. **First/last letter matching implemented**: WordPredictor already has priority system for endpoint matches
2. **Two-pass scoring system**: Separates priority (first+last) matches from other candidates
3. **Proper swipe duration tracking**: Fixed to use actual swipe time instead of arbitrary delay

### Algorithm Improvements Needed
1. **Configurable endpoint weights**: 
   - Separate weights for first letter match
   - Separate weights for last letter match
   - Bonus weight when both match
   - Option to restrict to ONLY endpoint matches
2. **Debug score display**: Show confidence values below predictions
3. **Weight integration**: Apply all Config weights to scoring algorithm
4. **DTW integration**: Use DTWPredictor for coordinate-based matching