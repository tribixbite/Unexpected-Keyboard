# Swipe Typing ML Training Pipeline

This directory contains the machine learning training pipeline for the Unexpected Keyboard swipe typing feature. The pipeline includes data preprocessing, model training, and comprehensive evaluation tools.

## Overview

The swipe typing ML system uses advanced neural network architectures to predict words from swipe gesture traces. The system supports:

- Advanced preprocessing with quality analysis and feature engineering
- Dual-branch GRU architecture with multi-head attention (advanced model)
- Legacy dual-branch model for compatibility (basic model)
- Data augmentation for improved robustness
- Comprehensive evaluation and validation metrics
- TensorFlow Lite model export for mobile deployment

## Quick Start

### 1. Export Training Data

First, export swipe data from your Android device:

```bash
# Make the export script executable
chmod +x export_training_data.sh

# Export data from connected device
./export_training_data.sh
```

This will create a timestamped directory with your training data in NDJSON format.

### 2. Setup Environment

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Train Model

#### Option A: Advanced Model (Recommended)
```bash
# Basic training with default parameters
python train_advanced_model.py --data data/20240101_120000/training_data.ndjson

# Advanced training with custom parameters
python train_advanced_model.py \
    --data data/20240101_120000/training_data.ndjson \
    --batch-size 64 \
    --epochs 50 \
    --attention-heads 8 \
    --hidden-units 256 \
    --dropout 0.3 \
    --output-dir models/advanced_v1
```

#### Option B: Legacy Model
```bash
# Train with sample data
python train_swipe_model.py --data sample_data.ndjson --epochs 50

# Train with real data
python train_swipe_model.py --data data/*/training_data.ndjson --dict dictionary.txt
```

### 4. Evaluate Model

```bash
# Comprehensive evaluation
python evaluate_model.py \
    --model models/advanced_v1/swipe_model.tflite \
    --test-data data/20240101_120000/test_data.ndjson \
    --vocabulary models/advanced_v1/vocabulary.json \
    --output-dir evaluation_results
```

### 5. Deploy Model

The training script produces:
- `models/swipe_model.tflite` - Quantized TFLite model for Android
- `models/vocabulary.json` - Word-to-index mappings
- `models/training_history.png` - Training metrics visualization
- `evaluation_results/` - Comprehensive performance analysis

## Files Description

### Core Scripts
- `train_advanced_model.py` - Advanced training with attention mechanisms (recommended)
- `train_swipe_model.py` - Legacy training script with dual-branch GRU architecture
- `preprocess_data.py` - Advanced data preprocessing and quality analysis
- `evaluate_model.py` - Comprehensive model evaluation and metrics
- `export_training_data.sh` - Export collected data from Android device
- `generate_sample_data.py` - Generate synthetic data for testing

### Model Architecture

#### Advanced Model (train_advanced_model.py)
The advanced model uses a dual-branch architecture with attention mechanisms:

**Branch 1: Spatial Features**
```
Trace Points (x,y) → Masking → GRU(hidden_units) → BatchNorm
```

**Branch 2: Velocity Features**  
```
Velocity (vx,vy) → Masking → GRU(hidden_units) → BatchNorm
```

**Attention and Fusion**
```
Spatial + Velocity → Multi-Head Attention(heads=4) → Feature Fusion → Dense Layers → Softmax
```

#### Legacy Model (train_swipe_model.py)
The legacy model uses the original dual-branch architecture:

```
Input A (Trace Path) → Masking → GRU(128) → 
                                              → Concat → Dense(256) → Dropout → Softmax
Input B (Key Sequence) → Embedding(32) → GlobalAvgPool →
```

#### Key Features
- **Data Augmentation**: Spatial jittering, temporal variations
- **Quality Analysis**: Trace validation and filtering
- **Advanced Preprocessing**: Interpolation, normalization, feature engineering
- **Comprehensive Evaluation**: Accuracy, top-k, confusion matrix, performance by word length

### Data Format

Training data is in NDJSON format with normalized coordinates:

```json
{
  "trace_id": "uuid",
  "target_word": "hello",
  "metadata": {
    "timestamp_utc": 1677610000000,
    "screen_width_px": 1080,
    "screen_height_px": 2400,
    "keyboard_height_px": 800,
    "collection_source": "calibration"
  },
  "trace_points": [
    {"x": 0.23, "y": 0.85, "t_delta_ms": 0},
    {"x": 0.25, "y": 0.84, "t_delta_ms": 16}
  ],
  "registered_keys": ["h", "e", "l", "l", "o"]
}
```

## Training Parameters

### Advanced Model Parameters
- `--data`: Path to training data (NDJSON format)
- `--test-split`: Test data split ratio (default: 0.2)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 30)
- `--hidden-units`: GRU hidden units (default: 128)
- `--attention-heads`: Multi-head attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.2)
- `--max-trace-length`: Maximum trace points (default: 50)
- `--learning-rate`: Adam learning rate (default: 0.001)
- `--augmentation-prob`: Data augmentation probability (default: 0.3)
- `--min-word-freq`: Minimum word frequency for vocabulary (default: 2)
- `--output-dir`: Output directory for models and artifacts

### Legacy Model Parameters (Default hyperparameters)
- **GRU Units**: 128
- **Embedding Dim**: 32
- **Dense Units**: 256
- **Dropout Rate**: 0.4
- **Batch Size**: 64
- **Learning Rate**: 1e-3

## Evaluation Metrics

The evaluation script provides comprehensive metrics:

### Accuracy Metrics
- Overall accuracy
- Precision, recall, F1-score
- Average confidence scores
- Prediction latency

### Top-K Accuracy
- Top-1, Top-3, Top-5, Top-10 accuracy
- Useful for keyboard suggestion systems

### Confusion Matrix Analysis
- Visual confusion matrix for top words
- Per-class precision and recall
- Misclassification patterns

### Performance Analysis
- Accuracy by word length
- Sample distribution statistics
- Performance plots and visualizations

## Performance Targets

- **Top-1 Accuracy**: >70%
- **Top-3 Accuracy**: >85%
- **Model Size**: <10MB (quantized)
- **Inference Time**: <50ms

## Workflow

### Data Collection → Training → Deployment

1. **Collect Data**
   - Calibration activity in app
   - User selections during normal use
   - Export via settings or ADB

2. **Train Model**
   - Load and preprocess data
   - Train dual-branch GRU model
   - Validate on test set
   - Convert to TFLite with INT8 quantization

3. **Deploy to App**
   - Copy `swipe_model.tflite` to `assets/`
   - Update vocabulary mapping
   - Test inference performance

## Advanced Usage

### Custom Training Parameters
```bash
python train_swipe_model.py \
  --data training_data.ndjson \
  --dict dictionary.txt \
  --epochs 100 \
  --batch-size 32 \
  --no-quantize  # Skip INT8 quantization
```

### Data Augmentation

The training script includes:
- Spatial jitter (noise on x,y coordinates)
- Temporal jitter (noise on time deltas)
- Configurable in `DataPreprocessor.augment_trace()`

### Model Evaluation

After training:
- Check `models/training_history.png` for loss/accuracy curves
- Review test set metrics in console output
- Test TFLite model accuracy

## Troubleshooting

### Low Accuracy
- Collect more training data (minimum 100 samples per word)
- Increase epochs or adjust learning rate
- Check data quality and distribution

### Model Too Large
- Ensure INT8 quantization is enabled
- Reduce GRU_UNITS or DENSE_UNITS
- Consider pruning techniques

### Slow Inference
- Use quantized model
- Reduce MAX_SEQ_LENGTH if possible
- Profile with TFLite benchmark tool

## Next Steps

1. Integrate TFLite model into Android app
2. Implement inference pipeline in Java
3. Add personalization layer
4. Set up continuous learning pipeline

## References

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Gemini's Architecture Recommendations](../memory/swipe.md)
- [Project Roadmap](../memory/pm.md)