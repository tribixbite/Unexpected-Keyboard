# Swipe Typing ML Training Pipeline

This directory contains the machine learning training pipeline for the Unexpected Keyboard swipe typing feature.

## Quick Start

### 1. Setup Environment

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Get Training Data

#### Option A: Export from Device
```bash
# Connect Android device with USB debugging enabled
./export_training_data.sh
```

#### Option B: Generate Sample Data (for testing)
```bash
python generate_sample_data.py --samples 1000 --dict
```

### 3. Train Model

```bash
# Train with sample data
python train_swipe_model.py --data sample_data.ndjson --epochs 50

# Train with real data
python train_swipe_model.py --data data/*/training_data.ndjson --dict dictionary.txt
```

### 4. Deploy Model

The training script produces:
- `models/swipe_model.tflite` - Quantized TFLite model for Android
- `models/word_vocab.txt` - Vocabulary mapping
- `models/training_history.png` - Training metrics visualization

## Files Description

### Core Scripts
- `train_swipe_model.py` - Main training script with dual-branch GRU architecture
- `export_training_data.sh` - Export collected data from Android device
- `generate_sample_data.py` - Generate synthetic data for testing

### Model Architecture

The model uses a dual-branch architecture as recommended by Gemini:

```
Input A (Trace Path) → Masking → GRU(128) → 
                                              → Concat → Dense(256) → Dropout → Softmax
Input B (Key Sequence) → Embedding(32) → GlobalAvgPool →
```

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

Default hyperparameters (tunable):
- **GRU Units**: 128
- **Embedding Dim**: 32
- **Dense Units**: 256
- **Dropout Rate**: 0.4
- **Batch Size**: 64
- **Learning Rate**: 1e-3

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