# ML Training Guide - Neural Swipe Typing

**Audience**: Developers and ML Engineers
**Purpose**: Train custom ONNX models for swipe prediction
**Difficulty**: Intermediate to Advanced

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Collection](#data-collection)
4. [Data Export](#data-export)
5. [Training Pipeline](#training-pipeline)
6. [Model Architecture](#model-architecture)
7. [ONNX Conversion](#onnx-conversion)
8. [Model Deployment](#model-deployment)
9. [Evaluation](#evaluation)
10. [Advanced Topics](#advanced-topics)

---

## Overview

### What You'll Build

A dual-branch encoder-decoder neural network model:
- **Encoder**: Processes touch trajectory coordinates
- **Decoder**: Generates word predictions with attention mechanism

**Model Format**: ONNX (Open Neural Network Exchange)
**Target Platform**: Android with ONNX Runtime
**Training Framework**: TensorFlow/Keras or PyTorch

### Performance Targets

- **Accuracy**: Top-1 >70%, Top-3 >85%, Top-5 >90%
- **Latency**: <50ms inference on mobile devices
- **Model Size**: Encoder <6MB, Decoder <8MB
- **Memory**: <20MB runtime usage

---

## Prerequisites

### Development Environment

```bash
# Python 3.8+
python --version

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Required packages
pip install tensorflow==2.13.0
pip install tf2onnx
pip install onnx
pip install onnxruntime
pip install numpy pandas scikit-learn
pip install matplotlib seaborn  # For visualization
```

### Training Hardware

**Minimum**:
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free

**Recommended**:
- GPU: NVIDIA with CUDA support
- RAM: 16GB+
- Storage: 50GB+ for large datasets

### Knowledge Requirements

- Python programming
- Neural networks basics (RNNs, attention)
- TensorFlow or PyTorch
- Data preprocessing
- Model evaluation

---

## Data Collection

### Enabling Data Collection

1. **Grant Consent**:
   - Settings → Neural ML & Swipe Typing → Privacy & Data
   - Tap "Data Collection Consent" → "Grant Consent"

2. **Enable Collection Types**:
   - ✅ Collect Swipe Data (essential)
   - ✅ Collect Performance Data (optional, for analysis)

3. **Configure Settings**:
   - Retention Period: 365 days (or never delete)
   - Auto-Delete: OFF (prevent premature deletion)
   - Local-Only Training: ON (keep data private)

### Data Collection Sources

#### Calibration Data (High Quality)

Best for initial training:

1. Launch **Swipe Calibration** activity
2. Swipe provided word prompts accurately
3. Collect 100-1000 samples
4. Covers common words with labeled ground truth

#### Usage Data (Real World)

Best for production models:

1. Use keyboard normally
2. Swipe naturally (no special effort)
3. Collect for weeks/months
4. Captures actual usage patterns

### Data Quality

**Good Quality**:
- Complete swipe gestures
- Correct word selections
- Natural timing and velocity
- Varied word lengths

**Poor Quality**:
- Accidental swipes
- Incorrect selections
- Extremely slow/fast gestures
- Single-letter swipes

---

## Data Export

### Export from App

1. **Settings** → **Neural ML & Swipe Typing** → **ML Data Management**
2. Tap **"Export ML Data"**
3. Choose export format:
   - **JSON**: Human-readable, larger files
   - **NDJSON**: Newline-delimited JSON, easier to process

4. **Share** exported file via:
   - Save to Files app
   - Send via email/messaging
   - Transfer to computer

### Data Format

#### JSON Structure

```json
{
  "timestamp": 1700000000000,
  "source": "usage",
  "word": "hello",
  "screen_width": 1080,
  "screen_height": 2340,
  "trace": {
    "points": [
      {"x": 0.123, "y": 0.456, "t_delta": 0},
      {"x": 0.234, "y": 0.567, "t_delta": 15},
      ...
    ],
    "keys": ["h", "e", "l", "l", "o"],
    "length_px": 234.5,
    "duration_ms": 450
  }
}
```

#### Field Descriptions

- **timestamp**: Unix timestamp (milliseconds)
- **source**: "calibration" or "usage"
- **word**: Ground truth word (what was typed)
- **screen_width/height**: Device resolution for normalization
- **trace.points**: Touch coordinates:
  - `x, y`: Normalized [0,1] coordinates
  - `t_delta`: Time delta from previous point (ms)
- **trace.keys**: Detected key sequence
- **trace.length_px**: Total gesture path length
- **trace.duration_ms**: Gesture duration

### Data Validation

```python
import json

def validate_sample(sample):
    """Validate a single training sample."""
    # Required fields
    assert 'word' in sample
    assert 'trace' in sample
    assert 'points' in sample['trace']

    # Data quality
    assert len(sample['word']) >= 2  # Minimum word length
    assert len(sample['trace']['points']) >= 3  # Minimum trajectory points
    assert sample['trace']['duration_ms'] > 0  # Valid duration

    # Coordinate normalization
    for pt in sample['trace']['points']:
        assert 0 <= pt['x'] <= 1
        assert 0 <= pt['y'] <= 1
        assert pt['t_delta'] >= 0

    return True

# Load and validate
with open('exported_data.json') as f:
    data = json.load(f)

valid_samples = [s for s in data if validate_sample(s)]
print(f"Valid samples: {len(valid_samples)}/{len(data)}")
```

---

## Training Pipeline

### Data Preprocessing

```python
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_dataset(samples, vocab_size=10000):
    """Convert raw samples to training tensors."""

    # Build vocabulary from most common words
    word_counts = {}
    for sample in samples:
        word = sample['word'].lower()
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency, keep top K
    vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = [word for word, _ in vocab[:vocab_size]]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Convert samples to tensors
    traces = []
    keys = []
    labels = []

    for sample in samples:
        word = sample['word'].lower()
        if word not in word_to_idx:
            continue  # Skip OOV words

        # Trace coordinates (x, y, t_delta)
        trace = np.array([[pt['x'], pt['y'], pt['t_delta']]
                          for pt in sample['trace']['points']])
        traces.append(trace)

        # Key sequence (convert letters to indices)
        key_seq = [ord(k) - ord('a') for k in sample['trace']['keys']
                   if 'a' <= k <= 'z']
        keys.append(np.array(key_seq))

        # Label (word index)
        labels.append(word_to_idx[word])

    return {
        'traces': traces,
        'keys': keys,
        'labels': np.array(labels),
        'vocab': vocab,
        'word_to_idx': word_to_idx
    }

# Load and preprocess
with open('exported_data.json') as f:
    samples = json.load(f)

data = preprocess_dataset(samples, vocab_size=10000)

# Split train/validation/test (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    list(zip(data['traces'], data['keys'])),
    data['labels'],
    test_size=0.3,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42
)
```

### Data Augmentation

```python
def augment_trajectory(trace, noise_std=0.01, time_stretch=0.1):
    """Apply data augmentation to trajectory."""
    trace = trace.copy()

    # Add Gaussian noise to coordinates
    trace[:, :2] += np.random.normal(0, noise_std, trace[:, :2].shape)

    # Clip to [0, 1]
    trace[:, :2] = np.clip(trace[:, :2], 0, 1)

    # Time stretching/compression
    time_factor = np.random.uniform(1 - time_stretch, 1 + time_stretch)
    trace[:, 2] *= time_factor

    return trace

def augment_dataset(traces, keys, labels, augment_factor=2):
    """Augment dataset by factor."""
    aug_traces = []
    aug_keys = []
    aug_labels = []

    for trace, key, label in zip(traces, keys, labels):
        # Original
        aug_traces.append(trace)
        aug_keys.append(key)
        aug_labels.append(label)

        # Augmented copies
        for _ in range(augment_factor - 1):
            aug_traces.append(augment_trajectory(trace))
            aug_keys.append(key)  # Key sequence unchanged
            aug_labels.append(label)

    return aug_traces, aug_keys, aug_labels
```

---

## Model Architecture

### Dual-Branch Encoder-Decoder

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(vocab_size=10000, max_trace_len=200, max_key_len=50):
    """Build dual-branch encoder-decoder model."""

    # === ENCODER ===

    # Branch A: Trace coordinates
    trace_input = keras.Input(shape=(None, 3), name='trace_input')
    trace_masked = layers.Masking(mask_value=0.0)(trace_input)
    trace_lstm = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False)
    )(trace_masked)

    # Branch B: Key sequence
    keys_input = keras.Input(shape=(None,), name='keys_input')
    keys_embed = layers.Embedding(
        input_dim=26,  # a-z
        output_dim=16,
        mask_zero=True
    )(keys_input)
    keys_lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False)
    )(keys_embed)

    # Merge branches
    encoder_output = layers.Concatenate()([trace_lstm, keys_lstm])
    encoder_output = layers.Dense(256, activation='relu')(encoder_output)
    encoder_output = layers.Dropout(0.3)(encoder_output)

    # === DECODER ===

    # Previous tokens input (for autoregressive generation)
    prev_tokens = keras.Input(shape=(None,), name='prev_tokens')
    token_embed = layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        mask_zero=True
    )(prev_tokens)

    # Decoder LSTM with encoder context
    decoder_lstm = layers.LSTM(256, return_sequences=True)(
        token_embed,
        initial_state=[encoder_output, encoder_output]
    )

    # Attention mechanism
    attention = layers.Attention()([decoder_lstm, decoder_lstm])
    decoder_output = layers.Add()([decoder_lstm, attention])

    # Output layer
    output = layers.TimeDistributed(
        layers.Dense(vocab_size, activation='softmax')
    )(decoder_output)

    # Build model
    model = keras.Model(
        inputs=[trace_input, keys_input, prev_tokens],
        outputs=output,
        name='swipe_predictor'
    )

    return model

# Create model
model = build_model(vocab_size=10000)
model.summary()
```

### Separate Encoder-Decoder for ONNX

For deployment, split into two models:

```python
def build_encoder(vocab_size=10000):
    """Encoder model for ONNX export."""
    trace_input = keras.Input(shape=(None, 3), name='trace')
    keys_input = keras.Input(shape=(None,), name='keys')

    # Trace branch
    trace_masked = layers.Masking(mask_value=0.0)(trace_input)
    trace_lstm = layers.Bidirectional(
        layers.LSTM(128)
    )(trace_masked)

    # Keys branch
    keys_embed = layers.Embedding(26, 16, mask_zero=True)(keys_input)
    keys_lstm = layers.Bidirectional(
        layers.LSTM(64)
    )(keys_embed)

    # Merge
    merged = layers.Concatenate()([trace_lstm, keys_lstm])
    output = layers.Dense(256, activation='relu', name='encoded')(merged)

    return keras.Model([trace_input, keys_input], output, name='encoder')

def build_decoder(vocab_size=10000, encoding_dim=256):
    """Decoder model for ONNX export."""
    encoding = keras.Input(shape=(encoding_dim,), name='encoding')
    prev_token = keras.Input(shape=(1,), name='prev_token', dtype='int32')
    hidden_state = keras.Input(shape=(256,), name='hidden_state')
    cell_state = keras.Input(shape=(256,), name='cell_state')

    # Token embedding
    token_embed = layers.Embedding(vocab_size, 128)(prev_token)
    token_embed = layers.Flatten()(token_embed)

    # Combine with encoding
    combined = layers.Concatenate()([token_embed, encoding])

    # LSTM step
    lstm_cell = layers.LSTMCell(256)
    output, [new_hidden, new_cell] = lstm_cell(
        combined,
        states=[hidden_state, cell_state]
    )

    # Output probabilities
    probs = layers.Dense(vocab_size, activation='softmax', name='probabilities')(output)

    return keras.Model(
        [encoding, prev_token, hidden_state, cell_state],
        [probs, new_hidden, new_cell],
        name='decoder'
    )
```

---

## ONNX Conversion

### TensorFlow to ONNX

```python
import tf2onnx

# Convert encoder
encoder_model = build_encoder(vocab_size=10000)
encoder_model.load_weights('encoder_weights.h5')

spec = (
    tf.TensorSpec((None, None, 3), tf.float32, name="trace"),
    tf.TensorSpec((None, None), tf.int32, name="keys")
)

encoder_onnx, _ = tf2onnx.convert.from_keras(
    encoder_model,
    input_signature=spec,
    opset=13,
    output_path="encoder.onnx"
)

# Convert decoder
decoder_model = build_decoder(vocab_size=10000)
decoder_model.load_weights('decoder_weights.h5')

spec = (
    tf.TensorSpec((None, 256), tf.float32, name="encoding"),
    tf.TensorSpec((None, 1), tf.int32, name="prev_token"),
    tf.TensorSpec((None, 256), tf.float32, name="hidden_state"),
    tf.TensorSpec((None, 256), tf.float32, name="cell_state")
)

decoder_onnx, _ = tf2onnx.convert.from_keras(
    decoder_model,
    input_signature=spec,
    opset=13,
    output_path="decoder.onnx"
)
```

### Verify ONNX Models

```python
import onnx
import onnxruntime as ort

# Load and check encoder
encoder = onnx.load("encoder.onnx")
onnx.checker.check_model(encoder)
print("Encoder ONNX valid!")

# Test inference
session = ort.InferenceSession("encoder.onnx")
print("Encoder inputs:", [i.name for i in session.get_inputs()])
print("Encoder outputs:", [o.name for o in session.get_outputs()])

# Same for decoder
decoder = onnx.load("decoder.onnx")
onnx.checker.check_model(decoder)
print("Decoder ONNX valid!")
```

### Optimize ONNX

```python
from onnxruntime.transformers import optimizer

# Optimize encoder
optimized_encoder = optimizer.optimize_model(
    "encoder.onnx",
    model_type='bert',  # Or appropriate type
    num_heads=0,
    hidden_size=256
)
optimized_encoder.save_model_to_file("encoder_optimized.onnx")

# Check size reduction
import os
original_size = os.path.getsize("encoder.onnx") / (1024 * 1024)
optimized_size = os.path.getsize("encoder_optimized.onnx") / (1024 * 1024)
print(f"Encoder: {original_size:.2f}MB → {optimized_size:.2f}MB")
```

---

## Model Deployment

### Loading Models in App

1. **Copy ONNX files to device**:
   ```bash
   adb push encoder.onnx /sdcard/Download/
   adb push decoder.onnx /sdcard/Download/
   ```

2. **Load via Settings**:
   - Settings → Neural ML & Swipe Typing → Neural Model Settings
   - Tap "Select Custom Encoder Model"
   - Choose encoder.onnx from Downloads
   - Tap "Select Custom Decoder Model"
   - Choose decoder.onnx from Downloads

3. **Verify Loading**:
   - Check "Model Status" shows both models loaded
   - Interface should auto-detect
   - Try swipe typing

### Model Interface Requirements

The app auto-detects two interface types:

**Built-in V2 Interface** (recommended):
```
Inputs:
- encoding: [batch_size, encoding_dim]
- prev_token: [batch_size, 1]
- hidden_state: [batch_size, hidden_dim]
- cell_state: [batch_size, hidden_dim]
- target_mask: [batch_size, seq_len, seq_len]  # Combined mask

Outputs:
- probabilities: [batch_size, vocab_size]
- new_hidden: [batch_size, hidden_dim]
- new_cell: [batch_size, hidden_dim]
```

**Custom Interface** (fallback):
```
Inputs:
- Same as above, but:
- target_padding_mask: [batch_size, seq_len]
- target_causal_mask: [batch_size, seq_len, seq_len]
(Two separate masks instead of combined)
```

The app checks `getInputNames()` and adapts automatically.

---

## Evaluation

### Metrics

```python
from sklearn.metrics import accuracy_score, top_k_accuracy_score

def evaluate_model(model, X_test, y_test, vocab):
    """Evaluate model performance."""
    traces, keys = zip(*X_test)

    # Predict
    predictions = model.predict([traces, keys])
    y_pred = np.argmax(predictions, axis=1)

    # Top-1 accuracy
    top1 = accuracy_score(y_test, y_pred)

    # Top-3 accuracy
    top3 = top_k_accuracy_score(y_test, predictions, k=3)

    # Top-5 accuracy
    top5 = top_k_accuracy_score(y_test, predictions, k=5)

    print(f"Top-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-3 Accuracy: {top3*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")

    return {'top1': top1, 'top3': top3, 'top5': top5}

# Run evaluation
metrics = evaluate_model(model, X_test, y_test, vocab)
```

### Confusion Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_errors(y_true, y_pred, vocab, top_n=20):
    """Analyze most common prediction errors."""
    errors = {}

    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            true_word = vocab[true_idx]
            pred_word = vocab[pred_idx]
            key = (true_word, pred_word)
            errors[key] = errors.get(key, 0) + 1

    # Sort by frequency
    top_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print("Most Common Prediction Errors:")
    for (true_word, pred_word), count in top_errors:
        print(f"  {true_word} → {pred_word}: {count} times")

    return top_errors
```

---

## Advanced Topics

### Class Weighting

Handle imbalanced data:

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(enumerate(weights))

# Use in training
model.fit(
    [traces_train, keys_train],
    y_train,
    class_weight=class_weights,
    epochs=50,
    validation_data=([traces_val, keys_val], y_val)
)
```

### Learning Rate Scheduling

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    """Decay learning rate over epochs."""
    initial_lr = 0.001
    decay_factor = 0.5
    decay_epochs = 10

    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

model.fit(..., callbacks=[lr_scheduler])
```

### Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

model.fit(..., callbacks=[early_stop])
```

### Model Quantization

Reduce model size:

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (CPU)
quantize_dynamic(
    "encoder.onnx",
    "encoder_quantized.onnx",
    weight_type=QuantType.QUInt8
)

# Check size reduction
original = os.path.getsize("encoder.onnx") / (1024 * 1024)
quantized = os.path.getsize("encoder_quantized.onnx") / (1024 * 1024)
print(f"Encoder: {original:.2f}MB → {quantized:.2f}MB ({quantized/original*100:.1f}%)")
```

---

## Troubleshooting

### Issue: Low Accuracy

**Causes**:
- Insufficient training data
- Class imbalance
- Poor data quality
- Model underfitting

**Solutions**:
- Collect more data (especially calibration data)
- Apply class weighting
- Filter low-quality samples
- Increase model capacity
- Try data augmentation

### Issue: High Latency

**Causes**:
- Model too large
- Inefficient architecture
- Unoptimized ONNX

**Solutions**:
- Reduce hidden dimensions
- Use GRU instead of LSTM
- Apply ONNX optimization
- Quantize model
- Profile with ONNX Runtime

### Issue: ONNX Conversion Fails

**Causes**:
- Unsupported operations
- Dynamic shapes
- Custom layers

**Solutions**:
- Use tf2onnx latest version
- Simplify model architecture
- Set concrete input shapes
- Avoid custom Keras layers

---

## Resources

### Documentation

- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [TensorFlow to ONNX](https://github.com/onnx/tensorflow-onnx)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

### Example Repositories

- [Unexpected Keyboard ML Training](https://github.com/Julow/Unexpected-Keyboard/tree/main/ml_training)
- [ONNX Model Zoo](https://github.com/onnx/models)

### Community

- [GitHub Discussions](https://github.com/Julow/Unexpected-Keyboard/discussions)
- [Issues Tracker](https://github.com/Julow/Unexpected-Keyboard/issues)

---

*Last Updated: 2025-11-27*
*For users: See [NEURAL_SWIPE_GUIDE.md](NEURAL_SWIPE_GUIDE.md)*
*For privacy: See [PRIVACY_POLICY.md](PRIVACY_POLICY.md)*
