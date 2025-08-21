#!/usr/bin/env python3
"""
Swipe Typing Neural Network Training Script
For Unexpected Keyboard Android App
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, Dense, Concatenate, Embedding,
    GlobalAveragePooling1D, Masking, Dropout, StringLookup
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# --- Configuration & Hyperparameters ---
# Data Parameters
MAX_SEQ_LENGTH = 120  # 95th percentile of trace lengths
MAX_KEYS_LENGTH = 20  # Max length of registered_keys
MIN_TRACE_POINTS = 3  # Minimum points for valid swipe

# Model Parameters
GRU_UNITS = 128
EMBEDDING_DIM = 32
DENSE_UNITS = 256
DROPOUT_RATE = 0.4

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
LEARNING_RATE = 1e-3

# Output paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class SwipeDataLoader:
    """Handles loading and preprocessing of swipe data."""
    
    def __init__(self, file_path: str, dictionary_path: str = None):
        self.file_path = file_path
        self.dictionary_path = dictionary_path
        self.word_frequencies = self._load_dictionary() if dictionary_path else {}
        
    def _load_dictionary(self) -> Dict[str, int]:
        """Load word frequencies from dictionary file."""
        frequencies = {}
        if self.dictionary_path and os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, freq = parts
                        frequencies[word.lower()] = int(freq)
        return frequencies
    
    def load_data(self) -> Tuple[List, List, List, Dict]:
        """
        Loads data from an NDJSON file.
        Returns: traces, registered_keys, target_words, metadata
        """
        traces, registered_keys, target_words = [], [], []
        metadata = {'sources': {}, 'screen_sizes': []}
        
        print(f"Loading data from {self.file_path}...")
        with open(self.file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line)
                    
                    # Skip invalid records
                    if len(record['trace_points']) < MIN_TRACE_POINTS:
                        continue
                    
                    # Extract x, y, t_delta_ms features
                    trace_points = np.array([
                        [p['x'], p['y'], p['t_delta_ms']] 
                        for p in record['trace_points']
                    ], dtype=np.float32)
                    
                    traces.append(trace_points)
                    registered_keys.append(record['registered_keys'])
                    target_words.append(record['target_word'].lower())
                    
                    # Collect metadata
                    source = record['metadata']['collection_source']
                    metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
                    
                    screen_info = (
                        record['metadata']['screen_width_px'],
                        record['metadata']['screen_height_px']
                    )
                    if screen_info not in metadata['screen_sizes']:
                        metadata['screen_sizes'].append(screen_info)
                        
                except Exception as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(traces)} valid samples")
        print(f"Sources: {metadata['sources']}")
        print(f"Unique words: {len(set(target_words))}")
        print(f"Screen sizes: {len(metadata['screen_sizes'])} unique")
        
        return traces, registered_keys, target_words, metadata


class DataPreprocessor:
    """Handles data preprocessing and augmentation."""
    
    def __init__(self, max_seq_length: int, max_keys_length: int):
        self.max_seq_length = max_seq_length
        self.max_keys_length = max_keys_length
        self.word_vocab_layer = None
        self.key_map = None
        
    def preprocess_and_split(self, traces, keys, words):
        """Pads sequences, encodes labels, and performs stratified split."""
        
        # Analyze sequence lengths for optimal padding
        trace_lengths = [len(t) for t in traces]
        print(f"Trace length stats: min={min(trace_lengths)}, "
              f"max={max(trace_lengths)}, "
              f"mean={np.mean(trace_lengths):.1f}, "
              f"p95={np.percentile(trace_lengths, 95):.1f}")
        
        # Pad trace sequences (post-padding, pre-truncating as recommended)
        padded_traces = pad_sequences(
            traces, 
            maxlen=self.max_seq_length, 
            dtype='float32', 
            padding='post', 
            truncating='pre', 
            value=-1.0
        )
        
        # Create vocabulary for words
        self.word_vocab_layer = StringLookup(mask_token=None)
        self.word_vocab_layer.adapt(words)
        
        # Create vocabulary for keys
        all_keys = sorted(list(set(char for key_list in keys for char in key_list)))
        self.key_map = {char: i + 1 for i, char in enumerate(all_keys)}  # 0 for padding
        
        # Encode and pad keys
        encoded_keys = [[self.key_map.get(k, 0) for k in key_list] for key_list in keys]
        padded_keys = pad_sequences(
            encoded_keys, 
            maxlen=self.max_keys_length, 
            padding='post', 
            value=0
        )
        
        # Encode target words
        word_labels = self.word_vocab_layer(words).numpy()
        
        # Stratified split
        indices = np.arange(len(words))
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            indices, 
            test_size=self.TEST_SPLIT, 
            stratify=word_labels, 
            random_state=42
        )
        
        # Prepare test set
        X_test = {
            'trace_input': padded_traces[test_indices],
            'keys_input': padded_keys[test_indices]
        }
        y_test = word_labels[test_indices]
        
        # Second split: train vs val
        X_train_val_trace = padded_traces[train_val_indices]
        X_train_val_keys = padded_keys[train_val_indices]
        y_train_val = word_labels[train_val_indices]
        
        train_indices, val_indices = train_test_split(
            np.arange(len(y_train_val)), 
            test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), 
            stratify=y_train_val, 
            random_state=42
        )
        
        X_train = {
            'trace_input': X_train_val_trace[train_indices],
            'keys_input': X_train_val_keys[train_indices]
        }
        y_train = y_train_val[train_indices]
        
        X_val = {
            'trace_input': X_train_val_trace[val_indices],
            'keys_input': X_train_val_keys[val_indices]
        }
        y_val = y_train_val[val_indices]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def augment_trace(self, trace, noise_std=0.01, time_noise_std=2.0):
        """
        Apply data augmentation to a trace.
        - Spatial jitter: Add noise to x, y coordinates
        - Temporal jitter: Add noise to time deltas
        """
        augmented = trace.copy()
        
        # Spatial noise (only on valid points, not padding)
        valid_mask = trace[:, 0] != -1.0
        spatial_noise = np.random.normal(0, noise_std, (len(trace), 2))
        augmented[valid_mask, :2] += spatial_noise[valid_mask]
        
        # Clip to valid range [0, 1]
        augmented[valid_mask, :2] = np.clip(augmented[valid_mask, :2], 0, 1)
        
        # Temporal noise
        time_noise = np.random.normal(0, time_noise_std, len(trace))
        augmented[valid_mask, 2] = np.maximum(0, augmented[valid_mask, 2] + time_noise[valid_mask])
        
        return augmented


def build_dual_branch_model(max_seq_len, num_features, max_keys_len, key_vocab_size, num_classes):
    """
    Builds the dual-branch GRU model as recommended by Gemini.
    
    Architecture:
    - Branch 1: Trace processing (GRU-128)
    - Branch 2: Key sequence processing (Embedding-32 → GlobalAvgPool)
    - Merge → Dense(256) → Dropout → Softmax
    """
    
    # Branch 1: Trace processing (temporal)
    trace_input = Input(shape=(max_seq_len, num_features), name='trace_input')
    masked_trace = Masking(mask_value=-1.0)(trace_input)
    gru_layer = GRU(GRU_UNITS, return_sequences=False, name='trace_gru')(masked_trace)
    
    # Branch 2: Registered keys processing (categorical)
    keys_input = Input(shape=(max_keys_len,), name='keys_input')
    key_embedding = Embedding(
        input_dim=key_vocab_size + 1,  # +1 for padding token
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name='key_embedding'
    )(keys_input)
    keys_pooled = GlobalAveragePooling1D(name='keys_pooling')(key_embedding)
    
    # Merge branches
    concatenated = Concatenate(name='merge')([gru_layer, keys_pooled])
    
    # Classification head
    x = Dense(DENSE_UNITS, activation='relu', name='dense_1')(concatenated)
    x = Dropout(DROPOUT_RATE, name='dropout')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=[trace_input, keys_input], outputs=output)
    return model


def calculate_class_weights(y_train, word_frequencies=None):
    """
    Calculate class weights based on training data distribution.
    Optionally incorporate dictionary word frequencies.
    """
    # Basic class weight calculation
    classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    
    # Convert to dictionary
    class_weight_dict = dict(enumerate(class_weights))
    
    # TODO: Incorporate word_frequencies if available
    # This would require mapping word indices back to words
    
    return class_weight_dict


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def convert_to_tflite(model, X_train, save_path='swipe_model.tflite', quantize=True):
    """
    Convert Keras model to TFLite with optional quantization.
    """
    print("\nConverting to TFLite...")
    
    if quantize:
        print("Using INT8 quantization...")
        
        # Representative dataset generator for quantization calibration
        def representative_dataset_gen():
            for i in range(min(100, len(X_train['trace_input']))):
                yield [
                    np.expand_dims(X_train['trace_input'][i], axis=0).astype(np.float32),
                    np.expand_dims(X_train['keys_input'][i], axis=0).astype(np.float32)
                ]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        
        # Full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("Using dynamic range quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    # Report size reduction
    keras_size = os.path.getsize('best_swipe_model.h5') / 1024 / 1024
    tflite_size = os.path.getsize(save_path) / 1024 / 1024
    print(f"Model size: {keras_size:.2f} MB → {tflite_size:.2f} MB")
    print(f"Size reduction: {(1 - tflite_size/keras_size)*100:.1f}%")
    print(f"TFLite model saved to {save_path}")
    
    return tflite_model


def evaluate_tflite_model(tflite_path, X_test, y_test, sample_size=100):
    """
    Evaluate TFLite model accuracy on test set.
    """
    print(f"\nEvaluating TFLite model on {sample_size} samples...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    for i in range(min(sample_size, len(y_test))):
        # Prepare inputs
        trace_input = np.expand_dims(X_test['trace_input'][i], axis=0).astype(np.float32)
        keys_input = np.expand_dims(X_test['keys_input'][i], axis=0).astype(np.float32)
        
        # Set inputs
        interpreter.set_tensor(input_details[0]['index'], trace_input)
        interpreter.set_tensor(input_details[1]['index'], keys_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted = np.argmax(output)
        
        if predicted == y_test[i]:
            correct += 1
    
    accuracy = correct / min(sample_size, len(y_test))
    print(f"TFLite accuracy: {accuracy:.4f}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train swipe typing neural network')
    parser.add_argument('--data', type=str, required=True, help='Path to NDJSON data file')
    parser.add_argument('--dict', type=str, help='Path to dictionary file with word frequencies')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--no-quantize', action='store_true', help='Skip INT8 quantization')
    args = parser.parse_args()
    
    # Load data
    loader = SwipeDataLoader(args.data, args.dict)
    traces, keys, words, metadata = loader.load_data()
    
    if len(traces) < 100:
        print("Warning: Very few samples. Consider collecting more data.")
    
    # Preprocess data
    preprocessor = DataPreprocessor(MAX_SEQ_LENGTH, MAX_KEYS_LENGTH)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.preprocess_and_split(
        traces, keys, words
    )
    
    # Save vocabulary for inference
    vocab_path = MODEL_DIR / 'word_vocab.txt'
    with open(vocab_path, 'w') as f:
        for word in preprocessor.word_vocab_layer.get_vocabulary():
            f.write(f"{word}\n")
    print(f"Vocabulary saved to {vocab_path}")
    
    # Build model
    num_classes = preprocessor.word_vocab_layer.vocabulary_size()
    num_trace_features = X_train['trace_input'].shape[-1]
    key_vocab_size = len(preprocessor.key_map)
    
    print(f"\nModel configuration:")
    print(f"  Classes (words): {num_classes}")
    print(f"  Trace features: {num_trace_features}")
    print(f"  Key vocabulary: {key_vocab_size}")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    
    model = build_dual_branch_model(
        max_seq_len=MAX_SEQ_LENGTH,
        num_features=num_trace_features,
        max_keys_len=MAX_KEYS_LENGTH,
        key_vocab_size=key_vocab_size,
        num_classes=num_classes
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    model.summary()
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train, loader.word_frequencies)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(str(MODEL_DIR / 'best_swipe_model.h5'), save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, str(MODEL_DIR / 'training_history.png'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
    
    # Convert to TFLite
    tflite_path = MODEL_DIR / 'swipe_model.tflite'
    convert_to_tflite(
        model, 
        X_train, 
        str(tflite_path),
        quantize=not args.no_quantize
    )
    
    # Evaluate TFLite model
    evaluate_tflite_model(str(tflite_path), X_test, y_test)
    
    print("\nTraining complete!")
    print(f"Models saved in {MODEL_DIR}/")


if __name__ == "__main__":
    main()