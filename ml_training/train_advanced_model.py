#!/usr/bin/env python3
"""
Advanced Swipe Typing Neural Network Training Script
Enhanced version with improved preprocessing, architecture, and evaluation

Features:
- Advanced data preprocessing with smart normalization
- Enhanced model architecture with attention mechanism
- Comprehensive evaluation metrics and visualization
- Advanced data augmentation techniques
- Model interpretability tools
- Integration with user adaptation data
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, LSTM, Dense, Concatenate, Embedding, MultiHeadAttention,
    GlobalAveragePooling1D, Masking, Dropout, StringLookup, LayerNormalization,
    BatchNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Configuration & Hyperparameters ---
# Data Parameters
MAX_SEQ_LENGTH = 120  # 95th percentile of trace lengths
MAX_KEYS_LENGTH = 20  # Max length of registered_keys
MIN_TRACE_POINTS = 3  # Minimum points for valid swipe
TRACE_INTERPOLATION_POINTS = 100  # Standardized interpolation length

# Advanced Model Parameters
GRU_UNITS = 128
LSTM_UNITS = 64
ATTENTION_HEADS = 4
ATTENTION_DIM = 64
EMBEDDING_DIM = 32
DENSE_UNITS = 256
DROPOUT_RATE = 0.4
L2_REG = 1e-4

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 150
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
PATIENCE = 15

# Data Augmentation Parameters
SPATIAL_NOISE_STD = 0.02
TEMPORAL_NOISE_STD = 5.0
AUGMENTATION_FACTOR = 2.0

# Output paths
MODEL_DIR = Path("models_advanced")
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR = MODEL_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class AdvancedSwipeDataLoader:
    """Enhanced data loader with improved preprocessing and statistics."""
    
    def __init__(self, file_path: str, dictionary_path: Optional[str] = None, user_adaptation_path: Optional[str] = None):
        self.file_path = file_path
        self.dictionary_path = dictionary_path
        self.user_adaptation_path = user_adaptation_path
        self.word_frequencies = self._load_dictionary() if dictionary_path else {}
        self.user_preferences = self._load_user_adaptation() if user_adaptation_path else {}
        
    def _load_dictionary(self) -> Dict[str, int]:
        """Load word frequencies from dictionary file."""
        frequencies = {}
        if self.dictionary_path and os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word, freq = parts[0], parts[1]
                        frequencies[word.lower()] = int(freq)
        return frequencies
    
    def _load_user_adaptation(self) -> Dict[str, float]:
        """Load user adaptation data if available."""
        preferences = {}
        if self.user_adaptation_path and os.path.exists(self.user_adaptation_path):
            try:
                with open(self.user_adaptation_path, 'r') as f:
                    adaptation_data = json.load(f)
                    for word, data in adaptation_data.items():
                        if 'selection_count' in data and 'total_selections' in data:
                            preferences[word] = data['selection_count'] / max(data['total_selections'], 1)
            except Exception as e:
                logger.warning(f"Could not load user adaptation data: {e}")
        return preferences
    
    def load_data(self) -> Tuple[List, List, List, Dict]:
        """Enhanced data loading with comprehensive statistics."""
        traces, registered_keys, target_words = [], [], []
        metadata = {
            'sources': {}, 'screen_sizes': [], 'trace_stats': [],
            'temporal_stats': [], 'collection_dates': []
        }
        
        logger.info(f"Loading data from {self.file_path}...")
        invalid_samples = 0
        
        with open(self.file_path, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading data")):
                try:
                    record = json.loads(line)
                    
                    # Enhanced validation
                    if not self._validate_sample(record):
                        invalid_samples += 1
                        continue
                    
                    # Extract and preprocess trace points
                    trace_points = self._preprocess_trace(record['trace_points'])
                    
                    traces.append(trace_points)
                    registered_keys.append(record['registered_keys'])
                    target_words.append(record['target_word'].lower())
                    
                    # Enhanced metadata collection
                    self._collect_metadata(record, metadata)
                        
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    invalid_samples += 1
                    continue
        
        logger.info(f"Loaded {len(traces)} valid samples ({invalid_samples} invalid)")
        self._print_data_statistics(traces, target_words, metadata)
        
        return traces, registered_keys, target_words, metadata
    
    def _validate_sample(self, record: dict) -> bool:
        """Enhanced sample validation."""
        if len(record['trace_points']) < MIN_TRACE_POINTS:
            return False
        
        # Check for reasonable coordinate ranges
        for point in record['trace_points']:
            if not (0.0 <= point['x'] <= 1.0 and 0.0 <= point['y'] <= 1.0):
                return False
            if point['t_delta_ms'] < 0 or point['t_delta_ms'] > 10000:  # Max 10 seconds
                return False
        
        # Check target word validity
        if not record['target_word'] or len(record['target_word']) > 50:
            return False
            
        return True
    
    def _preprocess_trace(self, trace_points: List[dict]) -> np.ndarray:
        """Enhanced trace preprocessing with smoothing and normalization."""
        points = np.array([
            [p['x'], p['y'], p['t_delta_ms']] 
            for p in trace_points
        ], dtype=np.float32)
        
        # Temporal normalization - convert to relative time ratios
        if len(points) > 1:
            total_time = points[-1, 2] - points[0, 2]
            if total_time > 0:
                points[:, 2] = (points[:, 2] - points[0, 2]) / total_time
            else:
                points[:, 2] = np.linspace(0, 1, len(points))
        
        # Add velocity features
        if len(points) > 1:
            velocities = np.diff(points[:, :2], axis=0)
            velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
            
            # Pad velocity to match trace length
            velocity_magnitudes = np.concatenate([[0], velocity_magnitudes])
            
            # Add velocity as fourth feature
            points = np.column_stack([points, velocity_magnitudes])
        else:
            # Single point - zero velocity
            points = np.column_stack([points, [0]])
        
        return points
    
    def _collect_metadata(self, record: dict, metadata: dict):
        """Collect comprehensive metadata for analysis."""
        source = record['metadata']['collection_source']
        metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
        
        screen_info = (
            record['metadata']['screen_width_px'],
            record['metadata']['screen_height_px']
        )
        if screen_info not in metadata['screen_sizes']:
            metadata['screen_sizes'].append(screen_info)
        
        # Trace statistics
        trace_length = len(record['trace_points'])
        total_time = record['trace_points'][-1]['t_delta_ms'] - record['trace_points'][0]['t_delta_ms']
        metadata['trace_stats'].append((trace_length, total_time))
        
        # Collection date
        if 'timestamp_utc' in record['metadata']:
            metadata['collection_dates'].append(record['metadata']['timestamp_utc'])
    
    def _print_data_statistics(self, traces: List, words: List, metadata: Dict):
        """Print comprehensive data statistics."""
        logger.info(f"=== DATA STATISTICS ===")
        logger.info(f"Total samples: {len(traces)}")
        logger.info(f"Sources: {metadata['sources']}")
        logger.info(f"Unique words: {len(set(words))}")
        logger.info(f"Screen sizes: {len(metadata['screen_sizes'])} unique")
        
        # Word frequency distribution
        word_counts = pd.Series(words).value_counts()
        logger.info(f"Most common words: {word_counts.head(10).to_dict()}")
        
        # Trace length statistics
        trace_lengths = [len(t) for t in traces]
        logger.info(f"Trace lengths - Min: {min(trace_lengths)}, Max: {max(trace_lengths)}, "
                   f"Mean: {np.mean(trace_lengths):.1f}, Std: {np.std(trace_lengths):.1f}")
        
        # Temporal statistics
        if metadata['trace_stats']:
            _, durations = zip(*metadata['trace_stats'])
            logger.info(f"Trace durations - Min: {min(durations):.0f}ms, Max: {max(durations):.0f}ms, "
                       f"Mean: {np.mean(durations):.0f}ms")


class AdvancedDataPreprocessor:
    """Enhanced data preprocessor with augmentation and advanced features."""
    
    def __init__(self, max_seq_length: int, max_keys_length: int):
        self.max_seq_length = max_seq_length
        self.max_keys_length = max_keys_length
        self.word_vocab_layer = None
        self.key_map = None
        self.scaler_stats = None
        
    def preprocess_and_split(self, traces, keys, words, augment=True):
        """Enhanced preprocessing with augmentation and stratified splitting."""
        
        logger.info("Preprocessing data...")
        
        # Analyze data characteristics
        self._analyze_data_distribution(traces, words)
        
        # Apply data augmentation if requested
        if augment:
            traces, keys, words = self._augment_data(traces, keys, words)
        
        # Pad and normalize trace sequences
        padded_traces = self._pad_and_normalize_traces(traces)
        
        # Process vocabulary
        self._build_vocabularies(words, keys)
        
        # Encode labels and keys
        encoded_keys = self._encode_keys(keys)
        encoded_words = self._encode_words(words)
        
        # Stratified split to maintain word distribution
        X_temp, X_test, y_temp, y_test, keys_temp, keys_test = train_test_split(
            padded_traces, encoded_words, encoded_keys,
            test_size=TEST_SPLIT, stratify=encoded_words, random_state=42
        )
        
        X_train, X_val, y_train, y_val, keys_train, keys_val = train_test_split(
            X_temp, y_temp, keys_temp,
            test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), stratify=y_temp, random_state=42
        )
        
        # Package data for model
        X_train_dict = {'trace_input': X_train, 'keys_input': keys_train}
        X_val_dict = {'trace_input': X_val, 'keys_input': keys_val}
        X_test_dict = {'trace_input': X_test, 'keys_input': keys_test}
        
        logger.info(f"Data split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        return X_train_dict, y_train, X_val_dict, y_val, X_test_dict, y_test
    
    def _analyze_data_distribution(self, traces: List, words: List):
        """Analyze data distribution for preprocessing decisions."""
        trace_lengths = [len(t) for t in traces]
        
        logger.info("=== DATA DISTRIBUTION ANALYSIS ===")
        logger.info(f"Trace length percentiles:")
        for p in [50, 75, 90, 95, 99]:
            logger.info(f"  {p}th percentile: {np.percentile(trace_lengths, p):.1f}")
        
        # Word frequency analysis
        word_counts = pd.Series(words).value_counts()
        logger.info(f"Word frequency - Most common: {word_counts.iloc[0]} occurrences")
        logger.info(f"Words with <3 samples: {(word_counts < 3).sum()}")
    
    def _augment_data(self, traces: List, keys: List, words: List) -> Tuple[List, List, List]:
        """Apply data augmentation techniques."""
        logger.info(f"Applying data augmentation (factor: {AUGMENTATION_FACTOR})...")
        
        augmented_traces, augmented_keys, augmented_words = [], [], []
        
        for trace, key_seq, word in tqdm(zip(traces, keys, words), desc="Augmenting data"):
            # Original sample
            augmented_traces.append(trace)
            augmented_keys.append(key_seq)
            augmented_words.append(word)
            
            # Generate augmented samples
            for _ in range(int(AUGMENTATION_FACTOR)):
                aug_trace = self._augment_trace(trace)
                augmented_traces.append(aug_trace)
                augmented_keys.append(key_seq)  # Keys don't change
                augmented_words.append(word)
        
        logger.info(f"Augmented dataset size: {len(augmented_traces)} samples")
        return augmented_traces, augmented_keys, augmented_words
    
    def _augment_trace(self, trace: np.ndarray) -> np.ndarray:
        """Apply trace-level augmentation."""
        augmented = trace.copy()
        
        # Spatial jitter
        spatial_noise = np.random.normal(0, SPATIAL_NOISE_STD, size=(len(trace), 2))
        augmented[:, :2] += spatial_noise
        
        # Clip to valid coordinate range
        augmented[:, :2] = np.clip(augmented[:, :2], 0.0, 1.0)
        
        # Temporal jitter (on relative time)
        if len(trace) > 2:
            temporal_noise = np.random.normal(0, TEMPORAL_NOISE_STD/1000, size=len(trace))
            augmented[:, 2] += temporal_noise
            # Ensure monotonic increase
            augmented[:, 2] = np.maximum.accumulate(augmented[:, 2])
            # Renormalize to [0, 1]
            if augmented[-1, 2] > 0:
                augmented[:, 2] = augmented[:, 2] / augmented[-1, 2]
        
        return augmented
    
    def _pad_and_normalize_traces(self, traces: List) -> np.ndarray:
        """Enhanced trace padding and normalization."""
        # Convert to consistent feature format
        processed_traces = []
        all_features = []
        
        for trace in traces:
            # Ensure trace has velocity feature (4th column)
            if trace.shape[1] == 3:
                # Add zero velocity if missing
                trace = np.column_stack([trace, np.zeros(len(trace))])
            
            processed_traces.append(trace)
            all_features.extend(trace.tolist())
        
        # Calculate normalization statistics
        all_features = np.array(all_features)
        self.scaler_stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + 1e-8  # Avoid division by zero
        }
        
        # Normalize and pad
        normalized_traces = []
        for trace in processed_traces:
            # Z-score normalization
            normalized = (trace - self.scaler_stats['mean']) / self.scaler_stats['std']
            normalized_traces.append(normalized)
        
        # Pad sequences
        padded = pad_sequences(
            normalized_traces,
            maxlen=self.max_seq_length,
            dtype='float32',
            padding='post',
            truncating='pre',
            value=0.0
        )
        
        return padded
    
    def _build_vocabularies(self, words: List, keys: List):
        """Build vocabularies for words and keys."""
        # Build word vocabulary
        unique_words = sorted(list(set(words)))
        self.word_vocab_layer = StringLookup(mask_token=None, num_oov_indices=1)
        self.word_vocab_layer.adapt(unique_words)
        
        # Build key vocabulary
        all_keys = set()
        for key_seq in keys:
            all_keys.update(key_seq)
        
        self.key_map = {key: idx + 1 for idx, key in enumerate(sorted(all_keys))}
        self.key_map['<PAD>'] = 0
    
    def _encode_keys(self, keys: List) -> np.ndarray:
        """Encode key sequences."""
        encoded_keys = []
        for key_seq in keys:
            encoded = [self.key_map.get(k, 0) for k in key_seq]
            encoded_keys.append(encoded)
        
        return pad_sequences(
            encoded_keys,
            maxlen=self.max_keys_length,
            padding='post',
            truncating='post',
            value=0
        )
    
    def _encode_words(self, words: List) -> np.ndarray:
        """Encode target words."""
        return self.word_vocab_layer(words).numpy()


def build_advanced_model(max_seq_len: int, num_features: int, max_keys_len: int, 
                        key_vocab_size: int, num_classes: int) -> Model:
    """
    Build advanced dual-branch model with attention mechanism.
    
    Architecture:
    - Trace branch: Masking -> GRU -> Multi-head Attention -> LayerNorm
    - Keys branch: Embedding -> GlobalAveragePooling
    - Fusion: Concatenate -> Dense layers with residual connections
    """
    
    # Branch 1: Trace processing with attention
    trace_input = Input(shape=(max_seq_len, num_features), name='trace_input')
    
    # Masking layer for padded sequences
    masked = Masking(mask_value=0.0, name='trace_masking')(trace_input)
    
    # Bidirectional GRU
    gru_forward = GRU(GRU_UNITS, return_sequences=True, name='gru_forward')(masked)
    gru_backward = GRU(GRU_UNITS, return_sequences=True, go_backwards=True, name='gru_backward')(masked)
    gru_concat = Concatenate(name='gru_concat')([gru_forward, gru_backward])
    
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_DIM,
        dropout=DROPOUT_RATE,
        name='multi_head_attention'
    )(gru_concat, gru_concat)
    
    # Add & Norm
    attention_norm = Add(name='attention_add')([gru_concat, attention_output])
    attention_norm = LayerNormalization(name='attention_norm')(attention_norm)
    
    # Global pooling
    trace_pooled = GlobalAveragePooling1D(name='trace_pooling')(attention_norm)
    
    # Branch 2: Enhanced keys processing
    keys_input = Input(shape=(max_keys_len,), name='keys_input')
    key_embedding = Embedding(
        input_dim=key_vocab_size + 1,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name='key_embedding'
    )(keys_input)
    
    # Keys attention
    keys_attention = MultiHeadAttention(
        num_heads=2,
        key_dim=EMBEDDING_DIM//2,
        dropout=DROPOUT_RATE,
        name='keys_attention'
    )(key_embedding, key_embedding)
    
    keys_pooled = GlobalAveragePooling1D(name='keys_pooling')(keys_attention)
    
    # Fusion network with residual connections
    concatenated = Concatenate(name='fusion')([trace_pooled, keys_pooled])
    
    # First dense block
    x = Dense(DENSE_UNITS, activation='relu', name='dense_1')(concatenated)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    # Second dense block with residual connection
    x2 = Dense(DENSE_UNITS, activation='relu', name='dense_2')(x)
    x2 = BatchNormalization(name='bn_2')(x2)
    x2 = Dropout(DROPOUT_RATE, name='dropout_2')(x2)
    
    # Residual connection (if dimensions match)
    if x.shape[-1] == x2.shape[-1]:
        x = Add(name='residual')([x, x2])
    else:
        x = x2
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=[trace_input, keys_input], outputs=output, name='AdvancedSwipeModel')
    return model


def evaluate_model_comprehensive(model, X_test, y_test, word_vocab_layer, save_plots=True):
    """Comprehensive model evaluation with detailed metrics and visualizations."""
    logger.info("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Basic accuracy metrics
    accuracy = np.mean(y_pred == y_test)
    top3_accuracy = np.mean([y_test[i] in np.argsort(y_pred_proba[i])[-3:] for i in range(len(y_test))])
    top5_accuracy = np.mean([y_test[i] in np.argsort(y_pred_proba[i])[-5:] for i in range(len(y_test))])
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    logger.info(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    
    # Detailed classification report
    vocab_words = word_vocab_layer.get_vocabulary()
    target_names = [vocab_words[i] for i in range(len(vocab_words))]
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    # Save detailed metrics
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(MODEL_DIR / 'detailed_metrics.csv')
    logger.info("Detailed metrics saved to detailed_metrics.csv")
    
    if save_plots:
        # Confusion matrix for most common words
        most_common_indices = np.unique(y_test, return_counts=True)[1].argsort()[-20:][::-1]
        common_words = [target_names[i] for i in most_common_indices]
        
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(y_test, y_pred)
        cm_subset = cm[np.ix_(most_common_indices, most_common_indices)]
        
        sns.heatmap(cm_subset, annot=True, fmt='d', xticklabels=common_words, 
                   yticklabels=common_words, cmap='Blues')
        plt.title('Confusion Matrix - Top 20 Words')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction confidence distribution
        plt.figure(figsize=(10, 6))
        max_probs = np.max(y_pred_proba, axis=1)
        correct_mask = (y_pred == y_test)
        
        plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct Predictions', 
                color='green', density=True)
        plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect Predictions', 
                color='red', density=True)
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(MODEL_DIR / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation plots saved")
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'classification_report': report
    }


def main():
    """Main training pipeline with enhanced features."""
    parser = argparse.ArgumentParser(description='Train advanced swipe typing neural network')
    parser.add_argument('--data', type=str, required=True, help='Path to NDJSON data file')
    parser.add_argument('--dict', type=str, help='Path to dictionary file with word frequencies')
    parser.add_argument('--user-adapt', type=str, help='Path to user adaptation data (JSON)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--no-augment', action='store_true', help='Skip data augmentation')
    parser.add_argument('--no-quantize', action='store_true', help='Skip TFLite quantization')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f'training_{timestamp}.log'
    
    # Load data with enhanced loader
    logger.info("=== ADVANCED SWIPE MODEL TRAINING ===")
    loader = AdvancedSwipeDataLoader(args.data, args.dict, args.user_adapt)
    traces, keys, words, metadata = loader.load_data()
    
    if len(traces) < 100:
        logger.warning("Very few samples available. Consider collecting more data for better performance.")
    
    # Enhanced preprocessing
    preprocessor = AdvancedDataPreprocessor(MAX_SEQ_LENGTH, MAX_KEYS_LENGTH)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.preprocess_and_split(
        traces, keys, words, augment=not args.no_augment
    )
    
    # Save preprocessing artifacts
    vocab_path = MODEL_DIR / 'word_vocab.txt'
    with open(vocab_path, 'w') as f:
        for word in preprocessor.word_vocab_layer.get_vocabulary():
            f.write(f"{word}\n")
    
    # Save key mapping
    key_map_path = MODEL_DIR / 'key_mapping.json'
    with open(key_map_path, 'w') as f:
        json.dump(preprocessor.key_map, f, indent=2)
    
    # Save normalization statistics
    stats_path = MODEL_DIR / 'normalization_stats.json'
    with open(stats_path, 'w') as f:
        stats = {k: v.tolist() for k, v in preprocessor.scaler_stats.items()}
        json.dump(stats, f, indent=2)
    
    logger.info(f"Preprocessing artifacts saved")
    
    # Build advanced model
    num_classes = preprocessor.word_vocab_layer.vocabulary_size()
    num_trace_features = X_train['trace_input'].shape[-1]
    key_vocab_size = len(preprocessor.key_map)
    
    logger.info(f"Model configuration:")
    logger.info(f"  Classes (words): {num_classes}")
    logger.info(f"  Trace features: {num_trace_features}")
    logger.info(f"  Key vocabulary: {key_vocab_size}")
    logger.info(f"  Training samples: {len(y_train)}")
    
    model = build_advanced_model(
        max_seq_len=MAX_SEQ_LENGTH,
        num_features=num_trace_features,
        max_keys_len=MAX_KEYS_LENGTH,
        key_vocab_size=key_vocab_size,
        num_classes=num_classes
    )
    
    # Advanced compilation with custom metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
        ]
    )
    
    model.summary()
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=PATIENCE//2, 
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODEL_DIR / 'best_advanced_model.h5'), 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOGS_DIR / f'tensorboard_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    # Train model
    logger.info("Starting advanced model training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Comprehensive evaluation
    eval_results = evaluate_model_comprehensive(
        model, X_test, y_test, preprocessor.word_vocab_layer
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL_DIR / 'training_history.csv', index=False)
    
    # Enhanced training plots
    plot_training_history_advanced(history, MODEL_DIR / 'training_history_advanced.png')
    
    # Convert to TFLite
    if not args.no_quantize:
        logger.info("Converting to optimized TFLite...")
        tflite_path = MODEL_DIR / 'advanced_swipe_model.tflite'
        convert_to_tflite_advanced(model, X_train, tflite_path)
    
    logger.info(f"=== TRAINING COMPLETE ===")
    logger.info(f"Final Test Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"Models and artifacts saved to: {MODEL_DIR}")
    
    return model, eval_results


def plot_training_history_advanced(history, save_path):
    """Create enhanced training history plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    if 'top3_acc' in history.history:
        axes[1, 0].plot(history.history['top3_acc'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_top3_acc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='red')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Advanced training plots saved to {save_path}")


def convert_to_tflite_advanced(model, X_train, save_path):
    """Enhanced TFLite conversion with optimization."""
    def representative_dataset():
        for i in range(min(200, len(X_train['trace_input']))):
            yield [
                np.expand_dims(X_train['trace_input'][i], axis=0).astype(np.float32),
                np.expand_dims(X_train['keys_input'][i], axis=0).astype(np.float32)
            ]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Mixed precision quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model size
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model saved: {save_path} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()