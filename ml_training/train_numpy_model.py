#!/usr/bin/env python3
"""
Ultra-simple swipe typing model using only NumPy
Implements a basic nearest neighbor classifier
"""

import json
import numpy as np
import pickle
from collections import defaultdict

def load_ndjson_data(file_path):
    """Load training data from NDJSON file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def extract_gesture_signature(trace_data):
    """Extract a simple signature from swipe gesture"""
    points = trace_data.get('trace_points', trace_data.get('points', []))
    if len(points) < 2:
        return None
    
    # Normalize points to [0, 1] range
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Avoid division by zero
    x_range = max_x - min_x if max_x > min_x else 1
    y_range = max_y - min_y if max_y > min_y else 1
    
    # Resample to fixed number of points (10)
    n_samples = 10
    indices = np.linspace(0, len(points) - 1, n_samples).astype(int)
    
    signature = []
    for i in indices:
        x_norm = (points[i]['x'] - min_x) / x_range
        y_norm = (points[i]['y'] - min_y) / y_range
        signature.extend([x_norm, y_norm])
    
    # Add gesture characteristics
    total_length = 0
    for i in range(1, len(points)):
        # Handle both formats
        x_key = 'x' if 'x' in points[i] else 'X'
        y_key = 'y' if 'y' in points[i] else 'Y'
        dx = points[i][x_key] - points[i-1][x_key]
        dy = points[i][y_key] - points[i-1][y_key]
        total_length += np.sqrt(dx**2 + dy**2)
    
    x_key = 'x' if 'x' in points[0] else 'X'
    y_key = 'y' if 'y' in points[0] else 'Y'
    direct_dist = np.sqrt(
        (points[-1][x_key] - points[0][x_key])**2 + 
        (points[-1][y_key] - points[0][y_key])**2
    )
    
    straightness = direct_dist / (total_length + 0.001)
    
    signature.extend([
        total_length / 1000.0,  # Normalized length
        straightness,
        len(points) / 100.0  # Normalized point count
    ])
    
    return np.array(signature)

class SimpleSwipeClassifier:
    """Simple nearest neighbor classifier for swipe gestures"""
    
    def __init__(self):
        self.templates = defaultdict(list)
        self.vocabulary = set()
    
    def fit(self, traces, labels):
        """Train by storing gesture templates"""
        for trace, label in zip(traces, labels):
            signature = extract_gesture_signature(trace)
            if signature is not None:
                self.templates[label].append(signature)
                self.vocabulary.add(label)
        
        # Convert to arrays for faster computation
        for word in self.templates:
            self.templates[word] = np.array(self.templates[word])
        
        print(f"Learned {len(self.vocabulary)} unique words")
        return self
    
    def predict(self, trace, top_k=5):
        """Predict word from gesture using nearest neighbor"""
        signature = extract_gesture_signature(trace)
        if signature is None:
            return []
        
        scores = []
        for word, templates in self.templates.items():
            # Find minimum distance to any template of this word
            distances = np.sum((templates - signature)**2, axis=1)
            min_distance = np.min(distances)
            scores.append((word, -min_distance))  # Negative for sorting
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scores[:top_k]]
    
    def evaluate(self, test_traces, test_labels):
        """Evaluate accuracy"""
        correct = 0
        top5_correct = 0
        
        for trace, true_label in zip(test_traces, test_labels):
            predictions = self.predict(trace, top_k=5)
            if predictions:
                if predictions[0] == true_label:
                    correct += 1
                if true_label in predictions:
                    top5_correct += 1
        
        total = len(test_labels)
        return {
            'top1_accuracy': correct / total if total > 0 else 0,
            'top5_accuracy': top5_correct / total if total > 0 else 0,
            'total_samples': total
        }

def main():
    print("Simple Swipe Typing Model Training")
    print("=" * 40)
    
    print("\nLoading training data...")
    data = load_ndjson_data('sample_data.ndjson')
    print(f"Loaded {len(data)} samples")
    
    # Split data manually (80/20 split)
    np.random.seed(42)
    np.random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_traces = train_data
    train_labels = [d.get('target_word', d.get('word', '')) for d in train_data]
    test_traces = test_data
    test_labels = [d.get('target_word', d.get('word', '')) for d in test_data]
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Train model
    print("\nTraining model...")
    model = SimpleSwipeClassifier()
    model.fit(train_traces, train_labels)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(test_traces, test_labels)
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.1%}")
    
    # Save model
    print("\nSaving model...")
    with open('swipe_model_simple.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to swipe_model_simple.pkl")
    
    # Test with a sample
    if test_data:
        print("\nSample prediction:")
        sample = test_data[0]
        predictions = model.predict(sample, top_k=3)
        print(f"True word: {sample.get('target_word', sample.get('word', 'unknown'))}")
        print(f"Predictions: {predictions}")
    
    print("\n" + "=" * 40)
    print("Training complete!")
    print("\nNote: This is a simple model for demonstration.")
    print("For production, use TensorFlow/PyTorch on a desktop/server.")

if __name__ == "__main__":
    main()