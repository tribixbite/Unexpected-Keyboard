#!/usr/bin/env python3
"""
Simple swipe typing model training using scikit-learn
Works on Termux without TensorFlow
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

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

def extract_features(trace_data):
    """Extract simple features from a swipe trace"""
    points = trace_data['points']
    if len(points) < 2:
        return None
    
    features = []
    
    # Total length
    total_length = 0
    for i in range(1, len(points)):
        dx = points[i]['x'] - points[i-1]['x']
        dy = points[i]['y'] - points[i-1]['y']
        total_length += np.sqrt(dx**2 + dy**2)
    features.append(total_length)
    
    # Direct distance (start to end)
    direct_dist = np.sqrt(
        (points[-1]['x'] - points[0]['x'])**2 + 
        (points[-1]['y'] - points[0]['y'])**2
    )
    features.append(direct_dist)
    
    # Straightness ratio
    straightness = direct_dist / (total_length + 0.001)
    features.append(straightness)
    
    # Number of points
    features.append(len(points))
    
    # Average velocity
    if 'time_deltas' in trace_data and len(trace_data['time_deltas']) > 0:
        avg_time = np.mean(trace_data['time_deltas'])
        avg_velocity = total_length / (sum(trace_data['time_deltas']) + 0.001)
        features.append(avg_velocity)
    else:
        features.append(0)
    
    # Direction changes (simplified)
    direction_changes = 0
    for i in range(2, len(points)):
        dx1 = points[i-1]['x'] - points[i-2]['x']
        dy1 = points[i-1]['y'] - points[i-2]['y']
        dx2 = points[i]['x'] - points[i-1]['x']
        dy2 = points[i]['y'] - points[i-1]['y']
        
        # Simple dot product to detect direction change
        dot = dx1 * dx2 + dy1 * dy2
        if dot < 0:
            direction_changes += 1
    features.append(direction_changes)
    
    # Add key positions (first 10 keys, padded with zeros)
    keys = trace_data.get('keys', [])
    for i in range(10):
        if i < len(keys):
            # Convert character to ASCII code
            features.append(ord(keys[i]) if isinstance(keys[i], str) else keys[i])
        else:
            features.append(0)
    
    return features

def prepare_dataset(data):
    """Prepare features and labels from raw data"""
    X = []
    y = []
    
    for item in data:
        features = extract_features(item)
        if features is not None:
            X.append(features)
            y.append(item['word'])
    
    return np.array(X), np.array(y)

def main():
    print("Loading training data...")
    data = load_ndjson_data('sample_data.ndjson')
    print(f"Loaded {len(data)} samples")
    
    print("Extracting features...")
    X, y = prepare_dataset(data)
    print(f"Prepared {len(X)} valid samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Save model
    print("\nSaving model...")
    with open('swipe_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder,
            'feature_dim': X.shape[1]
        }, f)
    print("Model saved to swipe_model.pkl")
    
    # Feature importance
    print("\nTop 5 Feature Importances:")
    importance_idx = np.argsort(model.feature_importances_)[::-1][:5]
    feature_names = ['total_length', 'direct_dist', 'straightness', 'num_points', 
                     'avg_velocity', 'direction_changes'] + [f'key_{i}' for i in range(10)]
    for idx in importance_idx:
        if idx < len(feature_names):
            print(f"  {feature_names[idx]}: {model.feature_importances_[idx]:.3f}")

if __name__ == "__main__":
    main()