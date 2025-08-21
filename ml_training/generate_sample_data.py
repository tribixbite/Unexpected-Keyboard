#!/usr/bin/env python3
"""
Generate sample swipe data for testing the ML training pipeline.
Creates realistic synthetic data following the actual data format.
"""

import json
import numpy as np
import random
import uuid
from datetime import datetime
from pathlib import Path

# Common words for testing
TEST_WORDS = [
    "the", "and", "you", "that", "this", "hello", "world", "thanks", 
    "keyboard", "android", "swipe", "typing", "test", "quick", "brown",
    "fox", "jumps", "over", "lazy", "dog", "phone", "message", "text",
    "good", "morning", "night", "today", "tomorrow", "please", "sorry"
]

# QWERTY layout for key mapping
KEYBOARD_LAYOUT = {
    'q': (0.05, 0.33), 'w': (0.15, 0.33), 'e': (0.25, 0.33), 'r': (0.35, 0.33), 
    't': (0.45, 0.33), 'y': (0.55, 0.33), 'u': (0.65, 0.33), 'i': (0.75, 0.33), 
    'o': (0.85, 0.33), 'p': (0.95, 0.33),
    'a': (0.10, 0.50), 's': (0.20, 0.50), 'd': (0.30, 0.50), 'f': (0.40, 0.50), 
    'g': (0.50, 0.50), 'h': (0.60, 0.50), 'j': (0.70, 0.50), 'k': (0.80, 0.50), 
    'l': (0.90, 0.50),
    'z': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67), 
    'b': (0.55, 0.67), 'n': (0.65, 0.67), 'm': (0.75, 0.67)
}


def generate_swipe_path(word, num_points=None):
    """
    Generate a realistic swipe path for a given word.
    """
    if num_points is None:
        # Variable number of points based on word length
        num_points = len(word) * random.randint(8, 15)
    
    # Get key positions
    key_positions = []
    registered_keys = []
    
    for char in word.lower():
        if char in KEYBOARD_LAYOUT:
            key_positions.append(KEYBOARD_LAYOUT[char])
            registered_keys.append(char)
    
    if len(key_positions) < 2:
        return None, None
    
    # Generate smooth path through keys with some noise
    trace_points = []
    timestamps = []
    current_time = 0
    
    for i in range(num_points):
        # Interpolate between key positions
        progress = i / (num_points - 1)
        key_index = min(int(progress * (len(key_positions) - 1)), len(key_positions) - 2)
        local_progress = (progress * (len(key_positions) - 1)) - key_index
        
        # Linear interpolation between keys
        start_pos = key_positions[key_index]
        end_pos = key_positions[key_index + 1]
        
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * local_progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * local_progress
        
        # Add some noise to make it realistic
        x += random.gauss(0, 0.02)
        y += random.gauss(0, 0.02)
        
        # Clamp to valid range
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Generate time delta (faster in middle of swipe)
        if i == 0:
            t_delta = 0
        else:
            speed_factor = 1.0 - abs(progress - 0.5) * 1.5  # Slower at start/end
            t_delta = random.uniform(10, 25) * speed_factor
            current_time += t_delta
        
        trace_points.append({
            'x': round(x, 4),
            'y': round(y, 4),
            't_delta_ms': round(t_delta, 1)
        })
    
    return trace_points, registered_keys


def generate_sample_data(num_samples=100, output_file='sample_data.ndjson'):
    """
    Generate sample training data in NDJSON format.
    """
    print(f"Generating {num_samples} sample swipe traces...")
    
    # Screen sizes to simulate
    screen_sizes = [
        (1080, 2400),  # Common phone
        (1440, 3200),  # High-res phone
        (720, 1600),   # Budget phone
    ]
    
    samples = []
    word_counts = {}
    
    for i in range(num_samples):
        # Select word (with some repetition for common words)
        if random.random() < 0.3 and word_counts:
            # Repeat a common word
            word = random.choice(['the', 'and', 'you', 'that', 'this'])
        else:
            word = random.choice(TEST_WORDS)
        
        word_counts[word] = word_counts.get(word, 0) + 1
        
        # Generate swipe path
        trace_points, registered_keys = generate_swipe_path(word)
        if not trace_points:
            continue
        
        # Select random screen size
        screen_width, screen_height = random.choice(screen_sizes)
        keyboard_height = int(screen_height * 0.35)  # Keyboard is ~35% of screen
        
        # Create sample record
        sample = {
            'trace_id': str(uuid.uuid4()),
            'target_word': word,
            'metadata': {
                'timestamp_utc': int(datetime.now().timestamp() * 1000),
                'screen_width_px': screen_width,
                'screen_height_px': screen_height,
                'keyboard_height_px': keyboard_height,
                'collection_source': random.choice(['calibration', 'user_selection'])
            },
            'trace_points': trace_points,
            'registered_keys': registered_keys
        }
        
        samples.append(sample)
    
    # Write to NDJSON file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} samples")
    print(f"Unique words: {len(word_counts)}")
    print(f"Word distribution: {dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    print(f"Saved to: {output_path}")
    
    return samples


def generate_dictionary_file(output_file='dictionary.txt'):
    """
    Generate a simple dictionary file with word frequencies.
    """
    # Simple frequency distribution (higher for common words)
    frequencies = {
        'the': 10000, 'and': 8000, 'you': 7000, 'that': 6000, 'this': 5500,
        'hello': 5000, 'world': 4500, 'thanks': 4000, 'keyboard': 3500,
        'android': 3000, 'swipe': 2800, 'typing': 2600, 'test': 2400,
        'quick': 2200, 'brown': 2000, 'fox': 1900, 'jumps': 1800,
        'over': 1700, 'lazy': 1600, 'dog': 1500, 'phone': 1400,
        'message': 1300, 'text': 1200, 'good': 1100, 'morning': 1000,
        'night': 900, 'today': 850, 'tomorrow': 800, 'please': 750, 'sorry': 700
    }
    
    with open(output_file, 'w') as f:
        for word, freq in frequencies.items():
            f.write(f"{word}\t{freq}\n")
    
    print(f"Dictionary saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample swipe data for testing')
    parser.add_argument('--samples', type=int, default=500, 
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='sample_data.ndjson',
                       help='Output file path')
    parser.add_argument('--dict', action='store_true',
                       help='Also generate dictionary file')
    
    args = parser.parse_args()
    
    # Generate sample data
    generate_sample_data(args.samples, args.output)
    
    # Optionally generate dictionary
    if args.dict:
        generate_dictionary_file('dictionary.txt')
    
    print("\nYou can now test the training script with:")
    print(f"  python train_swipe_model.py --data {args.output} --epochs 10")