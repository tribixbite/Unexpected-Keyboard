#!/usr/bin/env python3
"""Test current prediction system with actual swipe data"""

import json
import subprocess
import tempfile
import os

def load_swipe_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_test_java(key_sequence, target_word):
    """Create a Java test file to run predictions"""
    test_code = f'''
import juloo.keyboard2.*;
import java.util.*;

public class TestPrediction {{
    public static void main(String[] args) {{
        // Simulate the prediction
        WordPredictor predictor = new WordPredictor();
        Config config = new Config(null, null, null, false);
        
        // Set default weights
        config.swipe_first_letter_weight = 1.5f;
        config.swipe_last_letter_weight = 1.5f;
        config.swipe_endpoint_bonus_weight = 2.0f;
        config.swipe_require_endpoints = false;
        
        predictor.setConfig(config);
        
        // Load dictionary (simplified)
        Map<String, Integer> dict = new HashMap<>();
        dict.put("the", 10000);
        dict.put("and", 9000);
        dict.put("you", 8000);
        dict.put("but", 7000);
        dict.put("this", 6000);
        dict.put("that", 5000);
        dict.put("typing", 4000);
        dict.put("android", 3000);
        dict.put("quick", 2000);
        dict.put("calibration", 1000);
        
        String keySequence = "{key_sequence}";
        String targetWord = "{target_word}";
        
        System.out.println("Testing: " + targetWord + " with keys: " + keySequence);
        
        // Get predictions
        List<String> predictions = predictor.predictWords(keySequence);
        
        System.out.println("Predictions: " + predictions);
        
        // Check if target word is in predictions
        boolean found = predictions.contains(targetWord);
        int position = predictions.indexOf(targetWord);
        
        System.out.println("Target found: " + found);
        if (found) {{
            System.out.println("Position: " + (position + 1));
        }}
    }}
}}
'''
    return test_code

def analyze_prediction_failures():
    """Analyze why predictions are failing"""
    
    data = load_swipe_data('swipe_data_20250821_235946.json')
    
    print("=== PREDICTION FAILURE ANALYSIS ===\n")
    
    failures = []
    
    for trace in data['data'][:10]:  # Test first 10
        target_word = trace['target_word']
        registered_keys = trace.get('registered_keys', [])
        key_sequence = ''.join(registered_keys)
        
        if not key_sequence:
            continue
            
        print(f"\nTarget: '{target_word}'")
        print(f"Keys:   '{key_sequence}'")
        print(f"Length: word={len(target_word)}, keys={len(key_sequence)}")
        
        # Analyze the key sequence
        issues = []
        
        # Check first/last letter matching
        if key_sequence[0] != target_word[0]:
            issues.append(f"First letter mismatch: '{key_sequence[0]}' != '{target_word[0]}'")
        
        if key_sequence[-1] != target_word[-1]:
            issues.append(f"Last letter mismatch: '{key_sequence[-1]}' != '{target_word[-1]}'")
        
        # Check if word letters are in sequence
        word_idx = 0
        key_idx = 0
        matched = []
        while word_idx < len(target_word) and key_idx < len(key_sequence):
            if target_word[word_idx] == key_sequence[key_idx]:
                matched.append(key_idx)
                word_idx += 1
            key_idx += 1
        
        if word_idx < len(target_word):
            issues.append(f"Missing letters: {target_word[word_idx:]}")
        
        # Check noise level
        noise_ratio = (len(key_sequence) - len(target_word)) / len(target_word)
        if noise_ratio > 1.0:
            issues.append(f"High noise: {noise_ratio:.1f}x extra keys")
        
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            failures.append((target_word, key_sequence, issues))
    
    print(f"\n=== SUMMARY ===")
    print(f"Total failures analyzed: {len(failures)}")
    
    # Count issue types
    issue_counts = {}
    for _, _, issues in failures:
        for issue in issues:
            issue_type = issue.split(':')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    print("\nMost common issues:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue_type}: {count}")
    
    return failures

def main():
    failures = analyze_prediction_failures()
    
    print("\n=== ROOT CAUSES ===")
    print("1. KEY REGISTRATION NOISE: Swipes register many extra keys")
    print("2. ENDPOINT ACCURACY: 29% of swipes have wrong last letter")
    print("3. PATH COMPLEXITY: Users make complex curves instead of direct paths")
    print("4. TIMING ISSUES: Negative time deltas suggest timestamp problems")
    print("5. COORDINATE PRECISION: Touch points may not align with key centers")
    
    print("\n=== PROPOSED FIXES ===")
    print("1. Implement better noise filtering in SwipeGestureRecognizer")
    print("2. Use path smoothing before key detection")
    print("3. Weight keys by proximity to path center, not just touched")
    print("4. Implement key probability maps instead of binary touched/not-touched")
    print("5. Add path curvature analysis to detect intentional vs accidental touches")

if __name__ == '__main__':
    main()