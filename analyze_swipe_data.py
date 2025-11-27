#!/usr/bin/env python3
"""Analyze swipe data to identify performance issues"""

import json
import statistics
from collections import Counter, defaultdict
import math

def load_swipe_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_trace_points(trace):
    """Analyze a single swipe trace"""
    points = trace['trace_points']
    registered_keys = trace.get('registered_keys', [])
    target_word = trace['target_word']
    
    # Calculate path metrics
    path_length = 0
    velocities = []
    direction_changes = 0
    
    for i in range(1, len(points)):
        p1, p2 = points[i-1], points[i]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dist = math.sqrt(dx*dx + dy*dy)
        path_length += dist
        
        # Calculate velocity if time delta is positive
        if p2['t_delta_ms'] > 0:
            velocity = dist / (p2['t_delta_ms'] / 1000.0)
            velocities.append(velocity)
    
    # Calculate time metrics
    total_time = sum(abs(p['t_delta_ms']) for p in points) / 1000.0
    
    # Key sequence analysis
    key_sequence = ''.join(registered_keys) if registered_keys else ''
    
    return {
        'target_word': target_word,
        'key_sequence': key_sequence,
        'num_points': len(points),
        'path_length': path_length,
        'total_time': total_time,
        'avg_velocity': statistics.mean(velocities) if velocities else 0,
        'first_key': key_sequence[0] if key_sequence else '',
        'last_key': key_sequence[-1] if key_sequence else '',
        'target_first': target_word[0] if target_word else '',
        'target_last': target_word[-1] if target_word else '',
        'endpoint_match': (key_sequence and target_word and 
                          key_sequence[0] == target_word[0] and 
                          key_sequence[-1] == target_word[-1])
    }

def main():
    # Load data
    data = load_swipe_data('swipe_data_20250821_235946.json')
    print(f"Analyzing {data['total_samples']} swipe samples\n")
    
    # Analyze each trace
    analyses = []
    for trace in data['data']:
        analysis = analyze_trace_points(trace)
        analyses.append(analysis)
    
    # Summary statistics
    print("=== SWIPE DATA ANALYSIS ===\n")
    
    # Target words
    word_counts = Counter(a['target_word'] for a in analyses)
    print(f"Unique words: {len(word_counts)}")
    print(f"Most common words: {word_counts.most_common(10)}\n")
    
    # Key sequence accuracy
    exact_matches = sum(1 for a in analyses if a['key_sequence'] == a['target_word'])
    first_matches = sum(1 for a in analyses if a['first_key'] == a['target_first'])
    last_matches = sum(1 for a in analyses if a['last_key'] == a['target_last'])
    endpoint_matches = sum(1 for a in analyses if a['endpoint_match'])
    
    print("=== KEY MATCHING ACCURACY ===")
    print(f"Exact key sequence matches: {exact_matches}/{len(analyses)} ({100*exact_matches/len(analyses):.1f}%)")
    print(f"First letter matches: {first_matches}/{len(analyses)} ({100*first_matches/len(analyses):.1f}%)")
    print(f"Last letter matches: {last_matches}/{len(analyses)} ({100*last_matches/len(analyses):.1f}%)")
    print(f"Both endpoints match: {endpoint_matches}/{len(analyses)} ({100*endpoint_matches/len(analyses):.1f}%)\n")
    
    # Path metrics
    avg_points = statistics.mean(a['num_points'] for a in analyses)
    avg_path_length = statistics.mean(a['path_length'] for a in analyses)
    avg_time = statistics.mean(a['total_time'] for a in analyses)
    
    print("=== PATH METRICS ===")
    print(f"Average points per swipe: {avg_points:.1f}")
    print(f"Average path length: {avg_path_length:.3f}")
    print(f"Average swipe duration: {avg_time:.2f} seconds\n")
    
    # Problem cases
    print("=== PROBLEM CASES ===")
    problems = []
    for a in analyses:
        issues = []
        if not a['first_key'] == a['target_first']:
            issues.append("first_letter_mismatch")
        if not a['last_key'] == a['target_last']:
            issues.append("last_letter_mismatch")
        if len(a['key_sequence']) > len(a['target_word']) * 2:
            issues.append("too_many_keys")
        if len(a['key_sequence']) < len(a['target_word']) / 2:
            issues.append("too_few_keys")
        
        if issues:
            problems.append({
                'word': a['target_word'],
                'keys': a['key_sequence'],
                'issues': issues
            })
    
    print(f"Problematic swipes: {len(problems)}/{len(analyses)} ({100*len(problems)/len(analyses):.1f}%)")
    
    # Show some examples
    print("\nExample problems:")
    for p in problems[:10]:
        print(f"  Word: '{p['word']}' -> Keys: '{p['keys']}' | Issues: {', '.join(p['issues'])}")
    
    # Key sequence patterns
    print("\n=== KEY SEQUENCE PATTERNS ===")
    for a in analyses[:5]:
        print(f"Word: '{a['target_word']:10s}' -> Keys: '{a['key_sequence']}'")

if __name__ == '__main__':
    main()