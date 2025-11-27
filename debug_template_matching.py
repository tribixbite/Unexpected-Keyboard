#!/usr/bin/env python3
"""
Standalone Template Matching Debug Script
Generates templates and compares with your swipe data to fix template generation
"""

import math
import json
from typing import List, Tuple, Dict

# QWERTY layout coordinates (matching WordGestureTemplateGenerator.java)
QWERTY_COORDS = {
    # Top row
    'q': (100, 200), 'w': (200, 200), 'e': (300, 200), 'r': (400, 200), 't': (500, 200),
    'y': (600, 200), 'u': (700, 200), 'i': (800, 200), 'o': (900, 200), 'p': (950, 200),
    
    # Middle row  
    'a': (150, 500), 's': (250, 500), 'd': (350, 500), 'f': (450, 500), 'g': (550, 500),
    'h': (650, 500), 'j': (750, 500), 'k': (850, 500), 'l': (950, 500),
    
    # Bottom row
    'z': (200, 800), 'x': (300, 800), 'c': (400, 800), 'v': (500, 800), 'b': (600, 800),
    'n': (700, 800), 'm': (800, 800)
}

# Your actual swipe data from calibration
SWIPE_DATA = {
    'kick': {
        'user_points': 109,
        'user_start': (862, 247),
        'user_end': (933, 168),
        'user_length': 1397,
        'normalized_start': (862/1080*1000, 247/400*1000),  # Approximate normalization
        'normalized_end': (933/1080*1000, 168/400*1000)
    },
    'surgery': {
        'user_points': 207,
        'user_start': (215, 263),
        'user_end': (651, 95),
        'user_length': 2009
    },
    'market': {
        'user_points': 207,
        'user_start': (828, 394),
        'user_end': (597, 97),
        'user_length': 2598
    },
    'wonderful': {
        'user_points': 241,
        'user_start': (215, 252),
        'user_end': (290, 60),
        'user_length': 3937
    }
}

def generate_template(word: str) -> List[Tuple[float, float]]:
    """Generate template coordinates for a word"""
    points = []
    for char in word.lower():
        if char in QWERTY_COORDS:
            points.append(QWERTY_COORDS[char])
        else:
            print(f"Warning: No coordinate for character '{char}'")
            return None
    return points

def calculate_path_length(points: List[Tuple[float, float]]) -> float:
    """Calculate total path length"""
    if len(points) < 2:
        return 0
    
    length = 0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        length += math.sqrt(dx*dx + dy*dy)
    return length

def normalize_to_screen_space(template_points: List[Tuple[float, float]], 
                             screen_width: float, screen_height: float) -> List[Tuple[float, float]]:
    """Convert template coordinates to screen space for comparison"""
    normalized = []
    for x, y in template_points:
        screen_x = (x / 1000.0) * screen_width
        screen_y = (y / 1000.0) * screen_height
        normalized.append((screen_x, screen_y))
    return normalized

def analyze_word(word: str, swipe_data: Dict) -> Dict:
    """Analyze template vs user swipe for a word"""
    template_points = generate_template(word)
    if not template_points:
        return {'error': f'Cannot generate template for {word}'}
    
    template_length = calculate_path_length(template_points)
    template_start = template_points[0]
    template_end = template_points[-1]
    
    # Normalize template to screen space for comparison
    screen_template = normalize_to_screen_space(template_points, 1080, 400)  # Approximate screen size
    screen_length = calculate_path_length(screen_template)
    
    analysis = {
        'word': word,
        'template': {
            'points': len(template_points),
            'coordinates': template_points,
            'start': template_start,
            'end': template_end,
            'length': template_length,
            'screen_length': screen_length
        },
        'user': swipe_data,
        'comparison': {}
    }
    
    # Calculate coordinate differences
    if 'user_start' in swipe_data and 'user_end' in swipe_data:
        screen_start = screen_template[0]
        screen_end = screen_template[-1]
        
        start_diff_x = abs(screen_start[0] - swipe_data['user_start'][0])
        start_diff_y = abs(screen_start[1] - swipe_data['user_start'][1])
        end_diff_x = abs(screen_end[0] - swipe_data['user_end'][0])
        end_diff_y = abs(screen_end[1] - swipe_data['user_end'][1])
        
        analysis['comparison'] = {
            'start_diff': (start_diff_x, start_diff_y),
            'end_diff': (end_diff_x, end_diff_y),
            'length_ratio': screen_length / swipe_data['user_length'] if swipe_data['user_length'] > 0 else 0,
            'start_distance': math.sqrt(start_diff_x*start_diff_x + start_diff_y*start_diff_y),
            'end_distance': math.sqrt(end_diff_x*end_diff_x + end_diff_y*end_diff_y)
        }
    
    return analysis

def simulate_cgr_recognition(word: str, swipe_data: Dict) -> Dict:
    """Simulate CGR recognition with parameter analysis"""
    template_points = generate_template(word)
    if not template_points:
        return {'error': f'Cannot generate template for {word}'}
    
    # CGR Parameters (from original)
    DEFAULT_E_SIGMA = 200.0
    DEFAULT_BETA = 400.0
    DEFAULT_LAMBDA = 0.4
    DEFAULT_KAPPA = 1.0
    
    # Simulate distance calculations
    user_start = swipe_data.get('user_start', (0, 0))
    user_end = swipe_data.get('user_end', (0, 0))
    user_length = swipe_data.get('user_length', 0)
    
    # Convert to template coordinate space
    template_start = template_points[0]
    template_end = template_points[-1]
    template_length = calculate_path_length(template_points)
    
    # Normalize coordinates to 1000x1000 space
    normalized_user_start = (user_start[0] / 1080 * 1000, user_start[1] / 400 * 1000)
    normalized_user_end = (user_end[0] / 1080 * 1000, user_end[1] / 400 * 1000)
    
    # Calculate Euclidean distance (simplified)
    start_diff = math.sqrt((normalized_user_start[0] - template_start[0])**2 + 
                          (normalized_user_start[1] - template_start[1])**2)
    end_diff = math.sqrt((normalized_user_end[0] - template_end[0])**2 + 
                        (normalized_user_end[1] - template_end[1])**2)
    
    avg_euclidean_dist = (start_diff + end_diff) / 2
    
    # Simulate turning angle (simplified - would need actual path analysis)
    turning_angle_dist = abs(template_length - user_length) / max(template_length, user_length)
    
    # Calculate likelihood using CGR formula
    x_e = avg_euclidean_dist
    x_a = turning_angle_dist
    sigma_e = DEFAULT_E_SIGMA
    sigma_a = DEFAULT_E_SIGMA / DEFAULT_BETA
    
    likelihood = math.exp(-(x_e * x_e / (sigma_e * sigma_e) * DEFAULT_LAMBDA + 
                           x_a * x_a / (sigma_a * sigma_a) * (1 - DEFAULT_LAMBDA)))
    
    # Apply end-point bias
    complete_prob = likelihood  # Simplified
    x = 1 - complete_prob
    end_point_bias = 1 + DEFAULT_KAPPA * math.exp(-x * x)
    final_prob = end_point_bias * likelihood
    
    return {
        'word': word,
        'euclidean_distance': x_e,
        'turning_angle_distance': x_a,
        'likelihood': likelihood,
        'end_point_bias': end_point_bias,
        'final_probability': final_prob,
        'parameters': {
            'e_sigma': sigma_e,
            'a_sigma': sigma_a,
            'lambda': DEFAULT_LAMBDA,
            'kappa': DEFAULT_KAPPA
        }
    }

def main():
    print("=== CGR Parameter Analysis ===\n")
    
    # Analyze each word with CGR simulation
    for word, swipe_info in SWIPE_DATA.items():
        analysis = simulate_cgr_recognition(word, swipe_info)
        
        if 'error' in analysis:
            print(f"ERROR: {analysis['error']}")
            continue
            
        print(f"WORD: {word.upper()}")
        print(f"Euclidean distance: {analysis['euclidean_distance']:.2f}")
        print(f"Turning angle distance: {analysis['turning_angle_distance']:.4f}")
        print(f"Likelihood: {analysis['likelihood']:.8f}")
        print(f"End-point bias: {analysis['end_point_bias']:.4f}")
        print(f"Final probability: {analysis['final_probability']:.8f}")
        
        # Parameter analysis
        params = analysis['parameters']
        print(f"Parameters: σe={params['e_sigma']}, σa={params['a_sigma']:.2f}, λ={params['lambda']}, κ={params['kappa']}")
        
        # Diagnose issues
        if analysis['euclidean_distance'] > 500:
            print("⚠️ HIGH Euclidean distance - coordinate alignment issue")
        if analysis['likelihood'] < 1e-10:
            print("❌ EXTREMELY LOW likelihood - sigma parameters may be wrong")
        if analysis['final_probability'] < 1e-6:
            print("❌ ZERO final probability - CGR parameters need tuning")
        
        print("-" * 60)
    
    print("\n=== QWERTY Layout Analysis ===")
    print("Checking for layout issues...")
    
    # Check for duplicate coordinates
    coords_used = {}
    for char, coord in QWERTY_COORDS.items():
        if coord in coords_used:
            print(f"❌ DUPLICATE: {char} and {coords_used[coord]} both use {coord}")
        else:
            coords_used[coord] = char
    
    # Check coordinate ranges
    x_coords = [coord[0] for coord in QWERTY_COORDS.values()]
    y_coords = [coord[1] for coord in QWERTY_COORDS.values()]
    
    print(f"X range: {min(x_coords)} to {max(x_coords)}")
    print(f"Y range: {min(y_coords)} to {max(y_coords)}")
    
    if min(x_coords) < 0 or min(y_coords) < 0:
        print("❌ NEGATIVE COORDINATES FOUND")
    
    if max(x_coords) > 1000 or max(y_coords) > 1000:
        print("❌ COORDINATES EXCEED 1000x1000 SPACE")

if __name__ == "__main__":
    main()