#!/usr/bin/env python3

"""
Standalone CLI test for ONNX neural transformer models
Tests exact web demo flow to validate transformer pipeline
"""

import onnxruntime as ort
import numpy as np
import sys
import os

def setup_models():
    """Load ONNX models and create inference sessions"""
    encoder_path = "assets/models/swipe_encoder_android.onnx"
    decoder_path = "assets/models/swipe_decoder_android.onnx"
    
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder model not found: {encoder_path}")
        return None, None
        
    if not os.path.exists(decoder_path):
        print(f"❌ Decoder model not found: {decoder_path}")
        return None, None
    
    print("🔄 Loading ONNX models...")
    
    # Create inference sessions
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)
    
    print(f"✅ Encoder loaded")
    print(f"   Input names: {[inp.name for inp in encoder_session.get_inputs()]}")
    print(f"   Input shapes: {[inp.shape for inp in encoder_session.get_inputs()]}")
    print(f"   Input types: {[inp.type for inp in encoder_session.get_inputs()]}")
    
    print(f"✅ Decoder loaded")
    print(f"   Input names: {[inp.name for inp in decoder_session.get_inputs()]}")
    print(f"   Input shapes: {[inp.shape for inp in decoder_session.get_inputs()]}")
    print(f"   Input types: {[inp.type for inp in decoder_session.get_inputs()]}")
    
    return encoder_session, decoder_session

def create_qwerty_layout():
    """Create QWERTY keyboard layout (normalized 0-1)"""
    layout = {}

    # Row 1 (q-p): y=0.25
    row1 = "qwertyuiop"
    for i, char in enumerate(row1):
        layout[char] = (0.05 + i * 0.09, 0.25)

    # Row 2 (a-l): y=0.50, offset by 0.045
    row2 = "asdfghjkl"
    for i, char in enumerate(row2):
        layout[char] = (0.095 + i * 0.09, 0.50)

    # Row 3 (z-m): y=0.75, offset by 0.135
    row3 = "zxcvbnm"
    for i, char in enumerate(row3):
        layout[char] = (0.185 + i * 0.09, 0.75)

    return layout

def find_nearest_key(x, y, layout):
    """Find nearest key to point (x, y)"""
    min_dist = float('inf')
    nearest = 'a'  # default

    for char, (kx, ky) in layout.items():
        dist = (x - kx)**2 + (y - ky)**2
        if dist < min_dist:
            min_dist = dist
            nearest = char

    return nearest

def create_test_swipe(word="the"):
    """Create realistic swipe for a given word"""
    layout = create_qwerty_layout()
    test_points = []

    key_positions = [layout[c] for c in word]

    # Create smooth continuous path through ALL keys
    for i in range(len(key_positions) - 1):
        start_x, start_y = key_positions[i]
        end_x, end_y = key_positions[i + 1]

        # Interpolate between consecutive keys (40 points between each - much denser)
        for j in range(40):
            t = j / 40.0
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)

            # Find nearest key for this interpolated position
            nearest = find_nearest_key(x, y, layout)
            test_points.append((x, y, nearest))

    # Add final key position with more points
    final_x, final_y = key_positions[-1]
    for j in range(20):  # More points on the final key
        x = final_x + (j - 10) * 0.002
        y = final_y + (j - 10) * 0.002
        nearest = find_nearest_key(x, y, layout)
        test_points.append((x, y, nearest))

    return test_points

def create_encoder_inputs(test_points):
    """Create encoder input tensors for v4 model (250 seq len)"""
    MAX_SEQUENCE_LENGTH = 250
    
    # Initialize arrays
    trajectory_data = np.zeros((MAX_SEQUENCE_LENGTH, 6), dtype=np.float32)
    nearest_keys_data = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int32)
    
    # Character to index mapping (web demo style)
    char_to_idx = {chr(ord('a') + i): i + 4 for i in range(26)}
    char_to_idx.update({'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3})
    
    actual_length = len(test_points)
    
    for i in range(MAX_SEQUENCE_LENGTH):
        if i < len(test_points):
            x, y, key = test_points[i]
            
            # Position
            trajectory_data[i, 0] = x
            trajectory_data[i, 1] = y
            
            # Velocity
            if i > 0:
                prev_x, prev_y, _ = test_points[i-1] 
                trajectory_data[i, 2] = x - prev_x  # vx
                trajectory_data[i, 3] = y - prev_y  # vy
            
            # Acceleration  
            if i > 1:
                trajectory_data[i, 4] = trajectory_data[i, 2] - trajectory_data[i-1, 2]  # ax
                trajectory_data[i, 5] = trajectory_data[i, 3] - trajectory_data[i-1, 3]  # ay
            
            # Nearest key
            nearest_keys_data[i] = char_to_idx.get(key, 1)  # 1 = UNK
    
    print(f"📊 Encoder inputs created:")
    print(f"   Trajectory: shape {trajectory_data.shape}, actual_length: {actual_length}")
    print(f"   Nearest keys: shape {nearest_keys_data.shape}")
    print(f"   First 10 nearest keys: {nearest_keys_data[:10]}")

    # Decode nearest keys to show what characters
    idx_to_char = {i + 4: chr(ord('a') + i) for i in range(26)}
    idx_to_char.update({0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'})
    decoded = [idx_to_char.get(int(k), '?') for k in nearest_keys_data[:10]]
    print(f"   Decoded: {decoded}")
    
    return {
        'trajectory_features': trajectory_data.reshape(1, MAX_SEQUENCE_LENGTH, 6),
        'nearest_keys': nearest_keys_data.reshape(1, MAX_SEQUENCE_LENGTH),
        'actual_length': np.array([actual_length], dtype=np.int32)
    }

def run_encoder(encoder_session, inputs):
    """Run encoder inference"""
    print("\n🚀 Running encoder inference...")
    
    try:
        encoder_output = encoder_session.run(None, inputs)
        memory = encoder_output[0]  # First output is memory
        
        print(f"✅ Encoder successful - Memory shape: {memory.shape}")
        return memory
        
    except Exception as e:
        print(f"💥 Encoder failed: {e}")
        return None

def run_decoder_beam_search(decoder_session, memory, actual_src_length):
    """Run decoder with beam search using v4 interface"""
    print("\n🔍 Running decoder beam search...")

    # Constants
    DECODER_SEQ_LENGTH = 20
    VOCAB_SIZE = 30
    BEAM_WIDTH = 8
    MAX_LENGTH = 35

    # Special tokens
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

    # Initialize beams
    beams = [{'tokens': [SOS_IDX], 'score': 0.0, 'finished': False}]

    for step in range(MAX_LENGTH):
        candidates = []

        for beam in beams:
            if beam['finished']:
                candidates.append(beam)
                continue

            # Prepare decoder inputs
            padded_tokens = np.zeros(DECODER_SEQ_LENGTH, dtype=np.int32)

            # Copy beam tokens
            beam_len = len(beam['tokens'])
            for i in range(min(beam_len, DECODER_SEQ_LENGTH)):
                padded_tokens[i] = beam['tokens'][i]

            try:
                # Run decoder with v4 interface
                decoder_inputs = {
                    'memory': memory,
                    'target_tokens': padded_tokens.reshape(1, DECODER_SEQ_LENGTH),
                    'actual_src_length': actual_src_length
                }

                decoder_output = decoder_session.run(None, decoder_inputs)
                logits = decoder_output[0]  # Shape: [1, 20, 30]
                
                # Extract logits for next token position
                token_position = min(beam_len - 1, DECODER_SEQ_LENGTH - 1)
                relevant_logits = logits[0, token_position, :]
                
                # Apply softmax
                probs = softmax(relevant_logits)

                # Get top k
                top_k = np.argsort(probs)[::-1][:BEAM_WIDTH]

                for idx in top_k:
                    new_beam = {
                        'tokens': beam['tokens'] + [int(idx)],
                        'score': beam['score'] + np.log(probs[idx]),
                        'finished': (idx == EOS_IDX)
                    }
                    candidates.append(new_beam)
                    
            except Exception as e:
                print(f"💥 Decoder step failed: {e}")
                continue
        
        # Select top beams
        candidates.sort(key=lambda x: x['score'], reverse=True)
        beams = candidates[:BEAM_WIDTH]
        
        # Check termination
        finished_beams = [b for b in beams if b['finished']]
        if len(finished_beams) >= 3 or step >= 10:
            break
    
    return beams

def softmax(logits):
    """Apply softmax to logits"""
    logits = np.array(logits)
    max_logit = np.max(logits)
    exp_scores = np.exp(logits - max_logit)
    return exp_scores / np.sum(exp_scores)

def beams_to_words(beams):
    """Convert beam tokens to words"""
    # Character mapping
    idx_to_char = {i + 4: chr(ord('a') + i) for i in range(26)}
    idx_to_char.update({0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'})
    
    words = []
    for beam in beams:
        chars = []
        for token in beam['tokens']:
            if token in [0, 1, 2, 3]:  # Skip special tokens
                continue
            char = idx_to_char.get(token, '?')
            if not char.startswith('<'):
                chars.append(char)
        
        word = ''.join(chars)
        if len(word) > 0:
            words.append((word, beam['score']))
    
    return words

def main():
    print("🧪 Standalone ONNX Neural Test")
    print("==============================")
    
    # Setup models
    encoder_session, decoder_session = setup_models()
    if not encoder_session or not decoder_session:
        print("❌ Failed to load models")
        sys.exit(1)
    
    # --- START ENCODER DIAGNOSTIC ---
    print("\n--- Running Encoder Diagnostic ---")

    # 1. Generate and run for "the"
    print("\n1. Processing swipe for 'the'...")
    points_the = create_test_swipe(word="the")
    inputs_the = create_encoder_inputs(points_the)
    memory_the = run_encoder(encoder_session, inputs_the)

    # 2. Generate and run for "sad"
    print("\n2. Processing swipe for 'sad'...")
    points_sad = create_test_swipe(word="sad")
    inputs_sad = create_encoder_inputs(points_sad)
    memory_sad = run_encoder(encoder_session, inputs_sad)

    # 3. Compare memory tensors
    if memory_the is not None and memory_sad is not None:
        if np.array_equal(memory_the, memory_sad):
            print("\n\n❌ DIAGNOSIS: Encoder output is IDENTICAL for different inputs.")
            print("   This points to a problem with the encoder.onnx model itself.")
            sys.exit(1)
        else:
            print("\n\n✅ DIAGNOSIS: Encoder output is DIFFERENT for different inputs.")
            print(f"   Mean absolute difference: {np.mean(np.abs(memory_the - memory_sad)):.6f}")
            print("   The problem is likely within the decoder or its inputs.")
    else:
        print("❌ Encoder run failed, cannot perform diagnostic.")
        sys.exit(1)

    # Proceed with decoding using one of the results
    memory = memory_the
    actual_src_length = inputs_the['actual_length']
    # --- END ENCODER DIAGNOSTIC ---

    # Run decoder
    beams = run_decoder_beam_search(decoder_session, memory, actual_src_length)
    
    # Convert to words
    words = beams_to_words(beams)
    
    print(f"\n🎯 Final predictions:")
    for i, (word, score) in enumerate(words[:5]):
        print(f"   {i+1}. {word} (score: {score:.3f})")
    
    if words:
        print(f"✅ Neural prediction successful! Got {len(words)} predictions")
        print(f"   Best prediction: '{words[0][0]}'")
    else:
        print("❌ No predictions generated")
    
    return len(words) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)