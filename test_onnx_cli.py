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
    encoder_path = "assets/models/swipe_model_character_quant.onnx"
    decoder_path = "assets/models/swipe_decoder_character_quant.onnx"
    
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
    
    print(f"✅ Encoder loaded - Inputs: {encoder_session.get_inputs()}")
    print(f"   Input names: {[inp.name for inp in encoder_session.get_inputs()]}")
    print(f"   Input shapes: {[inp.shape for inp in encoder_session.get_inputs()]}")
    
    print(f"✅ Decoder loaded - Inputs: {decoder_session.get_inputs()}")
    print(f"   Input names: {[inp.name for inp in decoder_session.get_inputs()]}")
    print(f"   Input shapes: {[inp.shape for inp in decoder_session.get_inputs()]}")
    
    return encoder_session, decoder_session

def create_test_swipe():
    """Create test swipe data matching web demo format"""
    MAX_SEQUENCE_LENGTH = 150
    
    # Create simple horizontal swipe for word "hello"
    test_points = []
    
    # Simulate 5 key positions for h-e-l-l-o
    key_positions = [
        (0.3, 0.6),  # h
        (0.4, 0.3),  # e  
        (0.7, 0.6),  # l
        (0.7, 0.6),  # l (same position)
        (0.9, 0.3)   # o
    ]
    
    # Create smooth path between keys
    for i in range(len(key_positions) - 1):
        start_x, start_y = key_positions[i]
        end_x, end_y = key_positions[i + 1]
        
        # Interpolate between keys (10 points each)
        for j in range(10):
            t = j / 10.0
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            test_points.append((x, y, chr(ord('h') + i)))  # Associate with key
    
    return test_points[:66]  # Match log: 66 points

def create_encoder_inputs(test_points):
    """Create encoder input tensors exactly like web demo"""
    MAX_SEQUENCE_LENGTH = 150
    
    # Initialize arrays
    trajectory_data = np.zeros((MAX_SEQUENCE_LENGTH, 6), dtype=np.float32)
    nearest_keys_data = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
    src_mask_data = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint8)
    
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
        
        # Source mask: 1 for padded, 0 for real
        src_mask_data[i] = 1 if i >= actual_length else 0
    
    print(f"📊 Encoder inputs created:")
    print(f"   Trajectory: shape {trajectory_data.shape}, actual_length: {actual_length}")
    print(f"   Nearest keys: shape {nearest_keys_data.shape}")
    print(f"   Source mask: shape {src_mask_data.shape}, padded: {np.sum(src_mask_data)}")
    
    return {
        'trajectory_features': trajectory_data.reshape(1, MAX_SEQUENCE_LENGTH, 6),
        'nearest_keys': nearest_keys_data.reshape(1, MAX_SEQUENCE_LENGTH),
        'src_mask': src_mask_data.reshape(1, MAX_SEQUENCE_LENGTH).astype(bool)
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

def run_decoder_beam_search(decoder_session, memory, src_mask):
    """Run decoder with beam search exactly like web demo"""
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
            padded_tokens = np.zeros(DECODER_SEQ_LENGTH, dtype=np.int64)
            tgt_mask = np.zeros(DECODER_SEQ_LENGTH, dtype=np.uint8)
            
            # Copy beam tokens and create mask
            beam_len = len(beam['tokens'])
            for i in range(min(beam_len, DECODER_SEQ_LENGTH)):
                padded_tokens[i] = beam['tokens'][i]
            for i in range(beam_len, DECODER_SEQ_LENGTH):
                tgt_mask[i] = 1  # Mark padded positions
            
            # Create source mask (all zeros like web demo)
            src_mask_decoder = np.zeros(memory.shape[1], dtype=np.uint8)
            
            try:
                # Run decoder
                decoder_inputs = {
                    'memory': memory,
                    'target_tokens': padded_tokens.reshape(1, DECODER_SEQ_LENGTH),
                    'target_mask': tgt_mask.reshape(1, DECODER_SEQ_LENGTH).astype(bool),
                    'src_mask': src_mask_decoder.reshape(1, -1).astype(bool)
                }
                
                decoder_output = decoder_session.run(None, decoder_inputs)
                logits = decoder_output[0]  # Shape: [1, 20, 30]
                
                print(f"   Decoder output logits shape: {logits.shape}")
                
                # Extract logits for next token position (web demo style)
                token_position = min(beam_len - 1, DECODER_SEQ_LENGTH - 1)
                # Flatten and slice like web demo
                logits_flat = logits.flatten()
                start_idx = token_position * VOCAB_SIZE  
                end_idx = start_idx + VOCAB_SIZE
                relevant_logits = logits_flat[start_idx:end_idx]
                
                print(f"   Token position: {token_position}, extracted logits: {len(relevant_logits)}")
                
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
    
    # Create test data
    test_points = create_test_swipe()
    print(f"\n📝 Created test swipe: {len(test_points)} points")
    
    # Create encoder inputs
    encoder_inputs = create_encoder_inputs(test_points)
    
    # Run encoder
    memory = run_encoder(encoder_session, encoder_inputs)
    if memory is None:
        print("❌ Encoder failed")
        sys.exit(1)
    
    # Run decoder
    beams = run_decoder_beam_search(decoder_session, memory, encoder_inputs['src_mask'])
    
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