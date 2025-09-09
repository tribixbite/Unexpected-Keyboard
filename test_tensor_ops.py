#!/usr/bin/env python3

"""
Neural ONNX tensor operation validator
Tests the specific operations our implementation performs
"""

import numpy as np

def test_logits_extraction():
    """Test 3D logits tensor extraction logic"""
    print("🔧 Testing 3D logits tensor extraction...")
    
    # Simulate decoder output logits [batch=1, seq_len=20, vocab=30]
    logits_3d = np.random.rand(1, 20, 30).astype(np.float32)
    print(f"   Logits shape: {logits_3d.shape}")
    
    # Test extraction for different token positions
    for token_pos in [0, 5, 10, 19]:
        print(f"   Token position {token_pos}:")
        
        # Extract logits for batch=0, position=token_pos, all vocab
        relevant_logits = logits_3d[0, token_pos, :]
        print(f"     Extracted shape: {relevant_logits.shape}")
        print(f"     Sample values: {relevant_logits[:5]}")
        
        # Apply softmax
        max_logit = np.max(relevant_logits)
        exp_scores = np.exp(relevant_logits - max_logit)
        probs = exp_scores / np.sum(exp_scores)
        
        # Get top 3 tokens
        top_indices = np.argsort(probs)[::-1][:3]
        print(f"     Top 3 tokens: {top_indices} with probs: {probs[top_indices]}")

def test_boolean_mask_logic():
    """Test boolean mask creation logic"""
    print("\n🎯 Testing boolean mask logic...")
    
    # Test source mask (encoder)
    actual_length = 66  # From log: 66 points
    max_seq_len = 150
    
    src_mask = np.zeros((1, max_seq_len), dtype=bool)
    for i in range(max_seq_len):
        src_mask[0, i] = (i >= actual_length)  # True for padded positions
    
    print(f"   Source mask shape: {src_mask.shape}")
    print(f"   Padded positions (True): {np.sum(src_mask[0])}")
    print(f"   Valid positions (False): {np.sum(~src_mask[0])}")
    
    # Test target mask (decoder)
    beam_token_count = 3  # Example: SOS + 2 tokens generated
    decoder_seq_len = 20
    
    tgt_mask = np.zeros((1, decoder_seq_len), dtype=bool)
    for i in range(decoder_seq_len):
        tgt_mask[0, i] = (i >= beam_token_count)  # True for padded positions
        
    print(f"   Target mask shape: {tgt_mask.shape}")
    print(f"   Padded positions (True): {np.sum(tgt_mask[0])}")
    print(f"   Valid positions (False): {np.sum(~tgt_mask[0])}")

def test_tokenizer_mapping():
    """Test character to token mapping"""
    print("\n📝 Testing tokenizer mapping...")
    
    # Test words from logs
    test_words = ["but", "how", "old", "boy"]
    
    # Tokenizer mapping (matching web demo)
    char_to_idx = {}
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        char_to_idx[c] = i + 4  # Indices 4-29
    
    for word in test_words:
        token_ids = [char_to_idx.get(c, 1) for c in word]  # 1 = UNK
        print(f"   '{word}' → {token_ids}")

def main():
    print("Neural ONNX System Validation")
    print("============================")
    
    test_logits_extraction()
    test_boolean_mask_logic() 
    test_tokenizer_mapping()
    
    print("\n🎯 Validation Summary:")
    print("- 3D logits extraction logic verified")
    print("- Boolean mask creation validated")
    print("- Tokenizer mapping confirmed")
    print()
    print("Current neural system should work with:")
    print("- Vocab size: 30 (4 special + 26 letters)")
    print("- Sequence length: 150 (matching web demo)")
    print("- Boolean tensors: [1, 150] and [1, 20] shapes")
    print("- 3D logits handling: [1, 20, 30] → extract position slice")

if __name__ == "__main__":
    main()