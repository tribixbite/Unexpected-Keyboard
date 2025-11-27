#!/usr/bin/env python3
"""
Broadcast-Enabled ONNX Export Script for Batched Beam Search

This script exports the swipe typing model with explicit memory broadcasting support
for efficient batched inference. The key modification is that the decoder can accept
memory with batch=1 and expand it to match the target_tokens batch dimension (num_beams).

## Problem Solved
The standard decoder export expects memory and target_tokens to have the same batch size.
This causes errors during beam search:
  Input shape:{1,8,1,250}, requested shape:{5,8,-1,250}

This happens because:
1. Encoder processes input once -> memory shape [1, seq_len, d_model]
2. Beam search needs to evaluate multiple beams simultaneously -> target shape [num_beams, dec_seq]
3. The MultiheadAttention reshape fails when batch dimensions don't match

## Solution
This export script creates a decoder wrapper that:
1. Detects when memory has batch=1 and target has batch=num_beams
2. Explicitly broadcasts (expands) memory to match target batch dimension
3. Also broadcasts masks accordingly

## Usage
```bash
# Export with broadcasting support
python export_broadcast.py checkpoints/best.ckpt out_broadcast --targets android

# Verify models
python -c "import onnx; onnx.checker.check_model('out_broadcast/swipe_decoder_android.onnx'); print('OK')"
```

## Runtime Usage
With the exported model, beam search can:
1. Run encoder once: memory = encoder(input) -> [1, seq_len, d_model]
2. Run decoder with all beams: log_probs = decoder(memory, all_beam_tokens) -> [num_beams, dec_seq, vocab]

The decoder will internally expand memory from [1, seq, d_model] to [num_beams, seq, d_model].

## I/O Contract

**Encoder ONNX** (unchanged from standard export)
* Inputs:
  - trajectory_features: (B, max_seq_len, 6) float32
  - nearest_keys: (B, max_seq_len) int32
  - actual_length: (B,) int32
* Output:
  - encoder_output: (B, max_seq_len, d_model) float32

**Decoder ONNX** (with broadcast support)
* Inputs:
  - memory: (1, enc_seq, d_model) float32 - Single batch from encoder
  - target_tokens: (num_beams, dec_seq) int32 - Multiple beam hypotheses
  - actual_src_length: (1,) int32 - Source length (will be broadcast)
* Output:
  - log_probs: (num_beams, dec_seq, vocab) float32

Note: The decoder also works when memory batch matches target batch (no broadcast needed).
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import onnx
from onnx import checker
from onnxruntime.quantization import quantize_dynamic, QuantType

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BroadcastExporter")

# -----------------------------------------------------------------------------
# 1. MODEL DEFINITION (same as standard export)
# -----------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz")
        special = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.vocab = special + chars
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.pad_idx = self.char_to_idx["<pad>"]
        self.unk_idx = self.char_to_idx["<unk>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]
        self.vocab_size = len(self.vocab)

class CharacterLevelSwipeModel(nn.Module):
    def __init__(
        self,
        traj_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        kb_vocab_size: int = 30,
        char_vocab_size: int = 30,
        max_seq_len: int = 250,
        kb_pad_idx: int = 0,
        char_pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder
        self.traj_proj = nn.Linear(traj_dim, d_model // 2)
        self.kb_embedding = nn.Embedding(kb_vocab_size, d_model // 2, padding_idx=kb_pad_idx)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Positional Encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        self.char_embedding = nn.Embedding(char_vocab_size, d_model, padding_idx=char_pad_idx)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, char_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_trajectory(self, traj_features, nearest_keys, src_mask=None):
        batch_size, seq_len, _ = traj_features.shape
        traj_enc = self.traj_proj(traj_features)
        kb_enc = self.kb_embedding(nearest_keys)
        combined = torch.cat([traj_enc, kb_enc], dim=-1)
        combined = self.encoder_norm(combined)
        combined = combined + self.pe[:, :seq_len, :]
        memory = self.encoder(combined, src_key_padding_mask=src_mask)
        return memory

# -----------------------------------------------------------------------------
# 2. CONFIG LOADING LOGIC
# -----------------------------------------------------------------------------

def load_model_flexible(checkpoint_path: Path, cli_args: argparse.Namespace) -> Tuple[nn.Module, Dict, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    # Backward compatible load
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        logger.warning("torch.load failed with weights_only=False. Retrying legacy load.")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    ckpt_config = checkpoint.get('config', {})
    logger.info(f"Checkpoint config found: {ckpt_config}")

    def resolve(param_name, default):
        if hasattr(cli_args, param_name) and getattr(cli_args, param_name) is not None:
            return getattr(cli_args, param_name)
        if param_name in ckpt_config:
            return int(ckpt_config[param_name])
        return default

    tokenizer = CharTokenizer()

    model_params = {
        'traj_dim': resolve('traj_dim', 6),
        'd_model': resolve('d_model', 256),
        'nhead': resolve('nhead', 8),
        'num_encoder_layers': resolve('num_encoder_layers', 6),
        'num_decoder_layers': resolve('num_decoder_layers', 4),
        'dim_feedforward': resolve('dim_feedforward', 1024),
        'max_seq_len': resolve('max_seq_len', 250),
        'dropout': 0.0,
        'kb_vocab_size': tokenizer.vocab_size,
        'char_vocab_size': tokenizer.vocab_size,
        'kb_pad_idx': tokenizer.pad_idx,
        'char_pad_idx': tokenizer.pad_idx,
    }

    model = CharacterLevelSwipeModel(**model_params)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logger.error("Weight mismatch between Checkpoint and Code!")
        raise e

    model.eval()
    accuracy = checkpoint.get('val_word_acc', 0.0)

    full_config = model_params.copy()
    full_config['accuracy'] = accuracy
    full_config['max_word_len'] = resolve('max_word_len', 25)

    return model, full_config, accuracy

# -----------------------------------------------------------------------------
# 3. BROADCAST-ENABLED EXPORT WRAPPERS
# -----------------------------------------------------------------------------

def export_encoder(model: CharacterLevelSwipeModel, path: Path, opset: int):
    """Export encoder (unchanged from standard export)"""
    logger.info(f"Exporting Encoder -> {path}")

    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer('seq_range', torch.arange(self.m.pe.shape[1]))

        def forward(self, traj, keys, length):
            seq = self.seq_range[:traj.size(1)]
            mask = seq.unsqueeze(0) >= length.unsqueeze(1)
            return self.m.encode_trajectory(traj, keys.long(), mask)

    max_len = model.pe.shape[1]
    dummy_traj = torch.randn(1, max_len, model.traj_proj.in_features)
    dummy_keys = torch.randint(0, 30, (1, max_len), dtype=torch.int32)
    dummy_len = torch.tensor([50], dtype=torch.int32)

    torch.onnx.export(
        EncoderWrapper(model),
        (dummy_traj, dummy_keys, dummy_len),
        path,
        input_names=['trajectory_features', 'nearest_keys', 'actual_length'],
        output_names=['encoder_output'],
        dynamic_axes={
            'trajectory_features': {0: 'batch'},
            'nearest_keys': {0: 'batch'},
            'actual_length': {0: 'batch'},
            'encoder_output': {0: 'batch'}
        },
        opset_version=opset,
        do_constant_folding=True
    )

    checker.check_model(str(path))
    logger.info("  Encoder integrity check passed")


def export_decoder_with_broadcast(model: CharacterLevelSwipeModel, path: Path, opset: int):
    """
    Export decoder with explicit memory broadcasting support.

    This wrapper handles the case where:
    - memory has shape [1, enc_seq, d_model] (single encoder output)
    - target_tokens has shape [num_beams, dec_seq] (multiple beam hypotheses)

    The memory tensor is explicitly expanded to [num_beams, enc_seq, d_model]
    to match the batch dimension of target_tokens.
    """
    logger.info(f"Exporting Decoder with broadcast support -> {path}")

    class BroadcastDecoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.d = m.d_model
            self.register_buffer('seq_range', torch.arange(self.m.pe.shape[1]))
            self.pad_idx = int(m.char_embedding.padding_idx) if m.char_embedding.padding_idx is not None else 0

        def forward(self, mem, tokens, src_len):
            """
            Forward pass with memory broadcasting.

            Args:
                mem: Encoder memory, shape [mem_batch, enc_seq, d_model]
                     For batched beam search, mem_batch is typically 1
                tokens: Target tokens, shape [num_beams, dec_seq]
                src_len: Source lengths, shape [src_batch,]
                         For batched beam search, src_batch is typically 1

            Returns:
                log_probs: Shape [num_beams, dec_seq, vocab_size]
            """
            mem_batch = mem.shape[0]
            num_beams = tokens.shape[0]
            T = tokens.shape[1]
            enc_seq = mem.shape[1]

            # CRITICAL: Broadcast memory from [1, enc_seq, d_model] to [num_beams, enc_seq, d_model]
            # This allows a single encoder pass to be used with multiple beam hypotheses
            if mem_batch == 1 and num_beams > 1:
                # expand() creates a view with the new shape, repeating data along dim 0
                # This is memory-efficient as it doesn't copy data
                mem = mem.expand(num_beams, enc_seq, self.d)

            # 1. Causal Mask (Target side) - same for all batches
            causal = nn.Transformer.generate_square_subsequent_mask(T).to(mem.device)

            # 2. Padding Mask (Source/Memory side)
            # Need to broadcast src_len if it has batch=1 but we have num_beams
            seq = self.seq_range[:enc_seq]

            if src_len.shape[0] == 1 and num_beams > 1:
                # Expand src_len from [1] to [num_beams] for mask creation
                src_len_expanded = src_len.expand(num_beams)
            else:
                src_len_expanded = src_len

            # Create memory key padding mask: [num_beams, enc_seq]
            # True where position >= actual length (i.e., padding positions)
            src_mask = seq.unsqueeze(0) >= src_len_expanded.unsqueeze(1)

            # 3. Padding Mask (Target side)
            tgt_pad_mask = (tokens.long() == self.pad_idx)

            # Embed target tokens with positional encoding
            emb = self.m.char_embedding(tokens.long()) * math.sqrt(self.d)
            emb = emb + self.m.pe[:, :T, :]

            # Run decoder with expanded memory
            out = self.m.decoder(
                emb,
                mem,
                tgt_mask=causal,
                memory_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
            return torch.log_softmax(self.m.output_proj(out), dim=-1)

    # Use batch=1 for memory to demonstrate broadcast capability
    # The dynamic axes allow any batch size at runtime
    max_len = model.pe.shape[1]
    dummy_mem = torch.randn(1, max_len, model.d_model)  # batch=1 for memory
    dummy_tok = torch.randint(0, 30, (5, 20), dtype=torch.int32)  # batch=5 for beams
    dummy_len = torch.tensor([50], dtype=torch.int32)  # batch=1 for length

    torch.onnx.export(
        BroadcastDecoderWrapper(model),
        (dummy_mem, dummy_tok, dummy_len),
        path,
        input_names=['memory', 'target_tokens', 'actual_src_length'],
        output_names=['log_probs'],
        dynamic_axes={
            # Memory can have batch=1 (will be broadcast) or batch=num_beams
            'memory': {0: 'mem_batch', 1: 'enc_seq'},
            # Target tokens have batch=num_beams
            'target_tokens': {0: 'num_beams', 1: 'dec_seq'},
            # Source length can have batch=1 (will be broadcast) or batch=num_beams
            'actual_src_length': {0: 'src_batch'},
            # Output has batch=num_beams
            'log_probs': {0: 'num_beams', 1: 'dec_seq'}
        },
        opset_version=opset,
        do_constant_folding=True
    )

    checker.check_model(str(path))
    logger.info("  Decoder (broadcast) integrity check passed")

# -----------------------------------------------------------------------------
# 4. UTILS (same as standard export)
# -----------------------------------------------------------------------------

def simplify(path: Path):
    try:
        import onnxsim
        logger.info(f"Simplifying {path.name}...")
        model = onnx.load(str(path))
        model_simp, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simp, str(path))
    except ImportError:
        logger.warning("onnxsim not found. Skipping simplification.")

def quantize(path: Path, out_path: Path, target: str):
    logger.info(f"Quantizing {path.name} -> {target}")

    w_type = QuantType.QUInt8 if target == "web" else QuantType.QInt8
    per_channel = (target == "android")
    reduce_range = True if target != "web" else False

    quantize_dynamic(
        str(path), str(out_path),
        weight_type=w_type,
        per_channel=per_channel,
        reduce_range=reduce_range
    )
    return os.path.getsize(out_path) / (1024*1024)

def write_configs(out_dir: Path, config: Dict):
    tok = CharTokenizer()
    t_conf = {
        'vocab_size': tok.vocab_size,
        'special_tokens': {'pad': tok.pad_idx, 'sos': tok.sos_idx, 'eos': tok.eos_idx},
        'idx_to_char': tok.idx_to_char
    }
    with open(out_dir / 'tokenizer_config.json', 'w') as f:
        json.dump(t_conf, f, indent=2)

    m_conf = {
        'meta': {
            'accuracy': f"{config['accuracy']:.4f}",
            'd_model': config['d_model'],
            'nhead': config.get('nhead'),
            'num_encoder_layers': config.get('num_encoder_layers'),
            'num_decoder_layers': config.get('num_decoder_layers'),
            'dim_feedforward': config.get('dim_feedforward'),
            'broadcast_enabled': True  # Indicate this model supports broadcasting
        },
        'limits': {
            'max_seq_len': config.get('max_seq_len', 250),
            'max_word_len': config.get('max_word_len', 25)
        },
        'inputs': {
            'encoder': ['trajectory_features (f32)', 'nearest_keys (i32)', 'actual_length (i32)'],
            'decoder': [
                'memory (f32) - can be batch=1, will broadcast to num_beams',
                'target_tokens (i32) - batch=num_beams',
                'actual_src_length (i32) - can be batch=1, will broadcast'
            ]
        },
        'outputs': {'decoder': 'log_probs (f32) - batch=num_beams'}
    }
    with open(out_dir / 'model_config.json', 'w') as f:
        json.dump(m_conf, f, indent=2)

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Swipe Model Exporter with Broadcast Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export for Android with broadcast support
  python export_broadcast.py checkpoints/best.ckpt out_broadcast --targets android

  # Export for both web and android
  python export_broadcast.py checkpoints/best.ckpt out_broadcast --targets web android

  # Verify exported model supports batched inference
  python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('out_broadcast/swipe_decoder_android.onnx')

# Test broadcast: memory batch=1, tokens batch=5
mem = np.random.randn(1, 100, 256).astype('float32')  # batch=1
tok = np.random.randint(0, 30, (5, 10)).astype('int32')  # batch=5
src_len = np.array([50], dtype='int32')  # batch=1

result = sess.run(None, {
    'memory': mem,
    'target_tokens': tok,
    'actual_src_length': src_len
})
print(f'Output shape: {result[0].shape}')  # Should be (5, 10, 30)
print('Broadcast test passed!')
"
        """
    )
    parser.add_argument('checkpoint', type=str, help="Path to .ckpt")
    parser.add_argument('output_dir', type=str, help="Target folder")
    parser.add_argument('--targets', nargs='+', default=['android'], choices=['web', 'android'])
    parser.add_argument('--opset', type=int, default=17, help="ONNX Opset Version")

    group = parser.add_argument_group('Model Architecture Overrides')
    group.add_argument('--d_model', type=int, help="Override d_model")
    group.add_argument('--nhead', type=int, help="Override nhead")
    group.add_argument('--num_encoder_layers', type=int, help="Override encoder layers")
    group.add_argument('--num_decoder_layers', type=int, help="Override decoder layers")
    group.add_argument('--dim_feedforward', type=int, help="Override feedforward dim")
    group.add_argument('--max_seq_len', type=int, help="Override max sequence length")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    model, config, acc = load_model_flexible(ckpt_path, args)

    # 2. Export with broadcast support
    enc_p = out_dir / "encoder.onnx"
    dec_p = out_dir / "decoder.onnx"
    export_encoder(model, enc_p, args.opset)
    export_decoder_with_broadcast(model, dec_p, args.opset)

    # 3. Simplify
    simplify(enc_p)
    simplify(dec_p)

    # 4. Quantize
    print("-" * 40)
    for t in args.targets:
        e_q = out_dir / f"swipe_encoder_{t}.onnx"
        d_q = out_dir / f"swipe_decoder_{t}.onnx"
        sz_e = quantize(enc_p, e_q, t)
        sz_d = quantize(dec_p, d_q, t)
        print(f"  {t.upper()} Total: {sz_e + sz_d:.2f} MB")
    print("-" * 40)

    # 5. Cleanup & Config
    write_configs(out_dir, config)
    enc_p.unlink()
    dec_p.unlink()

    logger.info(f"Done. Output at: {out_dir}")
    logger.info("Models exported with broadcast support for batched beam search.")

if __name__ == "__main__":
    main()
