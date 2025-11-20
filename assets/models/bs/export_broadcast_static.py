#!/usr/bin/env python3
"""
Broadcast-Enabled Static-Quant ONNX Export (Android-optimized)

- Broadcast decoder: accepts encoder memory with batch=1 and expands to num_beams.
- Dynamic causal mask sliced at runtime (no fixed 20x20 issue).
- Static INT8 quantization for Android (weights per-channel INT8, activations UINT8),
  using simple synthetic calibration sets (covers typical shapes).

Usage:
  python export_broadcast_static.py checkpoints/full_character_model_standalone_hwsfuto10/last.ckpt exported_broadcast --opset 17

Notes:
  - If you have a small calibration manifest and featurizer, you can extend
    the CalibrationDataReader to use real data. This script uses synthetic
    calibration covering common shape ranges to keep it self-contained.
"""

import os
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import onnx
from onnx import checker
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    CalibrationMethod,
)


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
        char_pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model

        self.traj_proj = nn.Linear(traj_dim, d_model // 2)
        self.kb_embedding = nn.Embedding(kb_vocab_size, d_model // 2, padding_idx=kb_pad_idx)
        self.encoder_norm = nn.LayerNorm(d_model)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.char_embedding = nn.Embedding(char_vocab_size, d_model, padding_idx=char_pad_idx)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.output_proj = nn.Linear(d_model, char_vocab_size)

        # for completeness in case of CTC aux; not used in export
        self.ctc_head = nn.Linear(d_model, char_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_trajectory(self, traj_features, nearest_keys, src_mask=None):
        seq_len = traj_features.shape[1]
        traj_enc = self.traj_proj(traj_features)
        kb_enc = self.kb_embedding(nearest_keys)
        combined = torch.cat([traj_enc, kb_enc], dim=-1)
        combined = self.encoder_norm(combined)
        combined = combined + self.pe[:, :seq_len, :]
        memory = self.encoder(combined, src_key_padding_mask=src_mask)
        return memory


def load_model_flexible(checkpoint_path: Path, args: argparse.Namespace) -> Tuple[nn.Module, Dict, float]:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {})

    def resolve(name, default):
        if hasattr(args, name) and getattr(args, name) is not None:
            return getattr(args, name)
        if name in cfg:
            try:
                return int(cfg[name])
            except Exception:
                try:
                    return float(cfg[name])
                except Exception:
                    return default
        return default

    tok = CharTokenizer()
    params = {
        'traj_dim': resolve('traj_dim', 6),
        'd_model': resolve('d_model', 256),
        'nhead': resolve('nhead', 8),
        'num_encoder_layers': resolve('num_encoder_layers', 6),
        'num_decoder_layers': resolve('num_decoder_layers', 4),
        'dim_feedforward': resolve('dim_feedforward', 1024),
        'max_seq_len': resolve('max_seq_len', 250),
        'dropout': 0.0,
        'kb_vocab_size': tok.vocab_size,
        'char_vocab_size': tok.vocab_size,
        'kb_pad_idx': tok.pad_idx,
        'char_pad_idx': tok.pad_idx,
    }
    model = CharacterLevelSwipeModel(**params)
    # Older checkpoints may not have ctc_head; load non-strict.
    missing = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    params['accuracy'] = ckpt.get('val_word_acc', 0.0)
    params['max_word_len'] = resolve('max_word_len', 20)
    return model, params, params['accuracy']


def export_encoder(model: CharacterLevelSwipeModel, path: Path, opset: int):
    class Enc(nn.Module):
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
        Enc(model),
        (dummy_traj, dummy_keys, dummy_len),
        path,
        input_names=['trajectory_features', 'nearest_keys', 'actual_length'],
        output_names=['encoder_output'],
        dynamic_axes={
            'trajectory_features': {0: 'batch'},
            'nearest_keys': {0: 'batch'},
            'actual_length': {0: 'batch'},
            'encoder_output': {0: 'batch'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    checker.check_model(str(path))


def export_decoder_broadcast(model: CharacterLevelSwipeModel, path: Path, opset: int):
    class Dec(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.d = m.d_model
            self.register_buffer('seq_range', torch.arange(self.m.pe.shape[1]))
            max_dec = int(self.m.pe.shape[1])
            full = torch.triu(torch.full((max_dec, max_dec), float('-inf')), diagonal=1)
            self.register_buffer('full_causal', full)
            self.pad_idx = int(m.char_embedding.padding_idx) if m.char_embedding.padding_idx is not None else 0
        def forward(self, mem, tokens, src_len):
            beams = tokens.size(0)
            T = tokens.size(1)
            enc_seq = mem.size(1)
            if mem.size(0) == 1 and beams > 1:
                mem = mem.expand(beams, -1, -1)
            causal = self.full_causal[:T, :T]
            seq = self.seq_range[:enc_seq]
            if src_len.dim() == 1 and src_len.size(0) == 1 and beams > 1:
                src_len_exp = src_len.expand(beams)
            else:
                src_len_exp = src_len
            src_mask = seq.unsqueeze(0) >= src_len_exp.unsqueeze(1)
            tgt_pad_mask = (tokens.long() == self.pad_idx)
            emb = self.m.char_embedding(tokens.long()) * math.sqrt(self.d)
            emb = emb + self.m.pe[:, :T, :]
            out = self.m.decoder(
                emb,
                mem,
                tgt_mask=causal,
                memory_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )
            return torch.log_softmax(self.m.output_proj(out), dim=-1)

    max_len = model.pe.shape[1]
    dummy_mem = torch.randn(1, max_len, model.d_model)
    dummy_tok = torch.randint(0, 30, (5, 20), dtype=torch.int32)  # beams=5, dec_seq=20
    dummy_len = torch.tensor([50], dtype=torch.int32)
    torch.onnx.export(
        Dec(model),
        (dummy_mem, dummy_tok, dummy_len),
        path,
        input_names=['memory', 'target_tokens', 'actual_src_length'],
        output_names=['log_probs'],
        dynamic_axes={
            'memory': {0: 'mem_batch', 1: 'enc_seq'},
            'target_tokens': {0: 'num_beams', 1: 'dec_seq'},
            'actual_src_length': {0: 'src_batch'},
            'log_probs': {0: 'num_beams', 1: 'dec_seq'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    checker.check_model(str(path))


class EncoderCalib(CalibrationDataReader):
    def __init__(self, max_seq_len: int, traj_dim: int, num_batches: int = 10, batch: int = 4):
        import numpy as np
        self.batch = batch
        self.it = 0
        self.num = num_batches
        self.msl = max_seq_len
        self.traj_dim = traj_dim
        self.np = np
    def get_next(self):
        if self.it >= self.num:
            return None
        self.it += 1
        L = int(self.np.random.choice([64, 96, 128, 160, 200], 1)[0])
        traj = self.np.random.randn(self.batch, self.msl, self.traj_dim).astype('float32') * 0.5
        keys = self.np.random.randint(0, 30, size=(self.batch, self.msl)).astype('int32')
        lengths = self.np.full((self.batch,), L, dtype='int32')
        return {
            'trajectory_features': traj,
            'nearest_keys': keys,
            'actual_length': lengths,
        }


class DecoderCalib(CalibrationDataReader):
    def __init__(self, d_model: int, max_seq_len: int, num_batches: int = 12, fixed_dec_seq: int = 20):
        import numpy as np
        self.it = 0
        self.num = num_batches
        self.d = d_model
        self.msl = max_seq_len
        self.fixed_dec_seq = fixed_dec_seq
        self.np = np
    def get_next(self):
        if self.it >= self.num:
            return None
        self.it += 1
        enc_seq = int(self.np.random.choice([80, 120, 160, 200], 1)[0])
        beams = int(self.np.random.choice([4, 6, 8], 1)[0])
        dec_seq = int(self.fixed_dec_seq)
        mem = self.np.random.randn(1, self.msl, self.d).astype('float32') * 0.6
        toks = self.np.random.randint(0, 30, size=(beams, dec_seq)).astype('int32')
        src_len = self.np.array([enc_seq], dtype='int32')
        return {
            'memory': mem,
            'target_tokens': toks,
            'actual_src_length': src_len,
        }


def simplify(path: Path):
    try:
        import onnxsim
        mdl = onnx.load(str(path))
        simp, ok = onnxsim.simplify(mdl)
        if ok:
            onnx.save(simp, str(path))
    except Exception:
        pass


def write_configs(out_dir: Path, config: Dict):
    tok = CharTokenizer()
    with open(out_dir / 'tokenizer_config.json', 'w') as f:
        json.dump({
            'vocab_size': tok.vocab_size,
            'special_tokens': {'pad': tok.pad_idx, 'sos': tok.sos_idx, 'eos': tok.eos_idx},
            'idx_to_char': tok.idx_to_char,
        }, f, indent=2)
    with open(out_dir / 'model_config.json', 'w') as f:
        json.dump({
            'meta': {
                'accuracy': f"{config.get('accuracy', 0.0):.4f}",
                'd_model': config.get('d_model'),
                'nhead': config.get('nhead'),
                'num_encoder_layers': config.get('num_encoder_layers'),
                'num_decoder_layers': config.get('num_decoder_layers'),
                'dim_feedforward': config.get('dim_feedforward'),
                'broadcast_enabled': True,
            },
            'limits': {
                'max_seq_len': config.get('max_seq_len', 250),
                'max_word_len': config.get('max_word_len', 20),
            },
            'inputs': {
                'encoder': ['trajectory_features (f32)', 'nearest_keys (i32)', 'actual_length (i32)'],
                'decoder': ['memory (f32)', 'target_tokens (i32)', 'actual_src_length (i32)'],
            },
            'outputs': {'decoder': 'log_probs (f32)'}
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description='Broadcast-enabled static-quant export (Android)')
    ap.add_argument('checkpoint', type=str)
    ap.add_argument('output_dir', type=str)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--d_model', type=int)
    ap.add_argument('--nhead', type=int)
    ap.add_argument('--num_encoder_layers', type=int)
    ap.add_argument('--num_decoder_layers', type=int)
    ap.add_argument('--dim_feedforward', type=int)
    ap.add_argument('--max_seq_len', type=int)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, config, _acc = load_model_flexible(ckpt, args)

    enc_p = out / 'encoder.onnx'
    dec_p = out / 'decoder.onnx'
    export_encoder(model, enc_p, args.opset)
    export_decoder_broadcast(model, dec_p, args.opset)

    simplify(enc_p)
    simplify(dec_p)

    # Static quantization (Android-friendly): weights per-channel INT8, activations UINT8
    e_q = out / 'swipe_encoder_android.onnx'
    d_q = out / 'swipe_decoder_android.onnx'

    enc_reader = EncoderCalib(max_seq_len=model.pe.shape[1], traj_dim=model.traj_proj.in_features)
    dec_reader = DecoderCalib(d_model=model.d_model, max_seq_len=model.pe.shape[1])

    quantize_static(
        str(enc_p), str(e_q),
        calibration_data_reader=enc_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    quantize_static(
        str(dec_p), str(d_q),
        calibration_data_reader=dec_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

    # Cleanup source ONNX and write configs
    write_configs(out, config)
    try:
        enc_p.unlink()
        dec_p.unlink()
    except Exception:
        pass

    print(f"Export complete. Files written to: {out}")


if __name__ == '__main__':
    main()
