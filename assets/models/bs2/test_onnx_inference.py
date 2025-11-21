#!/usr/bin/env python3
"""
Quick validator for exported ONNX encoder/decoder.

Runs greedy decoding on N examples from a JSONL dataset and prints
GT vs prediction. Useful to sanity-check static-quant/broadcast models.

Usage:
  python test_onnx_inference.py \
    --encoder exported_broadcast_prev/swipe_encoder_android.onnx \
    --decoder exported_broadcast_prev/swipe_decoder_android.onnx \
    --data data/train_hwsfuto.jsonl \
    --limit 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import onnxruntime as ort


class KeyboardGrid:
    def __init__(self):
        self.width = 1.0
        self.height = 1.0
        row_h = 1.0 / 3.0
        key_w = 1.0 / 10.0
        top = list("qwertyuiop")
        mid = list("asdfghjkl")
        bot = list("zxcvbnm")
        top_x0, mid_x0, bot_x0 = 0.0, 0.05, 0.15
        self.key_positions = {}
        def add_row(keys, y0, x0):
            for i, k in enumerate(keys):
                x = x0 + i * key_w
                y = y0
                w = key_w
                h = row_h
                cx, cy = x + w/2.0, y + h/2.0
                self.key_positions[k] = (cx, cy)
        add_row(top, 0.0 * row_h, top_x0)
        add_row(mid, 1.0 * row_h, mid_x0)
        add_row(bot, 2.0 * row_h, bot_x0)
        self.key_positions["<unk>"] = (0.5, 0.5)
        self.key_positions["<pad>"] = (0.0, 0.0)
        self._labels = [k for k in self.key_positions.keys() if k not in ("<unk>", "<pad>")]
        self._pos = np.array([self.key_positions[k] for k in self._labels], dtype=np.float32)

    def nearest_batch(self, xs: np.ndarray, ys: np.ndarray) -> List[str]:
        if xs.size == 0:
            return []
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        diff = coords[:, None, :] - self._pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        return [self._labels[i] for i in idx]


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

    def decode(self, ids: List[int]) -> str:
        out = []
        for t in ids:
            if t == self.sos_idx:
                continue
            if t == self.eos_idx or t == self.pad_idx:
                break
            out.append(self.idx_to_char.get(t, '?'))
        return ''.join(out)


def load_examples(path: str, limit: int) -> List[Dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(items) >= limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Normalize to x,y,t,word
            if 'curve' in obj and 'word' in obj:
                curve = obj['curve']
                if all(k in curve for k in ('x','y','t')):
                    items.append({'x': curve['x'], 'y': curve['y'], 't': curve['t'], 'word': obj['word']})
            elif 'word_seq' in obj:
                ws = obj['word_seq']
                if all(k in ws for k in ('x','y','time')):
                    items.append({'x': ws['x'], 'y': ws['y'], 't': ws['time'], 'word': obj.get('word','')})
            elif 'points' in obj and isinstance(obj['points'], list):
                xs = [p['x'] for p in obj['points']]
                ys = [p['y'] for p in obj['points']]
                ts = [p['t'] for p in obj['points']]
                items.append({'x': xs, 'y': ys, 't': ts, 'word': obj.get('word','')})
            elif 'x' in obj and 'y' in obj and 't' in obj and 'word' in obj:
                items.append({'x': obj['x'], 'y': obj['y'], 't': obj['t'], 'word': obj['word']})
    return items


def build_features(ex: Dict, kb: KeyboardGrid, tok: CharTokenizer, max_seq_len: int = 250) -> Tuple[np.ndarray, np.ndarray, int]:
    xs = np.asarray(ex['x'], dtype=np.float32)
    ys = np.asarray(ex['y'], dtype=np.float32)
    ts = np.asarray(ex['t'], dtype=np.float32)
    # clamp to [0,1]
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    # velocities/accelerations
    dt = np.diff(ts, prepend=ts[0])
    dt = np.maximum(dt, 1e-6)
    vx = np.zeros_like(xs)
    vy = np.zeros_like(ys)
    vx[1:] = np.diff(xs) / dt[1:]
    vy[1:] = np.diff(ys) / dt[1:]
    ax = np.zeros_like(xs)
    ay = np.zeros_like(ys)
    ax[1:] = np.diff(vx) / dt[1:]
    ay[1:] = np.diff(vy) / dt[1:]
    vx = np.clip(vx, -10, 10)
    vy = np.clip(vy, -10, 10)
    ax = np.clip(ax, -10, 10)
    ay = np.clip(ay, -10, 10)
    seq_len = int(min(len(xs), max_seq_len))
    # nearest keys
    near = kb.nearest_batch(xs[:seq_len], ys[:seq_len])
    near_ids = [tok.char_to_idx.get(k, tok.unk_idx) for k in near]
    # features
    feats = np.stack([xs[:seq_len], ys[:seq_len], vx[:seq_len], vy[:seq_len], ax[:seq_len], ay[:seq_len]], axis=1)
    if seq_len < max_seq_len:
        pad = max_seq_len - seq_len
        feats = np.pad(feats, ((0,pad),(0,0)), mode='constant')
        near_ids = near_ids + [tok.pad_idx]*pad
    return feats.astype(np.float32), np.asarray(near_ids, dtype=np.int32), seq_len


def greedy_decode(sess_dec: ort.InferenceSession, memory: np.ndarray, src_len: int, tok: CharTokenizer, max_out: int = 20) -> str:
    """Greedy decode with fixed dec_seq=20 to satisfy traced reshape constraints.
    Fills a (1,20) token array with pads and updates positions incrementally.
    """
    tokens = np.full((1, max_out), tok.pad_idx, dtype=np.int32)
    tokens[0, 0] = tok.sos_idx
    src_len_arr = np.array([src_len], dtype=np.int32)
    pos = 1
    while pos < max_out:
        out = sess_dec.run(None, {
            'memory': memory,
            'target_tokens': tokens,
            'actual_src_length': src_len_arr,
        })
        # Use the log-probs at current position-1 (the last filled index)
        last_logp = out[0][:, pos-1, :]  # (1,V)
        next_tok = int(np.argmax(last_logp, axis=-1)[0])
        tokens[0, pos] = next_tok
        pos += 1
        if next_tok == tok.eos_idx:
            break
    return tok.decode(tokens.flatten().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--decoder', required=True)
    ap.add_argument('--data', default='data/train_hwsfuto.jsonl')
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()

    # ORT sessions
    so = ort.SessionOptions()
    enc_sess = ort.InferenceSession(args.encoder, so, providers=['CPUExecutionProvider'])
    dec_sess = ort.InferenceSession(args.decoder, so, providers=['CPUExecutionProvider'])

    tok = CharTokenizer()
    kb = KeyboardGrid()

    items = load_examples(args.data, args.limit)
    if not items:
        print('No examples loaded from', args.data)
        return

    max_seq_len = 250
    correct = 0
    for i, ex in enumerate(items, 1):
        feats, near_ids, L = build_features(ex, kb, tok, max_seq_len=max_seq_len)
        # Encoder
        mem = enc_sess.run(None, {
            'trajectory_features': feats[np.newaxis, ...].astype(np.float32),
            'nearest_keys': near_ids[np.newaxis, ...].astype(np.int32),
            'actual_length': np.array([L], dtype=np.int32),
        })[0]
        # Decoder greedy
        pred = greedy_decode(dec_sess, mem.astype(np.float32), L, tok, max_out=20)
        gt = (ex.get('word') or '').lower()
        ok = pred == gt
        correct += int(ok)
        print(f"[{i:03d}] ok={ok} gt='{gt}' pred='{pred}'")

    print(f"Accuracy on {len(items)} samples: {correct/len(items):.2%}")


if __name__ == '__main__':
    main()
