# Broadcast-Enabled Android ONNX Export (Calibrated)

This folder contains an Android-optimized encoder/decoder pair for swipe typing:

- `swipe_encoder_android.onnx` — static INT8 (per-channel) with real-data calibration
- `swipe_decoder_android.onnx` — dynamic (weights-only) INT8, broadcast-enabled
- `model_config.json` — model metadata and I/O contract
- `tokenizer_config.json` — tokenizer indices and special tokens

The decoder supports memory broadcasting and expects a fixed decode length of 20. The models are drop-in compatible with your existing Android integration.

---

## I/O Contract

Encoder ONNX
- Inputs:
  - `trajectory_features`: float32 `[B, max_seq_len, 6]`
  - `nearest_keys`: int32 `[B, max_seq_len]`
  - `actual_length`: int32 `[B]` (unpadded source lengths)
- Output:
  - `encoder_output`: float32 `[B, max_seq_len, d_model]`

Decoder ONNX (broadcast-enabled)
- Inputs:
  - `memory`: float32 `[mem_batch, enc_seq, d_model]` (typically `mem_batch=1`)
  - `target_tokens`: int32 `[num_beams, dec_seq]` (decoder inputs, dec_seq fixed to 20)
  - `actual_src_length`: int32 `[src_batch]` (typically `src_batch=1`)
- Output:
  - `log_probs`: float32 `[num_beams, dec_seq, vocab]`

Broadcast Rules
- If `memory` has batch=1 and `target_tokens` has batch=`num_beams`, the decoder internally expands `memory` to `[num_beams, enc_seq, d_model]`.
- If `actual_src_length` has batch=1, it is also expanded to `num_beams`.

Masks
- Source key padding mask is computed inside the decoder from `actual_src_length`.
- Target pad mask is computed from `target_tokens` (pad index from `tokenizer_config.json`).

Fixed Decode Length
- The decoder graph was traced with `dec_seq=20`. Always pass a `[num_beams, 20]` token array.
- Fill unused positions with pad id; the decoder will ignore pad positions.

---

## Tokenizer

See `tokenizer_config.json`:
- `special_tokens.pad`, `special_tokens.sos`, `special_tokens.eos` — indices.
- `idx_to_char` — integer indices as JSON keys.

You must:
- Start decoder inputs with `<sos>` at position 0.
- Fill the remainder with pad tokens.
- Stop when `<eos>` is predicted (or after 20 steps).

---

## Feature Preparation (Match Training)

For each swipe trajectory:
1. Normalize `x` and `y` to `[0,1]` and clamp to `[0,1]`.
2. Compute `vx, vy, ax, ay` with safe time-step differences; clip each to `[-10,10]`.
3. Compute nearest keyboard key per point (QWERTY grid centers used during training) to build `nearest_keys`.
4. Pad/truncate to `max_seq_len` (250).
5. Provide `actual_length` as the unpadded trajectory length.

Structure:
- `trajectory_features[b] = [x, y, vx, vy, ax, ay]` (float32)
- `nearest_keys[b]` as int32 indices (use tokenizer indices for letters; unknowns map to `<unk>`)

---

## Greedy Decoding (Single Beam)

Pseudocode (Python/ONNX Runtime style):

```
# Encode once per swipe
mem = enc_session.run(None, {
  'trajectory_features': feats[np.newaxis, ...].astype(np.float32),
  'nearest_keys': keys[np.newaxis, ...].astype(np.int32),
  'actual_length': np.array([L], dtype=np.int32),
})[0]  # shape [1, max_seq_len, d_model]

# Prepare fixed-length tokens (dec_seq=20)
T = 20
pad = tok.pad_idx
sos = tok.sos_idx

tokens = np.full((1, T), pad, dtype=np.int32)
tokens[0, 0] = sos
src_len = np.array([L], dtype=np.int32)

# Iteratively fill tokens[0, pos]
for pos in range(1, T):
  out = dec_session.run(None, {
    'memory': mem.astype(np.float32),
    'target_tokens': tokens,
    'actual_src_length': src_len,
  })
  next_logp = out[0][:, pos-1, :]  # last non-pad position
  next_tok = int(np.argmax(next_logp, axis=-1)[0])
  tokens[0, pos] = next_tok
  if next_tok == tok.eos_idx:
    break

# Convert tokens -> string (skip sos/pad, stop at eos)
word = decode(tokens[0], tok)
```

---

## Batched Beam Search (Broadcast)

Run beams together to minimize decoder invocations per step.

Setup
- `num_beams = K`
- `tokens`: int32 `[K, 20]`, each row starts with `<sos>` then pads.
- `memory`: float32 `[1, enc_seq, d_model]` (single encoder pass)
- `actual_src_length`: int32 `[1]`

Per-step loop
1. Call decoder once with current `tokens`:
   - `log_probs = dec(memory, tokens, actual_src_length)` → `[K, 20, V]`
2. For each beam, read `log_probs[b, pos-1, :]` to choose the next token (greedy or top-k expand your beam candidates).
3. Update `tokens[:, pos]` for all beams with their selected next tokens.
4. Stop beams on `<eos>`; continue others until `pos==19`.

Notes
- Keep padding in the unused token slots (positions > current step).
- Because the decoder slices an internal causal mask, you can pass the full `[K, 20]` tokens array each step and rely on the position index (`pos-1`) for the next-token distribution.

---

## Android Integration Tips

- Use a single `OrtSession` per model and reuse it for all swipes.
- Pre-allocate input buffers (direct byte buffers) and reuse them to avoid GC pressure.
- Keep `dec_seq=20` and pad remaining positions with pad id.
- Use broadcast: pass encoder `memory` batch=1; `target_tokens` batch=`num_beams`.
- Threads: start with `intra_op_num_threads = number_of_cores` and `inter_op_num_threads = 1`.
- Enable graph optimizations to `ORT_ENABLE_ALL` (default in Android builds).
- Prefer CPU EP with XNNPACK for these models (default in ORT Mobile). INT8 encoder uses per-channel weights; dynamic decoder quant only touches weights.

---

## Performance Guidance

- Broadcast reduces decoder work: one encoder pass per swipe and one decoder call per beam step.
- Static encoder quant (MatMul/Gemm only) gives fast load and good accuracy when calibrated on real data.
- Dynamic decoder quant shrinks size (~4.8 MB) while preserving accuracy (activations remain FP32).
- Keep feature preparation lean on-device: precompute nearest keys if possible or cache keyboard centers and use a simple L2 nearest.

---

## Sanity Check Locally

A small validator script is included at repo root:

```
python test_onnx_inference.py \
  --encoder exported_broadcast_prev_calib_dyn/swipe_encoder_android.onnx \
  --decoder exported_broadcast_prev_calib_dyn/swipe_decoder_android.onnx \
  --data data/train_hwsfuto.jsonl \
  --limit 100
```

This prints gt/pred pairs and a quick accuracy to confirm model integrity.

---

## Known Constraints

- Decoder expects `dec_seq=20` at runtime due to trace-time reshape constants.
- `actual_src_length` must reflect unpadded encoder time steps; the model builds masks internally from it.
- Token indices must match `tokenizer_config.json`.

---

## Version

- Encoder: static INT8 (MatMul/Gemm), calibrated on 10k real traces
- Decoder: dynamic (weights-only) INT8, broadcast-enabled
- Opset: 17
